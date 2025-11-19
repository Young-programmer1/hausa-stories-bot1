#!/usr/bin/env python3
"""
Hausa Stories Bot - COMPLETE MULTI-USER VERSION
Sends stories to ALL users who request them,
optimized by saving audio files locally to avoid repeated TTS generation.
"""

import asyncio
import logging
import sqlite3
import re
import os
import glob
import hashlib
import random
from datetime import datetime
from typing import List, Dict, Optional, Union
import io
from urllib.parse import urljoin
import requests
from bs4 import BeautifulSoup
from gtts import gTTS
from telegram import Update, InputFile, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, ContextTypes, CallbackQueryHandler, MessageHandler, filters
from concurrent.futures import ThreadPoolExecutor
from keep_alive import keep_alive  # Add this

# ========== RAILWAY COMPATIBILITY FIX ==========
import tempfile

# Check if we're running on Railway
if os.environ.get("RAILWAY_ENVIRONMENT"):
    # On Railway, use /tmp directory which persists between restarts
    MANUAL_STORIES_DIR = "/tmp/manual_stories"
    AUDIO_CACHE_DIR = "/tmp/audio_cache"
    print("🚂 Running on Railway - Using /tmp directories")
else:
    # Local development
    MANUAL_STORIES_DIR = "manual_stories"
    AUDIO_CACHE_DIR = "audio_cache"
    print("💻 Running locally - Using local directories")

# Create directories
os.makedirs(MANUAL_STORIES_DIR, exist_ok=True)
os.makedirs(AUDIO_CACHE_DIR, exist_ok=True)

# ========== CONFIGURATION ==========
BOT_TOKEN = os.environ.get("BOT_TOKEN", "YOUR_BOT_TOKEN_HERE")
POLL_INTERVAL = 3600  # 1 hour

# Database path compatibility
if os.environ.get("RAILWAY_ENVIRONMENT"):
    DB_PATH = "/tmp/hausa_stories_bots.db"  # Persists on Railway
else:
    DB_PATH = "hausa_stories_bots.db"  # Local development

ADMIN_USER_ID = 6484243337

# ========== LOGGING SETUP ==========
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Search categories users can choose from
SEARCH_CATEGORIES = {
    "comedy": ["ban dariya", "wasan kwaikwayo", "nishadi", "dariya", "abun dariya", "wasa", "kannywood"],
    "news": ["sabon labari", "labarai", "news", "sabuwar", "yau", "ranar"],
    "sports": ["wasanni", "kwallon kafa", "boxing", "tennis", "gasar", "zakara"],
    "politics": ["siyasa", "gwamnati", "shugaba", "jam'iyya", "zabe", "majalisa"],
    "health": ["lafiya", "kwayoyi", "asibiti", "likita", "cutar", "magani"],
    "education": ["ilimi", "makaranta", "dalibi", "malami", "jami'a", "karatu"],
    "religion": ["addini", "musulunci", "kirista", "salla", "azumi", "hajji"],
    "business": ["kasuwanci", "arzikin", "tattalin arziki", "ciniki", "kuɗi", "banki"],
    "manual": ["manual", "uploaded", "file"]
}

# ========== DATABASE FUNCTIONS ==========


def db_connect():
    """Returns a connection object to the SQLite database."""
    return sqlite3.connect(DB_PATH)


def get_story(story_id: int) -> Optional[Dict]:
    """Retrieves a single story's full details from the database."""
    conn = db_connect()
    c = conn.cursor()
    # Ensure you are selecting the audio_paths column (column 8 in this query)
    c.execute(
        "SELECT id, title, content, link, provider, category, published, is_manual, processed, audio_paths FROM stories WHERE id = ?",
        (story_id,)
    )
    row = c.fetchone()
    conn.close()

    if row:
        # Original query had 8 columns, this query has 10.
        # audio_paths is now at index 9
        audio_paths_list = row[9].split(',') if row[9] else None

        return {
            "id": row[0],
            "title": row[1],
            "content": row[2],
            "link": row[3],
            "provider": row[4],
            "category": row[5],
            "published": row[6],
            "is_manual": bool(row[7]),
            "processed": bool(row[8]),
            "audio_paths": audio_paths_list
        }
    return None


def init_db():
    """Initialize database tables and performs migration."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    # Stories table - CORRECTED: Keep only one CREATE TABLE statement
    c.execute("""CREATE TABLE IF NOT EXISTS stories 
                 (id INTEGER PRIMARY KEY,
                  title TEXT UNIQUE,
                  content TEXT,
                  link TEXT,
                  provider TEXT,
                  category TEXT,
                  published TEXT,
                  audio_sent INTEGER DEFAULT 0,
                  is_manual INTEGER DEFAULT 0,
                  processed INTEGER DEFAULT 0,  -- <--- CORRECT COLUMN
                  audio_paths TEXT DEFAULT '', 
                  created_at TEXT DEFAULT CURRENT_TIMESTAMP)""")

    # Users table
    c.execute("""CREATE TABLE IF NOT EXISTS users 
                 (user_id INTEGER PRIMARY KEY,
                  username TEXT,
                  first_name TEXT,
                  categories TEXT DEFAULT 'comedy,news,manual',
                  is_active INTEGER DEFAULT 1,
                  created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                  last_active TEXT DEFAULT CURRENT_TIMESTAMP)""")

    # Sent stories tracking
    c.execute("""CREATE TABLE IF NOT EXISTS sent_stories 
                 (id INTEGER PRIMARY KEY,
                  user_id INTEGER,
                  story_id INTEGER,
                  sent_at TEXT DEFAULT CURRENT_TIMESTAMP,
                  FOREIGN KEY(user_id) REFERENCES users(user_id),
                  FOREIGN KEY(story_id) REFERENCES stories(id))""")

    # Manual stories tracking
    c.execute("""CREATE TABLE IF NOT EXISTS processed_manual_stories 
                 (id INTEGER PRIMARY KEY,
                  filename TEXT UNIQUE,
                  processed_at TEXT DEFAULT CURRENT_TIMESTAMP)""")

    # --- START DATABASE MIGRATION FIX (Guarantees processed column exists) ---
    try:
        # Check if 'processed' column exists and add it if it doesn't
        c.execute("ALTER TABLE stories ADD COLUMN processed INTEGER DEFAULT 0")
        logger.info(
            "✅ Database migration: Added 'processed' column to 'stories' table.")
    except sqlite3.OperationalError as e:
        # Expected error if the column already exists; safe to ignore.
        if 'duplicate column' not in str(e) and 'already exists' not in str(e):
            logger.error(f"Migration error (processed): {e}")

    # Check if 'audio_paths' column exists and add it if it doesn't (migration for old schemas)
    try:
        c.execute("ALTER TABLE stories ADD COLUMN audio_paths TEXT DEFAULT ''")
        logger.info(
            "✅ Database migration: Added 'audio_paths' column to 'stories' table.")
    except sqlite3.OperationalError as e:
        if 'duplicate column' not in str(e) and 'already exists' not in str(e):
            logger.error(f"Migration error (audio_paths): {e}")
    # --- END DATABASE MIGRATION FIX ---

    conn.commit()
    conn.close()
    logger.info("Database initialized")


def register_user(user_id: int, username: str = "", first_name: str = ""):
    """Register a new user or update existing user (synchronous helper)"""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    try:
        c.execute("""INSERT OR REPLACE INTO users 
                    (user_id, username, first_name, last_active, is_active)
                    VALUES (?, ?, ?, datetime('now'), 1)""",
                  (user_id, username, first_name))
        conn.commit()
        return True
    except Exception as e:
        logger.error(f"Error registering user: {e}")
        return False
    finally:
        conn.close()


def get_active_users() -> List[Dict]:
    """Get all active users (synchronous helper)"""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute(
        "SELECT user_id, username, first_name, categories FROM users WHERE is_active = 1")
    rows = c.fetchall()
    conn.close()

    users = []
    for row in rows:
        users.append({
            "user_id": row[0],
            "username": row[1],
            "first_name": row[2],
            "categories": row[3].split(",") if row[3] else ["comedy", "news", "manual"]
        })
    return users


def update_user_categories(user_id: int, categories: List[str]):
    """Update user's preferred categories (synchronous helper)"""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    try:
        categories_str = ",".join(categories)
        c.execute("UPDATE users SET categories = ?, last_active = datetime('now') WHERE user_id = ?",
                  (categories_str, user_id))
        conn.commit()
        return True
    except Exception as e:
        logger.error(f"Error updating user categories: {e}")
        return False
    finally:
        conn.close()


def get_user_categories(user_id: int) -> List[str]:
    """Get user's preferred categories (synchronous helper)"""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT categories FROM users WHERE user_id = ?", (user_id,))
    row = c.fetchone()
    conn.close()

    if row and row[0]:
        return row[0].split(",")
    return ["comedy", "news", "manual"]


def deactivate_user(user_id: int):
    """Deactivate a user (synchronous helper)"""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("UPDATE users SET is_active = 0 WHERE user_id = ?", (user_id,))
    conn.commit()
    conn.close()

# ========== MANUAL STORIES FUNCTIONS ==========


def get_manual_story_files() -> List[str]:
    """Get all story files from manual stories directory (synchronous helper)"""
    story_files = []
    for ext in ['*.txt', '*.md', '*.story']:
        story_files.extend(glob.glob(os.path.join(MANUAL_STORIES_DIR, ext)))
    return story_files


def is_manual_story_processed(filename: str) -> bool:
    """Check if manual story file has been processed (synchronous helper)"""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute(
        "SELECT 1 FROM processed_manual_stories WHERE filename = ?", (filename,))
    res = c.fetchone()
    conn.close()
    return res is not None


def mark_manual_story_processed(filename: str):
    """Mark manual story file as processed (synchronous helper)"""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    try:
        c.execute(
            "INSERT INTO processed_manual_stories (filename) VALUES (?)", (filename,))
        conn.commit()
    except Exception as e:
        logger.error(f"Error marking manual story as processed: {e}")
    finally:
        conn.close()


def read_manual_story_file(filepath: str) -> Dict[str, str]:
    """Read and parse manual story file (synchronous helper)"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read().strip()

        # Simple parsing: first line is title, rest is content
        lines = content.split('\n')
        title = lines[0].strip()
        story_content = '\n'.join(lines[1:]).strip()

        if not title or not story_content:
            logger.error(f"Invalid story file format: {filepath}")
            return None

        return {
            "title": title,
            "content": story_content,
            "filename": os.path.basename(filepath)
        }
    except Exception as e:
        logger.error(f"Error reading manual story file {filepath}: {e}")
        return None


def process_manual_stories() -> List[Dict]:
    """Process all new manual story files (synchronous helper)"""
    new_stories = []
    story_files = get_manual_story_files()

    for filepath in story_files:
        filename = os.path.basename(filepath)

        if not is_manual_story_processed(filename):
            story_data = read_manual_story_file(filepath)
            if story_data:
                new_stories.append({
                    "title": story_data["title"],
                    "content": story_data["content"],
                    "link": f"file://{filename}",
                    "provider": "Manual Upload",
                    "category": "manual",
                    "published": datetime.utcnow().isoformat(),
                    "is_manual": True
                })
                mark_manual_story_processed(filename)
                logger.info(f"✅ Processed manual story: {story_data['title']}")

    return new_stories


def story_exists(title: str) -> bool:
    """Check if story already exists (synchronous helper)"""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT 1 FROM stories WHERE title = ?", (title,))
    res = c.fetchone()
    conn.close()
    return res is not None


def save_story(title: str, content: str, link: str, provider: str, category: str, published: str, is_manual: bool = False):
    """Save story to database (synchronous helper)"""
    if story_exists(title):
        return False

    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    try:
        c.execute("""INSERT INTO stories 
                    (title, content, link, provider, category, published, is_manual)
                    VALUES (?, ?, ?, ?, ?, ?, ?)""",
                  (title, content, link, provider, category, published, 1 if is_manual else 0))
        conn.commit()
        return c.lastrowid
    except sqlite3.IntegrityError:
        return False
    finally:
        conn.close()


def is_story_sent_to_user(story_id: int, user_id: int) -> bool:
    """Check if story was already sent to user (synchronous helper)"""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT 1 FROM sent_stories WHERE story_id = ? AND user_id = ?",
              (story_id, user_id))
    res = c.fetchone()
    conn.close()
    return res is not None


def mark_story_sent_to_user(story_id: int, user_id: int):
    """Mark story as sent to user (synchronous helper)"""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    try:
        c.execute(
            "INSERT INTO sent_stories (story_id, user_id) VALUES (?, ?)", (story_id, user_id))
        conn.commit()
    except Exception as e:
        logger.error(f"Error marking story as sent: {e}")
    finally:
        conn.close()


def get_unsent_stories_for_user(user_id: int, categories: List[str], limit: int = 5) -> List[Dict]:
    """Get stories that haven't been sent to user and match their categories (synchronous helper)"""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    placeholders = ",".join("?" * len(categories))

    c.execute(f"""
        SELECT s.id, s.title, s.content, s.link, s.provider, s.category, s.published, s.is_manual, s.audio_paths, s.processed
        FROM stories s
        WHERE s.processed = 1 -- Only send processed stories
        AND s.category IN ({placeholders})
        AND s.id NOT IN (
            SELECT story_id FROM sent_stories WHERE user_id = ?
        )
        ORDER BY s.id DESC
        LIMIT ?
    """, categories + [user_id, limit])

    rows = c.fetchall()
    conn.close()

    stories = []
    for row in rows:
        # Check index 8 for audio_paths
        audio_paths = row[8].split(",") if row[8] else []
        stories.append({
            "id": row[0],
            "title": row[1],
            "content": row[2],
            "link": row[3],
            "provider": row[4],
            "category": row[5],
            "published": row[6],
            "is_manual": bool(row[7]),
            "processed": bool(row[9]),
            # Only return existing paths
            "audio_paths": [p for p in audio_paths if os.path.exists(p)]
        })
    return stories


def get_new_stories(limit: int = 10) -> List[Dict]:
    """Get newly added stories that haven't been processed (synchronous helper)"""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        SELECT id, title, content, link, provider, category, published, is_manual
        FROM stories 
        WHERE processed = 0 
        ORDER BY id ASC
        LIMIT ?
    """, (limit,))

    rows = c.fetchall()
    conn.close()

    stories = []
    for row in rows:
        stories.append({
            "id": row[0],
            "title": row[1],
            "content": row[2],
            "link": row[3],
            "provider": row[4],
            "category": row[5],
            "published": row[6],
            "is_manual": bool(row[7])
        })
    return stories


def mark_story_processed_and_save_paths(story_id: int, audio_paths: List[str]):
    """Marks a story as processed and saves the file paths to the database. (synchronous helper)"""
    conn = db_connect()
    c = conn.cursor()
    # Serialize the list of paths into a single comma-separated string
    paths_str = ",".join(audio_paths)
    c.execute(
        # Set audio_sent to 1 after processing
        "UPDATE stories SET processed = 1, audio_paths = ?, audio_sent = 1 WHERE id = ?",
        (paths_str, story_id)
    )
    conn.commit()
    conn.close()


def get_story_history(user_id: int, limit: int = 10, offset: int = 0) -> List[Dict]:
    """Get user's story history (synchronous helper)"""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        SELECT s.id, s.title, s.content, s.link, s.provider, s.category, s.published, ss.sent_at, s.audio_paths
        FROM stories s
        JOIN sent_stories ss ON s.id = ss.story_id
        WHERE ss.user_id = ?
        ORDER BY ss.sent_at DESC
        LIMIT ? OFFSET ?
    """, (user_id, limit, offset))

    rows = c.fetchall()
    conn.close()

    stories = []
    for row in rows:
        audio_paths = row[8].split(",") if row[8] else []
        stories.append({
            "id": row[0],
            "title": row[1],
            "content": row[2],
            "link": row[3],
            "provider": row[4],
            "category": row[5],
            "published": row[6],
            "sent_at": row[7],
            "audio_paths": [p for p in audio_paths if os.path.exists(p)]
        })
    return stories

# ========== CONTENT FETCHING ==========


def fetch_complete_story_content(url: str) -> str:
    """Fetch complete story content from URL (synchronous helper)"""
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        }
        response = requests.get(url, timeout=15, headers=headers)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, "html.parser")

        for script in soup(["script", "style"]):
            script.decompose()

        article = soup.find("article") or soup.find(
            "div", class_=re.compile(r"article|content|story", re.I))

        if article:
            paragraphs = article.find_all("p")
            content = " ".join([p.get_text(strip=True)
                               for p in paragraphs if p.get_text(strip=True)])

            if len(content) > 100:
                return content

        text_elements = soup.find_all(
            ["p", "div"], class_=re.compile(r"content|text|body", re.I))
        content = " ".join([elem.get_text(strip=True) for elem in text_elements if len(
            elem.get_text(strip=True)) > 50])

        return content if len(content) > 100 else ""

    except Exception as e:
        logger.error(f"Error fetching story content from {url}: {e}")
        return ""


def fetch_hausa_stories() -> List[Dict]:
    """Fetch Hausa stories from various sources (synchronous helper)"""
    all_stories = []

    # Legit.ng Hausa
    try:
        url = "https://hausa.legit.ng"
        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(url, timeout=15, headers=headers)
        soup = BeautifulSoup(response.text, "html.parser")

        links = soup.find_all("a", href=True)

        for link in links[:25]:
            title = link.get_text(strip=True)
            href = link["href"]

            if title and len(title) > 20:
                if href.startswith("/"):
                    href = urljoin(url, href)
                elif not href.startswith("http"):
                    continue

                # FIX: Wrap synchronous network I/O
                content = fetch_complete_story_content(href)

                if content and not story_exists(title):
                    all_stories.append({
                        "title": title,
                        "content": content,
                        "link": href,
                        "provider": "Legit.ng Hausa",
                        "category": "news",
                        "published": datetime.utcnow().isoformat()
                    })

    except Exception as e:
        logger.error(f"Error fetching from Legit.ng: {e}")

    return all_stories

# ========== AUDIO GENERATION (MODIFIED TO SAVE FILES) ==========


def generate_and_save_audio(story: Dict) -> Optional[List[str]]:
    """Generate complete audio for a story and save it to the cache directory (synchronous helper)"""
    try:
        full_text = f"{story['title']}. {story['content']}"

        full_text = re.sub(r'http\S+', '', full_text)
        full_text = re.sub(r'\s+', ' ', full_text).strip()

        if len(full_text) < 50:
            return None

        # Use a hash of the content to generate a unique filename
        content_hash = hashlib.sha1(full_text.encode('utf-8')).hexdigest()

        # Handle very long stories by splitting
        if len(full_text) > 25000:
            chunks = []
            current_chunk = ""
            sentences = full_text.split('. ')
            for sentence in sentences:
                # INCREASED CHUNK SIZE: Each audio file will contain up to 4500 characters
                if len(current_chunk + sentence) < 4500:
                    current_chunk += sentence + '. '
                else:
                    if current_chunk:
                        chunks.append(current_chunk.strip())
                    current_chunk = sentence + '. '
            if current_chunk:
                chunks.append(current_chunk.strip())

            audio_paths = []
            for i, chunk in enumerate(chunks):
                if len(chunk) < 50:
                    continue

                tts = gTTS(text=chunk, lang='ha', slow=False)

                # Save the audio file
                filename = f"{content_hash}_part{i+1}.mp3"
                filepath = os.path.join(AUDIO_CACHE_DIR, filename)
                tts.save(filepath)
                audio_paths.append(filepath)

            logger.info(
                f"Generated and saved {len(audio_paths)} audio parts for: {story['title'][:40]}...")
            return audio_paths
        else:
            # Normal story length
            tts = gTTS(text=full_text, lang='ha', slow=False)

            # Save the single audio file
            filename = f"{content_hash}.mp3"
            filepath = os.path.join(AUDIO_CACHE_DIR, filename)
            tts.save(filepath)

            logger.info(
                f"Generated and saved audio for: {story['title'][:40]}...")
            return [filepath]

    except Exception as e:
        logger.error(f"Audio generation error: {e}")
        return None
# ========== BOT CORE FUNCTIONALITY ==========


class HausaStoriesBot:
    def __init__(self):
        self.application = None
        # self.background_task = None # REMOVED: Job Queue handles this
        self.generation_locks: Dict[int, asyncio.Lock] = {}
        self.executor = ThreadPoolExecutor(max_workers=4)

    async def check_and_send_stories(self):
        """Main function to check for and send stories to ALL users (fully non-blocking)"""
        logger.info("Hausa Stories Bot background task started...")

        try:
            # 1. Process manual stories (SLOW: File I/O, DB Read/Write)
            logger.info("Scanning and processing new manual stories...")
            # FIX 1: Wrap the entire synchronous function call.
            manual_stories = await asyncio.to_thread(process_manual_stories)

            for story in manual_stories:
                # FIX 2: Wrap the synchronous save_story DB write.
                story_id = await asyncio.to_thread(
                    save_story, story["title"], story["content"], story["link"],
                    story["provider"], story["category"], story["published"], True
                )
                if story_id:
                    logger.info(
                        f"✅ Saved manual story: {story['title'][:50]}...")

            # 2. Fetch web stories (SLOW: Network I/O)
            logger.info("Starting web scraping...")
            # FIX 3: Wrap the synchronous web scraping function.
            web_stories = await asyncio.to_thread(fetch_hausa_stories)

            for story in web_stories:
                # FIX 4: Wrap the synchronous save_story DB write.
                story_id = await asyncio.to_thread(
                    save_story, story["title"], story["content"], story["link"],
                    story["provider"], story["category"], story["published"], False
                )
                if story_id:
                    logger.info(
                        f"Saved web story: {story['title'][:50]}...")

            # 3. Process new stories (generate audio and save)
            # FIX 5: get_new_stories is a synchronous DB read, wrap it.
            new_stories_to_process = await asyncio.to_thread(get_new_stories, limit=10)

            for story in new_stories_to_process:
                # Audio generation (synchronous network/disk I/O moved to thread)
                audio_file_paths = await asyncio.to_thread(generate_and_save_audio, story)

                # 💡 CRUCIAL FIX for Freezing/Endless Retries:
                # Determine paths to save (empty list if audio failed)
                paths_to_save = audio_file_paths if audio_file_paths else []

                # Update DB with processed status and file paths (synchronous operation moved to thread).
                # This ensures the story is marked PROCESSED even if the audio failed, preventing endless retries.
                await asyncio.to_thread(mark_story_processed_and_save_paths,
                                        story["id"], paths_to_save)

                if paths_to_save:
                    logger.info(
                        f"✅ Processed and saved audio for story: {story['title'][:40]}...")
                else:
                    logger.warning(
                        f"❌ Failed to generate audio for story {story['id']}. Marking as processed to prevent endless retry.")

            # 4. Send stories to ALL active users
            # FIX 7: Wrap the synchronous DB read operation.
            active_users = await asyncio.to_thread(get_active_users)
            logger.info(f"📍 Found {len(active_users)} active users")

            total_stories_sent = 0

            for user in active_users:
                user_id = user["user_id"]
                categories = user["categories"]

                # FIX 8: Wrap the synchronous DB read operation.
                unsent_stories = await asyncio.to_thread(get_unsent_stories_for_user,
                                                         user_id, categories, limit=5)

                if unsent_stories:
                    logger.info(
                        f"Sending {len(unsent_stories)} stories to user {user_id}")

                for story in unsent_stories:
                    if not story.get("audio_paths"):
                        logger.warning(
                            f"Story {story['id']} has no valid audio paths. Skipping send.")
                        continue

                    # send_story_to_user contains its own lock logic
                    success = await self.send_story_to_user(story, user_id)

                    if success:
                        # FIX 9: Wrap the synchronous DB write operation.
                        await asyncio.to_thread(mark_story_sent_to_user, story["id"], user_id)
                        total_stories_sent += 1
                        logger.info(
                            f"✅ Sent to user {user_id}: {story['title'][:30]}...")
                    # Delay between sends (this is fine in async)
                    await asyncio.sleep(2)

            logger.info(
                f"🎉 Cycle complete. Sent {total_stories_sent} stories to {len(active_users)} users")

        except Exception as e:
            logger.error(f"Error in main loop: {e}")

    async def check_and_send_stories_job(self, context: ContextTypes.DEFAULT_TYPE):
        """Job Queue wrapper that calls the main checking logic."""
        await self.check_and_send_stories()

    async def post_init(self, application: Application) -> None:
        """Runs once the bot is fully initialized to schedule the background task (Job Queue FIX)."""
        # --- CORRECT JOB QUEUE IMPLEMENTATION ---
        self.application = application
        application.job_queue.run_repeating(
            self.check_and_send_stories_job,  # The wrapper function defined above
            interval=POLL_INTERVAL,  # Uses your POLL_INTERVAL (3600 seconds)
            first=5,  # Start 5 seconds after initialization
            name="stories_fetch_and_send_job"
        )
        logger.info("✅ Background job scheduled using Job Queue.")
        # --- END FIX ---

    async def send_story_to_user(self, story: Dict, user_id: int) -> bool:
        """Send story to specific user using cached audio files, with concurrency lock."""
        try:
            story_id = story.get("id")
            if not story_id:
                logger.error(
                    "Story dictionary is missing 'id'. Cannot process.")
                return False

            audio_paths = story.get("audio_paths")

            # --- START CONCURRENCY LOCK LOGIC ---
            if not audio_paths:
                lock = self.generation_locks.get(story_id)
                if not lock:
                    lock = asyncio.Lock()
                    self.generation_locks[story_id] = lock

                logger.info(
                    f"User {user_id} waiting for lock on story {story_id}...")

                async with lock:
                    logger.info(
                        f"User {user_id} acquired lock for story {story_id}.")

                    # FIX: Wrap synchronous DB read
                    story_update = await asyncio.to_thread(get_story, story_id)
                    audio_paths = story_update.get("audio_paths")

                    if not audio_paths or story_update.get("processed") == 0:
                        logger.info(
                            f"⏳ Story {story_id} not cached. Generating audio...")

                        # Audio generation is already wrapped
                        audio_paths = await asyncio.to_thread(generate_and_save_audio, story)

                        if audio_paths:
                            # FIX: Wrap synchronous DB write
                            await asyncio.to_thread(mark_story_processed_and_save_paths,
                                                    story_id, audio_paths)
                            logger.info(
                                f"✅ Generated and saved audio for story {story_id}.")
                    else:
                        logger.info(
                            f"✅ Story {story_id} generated by another thread. Sending cached.")

            # --- END CONCURRENCY LOCK LOGIC ---

            if audio_paths and self.application:
                if story.get("is_manual"):
                    emoji = "📝"
                    source = "Labarin da aka shigar"
                else:
                    emoji = "📰"
                    source = story["provider"]

                for i, audio_path in enumerate(audio_paths):
                    # NOTE: File open/read is fast, keeping it in the main thread for simplicity is often fine,
                    # but for absolute safety, it could be wrapped if files were extremely large.
                    with open(audio_path, 'rb') as audio_file:
                        if len(audio_paths) > 1:
                            caption = (
                                f"{emoji} *LABARI NA HAUSA* {emoji}\n\n"
                                f"*Take:* {story['title']}\n"
                                f"*Sashi:* {i+1} na {len(audio_paths)}\n"
                                f"*Mai Kawo:* {source}\n"
                                f"*Lokaci:* {datetime.now().strftime('%H:%M')}\n\n"
                                f"_An fassara duka labarin ta hanyar sauti_"
                            )
                        else:
                            caption = (
                                f"{emoji} *LABARI NA HAUSA* {emoji}\n\n"
                                f"*Take:* {story['title']}\n"
                                f"*Mai Kawo:* {source}\n"
                                f"*Lokaci:* {datetime.now().strftime('%H:%M')}\n\n"
                                f"_An fassara duka labarin ta hanyar sauti_"
                            )

                        await self.application.bot.send_audio(
                            chat_id=user_id,
                            audio=InputFile(
                                audio_file, filename=os.path.basename(audio_path)),
                            caption=caption,
                            parse_mode="Markdown"
                        )
                    await asyncio.sleep(1)

                return True
            else:
                return False

        except Exception as e:
            logger.error(f"Error sending to user {user_id}: {e}")
            return False

    async def start_cmd(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /start command"""
        user = update.effective_user

        # FIX: Wrap synchronous DB write
        await asyncio.to_thread(register_user, user.id, user.username, user.first_name)

        welcome_msg = (
            f"Barka da zuwa `{user.first_name}`! 🎧\n\n"
            "**Hausa Stories Bot** ya fara aiki!\n\n"
            "Zan tuntube ku da *sababbin labarai* a cikin harshen Hausa "
            "a matsayin sautin audio cikakke.\n\n"
            "Ku zaɓi nau'in labarai da kuke so:"
        )

        keyboard = [
            # CORRECTED EMOJIS:
            [InlineKeyboardButton("😂 Comedy", callback_data="cat_comedy"),  # ðŸ˜‚ -> 😂
             InlineKeyboardButton("📰 News", callback_data="cat_news")],  # ðŸ“° -> 📰
            [InlineKeyboardButton("⚽ Sports", callback_data="cat_sports"),  # âš½ -> ⚽
             InlineKeyboardButton("🏛️ Politics", callback_data="cat_politics")],  # ðŸ›ï¸ -> 🏛️
            [InlineKeyboardButton("📝 Manual Stories",  # ðŸ“ -> 📝
                                  callback_data="cat_manual")],
            # ðŸ’¼ -> 💼
            [InlineKeyboardButton("💼 Duka Labarai", callback_data="cat_all")]
        ]

        reply_markup = InlineKeyboardMarkup(keyboard)

        await update.message.reply_text(welcome_msg, reply_markup=reply_markup, parse_mode="Markdown")

    async def handle_category_selection(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle category selection"""
        query = update.callback_query
        await query.answer()

        user_id = query.from_user.id
        data = query.data

        if data == "cat_all":
            categories = ["comedy", "news", "sports", "politics", "manual"]
            # FIX: Wrap synchronous DB write
            await asyncio.to_thread(update_user_categories, user_id, categories)
            await query.edit_message_text(
                "✅ Kun zaɓi duka nau'in labarai! Zan tuntube ku da duk sababbin labarai.\n\n"
                "Yi amfani da /nauina don canza zaɓin ku ko /taimako don duk umarni."
            )
        elif data == "cat_manual":
            # FIX: Wrap synchronous DB write
            await asyncio.to_thread(update_user_categories, user_id, ["manual"])
            await query.edit_message_text(
                "✅ Kun zaɓi labaran da aka shigar! Zan tuntube ku da sababbin labaran da aka shigar.\n\n"
                "Yi amfani da /nauina don canza zaɓin ku ko /taimako don duk umarni."
            )
        else:
            category = data.replace("cat_", "")
            # FIX: Wrap synchronous DB write
            await asyncio.to_thread(update_user_categories, user_id, [category])
            await query.edit_message_text(
                f"✅ Kun zaɓi labaran {category}! Zan tuntube ku da sababbin labarai na {category}.\n\n"
                "Yi amfani da /nauina don canza zaɓin ku ko /taimako don duk umarni."
            )

        # Send immediate stories to this user
        await query.message.reply_text("🔍 Ina neman sababbin labarai don ku...")
        await self.send_immediate_stories(user_id)

    async def send_immediate_stories(self, user_id: int):
        """Send immediate stories to a specific user"""
        try:
            # Calls get_user_categories() which is synchronous DB call
            categories = await asyncio.to_thread(get_user_categories, user_id)

            # Calls get_unsent_stories_for_user() which is synchronous DB call
            unsent_stories = await asyncio.to_thread(get_unsent_stories_for_user,
                                                     user_id, categories, limit=3)

            for story in unsent_stories:
                if not story.get("audio_paths"):
                    # This could happen if the audio hasn't been processed yet
                    logger.info(
                        f"Story {story['id']} audio not ready for immediate send.")
                    continue

                success = await self.send_story_to_user(story, user_id)
                if success:
                    # Calls mark_story_sent_to_user() which is synchronous DB call
                    await asyncio.to_thread(mark_story_sent_to_user, story["id"], user_id)
                    logger.info(
                        f"📨 Immediate send to user {user_id}: {story['title'][:30]}...")
                await asyncio.sleep(2)

        except Exception as e:
            logger.error(
                f"Error sending immediate stories to user {user_id}: {e}")

    async def categories_cmd(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /nauina command - change categories"""
        keyboard = [
            [InlineKeyboardButton("😂 Comedy", callback_data="cat_comedy"),
             InlineKeyboardButton("📰 News", callback_data="cat_news")],
            [InlineKeyboardButton("⚽ Sports", callback_data="cat_sports"),
             InlineKeyboardButton("🏛️ Politics", callback_data="cat_politics")],
            [InlineKeyboardButton("📝 Manual Stories",
                                  callback_data="cat_manual")],
            [InlineKeyboardButton("💼 Duka Labarai", callback_data="cat_all")]
        ]

        reply_markup = InlineKeyboardMarkup(keyboard)

        await update.message.reply_text(
            "Zaɓi nau'in labarai da kuke so in tuntube ku da su:",
            reply_markup=reply_markup
        )

    async def search_cmd(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /nema command - search for stories immediately"""
        user_id = update.effective_user.id
        await update.message.reply_text("🔍 Ina neman sababbin labarai yanzu...")

        # 1. Force process manual stories immediately (save to DB) - WRAPPED FOR ASYNC
        manual_stories = await asyncio.to_thread(process_manual_stories)

        for story in manual_stories:
            story_id = await asyncio.to_thread(save_story, story["title"], story["content"], story["link"],
                                               story["provider"], story["category"], story["published"], True)
            if story_id:
                logger.info(
                    f"✅ Immediate save manual story: {story['title'][:50]}...")

        # 2. Process NEWLY saved stories (Generate and save audio, update DB)
        new_stories_to_process = await asyncio.to_thread(get_new_stories, limit=10)

        for story in new_stories_to_process:
            audio_file_paths = await asyncio.to_thread(generate_and_save_audio, story)

            # 💡 CRUCIAL FIX: Ensure the story is marked processed (processed=1) even if audio fails
            paths_to_save = audio_file_paths if audio_file_paths else []

            # Always call the function to update the DB, preventing endless retries
            await asyncio.to_thread(mark_story_processed_and_save_paths,
                                    story["id"], paths_to_save)

            if paths_to_save:
                logger.info(
                    f"✅ Processed audio immediately: {story['title'][:40]}...")
            else:
                logger.warning(
                    f"❌ Failed to generate audio for story {story['id']} during immediate search. Marking as processed.")  # New Log

        # 3. Send immediate stories to this user
        await self.send_immediate_stories(user_id)

        await update.message.reply_text("✅ An aiko muku da sababbin labarai!")

    async def history_cmd(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /tsoho command - show history"""
        user_id = update.effective_user.id
        # FIX: Wrap synchronous DB read
        stories = await asyncio.to_thread(get_story_history, user_id, limit=5)

        if not stories:
            await update.message.reply_text("📜 Babu labaran da kuka taɓa karɓa tukuna.")
            return

        keyboard = []
        for i, story in enumerate(stories, 1):
            short_title = story['title'][:35] + \
                "..." if len(story['title']) > 35 else story['title']
            keyboard.append([InlineKeyboardButton(
                f"{i}. {short_title}", callback_data=f"hist_{story['id']}")])

        keyboard.append([InlineKeyboardButton(
            "❌ Rufe", callback_data="close_history")])

        reply_markup = InlineKeyboardMarkup(keyboard)

        await update.message.reply_text(
            "📜 Zaɓi labarin da kuke so ku sake ji:",
            reply_markup=reply_markup
        )

    async def handle_history_selection(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle history story selection"""
        query = update.callback_query
        await query.answer()

        if query.data == "close_history":
            await query.message.delete()
            return

        story_id = int(query.data.replace("hist_", ""))
        user_id = query.from_user.id

        # FIX: Wrap synchronous DB read
        stories = await asyncio.to_thread(get_story_history, user_id, limit=50)
        target_story = next((s for s in stories if s['id'] == story_id), None)

        if target_story and target_story.get("audio_paths"):
            await query.message.reply_text("🎧 Ina aiko muku da labarin...")
            # Use the saved audio paths for resend
            await self.send_story_to_user(target_story, user_id)
            # FIX: Wrap synchronous DB write
            await asyncio.to_thread(mark_story_sent_to_user, story_id, user_id)
        else:
            await query.message.reply_text("❌ Labarin ya ɓace ko kuma sautinsa bai shiryu ba tukuna.")

        await query.message.delete()

    async def stop_cmd(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /daina command - stop receiving stories"""
        user_id = update.effective_user.id
        # FIX: Wrap synchronous DB write
        await asyncio.to_thread(deactivate_user, user_id)

        await update.message.reply_text(
            "✅ An daina aiko muku da labarai.\n\n"
            "Idan kuna son sake farawa, yi amfani da /fara"
        )

    async def stats_cmd(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /stats command - show bot statistics"""
        # FIX: Wrap synchronous DB read
        active_users = await asyncio.to_thread(get_active_users)
        await update.message.reply_text(
            f"📊 *Bayanan Bot*\n\n"
            f"• Masu amfani: *{len(active_users)}*\n"
            f"• Nau'in labarai: Comedy, News, Sports, Politics, Manual\n\n"
            f"Kowane mai amfani zai karɓi duk sababbin labarai!",
            parse_mode="Markdown"
        )

    async def help_cmd(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /taimako command"""
        help_text = (
            "🎧 *Hausa Stories Bot - Taimako*\n\n"
            "*Umarni:*\n"
            "/fara - Fara bot ko sake farawa\n"
            "/nauina - Canza nau'in labarai\n"
            "/nema - Nemi sababbin labarai yanzu\n"
            "/tsoho - Duba labaran da suka wuce\n"
            "/stats - Nuna bayanan bot\n"
            "/daina - Daina karɓar labarai\n"
            "/taimako - Wannan bayanin\n\n"
            "*Nau'in Labarai:*\n"
            "• Comedy - Labaran ban dariya\n"
            "• News - Sababbin labarai\n"
            "• Sports - Labaran wasanni\n"
            "• Politics - Labaran siyasa\n"
            "• Manual - Labaran da aka shigar\n\n"
            "**Kowane mai amfani zai karɓi duk sababbin labarai!** 🎯"
        )
        await update.message.reply_text(help_text, parse_mode="Markdown")

    async def add_story_cmd(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /shiga command - Admin command to add a story via Telegram."""
        user_id = update.effective_user.id

        # 1. ADMIN RESTRICTION CHECK (The core security feature)
        global ADMIN_USER_ID  # Need to access the global var
        if user_id != ADMIN_USER_ID:
            logger.warning(f"Unauthorized use of /shiga by user {user_id}")
            await update.message.reply_text("❌ Ba a yarda ka yi amfani da wannan umarnin ba.")
            return

        # 2. Argument Check
        if not context.args or '|' not in " ".join(context.args):
            await update.message.reply_text(
                "📝 *Yadda ake shigarwa:*\n\n"
                "Yi amfani da wannan tsari:\n"
                "`/shiga Take na Labari | Dukkanin rubutun labarin`\n\n"
                "Misali:\n"
                "`/shiga Kwallon Kafa | Labari ne game da gasar karshe a Kano...`",
                parse_mode="Markdown"
            )
            return

        # Combine all arguments into one string to ensure the full message is captured
        full_text = " ".join(context.args)

        try:
            # 3. Parse Title and Content
            title, content = full_text.split('|', 1)
            title = title.strip()
            content = content.strip()
        except ValueError:
            await update.message.reply_text("❌ Ba a iya raba Take da Rubutu ba. Tabbatar da amfani da '|' guda ɗaya.")
            return

        # Get the word count
        word_count = len(content.split())

        # Check if the story is rejected
        is_rejected = False
        rejection_reason = ""

        # Check 1: Title
        if not title.strip() or len(title.strip()) < 5:
            is_rejected = True
            rejection_reason = "Take babu ko gajere ne (ƙasa da haruffa 5)."
        # Check 2: Content Word Count
        elif word_count < 50:
            is_rejected = True
            rejection_reason = f"Rubutun gajere ne. An samu *kalmomi {word_count}* kawai. Labarin ya buƙaci aƙalla *kalmomi 50*."

        # --- NEW LOGIC: Only show error if rejected ---
        if is_rejected:
            await update.message.reply_text(
                f"❌ An kasa shigar da labari:\n\n{rejection_reason}",
                parse_mode="Markdown"
            )
            return

        # If it reaches here, the story is valid
        await update.message.reply_text(f"⏳ Labari yayi kyau. Ina shigarwa da fara fassara sauti na *{title}*...", parse_mode="Markdown")
        # --- NEW LOGIC: Only show error if rejected ---
        if is_rejected:
            await update.message.reply_text(
                f"❌ An kasa shigar da labari:\n\n{rejection_reason}",
                parse_mode="Markdown"
            )
            return

        # If it reaches here, word_count >= 50 and title is valid!
        await update.message.reply_text(f"⏳ Labari yayi kyau. Ina shigarwa da fara fassara sauti na *{title}*...", parse_mode="Markdown")
        # 4. Create Story Dictionary
        story = {
            "title": title,
            "content": content,
            "link": f"admin_upload://{datetime.utcnow().timestamp()}",
            "provider": f"Admin ({user_id})",
            "category": "manual",
            "published": datetime.utcnow().isoformat(),
            "is_manual": True,
            "id": None  # Will be populated after saving
        }

        # 5. Save to DB (Async Wrapper)
        story_id = await asyncio.to_thread(
            save_story, story["title"], story["content"], story["link"],
            story["provider"], story["category"], story["published"], True
        )

        if not story_id:
            await update.message.reply_text("❌ Labari ya riga ya wanzu a cikin database.")
            return

        story['id'] = story_id

        # 6. Generate Audio (Async Wrapper)
        audio_file_paths = await asyncio.to_thread(generate_and_save_audio, story)

        if audio_file_paths:
            # 7. Mark Processed and Save Paths (Async Wrapper)
            await asyncio.to_thread(
                mark_story_processed_and_save_paths, story_id, audio_file_paths
            )
            await update.message.reply_text(
                f"✅ Labari *{title}* ya shirya! An shigar da shi, an fassara shi zuwa sauti, kuma zai iya zama rarraba wa masu amfani.",
                parse_mode="Markdown"
            )
            # Send immediately to the admin for review
            story['audio_paths'] = audio_file_paths
            await self.send_story_to_user(story, user_id)
        else:
            await update.message.reply_text("❌ An shigar da Labari, amma an kasa samar da sauti. Yana iya zama gajere ko matsala ta gTTS.")

    async def add_story_file_cmd(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """
        Handles text file uploads from the admin using a REPLY command.
        RUNS HEAVY PROCESSING IN BACKGROUND TO PREVENT FREEZING
        """
        user_id = update.effective_user.id
        global ADMIN_USER_ID

        # 1. Get Title from command arguments
        title = " ".join(context.args).strip()

        if user_id != ADMIN_USER_ID:
            logger.warning(f"Unauthorized use of /littafi by user {user_id}")
            return

        # 2. Basic Command Validation
        if not title:
            await update.message.reply_text("❌ Ba a yarda. Dole ne ka saka taken labarin: `/littafi Take na Labari`.", parse_mode="Markdown")
            return

        # 3. Check for Reply and Document
        if not update.message.reply_to_message:
            await update.message.reply_text(
                "❌ Dole ne ka yi **reply** ga fayil din rubutu (.txt) tare da umarnin `/littafi Take na Labari`.",
                parse_mode="Markdown"
            )
            return

        replied_message = update.message.reply_to_message
        document = replied_message.document

        if not document:
            await update.message.reply_text("❌ Fayil din da ka yi reply ba file na rubutu bane. Dole ne ya zama file (.txt).")
            return

        # 4. File Type Validation
        if document.mime_type not in ('text/plain', 'application/octet-stream') and not document.file_name.lower().endswith(('.txt', '.rtf')):
            await update.message.reply_text("❌ Dole ne ya zama file na rubutu (.txt).")
            return

        # IMMEDIATE FEEDBACK: Tell the admin we started processing
        await update.message.reply_text(f"⏳ Labari yayi kyau. Ina saukewa da fara fassara sauti na file *{document.file_name}*...", parse_mode="Markdown")

        # STORE THE DATA FOR BACKGROUND PROCESSING
        file_data = {
            'title': title,
            'document': document,
            'user_id': user_id,
            'update': update
        }

        # RUN HEAVY PROCESSING IN BACKGROUND TO PREVENT FREEZING
        asyncio.create_task(self.process_file_in_background(file_data))

    async def process_file_in_background(self, file_data):
        """Process file upload in background without blocking other users"""
        try:
            title = file_data['title']
            document = file_data['document']
            user_id = file_data['user_id']
            update = file_data['update']

            # 5. File Download (Non-blocking)
            file_obj = await document.get_file()

            # Prepare an in-memory buffer
            file_buffer = io.BytesIO()

            # AWAITING download_to_memory DIRECTLY (Fixes RuntimeWarning)
            await file_obj.download_to_memory(file_buffer)

            # Reset the buffer's cursor to the beginning (Fixes "kalmomi 0")
            file_buffer.seek(0)

            # Get content from the buffer and decode it
            content = file_buffer.getvalue().decode('utf-8')

            # 6. Final Validation (Word Count Check)
            word_count = len(content.split())

            if len(title) < 5 or word_count < 50:
                await update.message.reply_text(
                    f"❌ Take babu/gajere (kasa da 5 haruffa), ko kuma labarin gajere ne. An samu *kalmomi {word_count}* kawai.",
                    parse_mode="Markdown"
                )
                return

            # 7. Create Story Dictionary
            content_hash = hashlib.sha1(content.encode('utf-8')).hexdigest()
            story = {
                "title": title,
                "content": content,
                "link": f"admin_file://{content_hash}",
                "provider": f"Admin File ({user_id})",
                "category": "manual",
                "published": datetime.utcnow().isoformat(),
                "is_manual": True,
                "id": None
            }

            # --- CONCURRENCY FIX: Using asyncio.to_thread to avoid PTB AttributeErrors ---
            # 8. Save Story to DB (Offload blocking task)
            story_id = await asyncio.to_thread(
                save_story, story["title"], story["content"], story["link"],
                story["provider"], story["category"], story["published"], True
            )

            if not story_id:
                await update.message.reply_text("❌ Labari ya riga ya wanzu a cikin database.")
                return

            story['id'] = story_id

            # 9. Generate Audio (Offload blocking task)
            audio_file_paths = await asyncio.to_thread(
                generate_and_save_audio, story
            )

            # 10. Mark as Processed (Offload blocking task)
            paths_to_save = audio_file_paths if audio_file_paths else []
            await asyncio.to_thread(
                mark_story_processed_and_save_paths, story_id, paths_to_save
            )
            # --------------------------------------------------------------------------

            if paths_to_save:
                await update.message.reply_text(
                    f"✅ Labari *{title}* ya shirya! An shigar da labarin file mai tsawon {word_count} kalmomi.",
                    parse_mode="Markdown"
                )
                # Send immediately to the admin for review
                story['audio_paths'] = paths_to_save
                await self.send_story_to_user(story, user_id)
            else:
                await update.message.reply_text("❌ An shigar da Labari, amma an kasa samar da sauti. Yana iya zama gajere ko matsala ta gTTS. An riga an yi alama a matsayin an sarrafa shi.")

        except Exception as e:
            logger.error(
                f"Error in process_file_in_background for user {user_id}: {e}")
            await update.message.reply_text(
                f"❌ Wata babbar matsala ta faru yayin sarrafa fayil din. Kuskure: {type(e).__name__}."
            )

    def run(self):
        """Main function to start the bot"""
        keep_alive()  # ADD THIS LINE - keeps bot awake on Railway

        if BOT_TOKEN == "YOUR_BOT_TOKEN_HERE" or not BOT_TOKEN:
            logger.error(
                "❌ BOT_TOKEN not set! Please set it as environment variable:")
            logger.error("   export BOT_TOKEN='your_actual_bot_token_here'")
            return

        # Initialize database (synchronous helper)
        init_db()

        # Create application
        application = Application.builder().token(BOT_TOKEN).build()

        # Add handlers
        application.add_handler(CommandHandler(
            ["fara", "start"], self.start_cmd))
        application.add_handler(CommandHandler("taimako", self.help_cmd))
        application.add_handler(CommandHandler("nauina", self.categories_cmd))
        application.add_handler(CommandHandler("nema", self.search_cmd))
        application.add_handler(CommandHandler("tsoho", self.history_cmd))
        application.add_handler(CommandHandler("stats", self.stats_cmd))
        application.add_handler(CommandHandler("daina", self.stop_cmd))

        # Callback handlers
        application.add_handler(CallbackQueryHandler(
            self.handle_category_selection, pattern="^cat_"))
        application.add_handler(CallbackQueryHandler(
            self.handle_history_selection, pattern="^hist_"))
        application.add_handler(CallbackQueryHandler(
            self.handle_history_selection, pattern="^close_history"))
        application.add_handler(CommandHandler("shiga", self.add_story_cmd))
        application.add_handler(CommandHandler(
            "littafi", self.add_story_file_cmd, filters=filters.User(ADMIN_USER_ID)))

        # Store application reference and set post_init
        self.application = application
        application.post_init = self.post_init

        logger.info("✅ Hausa Stories Bot with Manual Stories is starting...")
        print("=" * 60)
        print("🎧 HAUSA STORIES BOT - MULTI-USER VERSION (FINAL HIGH PERFORMANCE)")
        print("✅ FULL CONCURRENCY AND JOB QUEUE FIXES APPLIED.")
        print("=" * 60)

        try:
            application.run_polling()
        except KeyboardInterrupt:
            logger.info("Bot stopped by user")
        except Exception as e:
            logger.error(f"Failed to start bot: {e}")


if __name__ == "__main__":
    bot = HausaStoriesBot()
    bot.run()
