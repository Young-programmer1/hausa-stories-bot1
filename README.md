# Hausa Stories Bot 🎧

A Telegram bot that sends Hausa stories as audio to users.

## Features
- Multiple story categories (Comedy, News, Sports, Politics, Manual)
- Text-to-speech audio generation in Hausa
- Multi-user support
- Admin commands for adding stories
- Free deployment on Render

## Bot Commands
- `/start` - Start the bot and choose categories
- `/nema` - Search for new stories immediately  
- `/nauina` - Change your preferred categories
- `/tsoho` - View your story history
- `/stats` - View bot statistics
- `/daina` - Stop receiving stories
- `/taimako` - Show help message

## Admin Commands
- `/shiga Title | Story content` - Add story via text
- `/littafi Title` + reply to .txt file - Add story via file upload

## Deployment
Deployed on Render for 24/7 free operation.

## Technology
- Python 3
- python-telegram-bot
- gTTS for audio generation
- SQLite database