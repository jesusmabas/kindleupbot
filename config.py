# config.py
import os
from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    """
    Centralized application settings using Pydantic-Settings.
    Reads configuration from environment variables. Pydantic automatically
    matches snake_case field names to UPPER_SNAKE_CASE environment variables.
    """
    # --- Telegram Bot ---
    TELEGRAM_BOT_TOKEN: str

    # --- Gmail SMTP Credentials ---
    # Corresponds to GMAIL_USER and GMAIL_APP_PASSWORD env vars
    GMAIL_USER: str
    GMAIL_PASSWORD: str

    # --- Database ---
    DATABASE_URL: str

    # --- Admin ---
    ADMIN_USER_ID: Optional[int] = None

    # --- Application Behavior ---
    MAX_FILE_SIZE: int = 48 * 1024 * 1024  # 48MB default
    MAX_RETRIES: int = 3
    RETRY_DELAY: float = 1.0

    # --- Rate Limiting ---
    RATE_LIMIT_WINDOW: int = 60  # seconds
    RATE_LIMIT_MAX_REQUESTS: int = 10

    # --- Cache ---
    CACHE_DURATION: int = 300  # 5 minutes in seconds

    # --- SMTP Server ---
    SMTP_SERVER: str = 'smtp.gmail.com'
    SMTP_PORT: int = 587
    
    # --- Server ---
    HOST: str = "0.0.0.0"
    # Compatible with platforms like Heroku/Render which set a PORT env var
    PORT: int = int(os.getenv("PORT", 8080))

    # Pydantic V2 model_config dictionary
    model_config = SettingsConfigDict(
        env_file='.env',  # Optional: for local development, it can read from a .env file
        env_file_encoding='utf-8',
        case_sensitive=False # TELEGRAM_BOT_TOKEN matches telegram_bot_token
    )

# Create a single, reusable instance of the settings
# This will be imported by other modules.
# If a required environment variable is missing, this line will raise an error on startup.
settings = Settings()
