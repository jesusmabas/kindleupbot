import os
import secrets
from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    # --- Configuración Esencial ---
    TELEGRAM_BOT_TOKEN: str
    GMAIL_USER: str
    GMAIL_APP_PASSWORD: str
    DATABASE_URL: str
    ADMIN_USER_ID: Optional[int] = None
    
    # --- Configuración de Rendimiento y Límites ---
    MAX_FILE_SIZE: int = 48 * 1024 * 1024  # 48 MB
    MAX_RETRIES: int = 3
    RETRY_DELAY: float = 1.0
    RATE_LIMIT_WINDOW: int = 60
    RATE_LIMIT_MAX_REQUESTS: int = 20
    CACHE_DURATION: int = 300
    
    # --- Configuración SMTP ---
    SMTP_SERVER: str = 'smtp.gmail.com'
    SMTP_PORT: int = 465
    
    model_config = SettingsConfigDict(
        env_file='.env',
        env_file_encoding='utf-8',
        case_sensitive=False
    )

settings = Settings()