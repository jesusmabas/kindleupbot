# main.py - Versión final con conversión de MD a DOCX y corrección de error en PDF

import os
import logging
import smtplib
import mimetypes
import asyncio
import uvicorn
import pypandoc
from contextlib import asynccontextmanager
from typing import Optional, Tuple, Dict, Any, List
from datetime import datetime, timedelta
from collections import defaultdict
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email.mime.text import MIMEText
from email import encoders
import time
import hashlib
import json
from pathlib import Path
import re
import unicodedata
from email.mime.application import MIMEApplication

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, field_validator
import aiofiles

from config import settings, Settings
from database import (
    setup_database, set_user_email, get_user_email,
    get_metrics_from_db, save_metric, get_total_users,
    reset_metrics_table, log_admin_action
)
from telegram import Update, ReplyKeyboardMarkup, ReplyKeyboardRemove, ForceReply, InlineKeyboardMarkup, InlineKeyboardButton
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes, CallbackQueryHandler
from telegram.constants import ParseMode
from telegram.error import TelegramError

# Configuración de logging mejorada
def setup_logging():
    """Configura el sistema de logging con rotación de archivos"""
    from logging.handlers import RotatingFileHandler
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    file_handler = RotatingFileHandler(
        log_dir / "bot.log", maxBytes=10*1024*1024, backupCount=5
    )
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
    )
    file_handler.setFormatter(formatter)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.addHandler(file_handler)
    logger.addHandler(logging.StreamHandler())
    return logger

logger = setup_logging()

# Constantes
SUPPORTED_FORMATS = {
    '.epub': 'application/epub+zip', '.pdf': 'application/pdf', '.mobi': 'application/x-mobipocket-ebook',
    '.azw': 'application/vnd.amazon.ebook', '.doc': 'application/msword',
    '.docx': 'application/vnd.openxmlformats-officedocument.wordprocessingml.document', '.rtf': 'application/rtf',
    '.txt': 'text/plain', '.html': 'text/html', '.htm': 'text/html', '.jpg': 'image/jpeg',
    '.jpeg': 'image/jpeg', '.png': 'image/png', '.gif': 'image/gif', '.bmp': 'image/bmp',
    '.md': 'text/markdown'
}
PROMPT_SET_EMAIL = "📧 Por favor, introduce tu email de Kindle (ejemplo: usuario@kindle.com):"


# --- FUNCIÓN ASISTENTE ASÍNCRONA PARA LA BASE DE DATOS ---
async def get_total_users_async() -> int:
    """Obtiene el total de usuarios de forma no bloqueante."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, get_total_users)

# --- FUNCIÓN DE CONVERSIÓN MARKDOWN A DOCX ---
async def convert_markdown_to_docx(md_path: Path, title: str) -> Tuple[Optional[Path], Optional[str]]:
    """Convierte un archivo Markdown a DOCX usando Pandoc."""
    docx_path = md_path.with_suffix('.docx')
    base_title = Path(title).stem
    metadata_args = [f'--metadata=title:{base_title}']
    
    try:
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            None,
            lambda: pypandoc.convert_file(
                str(md_path), 
                'docx',  # Convertimos a docx
                outputfile=str(docx_path), 
                extra_args=['--standalone'] + metadata_args
            )
        )
        logger.info(f"Archivo Markdown convertido exitosamente a DOCX en: {docx_path}")
        return docx_path, None
    except Exception as e:
        logger.error(f"Error al convertir Markdown a DOCX con Pandoc: {e}", exc_info=True)
        return None, f"Error de Pandoc: {str(e)}"

# --- Clases de utilidad (Cache, RateLimiter, etc.) ---
class CacheManager:
    def __init__(self, default_ttl: int = 300):
        self.cache = {}
        self.ttl = default_ttl

    def get(self, key: str) -> Optional[Any]:
        if key in self.cache:
            data, timestamp = self.cache[key]
            if time.time() - timestamp < self.ttl:
                return data
            else:
                del self.cache[key]
        return None

    def set(self, key: str, value: Any, ttl: Optional[int] = None):
        self.cache[key] = (value, time.time())
        if ttl:
            asyncio.create_task(self._auto_expire(key, ttl))

    async def _auto_expire(self, key: str, ttl: int):
        await asyncio.sleep(ttl)
        self.cache.pop(key, None)

    def clear(self):
        self.cache.clear()

class RateLimiter:
    def __init__(self, max_requests: int, window: int):
        self.max_requests = max_requests
        self.window = window
        self.requests = defaultdict(list)

    def is_allowed(self, user_id: int) -> bool:
        now = time.time()
        user_requests = self.requests[user_id]
        user_requests[:] = [req_time for req_time in user_requests if now - req_time < self.window]
        if len(user_requests) >= self.max_requests:
            return False
        user_requests.append(now)
        return True

    def get_remaining_time(self, user_id: int) -> int:
        if not self.requests[user_id]:
            return 0
        oldest_request = min(self.requests[user_id])
        return max(0, int(self.window - (time.time() - oldest_request)))

class MetricsCollector:
    def __init__(self):
        self.start_time = time.time()
        self.metrics = defaultdict(int)
        self.user_metrics = defaultdict(lambda: defaultdict(int))
        self.error_log = []
        self.response_times = []
        self.daily_stats = defaultdict(lambda: defaultdict(int))
        self.lock = asyncio.Lock()

    def reset(self):
        """Reinicia todas las métricas en memoria a sus valores iniciales."""
        logger.warning("Reiniciando el colector de métricas en memoria.")
        self.metrics.clear()
        self.user_metrics.clear()
        self.error_log.clear()
        self.response_times.clear()
        self.daily_stats.clear()

    def load_from_db(self):
        try:
            logger.info("Cargando métricas históricas desde la base de datos...")
            historical_metrics = get_metrics_from_db()
            for metric_name, user_id, value in historical_metrics:
                self.metrics[metric_name] += value
                if user_id:
                    self.user_metrics[user_id][metric_name] += value
            logger.info(f"{len(historical_metrics)} registros de métricas cargados.")
        except Exception as e:
            logger.error(f"Error cargando métricas: {e}")

    async def _save_metric_async(self, metric_name: str, user_id: Optional[int], value: int):
        try:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, save_metric, metric_name, user_id, value)
        except Exception as e:
            logger.error(f"Error guardando métrica {metric_name}: {e}")

    async def increment(self, metric_name: str, user_id: Optional[int] = None, value: int = 1):
        async with self.lock:
            self.metrics[metric_name] += value
            if user_id:
                self.user_metrics[user_id][metric_name] += value
            today = datetime.now().strftime('%Y-%m-%d')
            self.daily_stats[today][metric_name] += value
            asyncio.create_task(self._save_metric_async(metric_name, user_id, value))

    def log_error(self, error_type: str, error_message: str, user_id: Optional[int] = None):
        error_data = {
            'timestamp': datetime.now().isoformat(),
            'type': error_type, 'message': error_message, 'user_id': user_id,
            'traceback': None
        }
        self.error_log.insert(0, error_data)
        if len(self.error_log) > 100: self.error_log.pop()
        asyncio.create_task(self.increment('errors_total'))
        asyncio.create_task(self.increment(f'error_{error_type}', user_id))

    def log_response_time(self, duration: float, operation: str):
        self.response_times.insert(0, {'duration': duration, 'operation': operation, 'timestamp': time.time()})
        if len(self.response_times) > 1000: self.response_times.pop()

    async def get_summary(self) -> Dict[str, Any]:
        uptime = time.time() - self.start_time
        avg_response_time = sum(r['duration'] for r in self.response_times) / len(self.response_times) if self.response_times else 0
        total_users_count = await get_total_users_async()
        return {
            'uptime_formatted': self._format_uptime(uptime),
            'uptime_seconds': uptime,
            'total_users': total_users_count,
            'total_documents_sent': self.metrics.get('document_sent', 0),
            'total_documents_received': self.metrics.get('document_received', 0),
            'total_errors': self.metrics.get('errors_total', 0),
            'commands_executed': self.metrics.get('commands_total', 0),
            'avg_response_time_ms': round(avg_response_time * 1000, 2),
            'recent_errors': self.error_log[:10],
            'top_formats': self._get_top_formats(),
            'daily_stats': dict(self.daily_stats),
            'success_rate': self._calculate_success_rate(),
        }

    def _calculate_success_rate(self) -> float:
        sent = self.metrics.get('document_sent', 0)
        received = self.metrics.get('document_received', 0)
        return 100.0 if received == 0 else round((sent / received) * 100, 2)

    def get_user_stats(self, user_id: int) -> Dict[str, Any]:
        user_data = self.user_metrics.get(user_id, {})
        return {
            'documents_sent': user_data.get('document_sent', 0),
            'documents_received': user_data.get('document_received', 0),
            'commands_used': user_data.get('commands_total', 0),
            'errors_encountered': sum(v for k, v in user_data.items() if k.startswith('error_')),
            'top_format': self._get_user_top_format(user_id),
            'success_rate': self._calculate_user_success_rate(user_id),
            'last_activity': "N/A",
        }

    def _calculate_user_success_rate(self, user_id: int) -> float:
        user_data = self.user_metrics.get(user_id, {})
        sent = user_data.get('document_sent', 0)
        received = user_data.get('document_received', 0)
        return 100.0 if received == 0 else round((sent / received) * 100, 2)

    def _format_uptime(self, seconds: float) -> str:
        days, rem = divmod(seconds, 86400)
        hours, rem = divmod(rem, 3600)
        minutes, _ = divmod(rem, 60)
        return f"{int(days)}d {int(hours)}h {int(minutes)}m"

    def _get_top_formats(self) -> List[Tuple[str, int]]:
        formats = {k.replace('format_', ''): v for k, v in self.metrics.items() if k.startswith('format_')}
        return sorted(formats.items(), key=lambda x: x[1], reverse=True)[:10]

    def _get_user_top_format(self, user_id: int) -> str:
        user_data = self.user_metrics.get(user_id, {})
        formats = {k.replace('format_', ''): v for k, v in user_data.items() if k.startswith('format_')}
        return max(formats.items(), key=lambda x: x[1])[0] if formats else "Ninguno"

class EmailValidator:
    @staticmethod
    def validate_email(email: str) -> bool:
        if not email or len(email) < 5: return False
        return bool(re.match(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$', email))
    @staticmethod
    def is_kindle_email(email: str) -> bool:
        return any(domain in email.lower() for domain in ['@kindle.com', '@free.kindle.com'])

class FileValidator:
    @staticmethod
    def validate_file(filename: str, file_size: int, max_size: int) -> Tuple[bool, str]:
        if not filename: return False, "Nombre de archivo vacío"
        ext = Path(filename).suffix.lower()
        if ext not in SUPPORTED_FORMATS: return False, f"Formato {ext} no soportado"
        if file_size > max_size: return False, f"Archivo muy grande ({file_size / 1024**2:.1f}MB > {max_size / 1024**2:.1f}MB)"
        return True, "OK"

# Instancias globales
metrics_collector = MetricsCollector()
cache_manager = CacheManager(default_ttl=settings.CACHE_DURATION)
rate_limiter = RateLimiter(max_requests=settings.RATE_LIMIT_MAX_REQUESTS, window=settings.RATE_LIMIT_WINDOW)

def track_metrics(operation_name: str):
    def decorator(func):
        async def wrapper(self, update: Update, *args, **kwargs):
            start_time = time.time()
            user_id = update.effective_user.id if update and hasattr(update, 'effective_user') else None
            if user_id and not rate_limiter.is_allowed(user_id):
                remaining_time = rate_limiter.get_remaining_time(user_id)
                await update.message.reply_text(f"🚫 Límite de solicitudes excedido. Intenta de nuevo en {remaining_time} segundos.")
                return
            await metrics_collector.increment('commands_total', user_id)
            await metrics_collector.increment(operation_name, user_id)
            try:
                result = await func(self, update, *args, **kwargs)
                await metrics_collector.increment(f'{operation_name}_success', user_id)
                return result
            except Exception as e:
                metrics_collector.log_error(operation_name, str(e), user_id)
                logger.error(f"Error en {operation_name} para usuario {user_id}: {e}", exc_info=True)
                if hasattr(update, 'message') and update.message:
                    await update.message.reply_text("😔 Ocurrió un error inesperado. El equipo técnico ha sido notificado.")
                raise
            finally:
                duration = time.time() - start_time
                metrics_collector.log_response_time(duration, operation_name)
        return wrapper
    return decorator

# --- CLASE PRINCIPAL DEL BOT ---
class KindleEmailBot:
    def __init__(self, config: Settings):
        self.config = config
        self.application = Application.builder().token(self.config.TELEGRAM_BOT_TOKEN).build()
        self.email_validator = EmailValidator()
        self.file_validator = FileValidator()
        self._reset_lock = asyncio.Lock()

        self.main_keyboard = ReplyKeyboardMarkup([
            ["📧 Configurar Email", "🔍 Ver Mi Email"],
            ["📊 Mis Estadísticas", "❓ Ayuda"],
            ["🎯 Formatos Soportados", "🚀 Consejos"]
        ], resize_keyboard=True)
        
        self.admin_keyboard = ReplyKeyboardMarkup([
            ["👑 Panel Admin", "📈 Métricas"],
            ["🧹 Limpiar Cache", "🔄 Reiniciar Stats"],
            ["👥 Usuarios", "🏠 Menú Principal"]
        ], resize_keyboard=True)

        self.confirm_keyboard = InlineKeyboardMarkup([
            [InlineKeyboardButton("✅ Confirmar", callback_data="confirm")],
            [InlineKeyboardButton("❌ Cancelar", callback_data="cancel")]
        ])

        self.confirm_reset_keyboard = InlineKeyboardMarkup([
            [InlineKeyboardButton("✅ Sí, borrar estadísticas", callback_data="confirm_reset_stats")],
            [InlineKeyboardButton("❌ No, cancelar", callback_data="cancel_action")]
        ])

    async def initialize_handlers(self):
        handlers = [
            CommandHandler("start", self.start), CommandHandler("help", self.help_command),
            CommandHandler("set_email", self.set_email_command), CommandHandler("my_email", self.my_email_command),
            CommandHandler("stats", self.stats_command), CommandHandler("admin", self.admin_command),
            CommandHandler("hide_keyboard", self.hide_keyboard_command), CommandHandler("formats", self.formats_command),
            CommandHandler("tips", self.tips_command), CommandHandler("clear_cache", self.clear_cache_command),
            CommandHandler("reset_stats", self.reset_stats_command),
            CallbackQueryHandler(self.handle_callback),
            MessageHandler(filters.TEXT & ~filters.COMMAND & filters.REPLY, self.handle_email_input),
            MessageHandler(filters.Document.ALL, self.handle_document),
            MessageHandler(filters.TEXT & ~filters.COMMAND, self.handle_text),
        ]
        for handler in handlers: self.application.add_handler(handler)
        await self.application.initialize()
        logger.info("🤖 Handlers del bot inicializados correctamente")

    async def shutdown(self):
        logger.info("Iniciando secuencia de apagado del bot...")
        if self.application:
            await self.application.shutdown()
        logger.info("🤖 Bot cerrado correctamente")

    async def get_bot_info(self):
        try:
            return await self.application.bot.get_me() if self.application else None
        except TelegramError as e:
            logger.error(f"Error obteniendo info del bot: {e}")
            return None
    
    async def _perform_stats_reset(self, query: 'Update.callback_query') -> bool:
        async with self._reset_lock:
            try:
                await query.edit_message_text("⏳ Borrando historial de la base de datos...")
                loop = asyncio.get_event_loop()
                
                db_success = await loop.run_in_executor(None, reset_metrics_table)
                
                if not db_success:
                    await query.edit_message_text("❌ <b>Error Crítico:</b> No se pudo reiniciar la base de datos. La operación ha sido cancelada.", parse_mode=ParseMode.HTML)
                    return False

                await query.edit_message_text("⏳ Reseteando contadores en memoria...")
                metrics_collector.reset()
                
                admin_id = query.from_user.id
                log_success = await loop.run_in_executor(
                    None, log_admin_action, admin_id, "RESET_STATS", f"Admin {admin_id} reinició todas las métricas."
                )
                if not log_success:
                    logger.error(f"Fallo al registrar la acción de reinicio de stats para el admin {admin_id}")

                await query.edit_message_text("✅ ¡Todas las estadísticas han sido reiniciadas exitosamente!")
                logger.warning(f"Estadísticas reiniciadas por el admin {admin_id}.")
                return True
            except Exception as e:
                logger.error(f"Excepción no controlada durante el reinicio de stats: {e}", exc_info=True)
                await query.edit_message_text("❌ Ocurrió un error inesperado durante el proceso de reinicio.")
                return False

    @track_metrics('command_reset_stats')
    async def reset_stats_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        user_id = update.effective_user.id
        if not self.config.ADMIN_USER_ID or user_id != self.config.ADMIN_USER_ID:
            await update.message.reply_text("🚫 Acceso denegado.")
            return