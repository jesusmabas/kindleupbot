# main.py - Versión final y estable para Render
import os
import logging
import smtplib
import mimetypes
import asyncio
import uvicorn
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

# --- Configuración de logging (sin cambios) ---
def setup_logging():
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

# --- Constantes (sin cambios) ---
SUPPORTED_FORMATS = {
    '.epub': 'application/epub+zip', '.pdf': 'application/pdf', '.mobi': 'application/x-mobipocket-ebook',
    '.azw': 'application/vnd.amazon.ebook', '.doc': 'application/msword',
    '.docx': 'application/vnd.openxmlformats-officedocument.wordprocessingml.document', '.rtf': 'application/rtf',
    '.txt': 'text/plain', '.html': 'text/html', '.htm': 'text/html', '.jpg': 'image/jpeg',
    '.jpeg': 'image/jpeg', '.png': 'image/png', '.gif': 'image/gif', '.bmp': 'image/bmp'
}
PROMPT_SET_EMAIL = "📧 Por favor, introduce tu email de Kindle (ejemplo: usuario@kindle.com):"

# --- Función Asistente Asíncrona para la DB ---
async def get_total_users_async() -> int:
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, get_total_users)

# --- Clases de utilidad (MetricsCollector corregida) ---
class CacheManager:
    # ... (sin cambios)
    def __init__(self, default_ttl: int = 300):
        self.cache = {}
        self.ttl = default_ttl
    def get(self, key: str) -> Optional[Any]:
        if key in self.cache:
            data, timestamp = self.cache[key]
            if time.time() - timestamp < self.ttl: return data
            else: del self.cache[key]
        return None
    def set(self, key: str, value: Any, ttl: Optional[int] = None):
        self.cache[key] = (value, time.time())
        if ttl: asyncio.create_task(self._auto_expire(key, ttl))
    async def _auto_expire(self, key: str, ttl: int):
        await asyncio.sleep(ttl)
        self.cache.pop(key, None)
    def clear(self): self.cache.clear()

class RateLimiter:
    # ... (sin cambios)
    def __init__(self, max_requests: int, window: int):
        self.max_requests = max_requests
        self.window = window
        self.requests = defaultdict(list)
    def is_allowed(self, user_id: int) -> bool:
        now = time.time()
        user_requests = self.requests[user_id]
        user_requests[:] = [req_time for req_time in user_requests if now - req_time < self.window]
        if len(user_requests) >= self.max_requests: return False
        user_requests.append(now)
        return True
    def get_remaining_time(self, user_id: int) -> int:
        if not self.requests[user_id]: return 0
        oldest_request = min(self.requests[user_id])
        return max(0, int(self.window - (time.time() - oldest_request)))

class MetricsCollector:
    # ... (get_summary corregido a async)
    def __init__(self):
        self.start_time = time.time()
        self.metrics = defaultdict(int)
        self.user_metrics = defaultdict(lambda: defaultdict(int))
        self.error_log = []
        self.response_times = []
        self.daily_stats = defaultdict(lambda: defaultdict(int))
        self.lock = asyncio.Lock()
    def reset(self):
        logger.warning("Reiniciando el colector de métricas en memoria.")
        self.metrics.clear(); self.user_metrics.clear(); self.error_log.clear(); self.response_times.clear(); self.daily_stats.clear()
    def load_from_db(self):
        try:
            logger.info("Cargando métricas históricas desde la base de datos...")
            historical_metrics = get_metrics_from_db()
            for metric_name, user_id, value in historical_metrics:
                self.metrics[metric_name] += value
                if user_id: self.user_metrics[user_id][metric_name] += value
            logger.info(f"{len(historical_metrics)} registros de métricas cargados.")
        except Exception as e: logger.error(f"Error cargando métricas: {e}")
    async def _save_metric_async(self, metric_name: str, user_id: Optional[int], value: int):
        try:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, save_metric, metric_name, user_id, value)
        except Exception as e: logger.error(f"Error guardando métrica {metric_name}: {e}")
    async def increment(self, metric_name: str, user_id: Optional[int] = None, value: int = 1):
        async with self.lock:
            self.metrics[metric_name] += value
            if user_id: self.user_metrics[user_id][metric_name] += value
            today = datetime.now().strftime('%Y-%m-%d')
            self.daily_stats[today][metric_name] += value
            asyncio.create_task(self._save_metric_async(metric_name, user_id, value))
    def log_error(self, error_type: str, error_message: str, user_id: Optional[int] = None):
        error_data = {'timestamp': datetime.now().isoformat(),'type': error_type, 'message': error_message, 'user_id': user_id,'traceback': None}
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
        return {'uptime_formatted': self._format_uptime(uptime),'uptime_seconds': uptime,'total_users': total_users_count,'total_documents_sent': self.metrics.get('document_sent', 0),'total_documents_received': self.metrics.get('document_received', 0),'total_errors': self.metrics.get('errors_total', 0),'commands_executed': self.metrics.get('commands_total', 0),'avg_response_time_ms': round(avg_response_time * 1000, 2),'recent_errors': self.error_log[:10],'top_formats': self._get_top_formats(),'daily_stats': dict(self.daily_stats),'success_rate': self._calculate_success_rate(),}
    def _calculate_success_rate(self) -> float:
        sent = self.metrics.get('document_sent', 0); received = self.metrics.get('document_received', 0)
        return 100.0 if received == 0 else round((sent / received) * 100, 2)
    def get_user_stats(self, user_id: int) -> Dict[str, Any]:
        user_data = self.user_metrics.get(user_id, {})
        return {'documents_sent': user_data.get('document_sent', 0),'documents_received': user_data.get('document_received', 0),'commands_used': user_data.get('commands_total', 0),'errors_encountered': sum(v for k, v in user_data.items() if k.startswith('error_')),'top_format': self._get_user_top_format(user_id),'success_rate': self._calculate_user_success_rate(user_id),'last_activity': "N/A",}
    def _calculate_user_success_rate(self, user_id: int) -> float:
        user_data = self.user_metrics.get(user_id, {}); sent = user_data.get('document_sent', 0); received = user_data.get('document_received', 0)
        return 100.0 if received == 0 else round((sent / received) * 100, 2)
    def _format_uptime(self, seconds: float) -> str:
        days, rem = divmod(seconds, 86400); hours, rem = divmod(rem, 3600); minutes, _ = divmod(rem, 60)
        return f"{int(days)}d {int(hours)}h {int(minutes)}m"
    def _get_top_formats(self) -> List[Tuple[str, int]]:
        formats = {k.replace('format_', ''): v for k, v in self.metrics.items() if k.startswith('format_')}
        return sorted(formats.items(), key=lambda x: x[1], reverse=True)[:10]
    def _get_user_top_format(self, user_id: int) -> str:
        user_data = self.user_metrics.get(user_id, {}); formats = {k.replace('format_', ''): v for k, v in user_data.items() if k.startswith('format_')}
        return max(formats.items(), key=lambda x: x[1])[0] if formats else "Ninguno"

class EmailValidator:
    # ... (sin cambios)
    @staticmethod
    def validate_email(email: str) -> bool:
        if not email or len(email) < 5: return False
        return bool(re.match(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$', email))
    @staticmethod
    def is_kindle_email(email: str) -> bool:
        return any(domain in email.lower() for domain in ['@kindle.com', '@free.kindle.com'])

class FileValidator:
    # ... (sin cambios)
    @staticmethod
    def validate_file(filename: str, file_size: int, max_size: int) -> Tuple[bool, str]:
        if not filename: return False, "Nombre de archivo vacío"
        ext = Path(filename).suffix.lower()
        if ext not in SUPPORTED_FORMATS: return False, f"Formato {ext} no soportado"
        if file_size > max_size: return False, f"Archivo muy grande ({file_size / 1024**2:.1f}MB > {max_size / 1024**2:.1f}MB)"
        return True, "OK"

# --- Instancias globales y Decorador (sin cambios) ---
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

# --- Clase del Bot (con todos los handlers corregidos) ---
class KindleEmailBot:
    def __init__(self, config: Settings):
        self.config = config
        self.application = Application.builder().token(self.config.TELEGRAM_BOT_TOKEN).build()
        self.email_validator = EmailValidator()
        self.file_validator = FileValidator()
        self._reset_lock = asyncio.Lock()
        self.main_keyboard = ReplyKeyboardMarkup([["📧 Configurar Email", "🔍 Ver Mi Email"],["📊 Mis Estadísticas", "❓ Ayuda"],["🎯 Formatos Soportados", "🚀 Consejos"]], resize_keyboard=True)
        self.admin_keyboard = ReplyKeyboardMarkup([["👑 Panel Admin", "📈 Métricas"],["🧹 Limpiar Cache", "🔄 Reiniciar Stats"],["👥 Usuarios", "🏠 Menú Principal"]], resize_keyboard=True)
        self.confirm_keyboard = InlineKeyboardMarkup([[InlineKeyboardButton("✅ Confirmar", callback_data="confirm")],[InlineKeyboardButton("❌ Cancelar", callback_data="cancel")]])
        self.confirm_reset_keyboard = InlineKeyboardMarkup([[InlineKeyboardButton("✅ Sí, borrar estadísticas", callback_data="confirm_reset_stats")],[InlineKeyboardButton("❌ No, cancelar", callback_data="cancel_action")]])

    async def initialize_handlers(self):
        # ... (sin cambios)
        handlers = [CommandHandler("start", self.start), CommandHandler("help", self.help_command),CommandHandler("set_email", self.set_email_command), CommandHandler("my_email", self.my_email_command),CommandHandler("stats", self.stats_command), CommandHandler("admin", self.admin_command),CommandHandler("hide_keyboard", self.hide_keyboard_command), CommandHandler("formats", self.formats_command),CommandHandler("tips", self.tips_command), CommandHandler("clear_cache", self.clear_cache_command),CommandHandler("reset_stats", self.reset_stats_command),CallbackQueryHandler(self.handle_callback),MessageHandler(filters.TEXT & ~filters.COMMAND & filters.REPLY, self.handle_email_input),MessageHandler(filters.Document.ALL, self.handle_document),MessageHandler(filters.TEXT & ~filters.COMMAND, self.handle_text),]
        for handler in handlers: self.application.add_handler(handler)
        await self.application.initialize()
        logger.info("🤖 Handlers del bot inicializados correctamente")

    async def shutdown(self):
        # ... (sin cambios)
        logger.info("Iniciando secuencia de apagado del bot...")
        if self.application: await self.application.shutdown()
        logger.info("🤖 Bot cerrado correctamente")

    async def get_bot_info(self):
        # ... (sin cambios)
        try: return await self.application.bot.get_me() if self.application else None
        except TelegramError as e: logger.error(f"Error obteniendo info del bot: {e}"); return None

    async def _perform_stats_reset(self, query: 'Update.callback_query') -> bool:
        # ... (sin cambios, ya era seguro)
        async with self._reset_lock:
            try:
                await query.edit_message_text("⏳ Borrando historial de la base de datos...")
                loop = asyncio.get_event_loop()
                db_success = await loop.run_in_executor(None, reset_metrics_table)
                if not db_success:
                    await query.edit_message_text("❌ <b>Error Crítico:</b> No se pudo reiniciar la base de datos...", parse_mode=ParseMode.HTML)
                    return False
                await query.edit_message_text("⏳ Reseteando contadores en memoria...")
                metrics_collector.reset()
                admin_id = query.from_user.id
                log_success = await loop.run_in_executor(None, log_admin_action, admin_id, "RESET_STATS", f"Admin {admin_id} reinició métricas.")
                if not log_success: logger.error(f"Fallo al registrar acción de reinicio para admin {admin_id}")
                await query.edit_message_text("✅ ¡Estadísticas reiniciadas exitosamente!")
                logger.warning(f"Estadísticas reiniciadas por el admin {admin_id}.")
                return True
            except Exception as e:
                logger.error(f"Excepción en reinicio de stats: {e}", exc_info=True)
                await query.edit_message_text("❌ Ocurrió un error inesperado durante el reinicio.")
                return False

    @track_metrics('command_reset_stats')
    async def reset_stats_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        # ... (sin cambios)
        user_id = update.effective_user.id
        if not self.config.ADMIN_USER_ID or user_id != self.config.ADMIN_USER_ID:
            await update.message.reply_text("🚫 Acceso denegado."); return
        await update.message.reply_html("<b>⚠️ ¿Seguro que quieres reiniciar TODAS las estadísticas?</b>\n\nEsta acción es permanente.",reply_markup=self.confirm_reset_keyboard)

    async def handle_callback(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        # ... (sin cambios)
        query = update.callback_query; await query.answer(); user_id = update.effective_user.id
        if query.data == "confirm":
            pending_email = context.user_data.get('pending_email')
            if pending_email and await self._save_user_email(user_id, pending_email):
                await metrics_collector.increment('email_set_success', user_id)
                await query.edit_message_text(f"✅ <b>Email configurado</b>\n\n📧 <code>{pending_email}</code>", parse_mode=ParseMode.HTML)
            else: await query.edit_message_text("❌ Error al guardar el email")
            context.user_data.pop('pending_email', None)
        elif query.data == "cancel":
            await query.edit_message_text("❌ Configuración cancelada"); context.user_data.pop('pending_email', None)
        elif query.data == "confirm_reset_stats":
            if not self.config.ADMIN_USER_ID or user_id != self.config.ADMIN_USER_ID:
                await query.edit_message_text("🚫 Acción no autorizada."); return
            await self._perform_stats_reset(query)
        elif query.data == "cancel_action": await query.edit_message_text("👍 Acción cancelada.")

    @track_metrics('handle_text')
    async def handle_text(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        # ... (corregido para ser no bloqueante)
        text = update.message.text; is_admin = self.config.ADMIN_USER_ID and update.effective_user.id == self.config.ADMIN_USER_ID
        async def show_total_users(upd: Update, ctx: ContextTypes.DEFAULT_TYPE):
            total = await get_total_users_async()
            await upd.message.reply_text(f"👥 Hay un total de {total} usuarios registrados.")
        command_map = {"📧 Configurar Email": self.set_email_command, "🔍 Ver Mi Email": self.my_email_command,"📊 Mis Estadísticas": self.stats_command, "❓ Ayuda": self.help_command,"🎯 Formatos Soportados": self.formats_command, "🚀 Consejos": self.tips_command}
        admin_command_map = {"👑 Panel Admin": self.admin_command, "📈 Métricas": self.admin_command,"🧹 Limpiar Cache": self.clear_cache_command,"🔄 Reiniciar Stats": self.reset_stats_command,"👥 Usuarios": show_total_users,"🏠 Menú Principal": lambda u, c: u.message.reply_text("Volviendo al menú principal...", reply_markup=self.main_keyboard)}
        if text in command_map: await command_map[text](update, context)
        elif is_admin and text in admin_command_map: await admin_command_map[text](update, context)
        else:
            text_lower = text.lower()
            if any(word in text_lower for word in ['hola', 'hello', 'hi', 'buenas']): await update.message.reply_text("¡Hola! 👋 Soy tu asistente de Kindle.\nEnvíame un documento para empezar.", reply_markup=self.main_keyboard)
            elif any(word in text_lower for word in ['ayuda', 'help', 'auxilio']): await self.help_command(update, context)
            elif any(word in text_lower for word in ['gracias', 'thanks', 'thank you']): await update.message.reply_text("¡De nada! 😊 Estoy aquí para ayudarte.")
            else: await update.message.reply_html("🤔 <b>No entiendo ese mensaje</b>\n\nUsa los botones del menú o envía un documento.", reply_markup=self.main_keyboard)

    @track_metrics('command_start')
    async def start(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        # ... (sin cambios)
        user = update.effective_user; is_admin = self.config.ADMIN_USER_ID and user.id == self.config.ADMIN_USER_ID
        welcome_message = f"🎉 ¡Bienvenido, {user.mention_html()}!\n\n📚 <b>Kindle Bot v2.2</b>\n\n1. Configura tu email\n2. Autoriza <code>{self.config.GMAIL_USER}</code> en Amazon\n3. ¡Envía tus documentos!\n\n{'👑 <b>Acceso de administrador detectado</b>' if is_admin else ''}"
        keyboard = self.admin_keyboard if is_admin else self.main_keyboard
        await update.message.reply_html(welcome_message, reply_markup=keyboard)
    
    # ... otros handlers sin cambios (help, formats, tips, set_email, etc.)
    @track_metrics('command_help')
    async def help_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        help_text = f"📖 <b>Guía del Bot</b>\n\n• <b>Configurar Email</b>: Guarda tu email de Kindle.\n• <b>Ver Mi Email</b>: Muestra el email configurado.\n• <b>Mis Estadísticas</b>: Tus métricas de uso.\n\n🔑 <b>Email a autorizar:</b>\n<code>{self.config.GMAIL_USER}</code>"
        await update.message.reply_html(help_text)
    @track_metrics('command_formats')
    async def formats_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        message = f"📋 <b>Formatos Soportados</b>\n\n<b>Libros:</b> .epub, .mobi, .azw\n<b>Documentos:</b> .pdf, .doc, .docx, .rtf, .txt, .html\n<b>Imágenes:</b> .jpg, .png, .gif, .bmp\n\n📊 <b>Límite:</b> {self.config.MAX_FILE_SIZE // 1024**2}MB"
        await update.message.reply_html(message)
    @track_metrics('command_tips')
    async def tips_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        await update.message.reply_html("🚀 <b>Consejos:</b>\n\n• Escribe 'convert' en la descripción de un PDF para optimizarlo.\n• Usa tu email @kindle.com, no @amazon.com.\n• Autoriza mi email en la sección 'Contenido y dispositivos' de Amazon.")
    @track_metrics('command_set_email')
    async def set_email_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        await update.message.reply_text(PROMPT_SET_EMAIL, reply_markup=ForceReply(selective=True, input_field_placeholder="usuario@kindle.com"))
    @track_metrics('handle_email_input')
    async def handle_email_input(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not (update.message.reply_to_message and update.message.reply_to_message.text == PROMPT_SET_EMAIL): return
        user_id = update.effective_user.id; kindle_email = update.message.text.strip()
        if not self.email_validator.validate_email(kindle_email):
            await metrics_collector.increment('email_validation_failed', user_id)
            await update.message.reply_html("❌ <b>Email inválido.</b>\nFormato: <code>usuario@kindle.com</code>"); return
        if not self.email_validator.is_kindle_email(kindle_email):
            await update.message.reply_html("⚠️ <b>Advertencia:</b> No parece un email de Kindle. ¿Seguro?", reply_markup=self.confirm_keyboard)
            context.user_data['pending_email'] = kindle_email; return
        if await self._save_user_email(user_id, kindle_email):
            await metrics_collector.increment('email_set_success', user_id)
            await update.message.reply_html(f"✅ <b>Email configurado:</b> <code>{kindle_email}</code>")
        else: await update.message.reply_html("❌ <b>Error al guardar el email.</b>")
    async def _save_user_email(self, user_id: int, email: str) -> bool:
        try:
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, set_user_email, user_id, email)
        except Exception as e: logger.error(f"Error guardando email para {user_id}: {e}"); return False
    @track_metrics('command_my_email')
    async def my_email_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        user_id = update.effective_user.id; cache_key = f"user_email_{user_id}"; email = cache_manager.get(cache_key)
        if email is None:
            email = await self._get_user_email_async(user_id)
            if email: cache_manager.set(cache_key, email, 300)
        if email:
            is_kindle = self.email_validator.is_kindle_email(email)
            status_icon, status_text = ("✅", "Válido") if is_kindle else ("⚠️", "No parece de Kindle")
            await update.message.reply_html(f"📧 <b>Tu email:</b> <code>{email}</code>\n{status_icon} <b>Estado:</b> {status_text}")
        else: await update.message.reply_html("❌ <b>No tienes un email configurado.</b>")
    async def _get_user_email_async(self, user_id: int) -> Optional[str]:
        try:
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, get_user_email, user_id)
        except Exception as e: logger.error(f"Error obteniendo email para {user_id}: {e}"); return None

    @track_metrics('command_stats')
    async def stats_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        # ... (corregido para ser no bloqueante)
        user_id = update.effective_user.id; stats = metrics_collector.get_user_stats(user_id)
        success_rate = stats['success_rate']; filled_bars = int((success_rate / 100) * 10); bar = "█" * filled_bars + "░" * (10 - filled_bars)
        total_users = await get_total_users_async()
        summary = await metrics_collector.get_summary()
        stats_message = f"📊 <b>Tus Estadísticas</b>\n\n• Enviados: {stats['documents_sent']} / Recibidos: {stats['documents_received']}\n• Tasa Éxito: {success_rate}% {bar}\n• Comandos: {stats['commands_used']}\n• Formato Fav: {stats['top_format']}\n\n🏆 Eres 1 de {total_users} usuarios"
        await update.message.reply_html(stats_message)

    @track_metrics('command_admin')
    async def admin_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        # ... (corregido para ser no bloqueante)
        user_id = update.effective_user.id
        if not self.config.ADMIN_USER_ID or user_id != self.config.ADMIN_USER_ID: await update.message.reply_text("🚫 Acceso denegado"); return
        summary = await metrics_collector.get_summary()
        success_rate = summary['success_rate']; filled_bars = int((success_rate / 100) * 10); bar = "█" * filled_bars + "░" * (10 - filled_bars)
        top_formats = "\n".join([f"  • <code>{f}:</code> {c}" for f, c in summary['top_formats'][:5]]) if summary['top_formats'] else "Ninguno"
        admin_message = f"👑 <b>Panel Admin</b>\n\n• Activo: {summary['uptime_formatted']}\n• Usuarios: {summary['total_users']}\n• Docs Enviados: {summary['total_documents_sent']}\n• Éxito: {success_rate}% {bar}\n• Errores: {summary['total_errors']}\n• T. Respuesta: {summary['avg_response_time_ms']}ms\n\n<b>Formatos Pop:</b>\n{top_formats}"
        await update.message.reply_html(admin_message, reply_markup=self.admin_keyboard)

    @track_metrics('command_clear_cache')
    async def clear_cache_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        # ... (sin cambios)
        user_id = update.effective_user.id
        if not self.config.ADMIN_USER_ID or user_id != self.config.ADMIN_USER_ID: await update.message.reply_text("🚫 Acceso denegado"); return
        cache_manager.clear(); await update.message.reply_text("🧹 Caché limpiado.")

    @track_metrics('command_hide_keyboard')
    async def hide_keyboard_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        # ... (sin cambios)
        await update.message.reply_text("🙈 Teclado ocultado.", reply_markup=ReplyKeyboardRemove())

    @track_metrics('handle_document')
    async def handle_document(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        # ... (corregido para manejo de memoria eficiente)
        user_id = update.effective_user.id; user_kindle_email = await self._get_user_email_async(user_id)
        if not user_kindle_email: await update.message.reply_html("⚠️ <b>Email no configurado.</b>"); return
        doc = update.message.document
        valid, error_msg = self.file_validator.validate_file(doc.file_name, doc.file_size, self.config.MAX_FILE_SIZE)
        if not valid: await update.message.reply_html(f"❌ <b>Error:</b> {error_msg}"); return
        ext = Path(doc.file_name).suffix.lower()
        await metrics_collector.increment('document_received', user_id); await metrics_collector.increment(f'format_{ext.replace(".", "")}', user_id)
        processing_msg = await update.message.reply_html(f"⏳ Procesando <code>{doc.file_name}</code>...")
        temp_dir = Path("/tmp/kindleupbot_downloads"); temp_dir.mkdir(exist_ok=True)
        temp_file_path = temp_dir / f"{doc.file_unique_id}{ext}"
        try:
            file_obj = await context.bot.get_file(doc.file_id)
            await file_obj.download_to_drive(temp_file_path)
            subject = "Convert" if doc.file_name.lower().endswith('.pdf') and update.message.caption and 'convert' in update.message.caption.lower() else ""
            await processing_msg.edit_text(f"📤 Enviando <code>{doc.file_name}</code>...", parse_mode=ParseMode.HTML)
            async with aiofiles.open(temp_file_path, 'rb') as f: file_data = await f.read()
            success, msg = await self._send_to_kindle_with_retries(user_kindle_email, file_data, doc.file_name, subject)
            if success:
                await metrics_collector.increment('document_sent', user_id)
                await processing_msg.edit_text(f"✅ ¡<b>{doc.file_name}</b> enviado!", parse_mode=ParseMode.HTML)
            else: await processing_msg.edit_text(f"❌ <b>Error al enviar:</b> <i>{msg}</i>", parse_mode=ParseMode.HTML)
        except Exception as e:
            logger.error(f"Error procesando documento para {user_id}: {e}", exc_info=True)
            await processing_msg.edit_text(f"❌ <b>Error inesperado procesando el archivo.</b>", parse_mode=ParseMode.HTML)
        finally:
            if temp_file_path.exists(): os.remove(temp_file_path)

    async def _send_to_kindle_with_retries(self, kindle_email: str, file_data: bytes, filename: str, subject: str) -> Tuple[bool, str]:
        # ... (sin cambios)
        for attempt in range(self.config.MAX_RETRIES):
            try:
                success, msg = await self._send_to_kindle_async(kindle_email, file_data, filename, subject)
                if success: return True, msg
                if attempt < self.config.MAX_RETRIES - 1: await asyncio.sleep(self.config.RETRY_DELAY * (2 ** attempt))
            except Exception as e:
                logger.warning(f"Intento {attempt + 1} fallido: {e}")
                if attempt < self.config.MAX_RETRIES - 1: await asyncio.sleep(self.config.RETRY_DELAY * (2 ** attempt))
                else: return False, f"Error tras {self.config.MAX_RETRIES} intentos: {str(e)}"
        return False, "Falló tras todos los reintentos"

    async def _send_to_kindle_async(self, kindle_email: str, file_data: bytes, filename: str, subject: str) -> Tuple[bool, str]:
        # ... (sin cambios)
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._send_to_kindle_sync, kindle_email, file_data, filename, subject)

    def _send_to_kindle_sync(self, kindle_email: str, file_data: bytes, filename: str, subject: str) -> Tuple[bool, str]:
        # ... (sin cambios)
        try:
            msg = MIMEMultipart(); msg['From'], msg['To'], msg['Subject'] = self.config.GMAIL_USER, kindle_email, subject or f"Doc: {filename}"
            msg.attach(MIMEText(f"Enviado desde tu Bot de Telegram. Archivo: {filename}"))
            ctype, encoding = mimetypes.guess_type(filename); ctype = 'application/octet-stream' if ctype is None or encoding is not None else ctype
            maintype, subtype = ctype.split('/', 1)
            attachment = MIMEBase(maintype, subtype); attachment.set_payload(file_data); encoders.encode_base64(attachment)
            attachment.add_header('Content-Disposition', f'attachment; filename="{filename}"'); msg.attach(attachment)
            with smtplib.SMTP(self.config.SMTP_SERVER, self.config.SMTP_PORT) as server:
                server.starttls(); server.login(self.config.GMAIL_USER, self.config.GMAIL_APP_PASSWORD); server.send_message(msg)
            logger.info(f"Documento {filename} enviado a {kindle_email}"); return True, "Enviado"
        except smtplib.SMTPAuthenticationError: return False, "Error de autenticación SMTP"
        except smtplib.SMTPRecipientsRefused: return False, "Email de destinatario rechazado"
        except Exception as e: return False, f"Error SMTP: {str(e)}"


# --- MODELOS PYDANTIC (sin cambios) ---
class StatusResponse(BaseModel):
    status: str; bot_username: Optional[str] = None; metrics: Optional[Dict[str, Any]] = None; version: str = "2.2.0"
class EmailValidationRequest(BaseModel):
    email: str
    @field_validator('email')
    def validate_email(cls, v):
        if not re.match(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$', v): raise ValueError('Email inválido')
        return v.lower()

# --- GESTOR DE CICLO DE VIDA (LIFESPAN) CORREGIDO ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info(f"🚀 (PID: {os.getpid()}) Iniciando servidor y configurando webhook...")
    setup_database()
    metrics_collector.load_from_db()
    
    bot_instance = KindleEmailBot(settings)
    await bot_instance.initialize_handlers()
    
    webhook_url = f"{settings.WEBHOOK_URL}/telegram"
    try:
        await bot_instance.application.bot.set_webhook(
            url=webhook_url,
            secret_token=settings.WEBHOOK_SECRET_TOKEN,
            allowed_updates=Update.ALL_TYPES,
            drop_pending_updates=True
        )
        logger.info(f"✅ (PID: {os.getpid()}) Webhook configurado en: {webhook_url}")

        app.state.bot = bot_instance
        app.state.config = settings
        app.state.metrics = metrics_collector
        app.state.cache = cache_manager

        logger.info(f"✅ (PID: {os.getpid()}) Servidor listo.")
        yield

    except Exception as e:
        logger.critical(f"CRITICAL: (PID: {os.getpid()}) Error en inicio y config de webhook: {e}", exc_info=True)
        raise
    finally:
        # CORREGIDO: Se elimina delete_webhook para evitar race conditions en deploys.
        # La nueva instancia simplemente sobrescribirá el webhook.
        logger.info(f"🛑 (PID: {os.getpid()}) Cerrando servidor...")
        if hasattr(app.state, 'bot'):
            await app.state.bot.shutdown()
        logger.info(f"🛑 (PID: {os.getpid()}) Cierre completo.")


# --- APLICACIÓN FASTAPI (con endpoints corregidos) ---
app = FastAPI(title="Kindle Bot API", version="2.2.0", description="Bot de Telegram para envío de documentos a Kindle", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])
templates = Jinja2Templates(directory="templates")

@app.post("/telegram")
async def telegram_webhook(request: Request):
    # ... (sin cambios)
    secret_token = request.headers.get("X-Telegram-Bot-Api-Secret-Token")
    if secret_token != settings.WEBHOOK_SECRET_TOKEN:
        logger.warning("Token secreto de webhook inválido.")
        raise HTTPException(status_code=403, detail="Token secreto inválido")
    try:
        bot_instance = request.app.state.bot
        data = await request.json()
        update = Update.de_json(data, bot_instance.application.bot)
        await bot_instance.application.process_update(update)
        return {"status": "ok"}
    except Exception as e:
        logger.error(f"Error procesando update de webhook: {e}", exc_info=True)
        return {"status": "error_processing"}, 200

@app.get("/", response_model=StatusResponse)
async def read_root():
    # ... (corregido para ser no bloqueante)
    try:
        bot_info = await app.state.bot.get_bot_info()
        summary = await metrics_collector.get_summary()
        return StatusResponse(status="✅ Bot activo y funcionando", bot_username=bot_info.username if bot_info else None, metrics=summary)
    except Exception as e:
        logger.error(f"Error en endpoint raíz: {e}")
        return StatusResponse(status="❌ Error en el servicio", metrics={"error": str(e)})

@app.api_route("/health", methods=["GET", "HEAD"])
async def health_check():
    # ... (sin cambios)
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard(request: Request):
    # ... (sin cambios)
    return templates.TemplateResponse("dashboard.html", {"request": request})

@app.get("/api/metrics-data", response_class=JSONResponse)
async def metrics_data():
    # ... (corregido para ser no bloqueante)
    return await metrics_collector.get_summary()

@app.post("/api/clear-cache")
async def clear_cache():
    # ... (sin cambios)
    cache_manager.clear()
    return {"message": "Caché limpiado exitosamente"}

# --- PUNTO DE ENTRADA (sin cambios) ---
if __name__ == "__main__":
    try:
        logger.info(f"Iniciando servidor en {settings.HOST}:{settings.PORT}")
        uvicorn.run(app, host=settings.HOST, port=settings.PORT, log_level="info", access_log=True)
    except Exception as e:
        logger.critical(f"CRITICAL: Error fatal al iniciar servidor: {e}", exc_info=True)
        raise
