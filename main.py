import os
import logging
from logging.handlers import RotatingFileHandler
import smtplib
import asyncio
import pypandoc
import re
import unicodedata
import time
from pathlib import Path
from collections import defaultdict
from typing import Optional, Tuple, Dict, Any, List

# --- Imports de Email ---
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email.mime.text import MIMEText
from email.mime.application import MIMEApplication
from email import encoders

# --- Imports de Configuración y Base de Datos ---
from config import settings, Settings
from database import (
    setup_database, set_user_email, get_user_email,
    get_metrics_from_db, save_metric, get_total_users,
    reset_metrics_table, log_admin_action
)

# --- Imports de Telegram ---
from telegram import Update, ReplyKeyboardMarkup, ReplyKeyboardRemove, ForceReply, InlineKeyboardMarkup, InlineKeyboardButton
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes, CallbackQueryHandler
from telegram.constants import ParseMode
from telegram.error import TelegramError

def setup_logging():
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

SUPPORTED_FORMATS = {
    '.epub': 'application/epub+zip', '.pdf': 'application/pdf', '.mobi': 'application/x-mobipocket-ebook',
    '.azw': 'application/vnd.amazon.ebook', '.doc': 'application/msword',
    '.docx': 'application/vnd.openxmlformats-officedocument.wordprocessingml.document', '.rtf': 'application/rtf',
    '.txt': 'text/plain', '.html': 'text/html', '.htm': 'text/html', '.jpg': 'image/jpeg',
    '.jpeg': 'image/jpeg', '.png': 'image/png', '.gif': 'image/gif', '.bmp': 'image/bmp',
    '.md': 'text/markdown'
}
PROMPT_SET_EMAIL = "📧 Por favor, introduce tu email de Kindle (ejemplo: usuario@kindle.com):"

async def convert_markdown_to_docx(md_path: Path, title: str) -> Tuple[Optional[Path], Optional[str]]:
    docx_path = md_path.with_suffix('.docx')
    base_title = Path(title).stem
    metadata_args = [f'--metadata=title:{base_title}']
    try:
        await asyncio.to_thread(
            pypandoc.convert_file,
            str(md_path),
            'docx',
            outputfile=str(docx_path),
            extra_args=['--standalone'] + metadata_args
        )
        logger.info(f"Archivo Markdown convertido exitosamente a DOCX en: {docx_path}")
        return docx_path, None
    except Exception as e:
        logger.error(f"Error al convertir Markdown a DOCX con Pandoc: {e}", exc_info=True)
        return None, f"Error de Pandoc: {str(e)}"

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
            await asyncio.to_thread(save_metric, metric_name, user_id, value)
        except Exception as e:
            logger.error(f"Error guardando métrica {metric_name}: {e}")
    async def increment(self, metric_name: str, user_id: Optional[int] = None, value: int = 1):
        async with self.lock:
            self.metrics[metric_name] += value
            if user_id:
                self.user_metrics[user_id][metric_name] += value
            today = time.strftime('%Y-%m-%d')
            self.daily_stats[today][metric_name] += value
            asyncio.create_task(self._save_metric_async(metric_name, user_id, value))
    def log_error(self, error_type: str, error_message: str, user_id: Optional[int] = None):
        error_data = {
            'timestamp': time.time(),
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
        total_users_count = await asyncio.to_thread(get_total_users)
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
        
        self._setup_handlers()

    def _setup_handlers(self):
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
        for handler in handlers:
            self.application.add_handler(handler)
        logger.info("🤖 Handlers del bot configurados.")

    async def _perform_stats_reset(self, query: 'Update.callback_query') -> bool:
        async with self._reset_lock:
            try:
                await query.edit_message_text("⏳ Borrando historial de la base de datos...")
                db_success = await asyncio.to_thread(reset_metrics_table)
                if not db_success:
                    await query.edit_message_text("❌ <b>Error Crítico:</b> No se pudo reiniciar la base de datos.", parse_mode=ParseMode.HTML)
                    return False
                
                await query.edit_message_text("⏳ Reseteando contadores en memoria...")
                metrics_collector.reset()
                
                admin_id = query.from_user.id
                log_success = await asyncio.to_thread(
                    log_admin_action, admin_id, "RESET_STATS", f"Admin {admin_id} reinició todas las métricas."
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
        await update.message.reply_html(
            "<b>⚠️ ¿Estás seguro de que quieres reiniciar TODAS las estadísticas?</b>\n\n"
            "Esta acción borrará permanentemente el historial de la base de datos y los contadores actuales.\n\n"
            "<i>Esta acción no se puede deshacer.</i>",
            reply_markup=self.confirm_reset_keyboard
        )

    @track_metrics('handle_callback')
    async def handle_callback(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        query = update.callback_query
        await query.answer()
        user_id = update.effective_user.id
        
        if query.data in ("pdf_convert_yes", "pdf_convert_no"):
            data = context.user_data.get('pending_pdf')
            if not data or data.get('user_id') != user_id:
                await query.edit_message_text("⚠️ Este menú ha expirado o no te pertenece.", reply_markup=None)
                return
            
            filename = data['filename']
            temp_path = Path(data['temp_path'])
            await query.edit_message_text(f"⏳ Preparando y enviando <code>{filename}</code>...", parse_mode=ParseMode.HTML)
            
            try:
                if not temp_path.exists():
                    await query.edit_message_text(f"❌ <b>Error:</b> El archivo temporal ya no existe.", parse_mode=ParseMode.HTML)
                    return
                
                file_data = await asyncio.to_thread(temp_path.read_bytes)
                subject = "Convert" if query.data == "pdf_convert_yes" else ""
                user_kindle_email = await self._get_user_email(user_id)
                
                if not user_kindle_email:
                    await query.edit_message_text("⚠️ Tu email de Kindle ya no está configurado.", parse_mode=ParseMode.HTML)
                    return
                
                success, msg = await self._send_to_kindle_with_retries(user_kindle_email, file_data, filename, subject)
                
                if success:
                    action = "(convertido)" if subject else "(sin conversión)"
                    await query.edit_message_text(f"✅ ¡<b>{filename}</b> enviado exitosamente {action}!", parse_mode=ParseMode.HTML)
                    await metrics_collector.increment('document_sent', user_id)
                    if subject:
                        await metrics_collector.increment('pdf_converted', user_id)
                else:
                    await query.edit_message_text(f"❌ <b>Error al enviar:</b> <i>{msg}</i>", parse_mode=ParseMode.HTML)
            except Exception as e:
                logger.error(f"Error en callback de PDF para {user_id}: {e}", exc_info=True)
                await query.edit_message_text("❌ <b>Error inesperado</b> durante el envío.", parse_mode=ParseMode.HTML)
            finally:
                context.user_data.pop('pending_pdf', None)
                if temp_path.exists():
                    await asyncio.to_thread(temp_path.unlink)
            return

        if query.data == "confirm":
            pending_email = context.user_data.get('pending_email')
            if pending_email and await self._save_user_email(user_id, pending_email):
                await metrics_collector.increment('email_set_success', user_id)
                await query.edit_message_text(f"✅ <b>Email configurado</b>\n\n📧 <code>{pending_email}</code>", parse_mode=ParseMode.HTML)
            else:
                await query.edit_message_text("❌ Error al guardar el email")
            context.user_data.pop('pending_email', None)
        elif query.data == "cancel":
            await query.edit_message_text("❌ Configuración cancelada")
            context.user_data.pop('pending_email', None)
        elif query.data == "confirm_reset_stats":
            if not self.config.ADMIN_USER_ID or user_id != self.config.ADMIN_USER_ID:
                await query.edit_message_text("🚫 Acción no autorizada.")
                return
            await self._perform_stats_reset(query)
        elif query.data == "cancel_action":
            await query.edit_message_text("👍 Acción cancelada.")

    @track_metrics('handle_text')
    async def handle_text(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        text = update.message.text
        is_admin = self.config.ADMIN_USER_ID and update.effective_user.id == self.config.ADMIN_USER_ID
        
        async def show_total_users(upd: Update, ctx: ContextTypes.DEFAULT_TYPE):
            total = await asyncio.to_thread(get_total_users)
            await upd.message.reply_text(f"👥 Hay un total de {total} usuarios registrados.")
            
        command_map = {
            "📧 Configurar Email": self.set_email_command, "🔍 Ver Mi Email": self.my_email_command,
            "📊 Mis Estadísticas": self.stats_command, "❓ Ayuda": self.help_command,
            "🎯 Formatos Soportados": self.formats_command, "🚀 Consejos": self.tips_command
        }
        admin_command_map = {
            "👑 Panel Admin": self.admin_command, "📈 Métricas": self.admin_command,
            "🧹 Limpiar Cache": self.clear_cache_command,
            "🔄 Reiniciar Stats": self.reset_stats_command,
            "👥 Usuarios": show_total_users,
            "🏠 Menú Principal": lambda u, c: u.message.reply_text("Volviendo al menú principal...", reply_markup=self.main_keyboard)
        }
        
        if text in command_map:
            await command_map[text](update, context)
        elif is_admin and text in admin_command_map:
            await admin_command_map[text](update, context)
        else:
            text_lower = text.lower()
            if any(word in text_lower for word in ['hola', 'hello', 'hi', 'buenas']):
                await update.message.reply_text("¡Hola! 👋 Soy tu asistente de Kindle.\nEnvíame un documento para empezar.", reply_markup=self.main_keyboard)
            elif any(word in text_lower for word in ['ayuda', 'help', 'auxilio']):
                await self.help_command(update, context)
            elif any(word in text_lower for word in ['gracias', 'thanks', 'thank you']):
                await update.message.reply_text("¡De nada! 😊 Estoy aquí para ayudarte.")
            else:
                await update.message.reply_html("🤔 <b>No entiendo ese mensaje</b>\n\n💡 <b>Puedo ayudarte con:</b>\n• Configurar tu email de Kindle\n• Enviar documentos a tu dispositivo\n• Mostrar estadísticas de uso\n\n📄 <b>Envía un documento</b> o usa los botones del menú", reply_markup=self.main_keyboard)

    @track_metrics('command_start')
    async def start(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        user = update.effective_user
        is_admin = self.config.ADMIN_USER_ID and user.id == self.config.ADMIN_USER_ID
        welcome_message = f"🎉 ¡Bienvenido, {user.mention_html()}!\n\n📚 <b>Kindle Bot v2.2</b> - Tu asistente personal para envío de documentos\n\n🚀 <b>Pasos para empezar:</b>\n1. Configura tu email con \"📧 Configurar Email\"\n2. Autoriza mi email en tu cuenta de Amazon\n3. ¡Envía tus documentos!\n\n📧 <b>Email a autorizar:</b> <code>{self.config.GMAIL_USER}</code>\n\n{'👑 <b>Acceso de administrador detectado</b>' if is_admin else ''}"
        keyboard = self.admin_keyboard if is_admin else self.main_keyboard
        await update.message.reply_html(welcome_message, reply_markup=keyboard)

    @track_metrics('command_help')
    async def help_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        intro = ("📖 <b>Guía Completa del KindleUp Bot</b>\n\n" "Domina el envío de tus documentos a Kindle en 3 pasos.")
        steps = [
            ("Configura tu Email de Kindle", "Usa /set_email o el botón 📧 para guardar tu dirección @kindle.com."),
            ("Autoriza mi Email en Amazon", f"Añade <code>{self.config.GMAIL_USER}</code> en tu cuenta de Amazon → " "'Gestionar contenido y dispositivos' → 'Preferencias'."),
            ("Envía tu Documento", "Arrastra y suelta un archivo aquí. Yo me encargo del resto.")
        ]
        pdf_flow = ("📄 <b>Flujo de PDF: ¡Tú eliges!</b>\n" "Tras enviarlo, te preguntaré qué hacer:\n" "  • <b>✅ Convertir:</b> Para texto adaptable (libros, artículos).\n" "  • <b>❌ Sin convertir:</b> Para mantener el diseño original (cómics, manuales).")
        commands = [
            ("/start", "Inicia la conversación y muestra el menú."),
            ("/help", "Muestra esta guía completa."),
            ("/set_email", "Configura o cambia tu email de Kindle."),
            ("/my_email", "Muestra tu email configurado."),
            ("/stats", "Muestra tus estadísticas de uso."),
            ("/formats", "Lista los formatos de archivo compatibles."),
            ("/tips", "Muestra consejos y trucos rápidos."),
            ("/hide_keyboard", "Oculta el teclado de botones del menú.")
        ]
        faq = [
            ("¿El documento no llega a mi Kindle?", "1. Asegúrate de haber autorizado mi email en Amazon.\n" "2. Comprueba la conexión Wi-Fi de tu Kindle.\n" "3. Dale unos minutos, a veces Amazon tarda un poco."),
            ("¿Recibo un error de 'email rechazado'?", "Tu email de Kindle es incorrecto. Verifícalo con /my_email y corrígelo con /set_email."),
            ("¿Mis archivos están seguros?", "Totalmente. Se borran de nuestros servidores temporales justo después de ser enviados. Nunca los almacenamos.")
        ]
        parts = [intro]
        parts.append("1️⃣ <b>PUESTA EN MARCHA</b>")
        for i, (title, desc) in enumerate(steps, 1):
            parts.append(f"<b>Paso {i}: {title}</b>\n{desc}")
        parts.append(pdf_flow)
        parts.append("🔧 <b>LISTA DE COMANDOS</b>")
        command_lines = [f"<code>{cmd}</code> - {desc}" for cmd, desc in commands]
        parts.append("\n".join(command_lines))
        parts.append("🤔 <b>SOLUCIÓN DE PROBLEMAS (FAQ)</b>")
        for q, a in faq:
            parts.append(f"<b>P: {q}</b>\nR: {a}")
        
        separator = "\n\n---\n\n"
        final_message = separator.join(parts)
        await update.message.reply_html(final_message, disable_web_page_preview=True)

    @track_metrics('command_tips')
    async def tips_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        tips_message = """
🚀 <b>Consejos y Trucos Rápidos</b>
🧠 <b>Markdown con Tablas</b>
Si envías un fichero <code>.md</code> con tablas o imágenes complejas, lo convertiré a <code>.docx</code> para asegurar la máxima fidelidad en tu Kindle.
📄 <b>Elige bien con los PDF</b>
• ¿Libro o artículo de texto? → <b>✅ Convertir</b>.
• ¿Manual con gráficos o cómic? → <b>❌ Sin convertir</b>.
Piénsalo así: si quisieras cambiar el tamaño de la letra en el documento, elige "Convertir".
⚡️ <b>El Formato Ideal</b>
Aunque el bot acepta muchos formatos, <code>.EPUB</code> es el rey para novelas y texto simple. Si tienes un libro en varios formatos, elige siempre la versión <code>.EPUB</code>.
🔄 <b>Reenvío Fácil desde otros Chats</b>
¿Te han enviado un documento en otro chat o canal? No hace falta que lo descargues y lo vuelvas a subir. Simplemente <b>reenvíamelo directamente</b> a este chat y yo me encargaré.
📂 <b>Gestiona Archivos Grandes</b>
El límite es de 48 MB. Si un archivo es más grande, es probable que Amazon lo rechace de todas formas. Considera comprimirlo o dividirlo si es posible.
"""
        await update.message.reply_html(tips_message)

    @track_metrics('command_formats')
    async def formats_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        formats_by_category = {
            "📚 Libros Electrónicos": [".epub", ".mobi", ".azw", ".md (convierte a docx)"],
            "📄 Documentos": [".pdf", ".doc", ".docx", ".rtf", ".txt", ".html"],
            "🖼️ Imágenes": [".jpg", ".jpeg", ".png", ".gif", ".bmp"]
        }
        message = f"📋 <b>Formatos Soportados</b>\n\n"
        for category, extensions in formats_by_category.items():
            message += f"<b>{category}:</b>\n • " + " • ".join(extensions) + "\n\n"
        message += f"📊 <b>Límite de tamaño:</b> {self.config.MAX_FILE_SIZE // 1024**2}MB"
        await update.message.reply_html(message)

    @track_metrics('command_set_email')
    async def set_email_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        await update.message.reply_text(PROMPT_SET_EMAIL, reply_markup=ForceReply(selective=True, input_field_placeholder="usuario@kindle.com"))

    @track_metrics('handle_email_input')
    async def handle_email_input(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not (update.message.reply_to_message and update.message.reply_to_message.text == PROMPT_SET_EMAIL): return
        user_id = update.effective_user.id
        kindle_email = update.message.text.strip()
        
        if not self.email_validator.validate_email(kindle_email):
            await metrics_collector.increment('email_validation_failed', user_id)
            await update.message.reply_html("❌ <b>Formato de email inválido</b>\n\n📧 Formato correcto: <code>usuario@kindle.com</code>\n🔄 Inténtalo de nuevo con /set_email")
            return
            
        if not self.email_validator.is_kindle_email(kindle_email):
            await update.message.reply_html("⚠️ <b>Advertencia:</b> Este no parece ser un email de Kindle\n\n📧 Los emails de Kindle terminan en:\n• @kindle.com\n• @free.kindle.com\n\n¿Estás seguro de que es correcto?", reply_markup=self.confirm_keyboard)
            context.user_data['pending_email'] = kindle_email
            return
            
        if await self._save_user_email(user_id, kindle_email):
            await metrics_collector.increment('email_set_success', user_id)
            await update.message.reply_html(f"✅ <b>Email configurado correctamente</b>\n\n📧 <b>Tu email:</b> <code>{kindle_email}</code>\n\n🔑 <b>Recuerda autorizar:</b> <code>{self.config.GMAIL_USER}</code>")
        else:
            await update.message.reply_html("❌ <b>Error al guardar el email</b>\n\n🔄 Por favor, inténtalo de nuevo")

    async def _save_user_email(self, user_id: int, email: str) -> bool:
        try:
            return await asyncio.to_thread(set_user_email, user_id, email)
        except Exception as e:
            logger.error(f"Error guardando email para usuario {user_id}: {e}")
            return False

    @track_metrics('command_my_email')
    async def my_email_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        user_id = update.effective_user.id
        cache_key = f"user_email_{user_id}"
        email = cache_manager.get(cache_key)
        if email is None:
            email = await self._get_user_email(user_id)
            if email:
                cache_manager.set(cache_key, email, 300)
                
        if email:
            is_kindle = self.email_validator.is_kindle_email(email)
            status_icon, status_text = ("✅", "Email de Kindle válido") if is_kindle else ("⚠️", "No es un email de Kindle")
            await update.message.reply_html(f"📧 <b>Tu email configurado:</b>\n\n<code>{email}</code>\n\n{status_icon} <b>Estado:</b> {status_text}\n\n🔑 <b>Email autorizado:</b> <code>{self.config.GMAIL_USER}</code>")
        else:
            await update.message.reply_html("❌ <b>No tienes un email configurado</b>\n\n📧 Usa el botón <b>Configurar Email</b> para empezar")

    async def _get_user_email(self, user_id: int) -> Optional[str]:
        try:
            return await asyncio.to_thread(get_user_email, user_id)
        except Exception as e:
            logger.error(f"Error obteniendo email para usuario {user_id}: {e}")
            return None

    @track_metrics('command_stats')
    async def stats_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        user_id = update.effective_user.id
        stats = metrics_collector.get_user_stats(user_id)
        success_rate = stats['success_rate']
        filled_bars = int((success_rate / 100) * 10)
        bar = "█" * filled_bars + "░" * (10 - filled_bars)
        total_users = await asyncio.to_thread(get_total_users)
        summary = await metrics_collector.get_summary()
        stats_message = f"📊 <b>Tus Estadísticas Personales</b>\n\n📄 <b>Documentos:</b>\n• Recibidos: {stats['documents_received']}\n• Enviados exitosamente: {stats['documents_sent']}\n• Tasa de éxito: {success_rate}% {bar}\n\n⚡ <b>Actividad:</b>\n• Comandos ejecutados: {stats['commands_used']}\n• Errores encontrados: {stats['errors_encountered']}\n• Formato preferido: {stats['top_format']}\n\n🏆 <b>Ranking:</b>\n• Eres uno de {total_users} usuarios totales\n• Tiempo promedio de respuesta: {summary['avg_response_time_ms']}ms"
        await update.message.reply_html(stats_message)

    @track_metrics('command_admin')
    async def admin_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        user_id = update.effective_user.id
        if not self.config.ADMIN_USER_ID or user_id != self.config.ADMIN_USER_ID:
            await update.message.reply_text("🚫 Acceso denegado")
            return
            
        summary = await metrics_collector.get_summary()
        success_rate = summary['success_rate']
        filled_bars = int((success_rate / 100) * 10)
        bar = "█" * filled_bars + "░" * (10 - filled_bars)
        top_formats = "\n".join([f"  • <code>{f}:</code> {c}" for f, c in summary['top_formats'][:5]]) if summary['top_formats'] else "Ninguno"
        admin_message = f"👑 <b>Panel de Administración</b>\n\n⏱️ <b>Sistema:</b>\n• Tiempo activo: {summary['uptime_formatted']}\n• Usuarios totales: {summary['total_users']}\n• Versión: 2.2.0\n\n📊 <b>Métricas:</b>\n• Documentos enviados: {summary['total_documents_sent']}\n• Documentos recibidos: {summary['total_documents_received']}\n• Tasa de éxito: {success_rate}% {bar}\n• Comandos ejecutados: {summary['commands_executed']}\n\n❌ <b>Errores:</b>\n• Total: {summary['total_errors']}\n• Últimos errores: {len(summary['recent_errors'])}\n\n⚡ <b>Rendimiento:</b>\n• Tiempo respuesta promedio: {summary['avg_response_time_ms']}ms\n\n📈 <b>Formatos populares:</b>\n{top_formats}"
        await update.message.reply_html(admin_message, reply_markup=self.admin_keyboard)

    @track_metrics('command_clear_cache')
    async def clear_cache_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        user_id = update.effective_user.id
        if not self.config.ADMIN_USER_ID or user_id != self.config.ADMIN_USER_ID:
            await update.message.reply_text("🚫 Acceso denegado")
            return
        cache_manager.clear()
        await update.message.reply_text("🧹 Caché limpiado exitosamente")

    @track_metrics('command_hide_keyboard')
    async def hide_keyboard_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        await update.message.reply_text("🙈 Teclado ocultado\n\n💡 Usa /start para mostrarlo de nuevo", reply_markup=ReplyKeyboardRemove())

    @track_metrics('handle_document')
    async def handle_document(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        user_id = update.effective_user.id
        user_kindle_email = await self._get_user_email(user_id)
        if not user_kindle_email:
            await update.message.reply_html("⚠️ <b>Email no configurado.</b>\n\nUsa /set_email o el botón del menú para empezar.")
            return
            
        doc = update.message.document
        valid, error_msg = self.file_validator.validate_file(
            doc.file_name, doc.file_size, self.config.MAX_FILE_SIZE
        )
        if not valid:
            await update.message.reply_html(f"❌ <b>Error:</b> {error_msg}")
            return
            
        ext = Path(doc.file_name).suffix.lower()
        await metrics_collector.increment('document_received', user_id)
        await metrics_collector.increment(f'format_{ext.replace(".", "")}', user_id)
        
        temp_dir = Path("/tmp/kindleupbot_downloads")
        temp_dir.mkdir(exist_ok=True)
        temp_file_path = temp_dir / f"{doc.file_unique_id}{ext}"
        
        processing_msg = None
        converted_path = None
        try:
            file_obj = await context.bot.get_file(doc.file_id)
            await file_obj.download_to_drive(temp_file_path)
            
            file_to_send_path = temp_file_path
            file_to_send_name = doc.file_name
            subject = ""
            
            if ext == '.md':
                processing_msg = await update.message.reply_html(f"⚙️ Convirtiendo <code>{doc.file_name}</code> a formato DOCX...")
                converted_path, convert_error = await convert_markdown_to_docx(temp_file_path, doc.file_name)
                if convert_error or not converted_path:
                    await processing_msg.edit_text(f"❌ <b>Error al convertir:</b>\n<i>{convert_error}</i>", parse_mode=ParseMode.HTML)
                    return
                file_to_send_path = converted_path
                file_to_send_name = Path(doc.file_name).with_suffix('.docx').name
                subject = f"Doc: {file_to_send_name}"
            elif ext == '.pdf':
                context.user_data['pending_pdf'] = {
                    'temp_path': str(temp_file_path),
                    'filename': doc.file_name,
                    'user_id': user_id
                }
                buttons = [
                    [InlineKeyboardButton("✅ Convertir (texto adaptable)", callback_data="pdf_convert_yes")],
                    [InlineKeyboardButton("❌ Sin convertir (diseño original)", callback_data="pdf_convert_no")]
                ]
                await update.message.reply_html(
                    f"📄 <b>{doc.file_name}</b>\n\n¿Quieres optimizar este PDF para Kindle?",
                    reply_markup=InlineKeyboardMarkup(buttons)
                )
                return
                
            file_data = await asyncio.to_thread(file_to_send_path.read_bytes)
            
            if not processing_msg:
                processing_msg = await update.message.reply_html(f"📤 Enviando <code>{file_to_send_name}</code>...")
            else:
                await processing_msg.edit_text(f"📤 Enviando <code>{file_to_send_name}</code>...", parse_mode=ParseMode.HTML)
                
            success, msg = await self._send_to_kindle_with_retries(user_kindle_email, file_data, file_to_send_name, subject)
            
            if success:
                await metrics_collector.increment('document_sent', user_id)
                if ext == '.md':
                    await metrics_collector.increment('md_converted_docx', user_id)
                await processing_msg.edit_text(f"✅ ¡<b>{file_to_send_name}</b> enviado!", parse_mode=ParseMode.HTML)
            else:
                await processing_msg.edit_text(f"❌ <b>Error al enviar:</b> <i>{msg}</i>", parse_mode=ParseMode.HTML)
        except Exception as e:
            logger.error(f"Error procesando documento para {user_id}: {e}", exc_info=True)
            error_message = "❌ <b>Error inesperado</b> al procesar el archivo."
            if processing_msg:
                await processing_msg.edit_text(error_message, parse_mode=ParseMode.HTML)
            else:
                await update.message.reply_html(error_message)
        finally:
            if ext != '.pdf' and temp_file_path.exists():
                await asyncio.to_thread(temp_file_path.unlink)
            if converted_path and converted_path.exists():
                await asyncio.to_thread(converted_path.unlink)

    async def _send_to_kindle_with_retries(self, kindle_email: str, file_data: bytes, filename: str, subject: str) -> Tuple[bool, str]:
        for attempt in range(self.config.MAX_RETRIES):
            try:
                success, msg = await self._send_to_kindle(kindle_email, file_data, filename, subject)
                if success: return True, msg
                if attempt < self.config.MAX_RETRIES - 1: await asyncio.sleep(self.config.RETRY_DELAY * (2 ** attempt))
            except Exception as e:
                logger.warning(f"Intento {attempt + 1} fallido: {e}")
                if attempt < self.config.MAX_RETRIES - 1: await asyncio.sleep(self.config.RETRY_DELAY * (2 ** attempt))
                else: return False, f"Error después de {self.config.MAX_RETRIES} intentos: {str(e)}"
        return False, "Falló después de todos los reintentos"

    async def _send_to_kindle(self, kindle_email: str, file_data: bytes, filename: str, subject: str) -> Tuple[bool, str]:
        return await asyncio.to_thread(
            self._send_to_kindle_sync, kindle_email, file_data, filename, subject
        )

    def _send_to_kindle_sync(self, kindle_email: str, file_data: bytes, filename: str, subject: str) -> Tuple[bool, str]:
        try:
            nfkd = unicodedata.normalize('NFKD', filename)
            safe_fn = ''.join(c for c in nfkd if unicodedata.category(c) != 'Mn')
            
            msg = MIMEMultipart()
            msg['From'] = self.config.GMAIL_USER
            msg['To'] = kindle_email
            msg['Subject'] = subject or f"Doc: {safe_fn}"
            msg.attach(MIMEText(f"Enviado desde tu Bot de Telegram. Archivo: {safe_fn}"))
            
            subtype = safe_fn.rsplit('.', 1)[-1]
            part = MIMEApplication(file_data, _subtype=subtype)
            part.add_header('Content-Disposition', 'attachment', filename=safe_fn)
            part.add_header('Content-Type', part.get_content_type(), name=safe_fn)
            msg.attach(part)
            
            # Usamos SMTP_SSL para el puerto 465, que es más seguro y directo
            with smtplib.SMTP_SSL(self.config.SMTP_SERVER, self.config.SMTP_PORT) as server:
                server.login(self.config.GMAIL_USER, self.config.GMAIL_APP_PASSWORD)
                server.send_message(msg)
                
            logger.info(f"Documento {safe_fn} enviado a {kindle_email}")
            return True, "Enviado"
            
        except smtplib.SMTPAuthenticationError:
            return False, "Error de autenticación SMTP"
        except smtplib.SMTPRecipientsRefused:
            return False, "Email de destinatario rechazado"
        except Exception as e:
            logger.error(f"Error SMTP al enviar a Kindle: {e}", exc_info=True)
            return False, f"Error SMTP: {str(e)}"

    def run(self):
        """Inicia el bot en modo Polling y se queda corriendo."""
        logger.info("Application started")
        self.application.run_polling()


def main() -> None:
    """Función principal para configurar e iniciar el bot."""
    logger.info("🚀 Iniciando bot...")
    
    # 1. Configurar la base de datos
    setup_database()
    logger.info("✅ Base de datos configurada.")

    # 2. Cargar métricas históricas
    metrics_collector.load_from_db()
    logger.info("✅ Métricas cargadas.")

    # 3. Crear una instancia del bot (esto también configura los handlers)
    bot = KindleEmailBot(settings)
    
    # 4. Iniciar el bot. Esta es una llamada bloqueante.
    bot.run()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.critical(f"CRITICAL: El bot ha fallado fatalmente: {e}", exc_info=True)