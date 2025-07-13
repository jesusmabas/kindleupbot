# main.py - Versión mejorada y corregida
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

from fastapi import FastAPI, HTTPException, Request, BackgroundTasks
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, field_validator
import aiofiles

# --- NUEVA IMPORTACIÓN DE CONFIGURACIÓN ---
from config import settings, Settings

from database import (
    setup_database, set_user_email, get_user_email,
    get_metrics_from_db, save_metric, get_total_users
)

from telegram import Update, ReplyKeyboardMarkup, ReplyKeyboardRemove, ForceReply, InlineKeyboardMarkup, InlineKeyboardButton
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes, CallbackQueryHandler
from telegram.constants import ParseMode
from telegram.error import TelegramError, RetryAfter

# Configuración de logging mejorada
def setup_logging():
    """Configura el sistema de logging con rotación de archivos"""
    from logging.handlers import RotatingFileHandler

    # Crear directorio de logs si no existe
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)

    # Configurar handler con rotación
    file_handler = RotatingFileHandler(
        log_dir / "bot.log",
        maxBytes=10*1024*1024,  # 10MB
        backupCount=5
    )

    # Formato mejorado con más información
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
    )
    file_handler.setFormatter(formatter)

    # Configurar logger principal
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.addHandler(file_handler)
    logger.addHandler(logging.StreamHandler())

    return logger

logger = setup_logging()

# Constantes mejoradas
SUPPORTED_FORMATS = {
    '.epub': 'application/epub+zip',
    '.pdf': 'application/pdf',
    '.mobi': 'application/x-mobipocket-ebook',
    '.azw': 'application/vnd.amazon.ebook',
    '.doc': 'application/msword',
    '.docx': 'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
    '.rtf': 'application/rtf',
    '.txt': 'text/plain',
    '.html': 'text/html',
    '.htm': 'text/html',
    '.jpg': 'image/jpeg',
    '.jpeg': 'image/jpeg',
    '.png': 'image/png',
    '.gif': 'image/gif',
    '.bmp': 'image/bmp'
}

PROMPT_SET_EMAIL = "📧 Por favor, introduce tu email de Kindle (ejemplo: usuario@kindle.com):"

# --- SISTEMA DE CACHE ---
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
            # Programar eliminación automática
            asyncio.create_task(self._auto_expire(key, ttl))

    async def _auto_expire(self, key: str, ttl: int):
        await asyncio.sleep(ttl)
        self.cache.pop(key, None)

    def clear(self):
        self.cache.clear()

# --- SISTEMA DE RATE LIMITING ---
class RateLimiter:
    def __init__(self, max_requests: int, window: int):
        self.max_requests = max_requests
        self.window = window
        self.requests = defaultdict(list)

    def is_allowed(self, user_id: int) -> bool:
        now = time.time()
        user_requests = self.requests[user_id]
        
        # Limpiar requests antiguos
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

# --- SISTEMA DE MÉTRICAS MEJORADO ---
class MetricsCollector:
    def __init__(self):
        self.start_time = time.time()
        self.metrics = defaultdict(int)
        self.user_metrics = defaultdict(lambda: defaultdict(int))
        self.error_log = []
        self.response_times = []
        self.daily_stats = defaultdict(lambda: defaultdict(int))
        self.lock = asyncio.Lock()

    def load_from_db(self):
        """Carga métricas históricas desde la base de datos"""
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
        """Guarda métricas en la base de datos de forma asíncrona"""
        try:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, save_metric, metric_name, user_id, value)
        except Exception as e:
            logger.error(f"Error guardando métrica {metric_name}: {e}")

    async def increment(self, metric_name: str, user_id: Optional[int] = None, value: int = 1):
        """Incrementa una métrica de forma thread-safe"""
        async with self.lock:
            self.metrics[metric_name] += value
            if user_id:
                self.user_metrics[user_id][metric_name] += value

            # Estadísticas diarias
            today = datetime.now().strftime('%Y-%m-%d')
            self.daily_stats[today][metric_name] += value

            # Guardar en BD en segundo plano
            asyncio.create_task(self._save_metric_async(metric_name, user_id, value))

    def log_error(self, error_type: str, error_message: str, user_id: Optional[int] = None):
        """Registra errores con más detalle"""
        error_data = {
            'timestamp': datetime.now().isoformat(),
            'type': error_type,
            'message': error_message,
            'user_id': user_id,
            'traceback': None  # Se podría agregar el traceback si es necesario
        }
        self.error_log.insert(0, error_data)
        if len(self.error_log) > 100:  # Aumentado de 50 a 100
            self.error_log.pop()

        asyncio.create_task(self.increment('errors_total'))
        asyncio.create_task(self.increment(f'error_{error_type}', user_id))

    def log_response_time(self, duration: float, operation: str):
        """Registra tiempos de respuesta"""
        self.response_times.insert(0, {
            'duration': duration,
            'operation': operation,
            'timestamp': time.time()
        })
        if len(self.response_times) > 1000:
            self.response_times.pop()

    def get_summary(self) -> Dict[str, Any]:
        """Obtiene resumen completo de métricas"""
        uptime = time.time() - self.start_time
        avg_response_time = sum(r['duration'] for r in self.response_times) / len(self.response_times) if self.response_times else 0

        return {
            'uptime_formatted': self._format_uptime(uptime),
            'uptime_seconds': uptime,
            'total_users': get_total_users(),
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
        """Calcula la tasa de éxito de envío de documentos"""
        sent = self.metrics.get('document_sent', 0)
        received = self.metrics.get('document_received', 0)
        if received == 0:
            return 100.0
        return round((sent / received) * 100, 2)

    def get_user_stats(self, user_id: int) -> Dict[str, Any]:
        """Obtiene estadísticas detalladas del usuario"""
        user_data = self.user_metrics.get(user_id, {})
        return {
            'documents_sent': user_data.get('document_sent', 0),
            'documents_received': user_data.get('document_received', 0),
            'commands_used': user_data.get('commands_total', 0),
            'errors_encountered': sum(v for k, v in user_data.items() if k.startswith('error_')),
            'top_format': self._get_user_top_format(user_id),
            'success_rate': self._calculate_user_success_rate(user_id),
            'last_activity': self._get_user_last_activity(user_id),
        }

    def _calculate_user_success_rate(self, user_id: int) -> float:
        """Calcula la tasa de éxito para un usuario específico"""
        user_data = self.user_metrics.get(user_id, {})
        sent = user_data.get('document_sent', 0)
        received = user_data.get('document_received', 0)
        if received == 0:
            return 100.0
        return round((sent / received) * 100, 2)

    def _get_user_last_activity(self, user_id: int) -> str:
        """Obtiene la fecha de última actividad del usuario"""
        # Esto sería mejor implementarlo en la base de datos
        return "N/A"

    def _format_uptime(self, seconds: float) -> str:
        """Formatea el tiempo de actividad"""
        days, rem = divmod(seconds, 86400)
        hours, rem = divmod(rem, 3600)
        minutes, seconds = divmod(rem, 60)
        return f"{int(days)}d {int(hours)}h {int(minutes)}m {int(seconds)}s"

    def _get_top_formats(self) -> List[Tuple[str, int]]:
        """Obtiene los formatos más populares"""
        formats = {k.replace('format_', ''): v for k, v in self.metrics.items() if k.startswith('format_')}
        return sorted(formats.items(), key=lambda x: x[1], reverse=True)[:10]

    def _get_user_top_format(self, user_id: int) -> str:
        """Obtiene el formato más usado por un usuario"""
        user_data = self.user_metrics.get(user_id, {})
        formats = {k.replace('format_', ''): v for k, v in user_data.items() if k.startswith('format_')}
        return max(formats.items(), key=lambda x: x[1])[0] if formats else "Ninguno"

# Instancias globales
metrics_collector = MetricsCollector()
cache_manager = CacheManager(default_ttl=settings.CACHE_DURATION)
rate_limiter = RateLimiter(
    max_requests=settings.RATE_LIMIT_MAX_REQUESTS,
    window=settings.RATE_LIMIT_WINDOW
)

# --- DECORADOR MEJORADO PARA MÉTRICAS ---
def track_metrics(operation_name: str):
    """Decorador mejorado para tracking de métricas"""
    def decorator(func):
        async def wrapper(self, update: Update, *args, **kwargs):
            start_time = time.time()
            user_id = update.effective_user.id if update and hasattr(update, 'effective_user') else None

            # Rate limiting
            if user_id and not rate_limiter.is_allowed(user_id):
                remaining_time = rate_limiter.get_remaining_time(user_id)
                await update.message.reply_text(
                    f"🚫 Has excedido el límite de solicitudes. "
                    f"Intenta de nuevo en {remaining_time} segundos."
                )
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

                # Mensaje de error más amigable
                if hasattr(update, 'message') and update.message:
                    await update.message.reply_text(
                        "😔 Ocurrió un error inesperado. El equipo técnico ha sido notificado."
                    )
                raise
            finally:
                duration = time.time() - start_time
                metrics_collector.log_response_time(duration, operation_name)

        return wrapper
    return decorator

# --- MODELOS PYDANTIC MEJORADOS ---
class StatusResponse(BaseModel):
    status: str
    bot_username: Optional[str] = None
    metrics: Optional[Dict[str, Any]] = None
    version: str = "2.2.0"

class EmailValidationRequest(BaseModel):
    email: str

    @field_validator('email')
    def validate_email(cls, v):
        email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        if not re.match(email_pattern, v):
            raise ValueError('Formato de email inválido')
        return v.lower()

# --- VALIDADORES MEJORADOS ---
class EmailValidator:
    @staticmethod
    def validate_email(email: str) -> bool:
        """Validación mejorada de email"""
        if not email or len(email) < 5:
            return False

        email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        return bool(re.match(email_pattern, email))

    @staticmethod
    def is_kindle_email(email: str) -> bool:
        """Verifica si es un email de Kindle"""
        kindle_domains = ['@kindle.com', '@free.kindle.com']
        return any(domain in email.lower() for domain in kindle_domains)

class FileValidator:
    @staticmethod
    def validate_file(filename: str, file_size: int, max_size: int) -> Tuple[bool, str]:
        """Validación completa de archivos"""
        if not filename:
            return False, "Nombre de archivo vacío"

        ext = Path(filename).suffix.lower()
        if ext not in SUPPORTED_FORMATS:
            return False, f"Formato {ext} no soportado"

        if file_size > max_size:
            return False, f"Archivo muy grande ({file_size / 1024**2:.1f}MB > {max_size / 1024**2:.1f}MB)"

        return True, "OK"

# --- GESTOR DE CICLO DE VIDA MEJORADO ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Gestor de ciclo de vida mejorado"""
    logger.info("🚀 Iniciando servidor...")

    try:
        # La configuración se valida automáticamente al importar 'settings' desde config.py
        logger.info("✅ Configuración cargada y validada")

        # Configurar base de datos
        setup_database()
        logger.info("✅ Base de datos configurada")

        # Cargar métricas
        metrics_collector.load_from_db()
        logger.info("✅ Métricas cargadas")

        # Inicializar bot
        bot_instance = KindleEmailBot(settings)
        await bot_instance.initialize()
        logger.info("✅ Bot inicializado")

        # Almacenar en el estado de la aplicación
        app.state.bot = bot_instance
        app.state.config = settings
        app.state.metrics = metrics_collector
        app.state.cache = cache_manager

        logger.info("✅ Servidor iniciado correctamente")
        yield

    except Exception as e:
        logger.error(f"Error durante el inicio: {e}", exc_info=True)
        raise
    finally:
        logger.info("🛑 Cerrando servidor...")
        if hasattr(app.state, 'bot'):
            await app.state.bot.shutdown()
        logger.info("✅ Servidor cerrado correctamente")

# --- APLICACIÓN FASTAPI MEJORADA ---
app = FastAPI(
    title="Kindle Bot API",
    version="2.2.0",
    description="Bot de Telegram para envío de documentos a Kindle",
    lifespan=lifespan
)

# Middleware CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

templates = Jinja2Templates(directory="templates")

@app.get("/", response_model=StatusResponse)
async def read_root():
    """Endpoint de estado principal"""
    try:
        bot_info = await app.state.bot.get_bot_info()
        summary = metrics_collector.get_summary()

        return StatusResponse(
            status="✅ Bot activo y funcionando",
            bot_username=bot_info.username if bot_info else None,
            metrics=summary
        )
    except Exception as e:
        logger.error(f"Error en endpoint raíz: {e}")
        return StatusResponse(
            status="❌ Error en el servicio",
            metrics={"error": str(e)}
        )

# --- CAMBIO APLICADO AQUÍ ---
@app.api_route("/health", methods=["GET", "HEAD"])
async def health_check():
    """Endpoint de health check (soporta GET y HEAD)"""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard(request: Request):
    """Dashboard web"""
    return templates.TemplateResponse("dashboard.html", {"request": request})

@app.get("/api/metrics-data", response_class=JSONResponse)
async def metrics_data():
    """Endpoint de métricas JSON"""
    return metrics_collector.get_summary()

@app.post("/api/clear-cache")
async def clear_cache():
    """Endpoint para limpiar caché"""
    cache_manager.clear()
    return {"message": "Caché limpiado exitosamente"}

# --- CLASE PRINCIPAL DEL BOT (MEJORADA) ---
class KindleEmailBot:
    def __init__(self, config: Settings):
        self.config = config
        self.application = Application.builder().token(self.config.TELEGRAM_BOT_TOKEN).build()
        self.email_validator = EmailValidator()
        self.file_validator = FileValidator()

        # Teclados mejorados
        self.main_keyboard = ReplyKeyboardMarkup([
            ["📧 Configurar Email", "🔍 Ver Mi Email"],
            ["📊 Mis Estadísticas", "❓ Ayuda"],
            ["🎯 Formatos Soportados", "🚀 Consejos"]
        ], resize_keyboard=True)

        self.admin_keyboard = ReplyKeyboardMarkup([
            ["👑 Panel Admin", "📈 Métricas"],
            ["🧹 Limpiar Cache", "🔄 Reiniciar"],
            ["👥 Usuarios", "🏠 Menú Principal"]
        ], resize_keyboard=True)

        # Teclado inline para confirmaciones
        self.confirm_keyboard = InlineKeyboardMarkup([
            [InlineKeyboardButton("✅ Confirmar", callback_data="confirm")],
            [InlineKeyboardButton("❌ Cancelar", callback_data="cancel")]
        ])

    async def initialize(self):
        """Inicialización mejorada del bot"""
        try:
            # Registrar handlers
            handlers = [
                CommandHandler("start", self.start),
                CommandHandler("help", self.help_command),
                CommandHandler("set_email", self.set_email_command),
                CommandHandler("my_email", self.my_email_command),
                CommandHandler("stats", self.stats_command),
                CommandHandler("admin", self.admin_command),
                CommandHandler("hide_keyboard", self.hide_keyboard_command),
                CommandHandler("formats", self.formats_command),
                CommandHandler("tips", self.tips_command),
                CommandHandler("clear_cache", self.clear_cache_command),

                # Handlers de callback
                CallbackQueryHandler(self.handle_callback),

                # Handlers de mensajes
                MessageHandler(filters.TEXT & ~filters.COMMAND & filters.REPLY, self.handle_email_input),
                MessageHandler(filters.Document.ALL, self.handle_document),
                MessageHandler(filters.TEXT & ~filters.COMMAND, self.handle_text),
            ]

            # Añadir handlers
            for handler in handlers:
                self.application.add_handler(handler)

            # Inicializar aplicación
            await self.application.initialize()
            await self.application.updater.start_polling(
                drop_pending_updates=True,
                allowed_updates=Update.ALL_TYPES
            )
            await self.application.start()

            logger.info("🤖 Bot inicializado correctamente")

        except Exception as e:
            logger.error(f"Error inicializando bot: {e}", exc_info=True)
            raise

    async def shutdown(self):
        """Cierre limpio del bot"""
        logger.info("Iniciando secuencia de apagado del bot...")
        try:
            if self.application and self.application.updater:
                if self.application.updater.is_running:
                    logger.info("Deteniendo el polling del updater...")
                    await self.application.updater.stop()
                else:
                    logger.warning("El updater no estaba corriendo al momento del apagado.")

            if self.application:
                logger.info("Deteniendo la aplicación de Telegram...")
                await self.application.stop()

                logger.info("Realizando el shutdown de la aplicación de Telegram...")
                await self.application.shutdown()

            logger.info("🤖 Bot cerrado correctamente")
        except RuntimeError as re:
            # Capturamos el error específico para dar un mensaje más claro,
            # aunque la lógica anterior debería prevenirlo.
            logger.error(f"Error de Runtime durante el apagado: {re}. Esto puede pasar si la secuencia se interrumpe.")
        except Exception as e:
            logger.error(f"Error inesperado cerrando el bot: {e}", exc_info=True)

    async def get_bot_info(self):
        """Obtiene información del bot"""
        try:
            return await self.application.bot.get_me() if self.application else None
        except Exception as e:
            logger.error(f"Error obteniendo info del bot: {e}")
            return None

    # Handlers mejorados
    @track_metrics('command_start')
    async def start(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Comando de inicio mejorado"""
        user = update.effective_user
        is_admin = self.config.ADMIN_USER_ID and user.id == self.config.ADMIN_USER_ID

        welcome_message = f"""
🎉 ¡Bienvenido, {user.mention_html()}!

📚 <b>Kindle Bot v2.2</b> - Tu asistente personal para envío de documentos

🚀 <b>Pasos para empezar:</b>
1. Configura tu email con "📧 Configurar Email"
2. Autoriza mi email en tu cuenta de Amazon
3. ¡Envía tus documentos!

📧 <b>Email a autorizar:</b> <code>{self.config.GMAIL_USER}</code>

{'👑 <b>Acceso de administrador detectado</b>' if is_admin else ''}
"""

        keyboard = self.admin_keyboard if is_admin else self.main_keyboard
        await update.message.reply_html(welcome_message, reply_markup=keyboard)

    @track_metrics('command_help')
    async def help_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Comando de ayuda mejorado"""
        help_text = f"""
📖 <b>Guía Completa del Kindle Bot</b>

🔧 <b>Comandos Principales:</b>
• <b>Configurar Email</b> - Establece tu email de Kindle
• <b>Ver Mi Email</b> - Muestra tu email actual
• <b>Mis Estadísticas</b> - Tus métricas personales
• <b>Formatos Soportados</b> - Lista de formatos válidos

📋 <b>Cómo usar:</b>
1. <b>Configura tu email</b> usando el botón correspondiente
2. <b>Autoriza mi email</b> en tu cuenta de Amazon Kindle
3. <b>Envía cualquier documento</b> compatible

💡 <b>Consejos Pro:</b>
• Usa "convert" en la descripción de PDFs para optimizar
• Los archivos se envían directamente a tu biblioteca
• Máximo {self.config.MAX_FILE_SIZE // 1024**2}MB por archivo

🔑 <b>Email a autorizar:</b>
<code>{self.config.GMAIL_USER}</code>

❓ <b>¿Problemas?</b> Verifica que el email esté autorizado en tu cuenta de Amazon.
"""
        await update.message.reply_html(help_text)

    @track_metrics('command_formats')
    async def formats_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Comando para mostrar formatos soportados"""
        formats_by_category = {
            "📚 Libros Electrónicos": [".epub", ".mobi", ".azw"],
            "📄 Documentos": [".pdf", ".doc", ".docx", ".rtf", ".txt", ".html"],
            "🖼️ Imágenes": [".jpg", ".jpeg", ".png", ".gif", ".bmp"]
        }

        message = "📋 <b>Formatos Soportados</b>\n\n"
        for category, extensions in formats_by_category.items():
            message += f"<b>{category}:</b>\n"
            message += " • " + " • ".join(extensions) + "\n\n"

        message += f"📊 <b>Límite de tamaño:</b> {self.config.MAX_FILE_SIZE // 1024**2}MB"

        await update.message.reply_html(message)

    @track_metrics('command_tips')
    async def tips_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Comando con consejos y trucos"""
        tips_message = """
🚀 <b>Consejos y Trucos</b>

💡 <b>Para PDFs:</b>
• Escribe "convert" en la descripción para optimizar lectura
• Los PDFs se convierten automáticamente al formato Kindle

📧 <b>Configuración de Email:</b>
• Usa tu email @kindle.com (no @amazon.com)
• Autoriza mi email en "Manage Your Content and Devices"
• Verifica tu configuración en Amazon

⚡ <b>Optimización:</b>
• Archivos más pequeños se envían más rápido
• Usa formatos nativos (.epub, .mobi) para mejor experiencia
• Los nombres de archivo se preservan

🔒 <b>Seguridad:</b>
• Tus archivos se envían directamente, no se almacenan
• Solo tú tienes acceso a tus documentos
• Configuración encriptada

📱 <b>Uso Móvil:</b>
• Funciona perfectamente desde móvil
• Envía desde cualquier chat con archivos
• Sincronización automática con todos tus dispositivos Kindle
"""
        await update.message.reply_html(tips_message)

    @track_metrics('command_set_email')
    async def set_email_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Comando mejorado para configurar email"""
        await update.message.reply_text(
            PROMPT_SET_EMAIL,
            reply_markup=ForceReply(
                selective=True,
                input_field_placeholder="usuario@kindle.com"
            )
        )

    @track_metrics('handle_email_input')
    async def handle_email_input(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Manejo mejorado de entrada de email"""
        if not (update.message.reply_to_message and
                update.message.reply_to_message.text == PROMPT_SET_EMAIL):
            return

        user_id = update.effective_user.id
        kindle_email = update.message.text.strip()

        # Validación mejorada
        if not self.email_validator.validate_email(kindle_email):
            await metrics_collector.increment('email_validation_failed', user_id)
            await update.message.reply_html(
                "❌ <b>Formato de email inválido</b>\n\n"
                "📧 Formato correcto: <code>usuario@kindle.com</code>\n"
                "🔄 Inténtalo de nuevo con /set_email"
            )
            return

        # Verificar si es un email de Kindle
        if not self.email_validator.is_kindle_email(kindle_email):
            await update.message.reply_html(
                "⚠️ <b>Advertencia:</b> Este no parece ser un email de Kindle\n\n"
                "📧 Los emails de Kindle terminan en:\n"
                "• @kindle.com\n"
                "• @free.kindle.com\n\n"
                "¿Estás seguro de que es correcto?",
                reply_markup=self.confirm_keyboard
            )
            # Guardar temporalmente para confirmar
            context.user_data['pending_email'] = kindle_email
            return

        # Guardar email
        if await self._save_user_email(user_id, kindle_email):
            await metrics_collector.increment('email_set_success', user_id)
            await update.message.reply_html(
                f"✅ <b>Email configurado correctamente</b>\n\n"
                f"📧 <b>Tu email:</b> <code>{kindle_email}</code>\n\n"
                f"🔑 <b>Recuerda autorizar:</b> <code>{self.config.GMAIL_USER}</code>"
            )
        else:
            await update.message.reply_html(
                "❌ <b>Error al guardar el email</b>\n\n"
                "🔄 Por favor, inténtalo de nuevo"
            )

    async def handle_callback(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Manejo de callbacks de botones inline"""
        query = update.callback_query
        await query.answer()

        if query.data == "confirm":
            user_id = update.effective_user.id
            pending_email = context.user_data.get('pending_email')

            if pending_email and await self._save_user_email(user_id, pending_email):
                await metrics_collector.increment('email_set_success', user_id)
                await query.edit_message_text(
                    f"✅ <b>Email configurado</b>\n\n"
                    f"📧 <code>{pending_email}</code>",
                    parse_mode=ParseMode.HTML
                )
            else:
                await query.edit_message_text("❌ Error al guardar el email")

            context.user_data.pop('pending_email', None)

        elif query.data == "cancel":
            await query.edit_message_text("❌ Configuración cancelada")
            context.user_data.pop('pending_email', None)

    async def _save_user_email(self, user_id: int, email: str) -> bool:
        """Guarda el email del usuario de forma asíncrona"""
        try:
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, set_user_email, user_id, email)
        except Exception as e:
            logger.error(f"Error guardando email para usuario {user_id}: {e}")
            return False

    @track_metrics('command_my_email')
    async def my_email_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Comando mejorado para ver email actual"""
        user_id = update.effective_user.id

        # Usar caché si está disponible
        cache_key = f"user_email_{user_id}"
        email = cache_manager.get(cache_key)

        if email is None:
            email = await self._get_user_email_async(user_id)
            if email:
                cache_manager.set(cache_key, email, 300)  # 5 minutos de caché

        if email:
            is_kindle = self.email_validator.is_kindle_email(email)
            status_icon = "✅" if is_kindle else "⚠️"
            status_text = "Email de Kindle válido" if is_kindle else "No es un email de Kindle"

            await update.message.reply_html(
                f"📧 <b>Tu email configurado:</b>\n\n"
                f"<code>{email}</code>\n\n"
                f"{status_icon} <b>Estado:</b> {status_text}\n\n"
                f"🔑 <b>Email autorizado:</b> <code>{self.config.GMAIL_USER}</code>"
            )
        else:
            await update.message.reply_html(
                "❌ <b>No tienes un email configurado</b>\n\n"
                "📧 Usa el botón <b>Configurar Email</b> para empezar"
            )

    async def _get_user_email_async(self, user_id: int) -> Optional[str]:
        """Obtiene el email del usuario de forma asíncrona"""
        try:
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, get_user_email, user_id)
        except Exception as e:
            logger.error(f"Error obteniendo email para usuario {user_id}: {e}")
            return None

    @track_metrics('command_stats')
    async def stats_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Comando mejorado de estadísticas"""
        user_id = update.effective_user.id
        stats = metrics_collector.get_user_stats(user_id)

        # Crear gráfico de barras simple para success rate
        success_rate = stats['success_rate']
        bar_length = 10
        filled_bars = int((success_rate / 100) * bar_length)
        bar = "█" * filled_bars + "░" * (bar_length - filled_bars)

        stats_message = f"""
📊 <b>Tus Estadísticas Personales</b>

📄 <b>Documentos:</b>
• Recibidos: {stats['documents_received']}
• Enviados exitosamente: {stats['documents_sent']}
• Tasa de éxito: {success_rate}% {bar}

⚡ <b>Actividad:</b>
• Comandos ejecutados: {stats['commands_used']}
• Errores encontrados: {stats['errors_encountered']}
• Formato preferido: {stats['top_format']}

🏆 <b>Ranking:</b>
• Eres uno de {get_total_users()} usuarios totales
• Tiempo promedio de respuesta: {metrics_collector.get_summary()['avg_response_time_ms']}ms
"""

        await update.message.reply_html(stats_message)

    @track_metrics('command_admin')
    async def admin_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Panel de administración mejorado"""
        user_id = update.effective_user.id

        if not self.config.ADMIN_USER_ID or user_id != self.config.ADMIN_USER_ID:
            await update.message.reply_text("🚫 Acceso denegado")
            return

        summary = metrics_collector.get_summary()

        # Crear gráfico de barras para success rate
        success_rate = summary['success_rate']
        bar_length = 10
        filled_bars = int((success_rate / 100) * bar_length)
        bar = "█" * filled_bars + "░" * (bar_length - filled_bars)

        # Top formatos
        top_formats = "\n".join([
            f"  • <code>{f}:</code> {c}" for f, c in summary['top_formats'][:5]
        ]) if summary['top_formats'] else "Ninguno"

        admin_message = f"""
👑 <b>Panel de Administración</b>

⏱️ <b>Sistema:</b>
• Tiempo activo: {summary['uptime_formatted']}
• Usuarios totales: {summary['total_users']}
• Versión: 2.2.0

📊 <b>Métricas:</b>
• Documentos enviados: {summary['total_documents_sent']}
• Documentos recibidos: {summary['total_documents_received']}
• Tasa de éxito: {success_rate}% {bar}
• Comandos ejecutados: {summary['commands_executed']}

❌ <b>Errores:</b>
• Total: {summary['total_errors']}
• Últimos errores: {len(summary['recent_errors'])}

⚡ <b>Rendimiento:</b>
• Tiempo respuesta promedio: {summary['avg_response_time_ms']}ms

📈 <b>Formatos populares:</b>
{top_formats}
"""

        await update.message.reply_html(admin_message, reply_markup=self.admin_keyboard)

    @track_metrics('command_clear_cache')
    async def clear_cache_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Comando para limpiar caché (solo admin)"""
        user_id = update.effective_user.id

        if not self.config.ADMIN_USER_ID or user_id != self.config.ADMIN_USER_ID:
            await update.message.reply_text("🚫 Acceso denegado")
            return

        cache_manager.clear()
        await update.message.reply_text("🧹 Caché limpiado exitosamente")

    @track_metrics('command_hide_keyboard')
    async def hide_keyboard_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Oculta el teclado"""
        await update.message.reply_text(
            "🙈 Teclado ocultado\n\n"
            "💡 Usa /start para mostrarlo de nuevo",
            reply_markup=ReplyKeyboardRemove()
        )

    @track_metrics('handle_document')
    async def handle_document(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Manejo mejorado de documentos"""
        user_id = update.effective_user.id

        # Verificar email configurado
        user_kindle_email = await self._get_user_email_async(user_id)
        if not user_kindle_email:
            await update.message.reply_html(
                "⚠️ <b>Email no configurado</b>\n\n"
                "📧 Usa <b>Configurar Email</b> primero"
            )
            return

        doc = update.message.document

        # Validar archivo
        valid, error_msg = self.file_validator.validate_file(
            doc.file_name, doc.file_size, self.config.MAX_FILE_SIZE
        )

        if not valid:
            await update.message.reply_html(f"❌ <b>Error:</b> {error_msg}")
            return

        # Detectar formato
        ext = Path(doc.file_name).suffix.lower()
        await metrics_collector.increment('document_received', user_id)
        await metrics_collector.increment(f'format_{ext.replace(".", "")}', user_id)

        # Mensaje de procesamiento
        processing_msg = await update.message.reply_html(
            f"⏳ <b>Procesando documento...</b>\n\n"
            f"📄 <b>Archivo:</b> <code>{doc.file_name}</code>\n"
            f"📊 <b>Tamaño:</b> {doc.file_size / 1024**2:.1f}MB\n"
            f"🎯 <b>Destino:</b> <code>{user_kindle_email}</code>"
        )

        try:
            # Descargar archivo
            download_start = time.time()
            file_obj = await context.bot.get_file(doc.file_id)
            file_data = await file_obj.download_as_bytearray()
            download_time = time.time() - download_start

            # Determinar si convertir PDF
            subject = ""
            if (doc.file_name.lower().endswith('.pdf') and
                update.message.caption and
                'convert' in update.message.caption.lower()):
                subject = "Convert"

            # Actualizar mensaje
            await processing_msg.edit_text(
                f"📤 <b>Enviando a Kindle...</b>\n\n"
                f"📄 <b>Archivo:</b> <code>{doc.file_name}</code>\n"
                f"⏱️ <b>Descarga:</b> {download_time:.1f}s\n"
                f"🎯 <b>Destino:</b> <code>{user_kindle_email}</code>",
                parse_mode=ParseMode.HTML
            )

            # Enviar con reintentos
            success, msg = await self._send_to_kindle_with_retries(
                user_kindle_email, file_data, doc.file_name, subject
            )

            if success:
                await metrics_collector.increment('document_sent', user_id)
                await processing_msg.edit_text(
                    f"✅ <b>¡Documento enviado exitosamente!</b>\n\n"
                    f"📄 <b>Archivo:</b> <code>{doc.file_name}</code>\n"
                    f"📧 <b>Enviado a:</b> <code>{user_kindle_email}</code>\n"
                    f"🚀 <b>Sincronizando con tu Kindle...</b>",
                    parse_mode=ParseMode.HTML
                )
            else:
                await processing_msg.edit_text(
                    f"❌ <b>Error al enviar documento</b>\n\n"
                    f"📄 <b>Archivo:</b> <code>{doc.file_name}</code>\n"
                    f"⚠️ <b>Error:</b> <i>{msg}</i>\n\n"
                    f"💡 <b>Verifica que el email esté autorizado</b>",
                    parse_mode=ParseMode.HTML
                )

        except Exception as e:
            logger.error(f"Error procesando documento para usuario {user_id}: {e}", exc_info=True)
            await processing_msg.edit_text(
                f"❌ <b>Error inesperado</b>\n\n"
                f"📄 <b>Archivo:</b> <code>{doc.file_name}</code>\n"
                f"🔧 <b>Error técnico registrado</b>",
                parse_mode=ParseMode.HTML
            )

    async def _send_to_kindle_with_retries(self, kindle_email: str, file_data: bytes,
                                          filename: str, subject: str) -> Tuple[bool, str]:
        """Envía documento con sistema de reintentos"""
        for attempt in range(self.config.MAX_RETRIES):
            try:
                success, msg = await self._send_to_kindle_async(
                    kindle_email, file_data, filename, subject
                )

                if success:
                    return True, msg

                # Si no es el último intento, esperar antes de reintentar
                if attempt < self.config.MAX_RETRIES - 1:
                    await asyncio.sleep(self.config.RETRY_DELAY * (2 ** attempt))

            except Exception as e:
                logger.warning(f"Intento {attempt + 1} fallido: {e}")
                if attempt < self.config.MAX_RETRIES - 1:
                    await asyncio.sleep(self.config.RETRY_DELAY * (2 ** attempt))
                else:
                    return False, f"Error después de {self.config.MAX_RETRIES} intentos: {str(e)}"

        return False, "Falló después de todos los reintentos"

    async def _send_to_kindle_async(self, kindle_email: str, file_data: bytes,
                                   filename: str, subject: str) -> Tuple[bool, str]:
        """Envía documento de forma asíncrona"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, self._send_to_kindle_sync, kindle_email, file_data, filename, subject
        )

    def _send_to_kindle_sync(self, kindle_email: str, file_data: bytes,
                            filename: str, subject: str) -> Tuple[bool, str]:
        """Envía documento de forma síncrona (mejorado)"""
        try:
            # Crear mensaje
            msg = MIMEMultipart()
            msg['From'] = self.config.GMAIL_USER
            msg['To'] = kindle_email
            msg['Subject'] = subject or f"Documento: {filename}"

            # Cuerpo del mensaje
            body = f"""
Documento enviado desde tu Bot de Telegram

📄 Archivo: {filename}
📅 Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
🤖 Enviado por: Kindle Bot v2.2

¡Disfruta tu lectura!
"""
            msg.attach(MIMEText(body, 'plain', 'utf-8'))

            # Adjuntar archivo
            ctype, encoding = mimetypes.guess_type(filename)
            if ctype is None or encoding is not None:
                ctype = 'application/octet-stream'

            maintype, subtype = ctype.split('/', 1)
            attachment = MIMEBase(maintype, subtype)
            attachment.set_payload(file_data)
            encoders.encode_base64(attachment)
            attachment.add_header(
                'Content-Disposition',
                f'attachment; filename="{filename}"'
            )
            msg.attach(attachment)

            # Enviar email
            with smtplib.SMTP(self.config.SMTP_SERVER, self.config.SMTP_PORT) as server:
                server.starttls()
                server.login(self.config.GMAIL_USER, self.config.GMAIL_APP_PASSWORD)
                server.send_message(msg)

            logger.info(f"Documento {filename} enviado exitosamente a {kindle_email}")
            return True, "Documento enviado exitosamente"

        except smtplib.SMTPAuthenticationError:
            error_msg = "Error de autenticación SMTP"
            logger.error(f"Error SMTP Auth enviando a {kindle_email}")
            return False, error_msg
        except smtplib.SMTPRecipientsRefused:
            error_msg = "Email de destinatario rechazado"
            logger.error(f"Email rechazado: {kindle_email}")
            return False, error_msg
        except Exception as e:
            error_msg = f"Error SMTP: {str(e)}"
            logger.error(f"Error SMTP enviando a {kindle_email}: {e}")
            return False, error_msg

    @track_metrics('handle_text')
    async def handle_text(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Manejo mejorado de texto que reconoce los botones del teclado."""
        text = update.message.text
        user_id = update.effective_user.id
        is_admin = self.config.ADMIN_USER_ID and user_id == self.config.ADMIN_USER_ID

        # ---- Lógica para los botones del menú principal ----
        if text == "📧 Configurar Email":
            await self.set_email_command(update, context)
        elif text == "🔍 Ver Mi Email":
            await self.my_email_command(update, context)
        elif text == "📊 Mis Estadísticas":
            await self.stats_command(update, context)
        elif text == "❓ Ayuda":
            await self.help_command(update, context)
        elif text == "🎯 Formatos Soportados":
            await self.formats_command(update, context)
        elif text == "🚀 Consejos":
            await self.tips_command(update, context)

        # ---- Lógica para los botones del menú de administrador ----
        elif is_admin and text == "👑 Panel Admin":
            await self.admin_command(update, context)
        elif is_admin and text == "📈 Métricas":
            await self.admin_command(update, context) # Llama al mismo panel
        elif is_admin and text == "🧹 Limpiar Cache":
            await self.clear_cache_command(update, context)
        elif is_admin and text == "🔄 Reiniciar":
            await update.message.reply_text("Esta función debe ser implementada por el administrador del servidor (ej. systemctl restart).")
        elif is_admin and text == "👥 Usuarios":
            await update.message.reply_text(f"👥 Hay un total de {get_total_users()} usuarios registrados.")
        elif is_admin and text == "🏠 Menú Principal":
            await update.message.reply_text("Volviendo al menú principal...", reply_markup=self.main_keyboard)

        # ---- Respuestas contextuales y mensaje por defecto ----
        else:
            # Convertimos a minúsculas solo para las palabras clave
            text_lower = text.lower()
            if any(word in text_lower for word in ['hola', 'hello', 'hi', 'buenas']):
                await update.message.reply_text(
                    "¡Hola! 👋 Soy tu asistente de Kindle.\n"
                    "Envíame un documento para empezar.",
                    reply_markup=self.main_keyboard
                )
            elif any(word in text_lower for word in ['ayuda', 'help', 'auxilio']):
                await self.help_command(update, context)
            elif any(word in text_lower for word in ['gracias', 'thanks', 'thank you']):
                await update.message.reply_text(
                    "¡De nada! 😊 Estoy aquí para ayudarte con tus documentos Kindle."
                )
            else:
                await update.message.reply_html(
                    "🤔 <b>No entiendo ese mensaje</b>\n\n"
                    "💡 <b>Puedo ayudarte con:</b>\n"
                    "• Configurar tu email de Kindle\n"
                    "• Enviar documentos a tu dispositivo\n"
                    "• Mostrar estadísticas de uso\n\n"
                    "📄 <b>Envía un documento</b> o usa los botones del menú",
                    reply_markup=self.main_keyboard
                )

# --- PUNTO DE ENTRADA ---
if __name__ == "__main__":
    try:
        logger.info(f"Iniciando servidor en {settings.HOST}:{settings.PORT}")

        uvicorn.run(
            app,
            host=settings.HOST,
            port=settings.PORT,
            log_level="info",
            access_log=True
        )
    except Exception as e:
        logger.error(f"Error fatal al iniciar servidor: {e}", exc_info=True)
        raise
