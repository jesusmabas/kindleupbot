# main.py
import os
import logging
import smtplib
import mimetypes
import asyncio
import uvicorn
from contextlib import asynccontextmanager
from typing import Optional, Tuple, Dict, Any
from dataclasses import dataclass
from datetime import datetime
from collections import defaultdict
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email.mime.text import MIMEText
from email import encoders
import time

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from database import setup_database, set_user_email, get_user_email, get_metrics_from_db, save_metric, get_total_users

from telegram import Update, ReplyKeyboardMarkup, ReplyKeyboardRemove, ForceReply
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
from telegram.constants import ParseMode
from telegram.error import TelegramError

# --- CONFIGURACIÓN Y CONSTANTES ---
@dataclass
class BotConfig:
    bot_token: str
    gmail_user: str
    gmail_password: str
    max_file_size: int = 48 * 1024 * 1024
    smtp_server: str = 'smtp.gmail.com'
    smtp_port: int = 587
    admin_user_id: Optional[int] = None

logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO,
    handlers=[logging.FileHandler('bot.log'), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

SUPPORTED_FORMATS = ('.epub', '.pdf', '.mobi', '.azw', '.doc', '.docx', '.rtf', '.txt', '.html', '.htm', '.jpg', '.jpeg', '.png', '.gif', '.bmp')
PROMPT_SET_EMAIL = "De acuerdo, por favor, introduce ahora tu email de Kindle:"

# --- SISTEMA DE MÉTRICAS (CORREGIDO) ---
class MetricsCollector:
    def __init__(self):
        self.start_time = time.time()
        self.metrics = defaultdict(int)
        self.user_metrics = defaultdict(lambda: defaultdict(int))
        self.error_log = []
        self.response_times = []

    def load_from_db(self):
        logger.info("Cargando métricas históricas desde la base de datos...")
        historical_metrics = get_metrics_from_db()
        for metric_name, user_id, value in historical_metrics:
            self.metrics[metric_name] += value
            if user_id:
                self.user_metrics[user_id][metric_name] += value
        logger.info(f"{len(historical_metrics)} registros de métricas cargados.")

    # --- CORRECCIÓN AQUÍ ---
    async def _save_metric_async(self, metric_name: str, user_id: Optional[int], value: int):
        """Ejecuta la función síncrona de guardado en un hilo separado."""
        try:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, save_metric, metric_name, user_id, value)
        except Exception as e:
            logger.error(f"Error en la tarea de guardado de métrica en segundo plano: {e}")

    def increment(self, metric_name: str, user_id: Optional[int] = None, value: int = 1):
        """Incrementa una métrica y la guarda en BD en segundo plano."""
        self.metrics[metric_name] += value
        if user_id:
            self.user_metrics[user_id][metric_name] += value
        # Ahora creamos una tarea a partir de nuestra corutina wrapper, que es lo correcto.
        asyncio.create_task(self._save_metric_async(metric_name, user_id, value))

    def log_error(self, error_type: str, error_message: str, user_id: Optional[int] = None):
        error_data = {'timestamp': datetime.now().isoformat(), 'type': error_type, 'message': error_message, 'user_id': user_id}
        self.error_log.insert(0, error_data)
        if len(self.error_log) > 50: self.error_log.pop()
        self.increment('errors_total')
        self.increment(f'error_{error_type}', user_id)

    def log_response_time(self, duration: float, operation: str):
        self.response_times.insert(0, {'duration': duration, 'operation': operation})
        if len(self.response_times) > 1000: self.response_times.pop()

    def get_summary(self) -> Dict[str, Any]:
        uptime = time.time() - self.start_time
        avg_response_time = sum(r['duration'] for r in self.response_times) / len(self.response_times) if self.response_times else 0
        return {
            'uptime_formatted': self._format_uptime(uptime),
            'total_users': get_total_users(),
            'total_documents_sent': self.metrics.get('document_sent', 0),
            'total_errors': self.metrics.get('errors_total', 0),
            'commands_executed': self.metrics.get('commands_total', 0),
            'avg_response_time_ms': round(avg_response_time * 1000, 2),
            'recent_errors': self.error_log[:10],
            'top_formats': self._get_top_formats(),
        }

    def get_user_stats(self, user_id: int) -> Dict[str, Any]:
        user_data = self.user_metrics.get(user_id, {})
        return {
            'documents_sent': user_data.get('document_sent', 0),
            'commands_used': user_data.get('commands_total', 0),
            'errors_encountered': sum(v for k, v in user_data.items() if k.startswith('error_')),
            'top_format': self._get_user_top_format(user_id),
        }

    def _format_uptime(self, seconds: float) -> str:
        days, rem = divmod(seconds, 86400)
        hours, rem = divmod(rem, 3600)
        minutes, _ = divmod(rem, 60)
        return f"{int(days)}d {int(hours)}h {int(minutes)}m"

    def _get_top_formats(self) -> list:
        formats = {k.replace('format_', ''): v for k, v in self.metrics.items() if k.startswith('format_')}
        return sorted(formats.items(), key=lambda x: x[1], reverse=True)[:5]

    def _get_user_top_format(self, user_id: int) -> str:
        user_data = self.user_metrics.get(user_id, {})
        formats = {k.replace('format_', ''): v for k, v in user_data.items() if k.startswith('format_')}
        return max(formats.items(), key=lambda x: x[1])[0] if formats else "Ninguno"

metrics_collector = MetricsCollector()

# (El resto del archivo `main.py` que te proporcioné anteriormente no necesita cambios)
# Puedes copiarlo desde la respuesta anterior, ya que el resto de la estructura
# con el decorador, la clase del bot, FastAPI y el ciclo de vida es correcta.
# Para evitar un bloque de código masivo, solo he puesto aquí la clase corregida.
# Si lo prefieres, te doy el archivo completo de nuevo.

# ... [PEGAR AQUÍ EL RESTO DEL ARCHIVO main.py DE LA RESPUESTA ANTERIOR] ...
# (Desde la definición del decorador `track_metrics` hasta el final)
# --- DECORADOR PARA MÉTRICAS ---
def track_metrics(operation_name: str):
    def decorator(func):
        async def wrapper(self, update: Update, *args, **kwargs):
            start_time = time.time()
            user_id = update.effective_user.id if update and hasattr(update, 'effective_user') else None
            metrics_collector.increment('commands_total', user_id)
            metrics_collector.increment(operation_name, user_id)
            try:
                return await func(self, update, *args, **kwargs)
            except Exception as e:
                metrics_collector.log_error(operation_name, str(e), user_id)
                logger.error(f"Error en {operation_name} para usuario {user_id}: {e}", exc_info=True)
                raise
            finally:
                duration = time.time() - start_time
                metrics_collector.log_response_time(duration, operation_name)
        return wrapper
    return decorator

# --- MODELOS PYDANTIC ---
class StatusResponse(BaseModel):
    status: str
    bot_username: Optional[str] = None
    metrics: Optional[Dict[str, Any]] = None

# --- VALIDACIÓN DE CONFIGURACIÓN ---
def validate_config() -> BotConfig:
    bot_token = os.getenv('TELEGRAM_BOT_TOKEN')
    gmail_user = os.getenv('GMAIL_USER')
    gmail_password = os.getenv('GMAIL_APP_PASSWORD')
    admin_user_id = os.getenv('ADMIN_USER_ID')
    
    if not all([bot_token, gmail_user, gmail_password]):
        raise ValueError("Faltan variables de entorno esenciales (BOT_TOKEN, GMAIL_USER, GMAIL_APP_PASSWORD)")
    
    return BotConfig(
        bot_token=bot_token, gmail_user=gmail_user, gmail_password=gmail_password,
        admin_user_id=int(admin_user_id) if admin_user_id else None
    )

# --- GESTOR DE CICLO DE VIDA ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("🚀 Iniciando servidor...")
    config = validate_config()
    setup_database()
    metrics_collector.load_from_db()
    bot_instance = KindleEmailBot(config)
    await bot_instance.initialize()
    app.state.bot = bot_instance
    app.state.config = config
    logger.info("✅ Servidor iniciado correctamente")
    yield
    logger.info("🛑 Cerrando servidor...")
    await bot_instance.shutdown()
    logger.info("✅ Servidor cerrado correctamente")

# --- APLICACIÓN FASTAPI ---
app = FastAPI(title="Kindle Bot API", version="2.1.0", lifespan=lifespan)
templates = Jinja2Templates(directory="templates")

@app.get("/", response_model=StatusResponse)
async def read_root():
    bot_info = await app.state.bot.get_bot_info()
    summary = metrics_collector.get_summary()
    return StatusResponse(
        status="✅ Bot activo y funcionando",
        bot_username=bot_info.username if bot_info else None,
        metrics={'uptime': summary['uptime_formatted'], **summary}
    )

@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard(request: Request):
    return templates.TemplateResponse("dashboard.html", {"request": request})

@app.get("/api/metrics-data", response_class=JSONResponse)
async def metrics_data():
    return metrics_collector.get_summary()

# --- CLASE PRINCIPAL DEL BOT (COMPLETADA) ---
class KindleEmailBot:
    def __init__(self, config: BotConfig):
        self.config = config
        self.application = Application.builder().token(self.config.bot_token).build()
        self.main_keyboard = ReplyKeyboardMarkup(
            [["/set_email 📧", "/my_email 🧐"], ["/help ❓", "/stats 📊"]], 
            resize_keyboard=True
        )
        
    async def initialize(self):
        handlers = [
            CommandHandler("start", self.start), CommandHandler("help", self.help_command),
            CommandHandler("set_email", self.set_email_command), CommandHandler("my_email", self.my_email_command),
            CommandHandler("stats", self.stats_command), CommandHandler("admin", self.admin_command),
            CommandHandler("hide_keyboard", self.hide_keyboard_command),
            MessageHandler(filters.TEXT & ~filters.COMMAND & filters.REPLY, self.handle_email_input),
            MessageHandler(filters.Document.ALL, self.handle_document),
            MessageHandler(filters.TEXT & ~filters.COMMAND, self.handle_text),
        ]
        self.application.add_handlers(handlers)
        await self.application.initialize()
        await self.application.updater.start_polling(drop_pending_updates=True)
        await self.application.start()
        logger.info("🤖 Bot inicializado correctamente")

    async def shutdown(self):
        if self.application: await self.application.shutdown()

    async def get_bot_info(self):
        return await self.application.bot.get_me() if self.application else None

    @track_metrics('command_start')
    async def start(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        await update.message.reply_html(f"""
👋 ¡Hola, {update.effective_user.mention_html()}!
📚 <b>Asistente de Envío a Kindle v2.1</b>
Para empezar:
1. Usa <b>/set_email 📧</b> para configurar tu email.
2. Autoriza mi email en Amazon: <code>{self.config.gmail_user}</code>
""", reply_markup=self.main_keyboard)

    @track_metrics('command_help')
    async def help_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        await update.message.reply_html(f"""
🤔 <b>Guía Completa</b>
<b>Uso:</b>
1. Configura tu email con <code>/set_email</code>.
2. Envía un documento.
<b>Comandos:</b>
• <code>/stats</code> - Tus estadísticas.
• <code>/my_email</code> - Ver tu email.
• <code>/hide_keyboard</code> - Ocultar botones.
<b>Tip PDF:</b> Escribe <code>convert</code> en la descripción para optimizar la lectura.
<b>¡Importante!</b> Autoriza mi email en Amazon: <code>{self.config.gmail_user}</code>
""")

    @track_metrics('command_set_email')
    async def set_email_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        await update.message.reply_text(PROMPT_SET_EMAIL, reply_markup=ForceReply(selective=True))

    @track_metrics('handle_email_input')
    async def handle_email_input(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not (update.message.reply_to_message and update.message.reply_to_message.text == PROMPT_SET_EMAIL): return
        user_id = update.effective_user.id
        kindle_email = update.message.text.strip()
        if not self._validate_email(kindle_email):
            metrics_collector.increment('email_validation_failed', user_id)
            return await update.message.reply_html("❌ Formato de email inválido. Inténtalo de nuevo.")
        if set_user_email(user_id, kindle_email):
            metrics_collector.increment('email_set_success', user_id)
            await update.message.reply_html(f"✅ Email guardado: <code>{kindle_email}</code>")
        else:
            await update.message.reply_html("❌ Error al guardar el email.")

    @track_metrics('command_my_email')
    async def my_email_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        email = get_user_email(update.effective_user.id)
        await update.message.reply_html(f"📧 Tu email es: <code>{email}</code>" if email else "❌ No has configurado un email.")

    @track_metrics('command_stats')
    async def stats_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        stats = metrics_collector.get_user_stats(update.effective_user.id)
        await update.message.reply_html(f"""
📊 <b>Tus Estadísticas</b>
• <b>Documentos enviados:</b> {stats['documents_sent']}
• <b>Comandos usados:</b> {stats['commands_used']}
• <b>Formato preferido:</b> {stats['top_format']}
• <b>Errores encontrados:</b> {stats['errors_encountered']}
""")

    @track_metrics('command_admin')
    async def admin_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        user_id = update.effective_user.id
        if not self.config.admin_user_id or user_id != self.config.admin_user_id:
            return await update.message.reply_text("🚫 Acceso denegado.")
        
        summary = metrics_collector.get_summary()
        top_formats = "\n".join([f"  - <code>{f}: {c}</code>" for f, c in summary['top_formats']]) if summary['top_formats'] else "Ninguno"
        await update.message.reply_html(f"""
👑 <b>Panel de Administrador</b>
• <b>Tiempo activo:</b> {summary['uptime_formatted']}
• <b>Usuarios totales:</b> {summary['total_users']}
• <b>Documentos totales:</b> {summary['total_documents_sent']}
• <b>Errores totales:</b> {summary['total_errors']}
• <b>T. Respuesta (avg):</b> {summary['avg_response_time_ms']} ms
<b>Formatos más populares:</b>
{top_formats}
""")

    @track_metrics('command_hide_keyboard')
    async def hide_keyboard_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        await update.message.reply_text("🙈 Teclado ocultado.", reply_markup=ReplyKeyboardRemove())

    @track_metrics('handle_document')
    async def handle_document(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        user_id = update.effective_user.id
        user_kindle_email = get_user_email(user_id)
        if not user_kindle_email:
            return await update.message.reply_html("⚠️ Usa <code>/set_email</code> primero.")
        
        doc = update.message.document
        if doc.file_size > self.config.max_file_size:
            return await update.message.reply_html(f"❌ Archivo muy grande (> {self.config.max_file_size // 1024**2}MB).")
        
        ext = os.path.splitext(doc.file_name)[1].lower()
        if ext not in SUPPORTED_FORMATS:
            return await update.message.reply_html("❌ Formato no soportado.")
        
        metrics_collector.increment('document_received', user_id)
        metrics_collector.increment(f'format_{ext.replace(".", "")}', user_id)
        
        processing_msg = await update.message.reply_html(f"⏳ Procesando <code>{doc.file_name}</code>...")
        try:
            file_data = await (await context.bot.get_file(doc.file_id)).download_as_bytearray()
            subject = "Convert" if doc.file_name.lower().endswith('.pdf') and update.message.caption and update.message.caption.lower() == 'convert' else ""
            success, msg = await self._send_to_kindle_async(user_kindle_email, file_data, doc.file_name, subject)
            if success:
                metrics_collector.increment('document_sent', user_id)
                await processing_msg.edit_text(f"✅ ¡<b>{doc.file_name}</b> enviado!", parse_mode=ParseMode.HTML)
            else:
                await processing_msg.edit_text(f"❌ <b>Error al enviar:</b> <i>{msg}</i>", parse_mode=ParseMode.HTML)
        except Exception as e:
            await processing_msg.edit_text("❌ Error inesperado durante el proceso.")
            raise e

    @track_metrics('handle_text')
    async def handle_text(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        await update.message.reply_html("🤔 No entiendo. Usa los botones o envíame un documento.", reply_markup=self.main_keyboard)

    def _validate_email(self, email: str) -> bool:
        return '@' in email and '.' in email and len(email) > 5

    async def _send_to_kindle_async(self, kindle_email: str, file_data: bytes, filename: str, subject: str) -> Tuple[bool, str]:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._send_to_kindle_sync, kindle_email, file_data, filename, subject)

    def _send_to_kindle_sync(self, kindle_email: str, file_data: bytes, filename: str, subject: str) -> Tuple[bool, str]:
        try:
            msg = MIMEMultipart()
            msg['From'] = self.config.gmail_user
            msg['To'] = kindle_email
            msg['Subject'] = subject or f"Documento Kindle: {filename}"
            msg.attach(MIMEText(f"Enviado por tu bot de Telegram.", 'plain'))
            ctype, _ = mimetypes.guess_type(filename)
            maintype, subtype = (ctype or 'application/octet-stream').split('/', 1)
            part = MIMEBase(maintype, subtype)
            part.set_payload(file_data)
            encoders.encode_base64(part)
            part.add_header('Content-Disposition', f'attachment; filename="{filename}"')
            msg.attach(part)
            with smtplib.SMTP(self.config.smtp_server, self.config.smtp_port) as server:
                server.starttls()
                server.login(self.config.gmail_user, self.config.gmail_password)
                server.send_message(msg)
            return True, "Enviado con éxito"
        except Exception as e:
            logger.error(f"Error SMTP enviando a {kindle_email}: {e}")
            return False, str(e)

# --- PUNTO DE ENTRADA ---
if __name__ == "__main__":
    port = int(os.getenv("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)