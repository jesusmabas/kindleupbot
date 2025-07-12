# main.py - Versión mejorada
import os
import logging
import smtplib
import mimetypes
import asyncio
import uvicorn
from contextlib import asynccontextmanager
from typing import Optional, Tuple, Dict, Any, List
from dataclasses import dataclass
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
from pydantic import BaseModel, validator
import aiofiles

from database import (
    setup_database, set_user_email, get_user_email, 
    get_metrics_from_db, save_metric, get_total_users
)

from telegram import Update, ReplyKeyboardMarkup, ReplyKeyboardRemove, ForceReply, InlineKeyboardMarkup, InlineKeyboardButton
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes, CallbackQueryHandler
from telegram.constants import ParseMode
from telegram.error import TelegramError, RetryAfter

# --- CONFIGURACIÓN MEJORADA ---
@dataclass
class BotConfig:
    bot_token: str
    gmail_user: str
    gmail_password: str
    max_file_size: int = 48 * 1024 * 1024
    smtp_server: str = 'smtp.gmail.com'
    smtp_port: int = 587
    admin_user_id: Optional[int] = None
    max_retries: int = 3
    retry_delay: float = 1.0
    rate_limit_window: int = 60  # segundos
    rate_limit_max_requests: int = 10
    cache_duration: int = 300  # 5 minutos

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
    def __init__(self, max_requests: int = 10, window: int = 60):
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
cache_manager = CacheManager()
rate_limiter = RateLimiter()

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
    
    @validator('email')
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

# --- CONFIGURACIÓN MEJORADA ---
def validate_config() -> BotConfig:
    """Validación mejorada de configuración"""
    required_vars = {
        'TELEGRAM_BOT_TOKEN': 'bot_token',
        'GMAIL_USER': 'gmail_user',
        'GMAIL_APP_PASSWORD': 'gmail_password'
    }
    
    config_data = {}
    missing_vars = []
    
    for env_var, config_key in required_vars.items():
        value = os.getenv(env_var)
        if not value:
            missing_vars.append(env_var)
        else:
            config_data[config_key] = value
    
    if missing_vars:
        raise ValueError(f"Faltan variables de entorno: {', '.join(missing_vars)}")
    
    # Variables opcionales
    admin_user_id = os.getenv('ADMIN_USER_ID')
    if admin_user_id:
        try:
            config_data['admin_user_id'] = int(admin_user_id)
        except ValueError:
            logger.warning("ADMIN_USER_ID no es un número válido")
    
    # Configuraciones adicionales
    config_data['max_file_size'] = int(os.getenv('MAX_FILE_SIZE', 50 * 1024 * 1024))  # 50MB por defecto
    config_data['max_retries'] = int(os.getenv('MAX_RETRIES', 3))
    config_data['rate_limit_max_requests'] = int(os.getenv('RATE_LIMIT_MAX_REQUESTS', 10))
    
    return BotConfig(**config_data)

# --- GESTOR DE CICLO DE VIDA MEJORADO ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Gestor de ciclo de vida mejorado"""
    logger.info("🚀 Iniciando servidor...")
    
    try:
        # Validar configuración
        config = validate_config()
        logger.info("✅ Configuración validada")
        
        # Configurar base de datos
        setup_database()
        logger.info("✅ Base de datos configurada")
        
        # Cargar métricas
        metrics_collector.load_from_db()
        logger.info("✅ Métricas cargadas")
        
        # Inicializar bot
        bot_instance = KindleEmailBot(config)
        await bot_instance.initialize()
        logger.info("✅ Bot inicializado")
        
        # Almacenar en el estado de la aplicación
        app.state.bot = bot_instance
        app.state.config = config
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

@app.get("/health")
async def health_check():
    """Endpoint de health check"""
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

# Continuará en la siguiente parte...

# --- CLASE PRINCIPAL DEL BOT (MEJORADA) ---
class KindleEmailBot:
    def __init__(self, config: BotConfig):
   