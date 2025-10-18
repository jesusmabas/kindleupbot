import logging
import asyncio
from typing import Optional

from telegram import Update, ReplyKeyboardMarkup, InlineKeyboardMarkup, InlineKeyboardButton, ReplyKeyboardRemove, ForceReply
from telegram.ext import Application, CommandHandler, MessageHandler, filters, CallbackQueryHandler, ContextTypes
from telegram.constants import ParseMode

from .config import Settings
from .database import set_user_email, get_user_email, reset_metrics_table, log_admin_action
from .core.metrics import metrics_collector
from .core.validators import EmailValidator, FileValidator
from .services.email_sender import EmailSender

# Importar los módulos de handlers que contienen la lógica de los comandos
from .handlers import commands, messages, callbacks

logger = logging.getLogger(__name__)

class KindleEmailBot:
    def __init__(self, config: Settings):
        self.config = config
        self.application = Application.builder().token(self.config.TELEGRAM_BOT_TOKEN).build()
        
        # Instanciar servicios y utilidades
        self.email_sender = EmailSender(config)
        self.email_validator = EmailValidator()
        self.file_validator = FileValidator()
        
        # Bloqueo para operaciones críticas
        self._reset_lock = asyncio.Lock()
        
        # Configurar teclados y handlers
        self._setup_keyboards()
        self._setup_handlers()

    def _setup_keyboards(self):
        """Define todos los teclados de la aplicación como atributos de instancia."""
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

    def _setup_handlers(self):
        """
        Registra todos los manejadores de eventos.
        Usa lambdas para pasar la instancia 'self' a cada función de handler,
        permitiéndoles acceder a la configuración, teclados y otros servicios del bot.
        """
        # --- Handlers de Comandos ---
        command_handlers = {
            "start": commands.start,
            "help": commands.help_command,
            "set_email": commands.set_email_command,
            "my_email": commands.my_email_command,
            "stats": commands.stats_command,
            "admin": commands.admin_command,
            "hide_keyboard": commands.hide_keyboard_command,
            "formats": commands.formats_command,
            "tips": commands.tips_command,
            "clear_cache": commands.clear_cache_command,
            "reset_stats": commands.reset_stats_command,
        }
        for command, handler_func in command_handlers.items():
            self.application.add_handler(CommandHandler(command, lambda u, c, hf=handler_func: hf(self, u, c)))

        # --- Handlers de Mensajes ---
        self.application.add_handler(MessageHandler(
            filters.TEXT & ~filters.COMMAND & filters.REPLY,
            lambda u, c: messages.handle_email_input(self, u, c)
        ))
        self.application.add_handler(MessageHandler(
            filters.Document.ALL,
            lambda u, c: messages.handle_document(self, u, c)
        ))
        self.application.add_handler(MessageHandler(
            filters.TEXT & ~filters.COMMAND,
            lambda u, c: messages.handle_text(self, u, c)
        ))

        # --- Handler de Callbacks (botones inline) ---
        self.application.add_handler(CallbackQueryHandler(lambda u, c: callbacks.handle_callback(self, u, c)))
        
        logger.info("🤖 Handlers del bot configurados.")

    # --- Métodos Auxiliares y Lógica Interna del Bot ---

    async def _save_user_email(self, user_id: int, email: str) -> bool:
        try:
            return await asyncio.to_thread(set_user_email, user_id, email)
        except Exception as e:
            logger.error(f"Error guardando email para usuario {user_id}: {e}")
            return False

    async def _get_user_email(self, user_id: int) -> Optional[str]:
        try:
            return await asyncio.to_thread(get_user_email, user_id)
        except Exception as e:
            logger.error(f"Error obteniendo email para usuario {user_id}: {e}")
            return None

    async def _perform_stats_reset(self, query: Update.callback_query) -> bool:
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

    def run(self):
        """Inicia el bot en modo Polling y se queda corriendo."""
        logger.info("Application started in polling mode.")
        self.application.run_polling()