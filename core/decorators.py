# kindleupbot/core/decorators.py
import time
import logging
from typing import TYPE_CHECKING

from telegram import Update

from core.metrics import metrics_collector
from core.limiter import rate_limiter

if TYPE_CHECKING:
    from bot import KindleEmailBot

logger = logging.getLogger(__name__)

def track_metrics(operation_name: str):
    def decorator(func):
        async def wrapper(bot: "KindleEmailBot", update: Update, *args, **kwargs):
            start_time = time.time()
            user_id = update.effective_user.id if update and hasattr(update, 'effective_user') else None

            if user_id and not rate_limiter.is_allowed(user_id):
                remaining_time = rate_limiter.get_remaining_time(user_id)
                if update.message:
                    await update.message.reply_text(f"🚫 Límite de solicitudes excedido. Intenta de nuevo en {remaining_time} segundos.")
                return

            await metrics_collector.increment('commands_total', user_id)
            await metrics_collector.increment(operation_name, user_id)

            try:
                result = await func(bot, update, *args, **kwargs)
                await metrics_collector.increment(f'{operation_name}_success', user_id)
                return result
            except Exception as e:
                metrics_collector.log_error(operation_name, str(e), user_id)
                logger.error(f"Error en {operation_name} para usuario {user_id}: {e}", exc_info=True)
                if hasattr(update, 'message') and update.message:
                    await update.message.reply_text("😔 Ocurrió un error inesperado. El equipo técnico ha sido notificado.")
                # No relanzamos la excepción para que el bot no se detenga
            finally:
                duration = time.time() - start_time
                metrics_collector.log_response_time(duration, operation_name)
        return wrapper
    return decorator