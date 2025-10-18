# kindleupbot/handlers/callbacks.py
import asyncio
from pathlib import Path
from typing import TYPE_CHECKING

from telegram import Update
from telegram.ext import ContextTypes
from telegram.constants import ParseMode

from core.metrics import metrics_collector
from core.decorators import track_metrics

if TYPE_CHECKING:
    from bot import KindleEmailBot

@track_metrics('handle_callback')
async def handle_callback(bot: "KindleEmailBot", update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    user_id = update.effective_user.id
    
    if query.data in ("pdf_convert_yes", "pdf_convert_no"):
        await handle_pdf_conversion_callback(bot, query, context)
        return

    if query.data == "confirm":
        pending_email = context.user_data.pop('pending_email', None)
        if pending_email and await bot._save_user_email(user_id, pending_email):
            await metrics_collector.increment('email_set_success', user_id)
            await query.edit_message_text(f"✅ <b>Email configurado</b>\n\n📧 <code>{pending_email}</code>", parse_mode=ParseMode.HTML)
        else:
            await query.edit_message_text("❌ Error al guardar el email")
    
    elif query.data == "cancel":
        context.user_data.pop('pending_email', None)
        await query.edit_message_text("❌ Configuración cancelada")
    
    elif query.data == "confirm_reset_stats":
        if not bot.config.ADMIN_USER_ID or user_id != bot.config.ADMIN_USER_ID:
            await query.edit_message_text("🚫 Acción no autorizada.")
            return
        await bot._perform_stats_reset(query)
    
    elif query.data == "cancel_action":
        await query.edit_message_text("👍 Acción cancelada.")

async def handle_pdf_conversion_callback(bot: "KindleEmailBot", query, context: ContextTypes.DEFAULT_TYPE):
    user_id = query.from_user.id
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
        user_kindle_email = await bot._get_user_email(user_id)
        
        if not user_kindle_email:
            await query.edit_message_text("⚠️ Tu email de Kindle ya no está configurado.", parse_mode=ParseMode.HTML)
            return
        
        success, msg = await bot.email_sender.send_to_kindle_with_retries(user_kindle_email, file_data, filename, subject)
        
        if success:
            action = "(convertido)" if subject else "(sin conversión)"
            await query.edit_message_text(f"✅ ¡<b>{filename}</b> enviado exitosamente {action}!", parse_mode=ParseMode.HTML)
            await metrics_collector.increment('document_sent', user_id)
            if subject:
                await metrics_collector.increment('pdf_converted', user_id)
        else:
            await query.edit_message_text(f"❌ <b>Error al enviar:</b> <i>{msg}</i>", parse_mode=ParseMode.HTML)
    finally:
        context.user_data.pop('pending_pdf', None)
        if temp_path.exists():
            await asyncio.to_thread(temp_path.unlink)