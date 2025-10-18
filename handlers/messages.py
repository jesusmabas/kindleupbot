# kindleupbot/handlers/messages.py
import asyncio
from pathlib import Path
from typing import TYPE_CHECKING

from telegram import Update, InlineKeyboardMarkup, InlineKeyboardButton
from telegram.ext import ContextTypes
from telegram.constants import ParseMode

from ..core.metrics import metrics_collector
from ..core.validators import PROMPT_SET_EMAIL
from ..core.decorators import track_metrics
from ..services.file_converter import convert_markdown_to_docx
from ..database import get_total_users

if TYPE_CHECKING:
    from ..bot import KindleEmailBot

@track_metrics('handle_email_input')
async def handle_email_input(bot: "KindleEmailBot", update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not (update.message.reply_to_message and update.message.reply_to_message.text == PROMPT_SET_EMAIL):
        return
    
    user_id = update.effective_user.id
    kindle_email = update.message.text.strip()
    
    if not bot.email_validator.validate_email(kindle_email):
        await metrics_collector.increment('email_validation_failed', user_id)
        await update.message.reply_html("❌ <b>Formato de email inválido</b>\n\n📧 Formato correcto: <code>usuario@kindle.com</code>\n🔄 Inténtalo de nuevo con /set_email")
        return
        
    if not bot.email_validator.is_kindle_email(kindle_email):
        await update.message.reply_html(
            "⚠️ <b>Advertencia:</b> Este no parece ser un email de Kindle\n\n"
            "📧 Los emails de Kindle terminan en:\n• @kindle.com\n• @free.kindle.com\n\n"
            "¿Estás seguro de que es correcto?",
            reply_markup=bot.confirm_keyboard
        )
        context.user_data['pending_email'] = kindle_email
        return
        
    if await bot._save_user_email(user_id, kindle_email):
        await metrics_collector.increment('email_set_success', user_id)
        await update.message.reply_html(
            f"✅ <b>Email configurado correctamente</b>\n\n"
            f"📧 <b>Tu email:</b> <code>{kindle_email}</code>\n\n"
            f"🔑 <b>Recuerda autorizar:</b> <code>{bot.config.GMAIL_USER}</code>"
        )
    else:
        await update.message.reply_html("❌ <b>Error al guardar el email</b>\n\n🔄 Por favor, inténtalo de nuevo")

@track_metrics('handle_document')
async def handle_document(bot: "KindleEmailBot", update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    user_kindle_email = await bot._get_user_email(user_id)
    if not user_kindle_email:
        await update.message.reply_html("⚠️ <b>Email no configurado.</b>\n\nUsa /set_email o el botón del menú para empezar.")
        return
        
    doc = update.message.document
    valid, error_msg = bot.file_validator.validate_file(
        doc.file_name, doc.file_size, bot.config.MAX_FILE_SIZE
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
            
        success, msg = await bot.email_sender.send_to_kindle_with_retries(user_kindle_email, file_data, file_to_send_name, subject)
        
        if success:
            await metrics_collector.increment('document_sent', user_id)
            if ext == '.md':
                await metrics_collector.increment('md_converted_docx', user_id)
            await processing_msg.edit_text(f"✅ ¡<b>{file_to_send_name}</b> enviado!", parse_mode=ParseMode.HTML)
        else:
            await processing_msg.edit_text(f"❌ <b>Error al enviar:</b> <i>{msg}</i>", parse_mode=ParseMode.HTML)
    finally:
        if ext != '.pdf' and temp_file_path.exists():
            await asyncio.to_thread(temp_file_path.unlink)
        if converted_path and converted_path.exists():
            await asyncio.to_thread(converted_path.unlink)

@track_metrics('handle_text')
async def handle_text(bot: "KindleEmailBot", update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = update.message.text
    is_admin = bot.config.ADMIN_USER_ID and update.effective_user.id == bot.config.ADMIN_USER_ID
    
    async def show_total_users(upd: Update, ctx: ContextTypes.DEFAULT_TYPE):
        total = await asyncio.to_thread(get_total_users)
        await upd.message.reply_text(f"👥 Hay un total de {total} usuarios registrados.")
        
    command_map = {
        "📧 Configurar Email": bot.set_email_command, "🔍 Ver Mi Email": bot.my_email_command,
        "📊 Mis Estadísticas": bot.stats_command, "❓ Ayuda": bot.help_command,
        "🎯 Formatos Soportados": bot.formats_command, "🚀 Consejos": bot.tips_command
    }
    admin_command_map = {
        "👑 Panel Admin": bot.admin_command, "📈 Métricas": bot.admin_command,
        "🧹 Limpiar Cache": bot.clear_cache_command,
        "🔄 Reiniciar Stats": bot.reset_stats_command,
        "👥 Usuarios": show_total_users,
        "🏠 Menú Principal": lambda u, c: u.message.reply_text("Volviendo al menú principal...", reply_markup=bot.main_keyboard)
    }
    
    if text in command_map:
        await command_map[text](update, context)
    elif is_admin and text in admin_command_map:
        await admin_command_map[text](update, context)
    else:
        text_lower = text.lower()
        if any(word in text_lower for word in ['hola', 'hello', 'hi', 'buenas']):
            await update.message.reply_text("¡Hola! 👋 Soy tu asistente de Kindle.\nEnvíame un documento para empezar.", reply_markup=bot.main_keyboard)
        elif any(word in text_lower for word in ['ayuda', 'help', 'auxilio']):
            await bot.help_command(update, context)
        elif any(word in text_lower for word in ['gracias', 'thanks', 'thank you']):
            await update.message.reply_text("¡De nada! 😊 Estoy aquí para ayudarte.")
        else:
            await update.message.reply_html("🤔 <b>No entiendo ese mensaje</b>\n\n💡 <b>Puedo ayudarte con:</b>\n• Configurar tu email de Kindle\n• Enviar documentos a tu dispositivo\n• Mostrar estadísticas de uso\n\n📄 <b>Envía un documento</b> o usa los botones del menú", reply_markup=bot.main_keyboard)