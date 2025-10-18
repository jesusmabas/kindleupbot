# kindleupbot/handlers/commands.py
import asyncio
from typing import TYPE_CHECKING

from telegram import Update, ReplyKeyboardRemove, ForceReply
from telegram.ext import ContextTypes

from core.metrics import metrics_collector
from core.cache import cache_manager
from core.validators import PROMPT_SET_EMAIL
from core.decorators import track_metrics
from database import get_total_users

if TYPE_CHECKING:
    from bot import KindleEmailBot

@track_metrics('command_start')
async def start(bot: "KindleEmailBot", update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = update.effective_user
    is_admin = bot.config.ADMIN_USER_ID and user.id == bot.config.ADMIN_USER_ID
    welcome_message = (
        f"🎉 ¡Bienvenido, {user.mention_html()}!\n\n"
        f"📚 <b>Kindle Bot v3.0</b> - Tu asistente personal para envío de documentos\n\n"
        "🚀 <b>Pasos para empezar:</b>\n"
        "1. Configura tu email con \"📧 Configurar Email\"\n"
        "2. Autoriza mi email en tu cuenta de Amazon\n"
        "3. ¡Envía tus documentos!\n\n"
        f"📧 <b>Email a autorizar:</b> <code>{bot.config.GMAIL_USER}</code>\n\n"
        f"{'👑 <b>Acceso de administrador detectado</b>' if is_admin else ''}"
    )
    keyboard = bot.admin_keyboard if is_admin else bot.main_keyboard
    await update.message.reply_html(welcome_message, reply_markup=keyboard)

@track_metrics('command_help')
async def help_command(bot: "KindleEmailBot", update: Update, context: ContextTypes.DEFAULT_TYPE):
    intro = "📖 <b>Guía Completa del KindleUp Bot</b>\n\nDomina el envío de tus documentos a Kindle en 3 pasos."
    steps = [
        ("Configura tu Email de Kindle", "Usa /set_email o el botón 📧 para guardar tu dirección @kindle.com."),
        ("Autoriza mi Email en Amazon", f"Añade <code>{bot.config.GMAIL_USER}</code> en tu cuenta de Amazon → 'Gestionar contenido y dispositivos' → 'Preferencias'."),
        ("Envía tu Documento", "Arrastra y suelta un archivo aquí. Yo me encargo del resto.")
    ]
    pdf_flow = "📄 <b>Flujo de PDF: ¡Tú eliges!</b>\nTras enviarlo, te preguntaré qué hacer:\n  • <b>✅ Convertir:</b> Para texto adaptable (libros, artículos).\n  • <b>❌ Sin convertir:</b> Para mantener el diseño original (cómics, manuales)."
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
        ("¿El documento no llega a mi Kindle?", "1. Asegúrate de haber autorizado mi email en Amazon.\n2. Comprueba la conexión Wi-Fi de tu Kindle.\n3. Dale unos minutos, a veces Amazon tarda un poco."),
        ("¿Recibo un error de 'email rechazado'?", "Tu email de Kindle es incorrecto. Verifícalo con /my_email y corrígelo con /set_email."),
        ("¿Mis archivos están seguros?", "Totalmente. Se borran de nuestros servidores temporales justo después de ser enviados. Nunca los almacenamos.")
    ]
    parts = [intro, "1️⃣ <b>PUESTA EN MARCHA</b>"]
    parts.extend([f"<b>Paso {i}: {title}</b>\n{desc}" for i, (title, desc) in enumerate(steps, 1)])
    parts.extend([pdf_flow, "🔧 <b>LISTA DE COMANDOS</b>", "\n".join([f"<code>{cmd}</code> - {desc}" for cmd, desc in commands]), "🤔 <b>SOLUCIÓN DE PROBLEMAS (FAQ)</b>"])
    parts.extend([f"<b>P: {q}</b>\nR: {a}" for q, a in faq])
    
    final_message = "\n\n---\n\n".join(parts)
    await update.message.reply_html(final_message, disable_web_page_preview=True)

@track_metrics('command_tips')
async def tips_command(bot: "KindleEmailBot", update: Update, context: ContextTypes.DEFAULT_TYPE):
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
async def formats_command(bot: "KindleEmailBot", update: Update, context: ContextTypes.DEFAULT_TYPE):
    formats_by_category = {
        "📚 Libros Electrónicos": [".epub", ".mobi", ".azw", ".md (convierte a docx)"],
        "📄 Documentos": [".pdf", ".doc", ".docx", ".rtf", ".txt", ".html"],
        "🖼️ Imágenes": [".jpg", ".jpeg", ".png", ".gif", ".bmp"]
    }
    message = "📋 <b>Formatos Soportados</b>\n\n"
    for category, extensions in formats_by_category.items():
        message += f"<b>{category}:</b>\n • " + " • ".join(extensions) + "\n\n"
    message += f"📊 <b>Límite de tamaño:</b> {bot.config.MAX_FILE_SIZE // 1024**2}MB"
    await update.message.reply_html(message)

@track_metrics('command_set_email')
async def set_email_command(bot: "KindleEmailBot", update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(PROMPT_SET_EMAIL, reply_markup=ForceReply(selective=True, input_field_placeholder="usuario@kindle.com"))

@track_metrics('command_my_email')
async def my_email_command(bot: "KindleEmailBot", update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    cache_key = f"user_email_{user_id}"
    email = cache_manager.get(cache_key)
    if email is None:
        email = await bot._get_user_email(user_id)
        if email:
            cache_manager.set(cache_key, email, 300)
            
    if email:
        is_kindle = bot.email_validator.is_kindle_email(email)
        status_icon, status_text = ("✅", "Email de Kindle válido") if is_kindle else ("⚠️", "No es un email de Kindle")
        await update.message.reply_html(f"📧 <b>Tu email configurado:</b>\n\n<code>{email}</code>\n\n{status_icon} <b>Estado:</b> {status_text}\n\n🔑 <b>Email autorizado:</b> <code>{bot.config.GMAIL_USER}</code>")
    else:
        await update.message.reply_html("❌ <b>No tienes un email configurado</b>\n\n📧 Usa el botón <b>Configurar Email</b> para empezar")

@track_metrics('command_stats')
async def stats_command(bot: "KindleEmailBot", update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    stats = metrics_collector.get_user_stats(user_id)
    success_rate = stats['success_rate']
    filled_bars = int((success_rate / 100) * 10)
    bar = "█" * filled_bars + "░" * (10 - filled_bars)
    total_users = await asyncio.to_thread(get_total_users)
    summary = await metrics_collector.get_summary()
    stats_message = (
        f"📊 <b>Tus Estadísticas Personales</b>\n\n"
        f"📄 <b>Documentos:</b>\n• Recibidos: {stats['documents_received']}\n• Enviados exitosamente: {stats['documents_sent']}\n• Tasa de éxito: {success_rate}% {bar}\n\n"
        f"⚡ <b>Actividad:</b>\n• Comandos ejecutados: {stats['commands_used']}\n• Errores encontrados: {stats['errors_encountered']}\n• Formato preferido: {stats['top_format']}\n\n"
        f"🏆 <b>Ranking:</b>\n• Eres uno de {total_users} usuarios totales\n• Tiempo promedio de respuesta: {summary['avg_response_time_ms']}ms"
    )
    await update.message.reply_html(stats_message)

@track_metrics('command_hide_keyboard')
async def hide_keyboard_command(bot: "KindleEmailBot", update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("🙈 Teclado ocultado\n\n💡 Usa /start para mostrarlo de nuevo", reply_markup=ReplyKeyboardRemove())

# --- Comandos de Administrador ---

@track_metrics('command_admin')
async def admin_command(bot: "KindleEmailBot", update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    if not bot.config.ADMIN_USER_ID or user_id != bot.config.ADMIN_USER_ID:
        await update.message.reply_text("🚫 Acceso denegado")
        return
        
    summary = await metrics_collector.get_summary()
    success_rate = summary['success_rate']
    filled_bars = int((success_rate / 100) * 10)
    bar = "█" * filled_bars + "░" * (10 - filled_bars)
    top_formats = "\n".join([f"  • <code>{f}:</code> {c}" for f, c in summary['top_formats'][:5]]) if summary['top_formats'] else "Ninguno"
    admin_message = (
        f"👑 <b>Panel de Administración</b>\n\n"
        f"⏱️ <b>Sistema:</b>\n• Tiempo activo: {summary['uptime_formatted']}\n• Usuarios totales: {summary['total_users']}\n• Versión: 3.0\n\n"
        f"📊 <b>Métricas:</b>\n• Documentos enviados: {summary['total_documents_sent']}\n• Documentos recibidos: {summary['total_documents_received']}\n• Tasa de éxito: {success_rate}% {bar}\n• Comandos ejecutados: {summary['commands_executed']}\n\n"
        f"❌ <b>Errores:</b>\n• Total: {summary['total_errors']}\n• Últimos errores: {len(summary['recent_errors'])}\n\n"
        f"⚡ <b>Rendimiento:</b>\n• Tiempo respuesta promedio: {summary['avg_response_time_ms']}ms\n\n"
        f"📈 <b>Formatos populares:</b>\n{top_formats}"
    )
    await update.message.reply_html(admin_message, reply_markup=bot.admin_keyboard)

@track_metrics('command_clear_cache')
async def clear_cache_command(bot: "KindleEmailBot", update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    if not bot.config.ADMIN_USER_ID or user_id != bot.config.ADMIN_USER_ID:
        await update.message.reply_text("🚫 Acceso denegado")
        return
    cache_manager.clear()
    await update.message.reply_text("🧹 Caché limpiado exitosamente")

@track_metrics('command_reset_stats')
async def reset_stats_command(bot: "KindleEmailBot", update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    if not bot.config.ADMIN_USER_ID or user_id != bot.config.ADMIN_USER_ID:
        await update.message.reply_text("🚫 Acceso denegado.")
        return
    await update.message.reply_html(
        "<b>⚠️ ¿Estás seguro de que quieres reiniciar TODAS las estadísticas?</b>\n\n"
        "Esta acción borrará permanentemente el historial de la base de datos y los contadores actuales.\n\n"
        "<i>Esta acción no se puede deshacer.</i>",
        reply_markup=bot.confirm_reset_keyboard
    )