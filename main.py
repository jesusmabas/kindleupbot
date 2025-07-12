# main.py
import os
import logging
import smtplib
import mimetypes
import asyncio # <--- NUEVA IMPORTACIÓN

# --- NUEVAS IMPORTACIONES PARA EL SERVIDOR WEB ---
import uvicorn
from fastapi import FastAPI
# -----------------------------------------------

from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email.mime.text import MIMEText
from email import encoders

from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
from telegram.constants import ParseMode

# --- AÑADIMOS EL SERVIDOR WEB FALSO ---
# Esto es lo que responderá a los chequeos de salud de Render
app = FastAPI()

@app.get("/")
async def read_root():
    return {"status": "Bot is alive and running!"}
# -----------------------------------------

# (El resto de tu código del bot va aquí sin cambios)
# ...
# Configuración de logging para ver qué hace el bot
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

SUPPORTED_FORMATS = (
    '.epub', '.pdf', '.mobi', '.azw', '.doc', '.docx', 
    '.rtf', '.txt', '.html', '.htm', '.jpg', '.jpeg', 
    '.png', '.gif', '.bmp'
)

class KindleEmailBot:
    def __init__(self, bot_token, gmail_user, gmail_password, kindle_email):
        self.bot_token = bot_token
        self.gmail_user = gmail_user
        self.gmail_password = gmail_password
        self.kindle_email = kindle_email
        
    async def start(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        user = update.effective_user
        await update.message.reply_html(
            f"¡Hola {user.mention_html()}!\n\n"
            f"📚 <b>Bot de Envío a Kindle (Multi-Formato)</b>\n\n"
            f"Reenvío tus documentos directamente a tu email de Kindle. Amazon se encarga de la conversión.\n\n"
            f"<b>Formatos Soportados:</b>\n"
            f"<code>.epub, .pdf, .docx, .mobi, .txt, imágenes y más...</code>\n\n"
            f"Envíame un archivo y lo mandaré a tu Kindle. ¡Así de fácil!\n\n"
            f"Usa /help para trucos y más información."
        )

    async def help_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        supported_extensions = ", ".join([f"<code>{ext}</code>" for ext in SUPPORTED_FORMATS])
        help_text = f"""
📚 <b>Bot de Envío a Kindle - Ayuda</b>

<b>Funcionamiento:</b>
Este bot envía cualquier documento compatible directamente a tu dirección de email de Kindle (<code>{self.kindle_email}</code>). Amazon procesará el archivo y lo añadirá a tu biblioteca.

<b>Formatos Soportados:</b>
{supported_extensions}

⭐ <b>TRUCO PARA ARCHIVOS PDF:</b>
Por defecto, los PDF se envían tal cual. Si quieres que Amazon convierta el PDF a un formato de libro electrónico (texto ajustable), adjunta el archivo y escribe en el pie de foto la palabra: <b>convert</b>

<b>Requisito Clave:</b>
Recuerda haber añadido tu email <code>{self.gmail_user}</code> a la "Lista de e-mails de documentos personales aprobados" en tu cuenta de Amazon.
        """
        await update.message.reply_html(help_text)

    async def status_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        status_text = f"""
📊 <b>Estado del Servicio</b>

<b>Configuración de Envío:</b>
• Email de envío: <code>{self.gmail_user}</code>
• Email de destino Kindle: <code>{self.kindle_email}</code>

<b>Sistema:</b>
• Servidor: ✅ Online
• Lógica de envío: ✅ Activa

<i>El bot está listo para recibir tus archivos.</i>
        """
        await update.message.reply_html(status_text)

    def send_to_kindle(self, file_data, filename, subject=""):
        try:
            msg = MIMEMultipart()
            msg['From'] = self.gmail_user
            msg['To'] = self.kindle_email
            msg['Subject'] = subject if subject else f"Documento para tu Kindle: {filename}"
            body = f"Este documento ha sido enviado automáticamente a través de tu bot de Telegram."
            msg.attach(MIMEText(body, 'plain'))
            ctype, encoding = mimetypes.guess_type(filename)
            if ctype is None or encoding is not None:
                ctype = 'application/octet-stream'
            maintype, subtype = ctype.split('/', 1)
            part = MIMEBase(maintype, subtype)
            part.set_payload(file_data)
            encoders.encode_base64(part)
            part.add_header('Content-Disposition', f'attachment; filename="{filename}"')
            msg.attach(part)
            server = smtplib.SMTP('smtp.gmail.com', 587)
            server.starttls()
            server.login(self.gmail_user, self.gmail_password)
            server.send_message(msg)
            server.quit()
            return True, "Documento enviado exitosamente a Kindle."
        except Exception as e:
            logger.error(f"Error al enviar email: {e}")
            return False, f"Error al enviar el email: {str(e)}"

    async def handle_document(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        document = update.message.document
        filename = document.file_name
        if not any(filename.lower().endswith(ext) for ext in SUPPORTED_FORMATS):
            await update.message.reply_html(
                f"❌ <b>Formato no soportado</b>\n\nEl archivo <code>{filename}</code> no parece ser compatible."
            )
            return
        if document.file_size > 48 * 1024 * 1024:
            await update.message.reply_html("❌ <b>Archivo demasiado grande</b>")
            return
        processing_msg = await update.message.reply_html("✅ <b>Archivo recibido...</b>")
        try:
            file = await context.bot.get_file(document.file_id)
            file_data = await file.download_as_bytearray()
            email_subject = ""
            if filename.lower().endswith('.pdf') and update.message.caption and update.message.caption.lower().strip() == 'convert':
                email_subject = "Convert"
                await processing_msg.edit_text("📧 <b>Enviando a Kindle (con conversión de PDF)...</b>", parse_mode=ParseMode.HTML)
            else:
                await processing_msg.edit_text("📧 <b>Enviando a tu Kindle...</b>", parse_mode=ParseMode.HTML)
            success, message = self.send_to_kindle(file_data, filename, subject=email_subject)
            if success:
                await processing_msg.edit_text(f"✅ <b>¡Enviado con éxito!</b>\n\n📖 <b>{filename}</b>", parse_mode=ParseMode.HTML)
            else:
                await processing_msg.edit_text(f"❌ <b>Error al enviar</b>\n\n<i>{message}</i>", parse_mode=ParseMode.HTML)
        except Exception as e:
            logger.error(f"Error procesando documento: {e}")
            await processing_msg.edit_text(f"❌ <b>Error inesperado</b>", parse_mode=ParseMode.HTML)

    async def handle_text(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        await update.message.reply_html("Hola, envíame un archivo compatible (<code>.epub</code>, <code>.pdf</code>, etc.).")


# --- NUEVA FUNCIÓN PRINCIPAL ASÍNCRONA ---
async def main():
    """Función principal que ahora corre el bot y el servidor web a la vez."""
    BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')
    GMAIL_USER = os.getenv('GMAIL_USER')
    GMAIL_APP_PASSWORD = os.getenv('GMAIL_APP_PASSWORD')
    KINDLE_EMAIL = os.getenv('KINDLE_EMAIL')
    
    if not all([BOT_TOKEN, GMAIL_USER, GMAIL_APP_PASSWORD, KINDLE_EMAIL]):
        logger.critical("Faltan variables de entorno. El bot no puede iniciar.")
        return
    
    # Configuración del bot de Telegram
    bot_instance = KindleEmailBot(BOT_TOKEN, GMAIL_USER, GMAIL_APP_PASSWORD, KINDLE_EMAIL)
    application = Application.builder().token(BOT_TOKEN).build()
    
    application.add_handler(CommandHandler("start", bot_instance.start))
    application.add_handler(CommandHandler("help", bot_instance.help_command))
    application.add_handler(CommandHandler("status", bot_instance.status_command))
    application.add_handler(MessageHandler(filters.Document.ALL, bot_instance.handle_document))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, bot_instance.handle_text))
    
    # Configuración del servidor web uvicorn
    # Render nos da la variable de entorno PORT. Escuchamos en 0.0.0.0
    port = int(os.getenv('PORT', 8080))
    web_server_config = uvicorn.Config(app, host="0.0.0.0", port=port)
    web_server = uvicorn.Server(web_server_config)
    
    # Ejecutar el bot y el servidor web concurrentemente
    logger.info(f"Iniciando el bot y el servidor web en el puerto {port}...")
    await asyncio.gather(
        application.run_polling(allowed_updates=Update.ALL_TYPES),
        web_server.serve()
    )

if __name__ == '__main__':
    # Usamos asyncio.run() para iniciar nuestra función main asíncrona
    asyncio.run(main())