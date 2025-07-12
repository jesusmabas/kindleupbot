# main.py
import os
import logging
import smtplib
import mimetypes # Importamos esta librería para detectar el tipo de archivo
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email.mime.text import MIMEText
from email import encoders

from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
from telegram.constants import ParseMode

# Configuración de logging para ver qué hace el bot
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Lista de formatos de archivo soportados por el servicio "Enviar a Kindle"
# Añade o quita extensiones según necesites
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
        """Comando /start. Da la bienvenida al usuario."""
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
        """Comando /help. Explica el funcionamiento y los formatos."""
        supported_extensions = ", ".join([f"<code>{ext}</code>" for ext in SUPPORTED_FORMATS])
        help_text = f"""
📚 <b>Bot de Envío a Kindle - Ayuda</b>

<b>Funcionamiento:</b>
Este bot envía cualquier documento compatible directamente a tu dirección de email de Kindle (<code>{self.kindle_email}</code>). Amazon procesará el archivo y lo añadirá a tu biblioteca.

<b>Formatos Soportados:</b>
{supported_extensions}

⭐ <b>TRUCO PARA ARCHIVOS PDF:</b>
Por defecto, los PDF se envían tal cual, manteniendo su diseño original (bueno para artículos con muchas imágenes).

Si quieres que Amazon convierta el PDF a un formato de libro electrónico (texto ajustable, que se lee mejor en Kindle), haz lo siguiente:
1. Adjunta el archivo PDF.
2. Antes de enviarlo, en el campo de texto o "pie de foto", escribe la palabra: <b>convert</b>

El bot detectará la palabra y le pedirá a Amazon que realice la conversión.

<b>Requisito Clave:</b>
Recuerda haber añadido tu email <code>{self.gmail_user}</code> a la "Lista de e-mails de documentos personales aprobados" en la configuración de tu cuenta de Amazon.
        """
        await update.message.reply_html(help_text)

    async def status_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Comando /status. Muestra la configuración actual."""
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
        """Envía el archivo al Kindle. Ahora detecta el tipo de archivo y acepta un asunto personalizado."""
        try:
            msg = MIMEMultipart()
            msg['From'] = self.gmail_user
            msg['To'] = self.kindle_email
            msg['Subject'] = subject if subject else f"Documento para tu Kindle: {filename}"
            
            body = f"Este documento ha sido enviado automáticamente a través de tu bot de Telegram."
            msg.attach(MIMEText(body, 'plain'))
            
            # Detectar el tipo de archivo (MIME type) para un adjunto correcto
            ctype, encoding = mimetypes.guess_type(filename)
            if ctype is None or encoding is not None:
                ctype = 'application/octet-stream' # Tipo por defecto si no se reconoce
            
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
        """Maneja cualquier tipo de documento soportado."""
        document = update.message.document
        filename = document.file_name
        
        # Comprobar si la extensión del archivo está en nuestra lista de soportados
        if not any(filename.lower().endswith(ext) for ext in SUPPORTED_FORMATS):
            await update.message.reply_html(
                f"❌ <b>Formato no soportado</b>\n\nEl archivo <code>{filename}</code> no parece ser compatible. "
                f"Por favor, envía uno de los siguientes tipos: {', '.join(SUPPORTED_FORMATS)}"
            )
            return
        
        if document.file_size > 48 * 1024 * 1024:
            await update.message.reply_html(
                "❌ <b>Archivo demasiado grande</b>\n\nEl límite de Amazon es 50MB."
            )
            return
        
        processing_msg = await update.message.reply_html(
            "✅ <b>Archivo recibido...</b>\n\n"
            "📥 Descargando y preparando para el envío..."
        )
        
        try:
            file = await context.bot.get_file(document.file_id)
            file_data = await file.download_as_bytearray()
            
            # Lógica para el asunto del email
            email_subject = ""
            # Si es un PDF y el usuario ha escrito "convert" en el pie de foto
            if filename.lower().endswith('.pdf') and update.message.caption and update.message.caption.lower().strip() == 'convert':
                email_subject = "Convert"
                await processing_msg.edit_text(
                    "📧 <b>Enviando a tu Kindle...</b>\n\n"
                    "<i>Se ha solicitado la conversión del PDF a formato de libro electrónico.</i>",
                    parse_mode=ParseMode.HTML
                )
            else:
                await processing_msg.edit_text(
                    "📧 <b>Enviando a tu Kindle...</b>\n\n"
                    "<i>Amazon lo procesará y aparecerá en tu biblioteca en unos minutos.</i>",
                    parse_mode=ParseMode.HTML
                )

            success, message = self.send_to_kindle(file_data, filename, subject=email_subject)
            
            if success:
                await processing_msg.edit_text(
                    f"✅ <b>¡Enviado con éxito!</b>\n\n"
                    f"📖 <b>{filename}</b>\n"
                    f"DESTINO: <code>{self.kindle_email}</code>\n\n"
                    f"Revisa la biblioteca de tu Kindle en 5-10 minutos.",
                    parse_mode=ParseMode.HTML
                )
            else:
                await processing_msg.edit_text(
                    f"❌ <b>Error al enviar</b>\n\n<i>{message}</i>",
                    parse_mode=ParseMode.HTML
                )
                
        except Exception as e:
            logger.error(f"Error procesando documento: {e}")
            await processing_msg.edit_text(
                f"❌ <b>Error inesperado</b>\n\n<i>{str(e)}</i>",
                parse_mode=ParseMode.HTML
            )

    async def handle_text(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Maneja mensajes de texto que no son comandos."""
        await update.message.reply_html(
            "Hola, para usarme, simplemente envíame un archivo compatible (como <code>.epub</code>, <code>.pdf</code>, etc.)."
        )

def main():
    """Función principal que inicia el bot."""
    BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')
    GMAIL_USER = os.getenv('GMAIL_USER')
    GMAIL_APP_PASSWORD = os.getenv('GMAIL_APP_PASSWORD')
    KINDLE_EMAIL = os.getenv('KINDLE_EMAIL')
    
    if not all([BOT_TOKEN, GMAIL_USER, GMAIL_APP_PASSWORD, KINDLE_EMAIL]):
        logger.critical("Faltan una o más variables de entorno. El bot no puede iniciar.")
        return
    
    bot = KindleEmailBot(BOT_TOKEN, GMAIL_USER, GMAIL_APP_PASSWORD, KINDLE_EMAIL)
    
    application = Application.builder().token(BOT_TOKEN).build()
    
    application.add_handler(CommandHandler("start", bot.start))
    application.add_handler(CommandHandler("help", bot.help_command))
    application.add_handler(CommandHandler("status", bot.status_command))
    application.add_handler(MessageHandler(filters.Document.ALL, bot.handle_document))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, bot.handle_text))
    
    logger.info("Iniciando el bot multi-formato...")
    application.run_polling(allowed_updates=Update.ALL_TYPES)

if __name__ == '__main__':
    main()
