# main.py
import os
import logging
import smtplib
import mimetypes
import asyncio
import uvicorn
from fastapi import FastAPI

# --- IMPORTAMOS NUESTRAS FUNCIONES DE BASE DE DATOS ---
from database import setup_database, set_user_email, get_user_email

from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email.mime.text import MIMEText
from email import encoders

from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
from telegram.constants import ParseMode

app = FastAPI()

@app.get("/")
async def read_root():
    return {"status": "Bot is alive and running!"}

logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

SUPPORTED_FORMATS = ('.epub', '.pdf', '.mobi', '.azw', '.doc', '.docx', '.rtf', '.txt', '.html', '.htm', '.jpg', '.jpeg', '.png', '.gif', '.bmp')

class KindleEmailBot:
    def __init__(self, bot_token, gmail_user, gmail_password):
        self.bot_token = bot_token
        self.gmail_user = gmail_user
        self.gmail_password = gmail_password
    
    # --- NUEVO COMANDO: /set_email ---
    async def set_email_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Permite al usuario guardar su email de Kindle."""
        user_id = update.effective_user.id
        # context.args contiene las palabras después del comando, separadas por espacios
        if not context.args:
            await update.message.reply_html(
                "Por favor, proporciona tu email de Kindle después del comando.\n"
                "Ejemplo: <code>/set_email tu_email@kindle.com</code>"
            )
            return
        
        kindle_email = context.args[0]
        # Validación simple de que el email parece un email
        if '@' not in kindle_email or '.' not in kindle_email:
            await update.message.reply_html("El formato del email no parece válido. Por favor, inténtalo de nuevo.")
            return

        if set_user_email(user_id, kindle_email):
            await update.message.reply_html(f"✅ ¡Perfecto! He guardado tu email de Kindle como: <code>{kindle_email}</code>")
        else:
            await update.message.reply_html("❌ Hubo un error al intentar guardar tu email. Por favor, contacta al administrador.")

    async def start(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        user = update.effective_user
        await update.message.reply_html(
            f"¡Hola {user.mention_html()}!\n\n"
            f"📚 <b>Bot de Envío a Kindle</b>\n\n"
            f"Este bot te permite enviar documentos a tu Kindle personal.\n\n"
            f"<b>PRIMER PASO:</b>\n"
            f"Configura tu email de Kindle con el comando:\n"
            f"<code>/set_email tu_direccion@kindle.com</code>\n\n"
            f"Usa /help para más información."
        )

    async def help_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        help_text = """
        📚 <b>Ayuda del Bot de Envío a Kindle</b>

        1️⃣ <b>Configura tu Email (Solo una vez)</b>
        Usa el comando <code>/set_email</code> para decirme a dónde enviar tus libros.
        Ejemplo: <code>/set_email mi_usuario_123@kindle.com</code>

        2️⃣ <b>Envía un Documento</b>
        Simplemente comparte conmigo un archivo compatible. ¡Yo me encargo del resto!
        Formatos: <code>.epub, .pdf, .docx, .mobi, etc.</code>

        ⭐ <b>TRUCO PARA PDF</b>
        Si envías un PDF y quieres que se convierta a formato de libro electrónico, escribe <b>convert</b> en el pie de foto del archivo.

        🔧 <b>Requisito Importante</b>
        Asegúrate de haber autorizado el email del bot (<code>{}</code>) en la configuración de tu cuenta de Amazon para que acepte los envíos.
        """.format(self.gmail_user)
        await update.message.reply_html(help_text)

    # El comando /status ya no tiene sentido porque el email es por usuario. Podríamos quitarlo o cambiarlo.
    # Por ahora, lo comentamos para simplificar.
    # async def status_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE): ...

    def send_to_kindle(self, kindle_email_destino, file_data, filename, subject=""):
        # La función ahora recibe el email de destino como parámetro
        try:
            msg = MIMEMultipart()
            msg['From'] = self.gmail_user
            msg['To'] = kindle_email_destino
            # ... (el resto de la función send_to_kindle es igual)
            msg['Subject'] = subject if subject else f"Documento para tu Kindle: {filename}"
            body = f"Este documento ha sido enviado automáticamente a través de tu bot de Telegram."
            msg.attach(MIMEText(body, 'plain'))
            ctype, encoding = mimetypes.guess_type(filename)
            if ctype is None or encoding is not None: ctype = 'application/octet-stream'
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
        user_id = update.effective_user.id
        
        # --- LÓGICA CLAVE: OBTENER EMAIL DEL USUARIO ---
        user_kindle_email = get_user_email(user_id)
        
        if not user_kindle_email:
            await update.message.reply_html(
                "⚠️ No he encontrado tu email de Kindle.\n\n"
                "Por favor, configúralo primero con el comando:\n"
                "<code>/set_email tu_email@kindle.com</code>"
            )
            return

        document = update.message.document
        filename = document.file_name
        
        if not any(filename.lower().endswith(ext) for ext in SUPPORTED_FORMATS):
            await update.message.reply_html(f"❌ <b>Formato no soportado</b>")
            return
        
        # ... (el resto de la función es casi igual)
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
            
            # --- Pasamos el email del usuario a la función de envío ---
            success, message = self.send_to_kindle(user_kindle_email, file_data, filename, subject=email_subject)
            
            if success:
                await processing_msg.edit_text(f"✅ <b>¡Enviado con éxito a <code>{user_kindle_email}</code>!</b>", parse_mode=ParseMode.HTML)
            else:
                await processing_msg.edit_text(f"❌ <b>Error al enviar</b>\n\n<i>{message}</i>", parse_mode=ParseMode.HTML)
                
        except Exception as e:
            logger.error(f"Error procesando documento: {e}")
            await processing_msg.edit_text(f"❌ <b>Error inesperado</b>", parse_mode=ParseMode.HTML)

    async def handle_text(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        await update.message.reply_html("Hola, para empezar, configura tu email con <code>/set_email</code> o envíame un documento.")


async def main():
    # --- INICIALIZAMOS LA BASE DE DATOS AL ARRANCAR ---
    setup_database()

    # Ya no necesitamos KINDLE_EMAIL de las variables de entorno
    BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')
    GMAIL_USER = os.getenv('GMAIL_USER')
    GMAIL_APP_PASSWORD = os.getenv('GMAIL_APP_PASSWORD')
    
    if not all([BOT_TOKEN, GMAIL_USER, GMAIL_APP_PASSWORD]):
        logger.critical("Faltan variables de entorno (BOT_TOKEN, GMAIL_USER, GMAIL_APP_PASSWORD). El bot no puede iniciar.")
        return
    
    bot_instance = KindleEmailBot(BOT_TOKEN, GMAIL_USER, GMAIL_APP_PASSWORD)
    application = Application.builder().token(BOT_TOKEN).build()
    
    # --- AÑADIMOS EL NUEVO HANDLER ---
    application.add_handler(CommandHandler("set_email", bot_instance.set_email_command))
    
    application.add_handler(CommandHandler("start", bot_instance.start))
    application.add_handler(CommandHandler("help", bot_instance.help_command))
    application.add_handler(MessageHandler(filters.Document.ALL, bot_instance.handle_document))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, bot_instance.handle_text))
    
    port = int(os.getenv('PORT', 8080))
    web_server_config = uvicorn.Config(app, host="0.0.0.0", port=port)
    web_server = uvicorn.Server(web_server_config)
    
    logger.info(f"Iniciando el bot y el servidor web en el puerto {port}...")
    await asyncio.gather(
        application.run_polling(allowed_updates=Update.ALL_TYPES),
        web_server.serve()
    )

if __name__ == '__main__':
    asyncio.run(main())