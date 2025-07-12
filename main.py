# main.py
import os
import logging
import smtplib
import mimetypes
import asyncio
import uvicorn
from fastapi import FastAPI

from database import setup_database, set_user_email, get_user_email

from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email.mime.text import MIMEText
from email import encoders

from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
from telegram.constants import ParseMode

# --- CONFIGURACIÓN DE LA APLICACIÓN WEB Y EL BOT ---
# Le decimos a FastAPI que use un "ciclo de vida" para gestionar el bot
app = FastAPI()

# Creamos la instancia de la aplicación del bot fuera para poder acceder a ella
# en los eventos de inicio y apagado.
BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')
application = Application.builder().token(BOT_TOKEN).build()

# --------------------------------------------------------

logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# --- EVENTO DE INICIO: AQUÍ ARRANCAMOS TODO ---
@app.on_event("startup")
async def startup_event():
    logger.info("El servidor web está arrancando...")
    
    # 1. Configurar la base de datos
    setup_database()
    
    # 2. Obtener las variables de entorno para la instancia del bot
    gmail_user = os.getenv('GMAIL_USER')
    gmail_password = os.getenv('GMAIL_APP_PASSWORD')

    # 3. Crear la instancia de nuestra clase de lógica del bot
    bot_instance = KindleEmailBot(BOT_TOKEN, gmail_user, gmail_password)

    # 4. Registrar todos los manejadores de comandos y mensajes
    application.add_handler(CommandHandler("set_email", bot_instance.set_email_command))
    application.add_handler(CommandHandler("start", bot_instance.start))
    application.add_handler(CommandHandler("help", bot_instance.help_command))
    application.add_handler(MessageHandler(filters.Document.ALL, bot_instance.handle_document))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, bot_instance.handle_text))
    
    # 5. Iniciar el bot en segundo plano. Esto es NO bloqueante.
    await application.initialize()
    await application.updater.start_polling()
    await application.start()
    
    logger.info("El bot de Telegram ha sido inicializado y está funcionando en segundo plano.")

# --- EVENTO DE APAGADO: AQUÍ PARAMOS EL BOT DE FORMA SEGURA ---
@app.on_event("shutdown")
async def shutdown_event():
    logger.info("El servidor web se está apagando...")
    
    await application.updater.stop()
    await application.stop()
    await application.shutdown()
    
    logger.info("El bot de Telegram se ha detenido de forma segura.")

# Ruta principal para que Render vea que el servidor está vivo
@app.get("/")
async def read_root():
    return {"status": "Bot is alive and running!"}

# --- EL CÓDIGO DE LA LÓGICA DEL BOT (SIN CAMBIOS) ---
SUPPORTED_FORMATS = ('.epub', '.pdf', '.mobi', '.azw', '.doc', '.docx', '.rtf', '.txt', '.html', '.htm', '.jpg', '.jpeg', '.png', '.gif', '.bmp')

class KindleEmailBot:
    def __init__(self, bot_token, gmail_user, gmail_password):
        self.bot_token = bot_token
        self.gmail_user = gmail_user
        self.gmail_password = gmail_password
    
    async def set_email_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        user_id = update.effective_user.id
        if not context.args:
            await update.message.reply_html("Ejemplo: <code>/set_email tu_email@kindle.com</code>")
            return
        kindle_email = context.args[0]
        if '@' not in kindle_email or '.' not in kindle_email:
            await update.message.reply_html("El formato del email no parece válido.")
            return
        if set_user_email(user_id, kindle_email):
            await update.message.reply_html(f"✅ Email guardado: <code>{kindle_email}</code>")
        else:
            await update.message.reply_html("❌ Hubo un error al guardar tu email.")

    async def start(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        user = update.effective_user
        await update.message.reply_html(
            f"¡Hola {user.mention_html()}!\n\n"
            f"<b>PRIMER PASO:</b> Configura tu email de Kindle con:\n"
            f"<code>/set_email tu_direccion@kindle.com</code>"
        )

    async def help_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        help_text = """
        📚 <b>Ayuda del Bot</b>
        1️⃣ <b>Configura tu Email:</b> <code>/set_email tu_email@kindle.com</code>
        2️⃣ <b>Envía un Documento:</b> Simplemente comparte un archivo compatible.
        ⭐ <b>TRUCO PDF:</b> Escribe <b>convert</b> en el pie de foto para convertir PDFs a formato de libro.
        🔧 <b>Importante:</b> Autoriza <code>{}</code> en tu cuenta de Amazon.
        """.format(self.gmail_user)
        await update.message.reply_html(help_text)

    def send_to_kindle(self, kindle_email_destino, file_data, filename, subject=""):
        try:
            msg = MIMEMultipart()
            msg['From'] = self.gmail_user
            msg['To'] = kindle_email_destino
            msg['Subject'] = subject if subject else f"Documento para Kindle: {filename}"
            msg.attach(MIMEText(f"Enviado por tu bot de Telegram.", 'plain'))
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
            return True, "Enviado."
        except Exception as e:
            logger.error(f"Error al enviar email: {e}")
            return False, str(e)

    async def handle_document(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        user_id = update.effective_user.id
        user_kindle_email = get_user_email(user_id)
        if not user_kindle_email:
            await update.message.reply_html("⚠️ No has configurado tu email. Usa <code>/set_email</code>.")
            return
        document = update.message.document
        filename = document.file_name
        if not any(filename.lower().endswith(ext) for ext in SUPPORTED_FORMATS):
            await update.message.reply_html(f"❌ Formato no soportado.")
            return
        if document.file_size > 48 * 1024 * 1024:
            await update.message.reply_html("❌ Archivo demasiado grande.")
            return
        processing_msg = await update.message.reply_html("✅ Recibido. Procesando...")
        try:
            file = await context.bot.get_file(document.file_id)
            file_data = await file.download_as_bytearray()
            email_subject = "Convert" if filename.lower().endswith('.pdf') and update.message.caption and update.message.caption.lower().strip() == 'convert' else ""
            success, message = self.send_to_kindle(user_kindle_email, file_data, filename, subject=email_subject)
            if success:
                await processing_msg.edit_text(f"✅ <b>¡Enviado a <code>{user_kindle_email}</code>!</b>", parse_mode=ParseMode.HTML)
            else:
                await processing_msg.edit_text(f"❌ <b>Error:</b> <i>{message}</i>", parse_mode=ParseMode.HTML)
        except Exception as e:
            logger.error(f"Error procesando documento: {e}")
            await processing_msg.edit_text(f"❌ Error inesperado.", parse_mode=ParseMode.HTML)

    async def handle_text(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        await update.message.reply_html("Hola, configura tu email con <code>/set_email</code> o envíame un documento.")


# --- EL PUNTO DE ENTRADA AHORA SOLO INICIA EL SERVIDOR WEB ---
if __name__ == "__main__":
    # uvicorn se encargará de llamar a los eventos de startup y shutdown.
    port = int(os.getenv("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)