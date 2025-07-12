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

from telegram import Update, ReplyKeyboardMarkup, KeyboardButton, ReplyKeyboardRemove, ForceReply
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
from telegram.constants import ParseMode

# --- CONFIGURACIÓN DE LA APLICACIÓN WEB Y EL BOT ---
app = FastAPI()

BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')
application = Application.builder().token(BOT_TOKEN).build()
# --------------------------------------------------------

logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# --- Texto de nuestros prompts para poder identificarlos ---
PROMPT_SET_EMAIL = "De acuerdo, por favor, introduce ahora tu email de Kindle:"

# --- EVENTOS DE CICLO DE VIDA ---
@app.on_event("startup")
async def startup_event():
    logger.info("El servidor web está arrancando...")
    setup_database()
    
    gmail_user = os.getenv('GMAIL_USER')
    gmail_password = os.getenv('GMAIL_APP_PASSWORD')

    bot_instance = KindleEmailBot(BOT_TOKEN, gmail_user, gmail_password)

    # Registramos todos los handlers. El orden es importante.
    application.add_handler(CommandHandler("start", bot_instance.start))
    application.add_handler(CommandHandler("help", bot_instance.help_command))
    application.add_handler(CommandHandler("set_email", bot_instance.set_email_command))
    application.add_handler(CommandHandler("my_email", bot_instance.my_email_command))
    application.add_handler(CommandHandler("hide_keyboard", bot_instance.hide_keyboard_command))
    
    # NUEVO HANDLER: Este se encargará de capturar la respuesta del email
    application.add_handler(MessageHandler(
        filters.TEXT & ~filters.COMMAND & filters.REPLY, 
        bot_instance.handle_email_input
    ))

    # El handler de documentos y texto genérico van al final
    application.add_handler(MessageHandler(filters.Document.ALL, bot_instance.handle_document))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, bot_instance.handle_text))
    
    await application.initialize()
    await application.updater.start_polling(drop_pending_updates=True)
    await application.start()
    
    logger.info("El bot de Telegram ha sido inicializado y está funcionando.")

@app.on_event("shutdown")
async def shutdown_event():
    logger.info("El servidor web se está apagando...")
    await application.updater.stop()
    await application.stop()
    await application.shutdown()
    logger.info("El bot de Telegram se ha detenido de forma segura.")

@app.get("/")
async def read_root():
    return {"status": "Bot is alive and running!"}

# --- CLASE PRINCIPAL DEL BOT CON LA LÓGICA ---
SUPPORTED_FORMATS = ('.epub', '.pdf', '.mobi', '.azw', '.doc', '.docx', '.rtf', '.txt', '.html', '.htm', '.jpg', '.jpeg', '.png', '.gif', '.bmp')

class KindleEmailBot:
    def __init__(self, bot_token, gmail_user, gmail_password):
        self.bot_token = bot_token
        self.gmail_user = gmail_user
        self.gmail_password = gmail_password
        self.main_keyboard = ReplyKeyboardMarkup(
            [["/set_email 📧", "/my_email 🧐"], ["/help ❓"]], 
            resize_keyboard=True
        )

    async def start(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        user = update.effective_user
        start_text = f"""
👋 ¡Hola, {user.mention_html()}!
📚 <b>Bienvenido al Asistente de Envío a Kindle</b>
Conmigo, tu Telegram se convierte en un portal directo a tu biblioteca Kindle.
🚀 <b>Primer Paso: Configurar tu Email</b>
Usa el botón <b>/set_email 📧</b> de abajo para empezar. Te pediré tu email a continuación.
Una vez configurado, simplemente envíame cualquier documento compatible.
        """
        await update.message.reply_html(start_text, reply_markup=self.main_keyboard)

    async def help_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        help_text = f"""
🤔 <b>Guía Completa del Bot</b> 🤔
<b>📖 CÓMO FUNCIONA</b>
1.  Pulsa <b>/set_email 📧</b> y envíame tu email cuando te lo pida.
2.  <b>Envía un archivo</b> compatible.
3.  <b>¡Lee en tu Kindle!</b>
<b>⚙️ COMANDOS DISPONIBLES</b>
• <code>/start</code> - Mensaje de bienvenida.
• <code>/help</code> - Muestra esta guía.
• <code>/set_email</code> - Inicia el proceso para guardar tu email de Kindle.
• <code>/my_email</code> - Muestra tu email configurado.
• <code>/hide_keyboard</code> - Oculta los botones.
⭐ <b>TRUCO PARA PDFS</b>
Escribe <code>convert</code> en el pie de foto de un PDF para convertirlo a formato de libro.
<b>🔒 REQUISITO IMPORTANTE</b>
Debes autorizar mi dirección de correo en tu cuenta de Amazon:
<code>{self.gmail_user}</code>
        """
        await update.message.reply_html(help_text, reply_markup=self.main_keyboard)

    async def set_email_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Inicia la conversación para pedir el email."""
        await update.message.reply_text(
            PROMPT_SET_EMAIL, # Usamos la constante que definimos
            reply_markup=ForceReply(selective=True, input_field_placeholder="tu_email@kindle.com")
        )

    async def handle_email_input(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Maneja la respuesta del usuario cuando introduce su email."""
        # Verificamos que el usuario está respondiendo a nuestro mensaje de petición de email
        if update.message.reply_to_message and update.message.reply_to_message.text == PROMPT_SET_EMAIL:
            user_id = update.effective_user.id
            kindle_email = update.message.text

            if '@' not in kindle_email or '.' not in kindle_email:
                await update.message.reply_html(
                    "El formato del email no parece válido. Por favor, inténtalo de nuevo usando el comando /set_email 📧"
                )
                return
            
            if set_user_email(user_id, kindle_email):
                await update.message.reply_html(
                    f"✅ ¡Genial! Tu email de Kindle ha sido guardado como:\n<code>{kindle_email}</code>"
                )
            else:
                await update.message.reply_html("❌ Hubo un error al guardar tu email.")

    async def my_email_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        user_id = update.effective_user.id
        saved_email = get_user_email(user_id)
        if saved_email:
            await update.message.reply_html(f"Tu email de Kindle configurado es:\n<code>{saved_email}</code>")
        else:
            await update.message.reply_html("Aún no has configurado tu email. Usa <b>/set_email 📧</b> para empezar.", parse_mode=ParseMode.HTML)

    async def hide_keyboard_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        await update.message.reply_text("Teclado ocultado.", reply_markup=ReplyKeyboardRemove())

    def send_to_kindle(self, kindle_email_destino, file_data, filename, subject=""):
        # (Esta función no necesita cambios)
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
            return True, "Enviado con éxito."
        except Exception as e:
            logger.error(f"Error al enviar email: {e}")
            return False, str(e)

    async def handle_document(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        # (Esta función no necesita cambios)
        user_id = update.effective_user.id
        user_kindle_email = get_user_email(user_id)
        if not user_kindle_email:
            await update.message.reply_html("⚠️ No has configurado tu email. Usa <code>/set_email tu_email@kindle.com</code>.")
            return
        document = update.message.document
        filename = document.file_name
        if not any(filename.lower().endswith(ext) for ext in SUPPORTED_FORMATS):
            await update.message.reply_html(f"❌ Formato de archivo no soportado.")
            return
        if document.file_size > 48 * 1024 * 1024:
            await update.message.reply_html("❌ Archivo demasiado grande.")
            return
        processing_msg = await update.message.reply_html("✅ Recibido. Enviando a Kindle...")
        try:
            file = await context.bot.get_file(document.file_id)
            file_data = await file.download_as_bytearray()
            email_subject = "Convert" if filename.lower().endswith('.pdf') and update.message.caption and update.message.caption.lower().strip() == 'convert' else ""
            success, message = self.send_to_kindle(user_kindle_email, file_data, filename, subject=email_subject)
            if success:
                await processing_msg.edit_text(f"✅ <b>¡Enviado a <code>{user_kindle_email}</code>!</b>", parse_mode=ParseMode.HTML)
            else:
                await processing_msg.edit_text(f"❌ <b>Error al enviar:</b> <i>{message}</i>", parse_mode=ParseMode.HTML)
        except Exception as e:
            logger.error(f"Error procesando documento: {e}")
            await processing_msg.edit_text(f"❌ Error inesperado.", parse_mode=ParseMode.HTML)

    async def handle_text(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        await update.message.reply_html("No he entendido eso. Usa los botones de abajo o envíame un documento.", reply_markup=self.main_keyboard)

# --- PUNTO DE ENTRADA ---
if __name__ == "__main__":
    port = int(os.getenv("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)