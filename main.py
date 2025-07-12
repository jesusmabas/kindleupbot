# main.py
import os
import logging
import smtplib
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
            f"📚 <b>Bot de Envío a Kindle</b>\n\n"
            f"Este bot reenvía tus libros <code>.epub</code> directamente a tu email de Kindle. "
            f"Amazon se encargará de la conversión automática.\n\n"
            f"<b>¿Cómo funciona?</b>\n"
            f"1. Envíame un archivo <code>.epub</code>.\n"
            f"2. Lo enviaré a tu dirección de Kindle configurada.\n"
            f"3. ¡Aparecerá en tu biblioteca en minutos!\n\n"
            f"Usa /help para más información."
        )

    async def help_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Comando /help. Explica el funcionamiento."""
        help_text = f"""
📚 <b>Bot de Envío a Kindle - Ayuda</b>

<b>Funcionamiento Principal:</b>
Este bot actúa como un intermediario para el servicio oficial "Enviar a Kindle" de Amazon.

1.  <b>Envía un archivo <code>.epub</code></b>: Simplemente comparte el documento conmigo.
2.  <b>Envío automático</b>: El bot adjuntará el archivo a un correo y lo enviará a <code>{self.kindle_email}</code>.
3.  <b>Conversión por Amazon</b>: El propio servicio de Amazon recibirá el EPUB, lo convertirá a un formato compatible y lo añadirá a tu biblioteca Kindle.

<b>Ventajas:</b>
• <b>Sincronización (Whispersync)</b>: Tus notas y progreso de lectura se sincronizarán en todos tus dispositivos.
• <b>Gratuito y sin límites de API</b>: No dependemos de servicios externos de conversión.
• <b>Fácil y directo</b>: No necesitas abrir tu cliente de correo.

<b>Requisitos de configuración (¡importante!):</b>
• Debes haber añadido tu dirección de correo <code>{self.gmail_user}</code> a la lista de remitentes aprobados en tu cuenta de Amazon.
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

<i>El bot está listo para recibir tus archivos <code>.epub</code>.</i>
        """
        await update.message.reply_html(status_text)

    def send_to_kindle(self, file_data, filename):
        """Envía el archivo original al Kindle por email."""
        try:
            msg = MIMEMultipart()
            msg['From'] = self.gmail_user
            msg['To'] = self.kindle_email
            msg['Subject'] = f"Libro para tu Kindle: {filename}"
            
            # Cuerpo del mensaje (opcional, pero buena práctica)
            body = f"Este libro ha sido enviado automáticamente a través de tu bot de Telegram.\n\nDisfruta de la lectura."
            msg.attach(MIMEText(body, 'plain'))
            
            # Adjuntar el archivo .epub original
            part = MIMEBase('application', 'epub+zip')
            part.set_payload(file_data)
            encoders.encode_base64(part)
            part.add_header('Content-Disposition', f'attachment; filename="{filename}"')
            msg.attach(part)
            
            # Conectar y enviar email usando el servidor SMTP de Gmail
            server = smtplib.SMTP('smtp.gmail.com', 587)
            server.starttls()
            server.login(self.gmail_user, self.gmail_password)
            server.send_message(msg)
            server.quit()
            
            return True, "Libro enviado exitosamente a Kindle."
            
        except Exception as e:
            logger.error(f"Error al enviar email: {e}")
            return False, f"Error al enviar el email: {str(e)}"

    async def handle_document(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Maneja documentos enviados al bot."""
        document = update.message.document
        
        if not document.file_name.lower().endswith('.epub'):
            await update.message.reply_html(
                "❌ <b>Formato no soportado</b>\n\nSolo acepto archivos <code>.epub</code>."
            )
            return
        
        if document.file_size > 48 * 1024 * 1024:  # Amazon limita a 50MB, usamos 48MB como margen seguro
            await update.message.reply_html(
                "❌ <b>Archivo demasiado grande</b>\n\nEl límite de Amazon es 50MB. Este archivo es demasiado pesado."
            )
            return
        
        processing_msg = await update.message.reply_html(
            "✅ <b>Archivo recibido...</b>\n\n"
            "📥 Descargando y preparando para el envío..."
        )
        
        try:
            # Descargar el archivo a la memoria
            file = await context.bot.get_file(document.file_id)
            file_data = await file.download_as_bytearray()
            
            await processing_msg.edit_text(
                "📧 <b>Enviando a tu Kindle...</b>\n\n"
                "<i>Amazon lo procesará y aparecerá en tu biblioteca en unos minutos.</i>",
                parse_mode=ParseMode.HTML
            )
            
            # Llamar a la función de envío de email
            success, message = self.send_to_kindle(file_data, document.file_name)
            
            if success:
                await processing_msg.edit_text(
                    f"✅ <b>¡Enviado con éxito!</b>\n\n"
                    f"📖 <b>{document.file_name}</b>\n"
                    f"DESTINO: <code>{self.kindle_email}</code>\n\n"
                    f"Revisa la biblioteca de tu Kindle en 5-10 minutos.",
                    parse_mode=ParseMode.HTML
                )
            else:
                await processing_msg.edit_text(
                    f"❌ <b>Error al enviar</b>\n\n<i>{message}</i>\n\n"
                    f"Verifica que la contraseña de aplicación de Gmail sea correcta y que tu email esté aprobado en Amazon.",
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
            "Hola, para usarme, simplemente envíame un archivo con la extensión <code>.epub</code>."
        )

def main():
    """Función principal que inicia el bot."""
    # Obtener variables de entorno (la forma segura de guardar tus secretos)
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
    
    logger.info("Iniciando el bot...")
    application.run_polling(allowed_updates=Update.ALL_TYPES)

if __name__ == '__main__':
    main()
