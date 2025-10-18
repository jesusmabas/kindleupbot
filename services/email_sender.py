# kindleupbot/services/email_sender.py
import smtplib
import unicodedata
import logging
import asyncio
from typing import Tuple

from email.mime.multipart import MIMEMultipart
from email.mime.application import MIMEApplication
from email.mime.text import MIMEText

from config import Settings

logger = logging.getLogger(__name__)

class EmailSender:
    def __init__(self, config: Settings):
        self.config = config

    async def send_to_kindle_with_retries(self, kindle_email: str, file_data: bytes, filename: str, subject: str) -> Tuple[bool, str]:
        for attempt in range(self.config.MAX_RETRIES):
            try:
                success, msg = await self._send_to_kindle(kindle_email, file_data, filename, subject)
                if success:
                    return True, msg
                if attempt < self.config.MAX_RETRIES - 1:
                    await asyncio.sleep(self.config.RETRY_DELAY * (2 ** attempt))
            except Exception as e:
                logger.warning(f"Intento de envío {attempt + 1} fallido: {e}")
                if attempt < self.config.MAX_RETRIES - 1:
                    await asyncio.sleep(self.config.RETRY_DELAY * (2 ** attempt))
                else:
                    return False, f"Error después de {self.config.MAX_RETRIES} intentos: {str(e)}"
        return False, "Falló después de todos los reintentos"

    async def _send_to_kindle(self, kindle_email: str, file_data: bytes, filename: str, subject: str) -> Tuple[bool, str]:
        return await asyncio.to_thread(
            self._send_to_kindle_sync, kindle_email, file_data, filename, subject
        )

    def _send_to_kindle_sync(self, kindle_email: str, file_data: bytes, filename: str, subject: str) -> Tuple[bool, str]:
        try:
            nfkd = unicodedata.normalize('NFKD', filename)
            safe_fn = ''.join(c for c in nfkd if unicodedata.category(c) != 'Mn')
            
            msg = MIMEMultipart()
            msg['From'] = self.config.GMAIL_USER
            msg['To'] = kindle_email
            msg['Subject'] = subject or f"Doc: {safe_fn}"
            msg.attach(MIMEText(f"Enviado desde tu Bot de Telegram. Archivo: {safe_fn}"))
            
            subtype = safe_fn.rsplit('.', 1)[-1]
            part = MIMEApplication(file_data, _subtype=subtype)
            part.add_header('Content-Disposition', 'attachment', filename=safe_fn)
            msg.attach(part)
            
            with smtplib.SMTP_SSL(self.config.SMTP_SERVER, self.config.SMTP_PORT) as server:
                server.login(self.config.GMAIL_USER, self.config.GMAIL_APP_PASSWORD)
                server.send_message(msg)
                
            logger.info(f"Documento {safe_fn} enviado a {kindle_email}")
            return True, "Enviado"
            
        except smtplib.SMTPAuthenticationError:
            logger.error("Error de autenticación SMTP. Revisa GMAIL_USER y GMAIL_APP_PASSWORD.")
            return False, "Error de autenticación SMTP"
        except smtplib.SMTPRecipientsRefused:
            logger.warning(f"El email del destinatario {kindle_email} fue rechazado.")
            return False, "Email de destinatario rechazado"
        except Exception as e:
            logger.error(f"Error SMTP inesperado al enviar a Kindle: {e}", exc_info=True)
            return False, f"Error SMTP: {str(e)}"