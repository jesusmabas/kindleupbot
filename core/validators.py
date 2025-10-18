# kindleupbot/core/validators.py
import re
from pathlib import Path
from typing import Tuple

# Definimos aquí las constantes que usan los validadores
SUPPORTED_FORMATS = {
    '.epub': 'application/epub+zip', '.pdf': 'application/pdf', '.mobi': 'application/x-mobipocket-ebook',
    '.azw': 'application/vnd.amazon.ebook', '.doc': 'application/msword',
    '.docx': 'application/vnd.openxmlformats-officedocument.wordprocessingml.document', '.rtf': 'application/rtf',
    '.txt': 'text/plain', '.html': 'text/html', '.htm': 'text/html', '.jpg': 'image/jpeg',
    '.jpeg': 'image/jpeg', '.png': 'image/png', '.gif': 'image/gif', '.bmp': 'image/bmp',
    '.md': 'text/markdown'
}
PROMPT_SET_EMAIL = "📧 Por favor, introduce tu email de Kindle (ejemplo: usuario@kindle.com):"

class EmailValidator:
    @staticmethod
    def validate_email(email: str) -> bool:
        if not email or len(email) < 5: return False
        return bool(re.match(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$', email))

    @staticmethod
    def is_kindle_email(email: str) -> bool:
        return any(domain in email.lower() for domain in ['@kindle.com', '@free.kindle.com'])

class FileValidator:
    @staticmethod
    def validate_file(filename: str, file_size: int, max_size: int) -> Tuple[bool, str]:
        if not filename: return False, "Nombre de archivo vacío"
        ext = Path(filename).suffix.lower()
        if ext not in SUPPORTED_FORMATS: return False, f"Formato {ext} no soportado"
        if file_size > max_size: return False, f"Archivo muy grande ({file_size / 1024**2:.1f}MB > {max_size / 1024**2:.1f}MB)"
        return True, "OK"