# database.py
import sqlite3
import os
import logging

# Render nos dará un disco persistente en /data. Usaremos esa ruta.
# Si la variable de entorno RENDER no existe (ej. probando en local), usa el directorio actual.
DB_DIR = "/data" if os.getenv("RENDER") else "."
DB_PATH = os.path.join(DB_DIR, "users.db")

def setup_database():
    """Crea la tabla de usuarios si no existe."""
    # Asegurarse de que el directorio de la base de datos exista en Render
    if not os.path.exists(DB_DIR):
        os.makedirs(DB_DIR)
        
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        # Creamos una tabla para guardar el ID de usuario de Telegram y su email de Kindle
        # user_id es INTEGER y PRIMARY KEY para que sea único.
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS users (
                user_id INTEGER PRIMARY KEY,
                kindle_email TEXT NOT NULL
            )
        """)
        conn.commit()
        conn.close()
        logging.info("Base de datos configurada y lista en: %s", DB_PATH)
    except Exception as e:
        logging.error("Error al configurar la base de datos: %s", e)

def set_user_email(user_id, kindle_email):
    """Guarda o actualiza el email de Kindle para un usuario."""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        # "INSERT OR REPLACE" es muy útil: si el user_id ya existe, actualiza su email.
        # Si no existe, lo inserta como una nueva fila.
        cursor.execute("INSERT OR REPLACE INTO users (user_id, kindle_email) VALUES (?, ?)", (user_id, kindle_email))
        conn.commit()
        conn.close()
        return True
    except Exception as e:
        logging.error("Error al guardar el email para el usuario %s: %s", user_id, e)
        return False

def get_user_email(user_id):
    """Obtiene el email de Kindle para un usuario."""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT kindle_email FROM users WHERE user_id = ?", (user_id,))
        result = cursor.fetchone() # fetchone() devuelve una tupla (email,) o None si no encuentra nada
        conn.close()
        
        if result:
            return result[0] # Devolvemos solo el email (el primer elemento de la tupla)
        else:
            return None # El usuario no ha configurado su email todavía
    except Exception as e:
        logging.error("Error al obtener el email para el usuario %s: %s", user_id, e)
        return None