# database.py
import os
import logging
import psycopg2 # La nueva librería para PostgreSQL

# Render nos dará la URL de conexión en esta variable de entorno
DATABASE_URL = os.getenv("DATABASE_URL")

def get_db_connection():
    """Establece una conexión con la base de datos PostgreSQL."""
    try:
        conn = psycopg2.connect(DATABASE_URL)
        return conn
    except psycopg2.OperationalError as e:
        logging.error("Error crítico de conexión a la base de datos: %s", e)
        # Si la base de datos no está lista o la URL es incorrecta, el bot no puede funcionar.
        raise e

def setup_database():
    """Crea la tabla de usuarios en PostgreSQL si no existe."""
    try:
        conn = get_db_connection()
        with conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    user_id BIGINT PRIMARY KEY,
                    kindle_email TEXT NOT NULL
                )
            """)
        conn.commit()
        conn.close()
        logging.info("Tabla 'users' verificada/creada en la base de datos PostgreSQL.")
    except Exception as e:
        logging.error("Error al configurar la tabla en la base de datos: %s", e)

def set_user_email(user_id, kindle_email):
    """Guarda o actualiza el email de Kindle para un usuario en PostgreSQL."""
    # La sintaxis para "insertar o actualizar" es diferente en PostgreSQL
    sql = """
        INSERT INTO users (user_id, kindle_email) VALUES (%s, %s)
        ON CONFLICT (user_id) DO UPDATE SET kindle_email = EXCLUDED.kindle_email;
    """
    try:
        conn = get_db_connection()
        with conn.cursor() as cur:
            cur.execute(sql, (user_id, kindle_email))
        conn.commit()
        conn.close()
        return True
    except Exception as e:
        logging.error("Error al guardar email para el usuario %s: %s", user_id, e)
        return False

def get_user_email(user_id):
    """Obtiene el email de Kindle para un usuario desde PostgreSQL."""
    sql = "SELECT kindle_email FROM users WHERE user_id = %s;"
    try:
        conn = get_db_connection()
        with conn.cursor() as cur:
            cur.execute(sql, (user_id,))
            result = cur.fetchone()
        conn.close()
        
        if result:
            return result[0]
        else:
            return None
    except Exception as e:
        logging.error("Error al obtener email para el usuario %s: %s", user_id, e)
        return None