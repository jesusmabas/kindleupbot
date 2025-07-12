# database.py
import os
import logging
import psycopg2
from typing import List, Tuple, Any

# --- CONFIGURACIÓN ---
DATABASE_URL = os.getenv("DATABASE_URL")
logger = logging.getLogger(__name__)

# --- FUNCIONES DE CONEXIÓN Y CONFIGURACIÓN ---
def get_db_connection():
    """Establece una conexión con la base de datos PostgreSQL."""
    try:
        conn = psycopg2.connect(DATABASE_URL)
        return conn
    except psycopg2.OperationalError as e:
        logger.error("Error crítico de conexión a la base de datos: %s", e)
        raise e

def setup_database():
    """Crea las tablas 'users' y 'metrics' si no existen."""
    try:
        conn = get_db_connection()
        with conn.cursor() as cur:
            # Tabla de usuarios
            cur.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    user_id BIGINT PRIMARY KEY,
                    kindle_email TEXT NOT NULL,
                    created_at TIMESTAMPTZ DEFAULT NOW(),
                    last_activity_at TIMESTAMPTZ
                )
            """)
            # Nueva tabla para métricas
            cur.execute("""
                CREATE TABLE IF NOT EXISTS metrics (
                    id SERIAL PRIMARY KEY,
                    timestamp TIMESTAMPTZ DEFAULT NOW(),
                    metric_name TEXT NOT NULL,
                    user_id BIGINT,
                    value INTEGER DEFAULT 1
                )
            """)
        conn.commit()
        conn.close()
        logger.info("Base de datos y tablas ('users', 'metrics') verificadas/creadas.")
    except Exception as e:
        logger.error("Error al configurar la base de datos: %s", e)

# --- FUNCIONES DE USUARIO ---
def set_user_email(user_id: int, kindle_email: str) -> bool:
    """Guarda o actualiza el email de Kindle y la actividad del usuario."""
    sql = """
        INSERT INTO users (user_id, kindle_email, last_activity_at) VALUES (%s, %s, NOW())
        ON CONFLICT (user_id) DO UPDATE SET 
            kindle_email = EXCLUDED.kindle_email,
            last_activity_at = NOW();
    """
    try:
        conn = get_db_connection()
        with conn.cursor() as cur:
            cur.execute(sql, (user_id, kindle_email))
        conn.commit()
        conn.close()
        return True
    except Exception as e:
        logger.error("Error al guardar email para el usuario %s: %s", user_id, e)
        return False

def get_user_email(user_id: int) -> str or None:
    """Obtiene el email de Kindle para un usuario y actualiza su actividad."""
    email = None
    try:
        conn = get_db_connection()
        with conn.cursor() as cur:
            cur.execute("SELECT kindle_email FROM users WHERE user_id = %s;", (user_id,))
            result = cur.fetchone()
            email = result[0] if result else None
            
            # Actualizar la última actividad
            if email:
                cur.execute("UPDATE users SET last_activity_at = NOW() WHERE user_id = %s;", (user_id,))
                conn.commit()
    except Exception as e:
        logger.error("Error al obtener email para el usuario %s: %s", user_id, e)
    finally:
        if conn:
            conn.close()
    return email

def get_total_users() -> int:
    """Obtiene el número total de usuarios registrados."""
    sql = "SELECT COUNT(*) FROM users;"
    try:
        conn = get_db_connection()
        with conn.cursor() as cur:
            cur.execute(sql)
            result = cur.fetchone()
        conn.close()
        return result[0] if result else 0
    except Exception as e:
        logger.error(f"Error al contar usuarios: {e}")
        return 0

# --- FUNCIONES DE MÉTRICAS ---
def save_metric(metric_name: str, user_id: int or None, value: int):
    """Guarda un evento de métrica en la base de datos."""
    sql = "INSERT INTO metrics (metric_name, user_id, value) VALUES (%s, %s, %s);"
    try:
        conn = get_db_connection()
        with conn.cursor() as cur:
            cur.execute(sql, (metric_name, user_id, value))
        conn.commit()
        conn.close()
    except Exception as e:
        logger.error(f"Error al guardar métrica '{metric_name}': {e}")

def get_metrics_from_db() -> List[Tuple[Any, ...]]:
    """Obtiene todos los registros de métricas de la base de datos."""
    sql = "SELECT metric_name, user_id, value FROM metrics;"
    try:
        conn = get_db_connection()
        with conn.cursor() as cur:
            cur.execute(sql)
            results = cur.fetchall()
        conn.close()
        return results
    except Exception as e:
        logger.error(f"Error al obtener métricas de la BD: {e}")
        return []
