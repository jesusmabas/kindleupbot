# database.py - VERSIÓN MEJORADA Y CORREGIDA
import logging
import psycopg2
from typing import List, Tuple, Any, Optional
from contextlib import contextmanager
from config import settings  # <-- IMPORTA LA CONFIGURACIÓN CENTRALIZADA

logger = logging.getLogger(__name__)

# --- GESTOR DE CONTEXTO PARA CONEXIONES ---
@contextmanager
def get_db_connection():
    """
    Proporciona una conexión a la base de datos como un gestor de contexto,
    asegurando que se cierre correctamente.
    """
    # Pydantic Settings ya se asegura de que la variable exista al iniciar.
    # Esta comprobación es una salvaguarda extra.
    if not settings.DATABASE_URL:
        logger.error("La variable de entorno DATABASE_URL no está configurada.")
        raise ValueError("DATABASE_URL no está configurada.")
    
    conn = None
    try:
        # Usa la URL de la configuración
        conn = psycopg2.connect(settings.DATABASE_URL)
        yield conn
        conn.commit()
    except Exception as e:
        if conn:
            conn.rollback() # Revierte los cambios si hubo un error
        logger.error("Error en la transacción de la base de datos: %s", e)
        raise # Vuelve a lanzar la excepción para que el llamador la maneje
    finally:
        if conn:
            conn.close()

def setup_database():
    """Crea las tablas 'users' y 'metrics' si no existen."""
    try:
        with get_db_connection() as conn:
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
        logger.info("Base de datos y tablas ('users', 'metrics') verificadas/creadas.")
    except Exception as e:
        # El error ya se loguea en get_db_connection, aquí solo informamos del contexto.
        logger.error("Fallo al ejecutar setup_database: %s", e)
        # Es un error crítico, así que lo relanzamos para detener la app si es necesario
        raise

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
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, (user_id, kindle_email))
        return True
    except Exception as e:
        # El error ya se loguea, aquí solo indicamos que la operación falló.
        return False

def get_user_email(user_id: int) -> Optional[str]:
    """Obtiene el email de Kindle para un usuario y actualiza su actividad."""
    sql_select = "SELECT kindle_email FROM users WHERE user_id = %s;"
    sql_update = "UPDATE users SET last_activity_at = NOW() WHERE user_id = %s;"
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(sql_select, (user_id,))
                result = cur.fetchone()
                email = result[0] if result else None
                
                # Actualizar la última actividad solo si se encontró el email
                if email:
                    cur.execute(sql_update, (user_id,))
        return email
    except Exception as e:
        return None

def get_total_users() -> int:
    """Obtiene el número total de usuarios registrados."""
    sql = "SELECT COUNT(*) FROM users;"
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(sql)
                result = cur.fetchone()
        return result[0] if result else 0
    except Exception as e:
        return 0

# --- FUNCIONES DE MÉTRICAS ---
def save_metric(metric_name: str, user_id: Optional[int], value: int):
    """Guarda un evento de métrica en la base de datos."""
    sql = "INSERT INTO metrics (metric_name, user_id, value) VALUES (%s, %s, %s);"
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, (metric_name, user_id, value))
    except Exception as e:
        # Evitamos que un fallo al guardar métricas detenga la app
        pass

def get_metrics_from_db() -> List[Tuple[Any, ...]]:
    """Obtiene todos los registros de métricas de la base de datos."""
    sql = "SELECT metric_name, user_id, value FROM metrics;"
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(sql)
                return cur.fetchall()
    except Exception as e:
        return []
