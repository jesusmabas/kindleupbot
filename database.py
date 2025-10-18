# database.py - VERSIÓN MEJORADA Y CORREGIDA
import logging
import psycopg2
from typing import List, Tuple, Any, Optional
from contextlib import contextmanager
from .config import settings

logger = logging.getLogger(__name__)

# --- GESTOR DE CONTEXTO PARA CONEXIONES ---
@contextmanager
def get_db_connection():
    """Proporciona una conexión a la base de datos como un gestor de contexto."""
    if not settings.DATABASE_URL:
        logger.error("La variable de entorno DATABASE_URL no está configurada.")
        raise ValueError("DATABASE_URL no está configurada.")
    conn = None
    try:
        conn = psycopg2.connect(settings.DATABASE_URL)
        yield conn
        conn.commit()
    except Exception as e:
        if conn:
            conn.rollback()
        logger.error("Error en la transacción de la base de datos: %s", e)
        raise
    finally:
        if conn:
            conn.close()

def setup_database():
    """Crea todas las tablas necesarias si no existen."""
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
                # Tabla de métricas
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS metrics (
                        id SERIAL PRIMARY KEY,
                        timestamp TIMESTAMPTZ DEFAULT NOW(),
                        metric_name TEXT NOT NULL,
                        user_id BIGINT,
                        value INTEGER DEFAULT 1
                    )
                """)
                # --- NUEVA TABLA DE AUDITORÍA ---
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS admin_logs (
                        id SERIAL PRIMARY KEY,
                        timestamp TIMESTAMPTZ DEFAULT NOW(),
                        admin_user_id BIGINT NOT NULL,
                        action TEXT NOT NULL,
                        details TEXT
                    )
                """)
        logger.info("Base de datos y tablas ('users', 'metrics', 'admin_logs') verificadas/creadas.")
    except Exception as e:
        logger.error("Fallo al ejecutar setup_database: %s", e)
        raise

# --- FUNCIONES DE USUARIO (sin cambios) ---
def set_user_email(user_id: int, kindle_email: str) -> bool:
    # ... (código existente)
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
    except Exception:
        return False

def get_user_email(user_id: int) -> Optional[str]:
    # ... (código existente)
    sql_select = "SELECT kindle_email FROM users WHERE user_id = %s;"
    sql_update = "UPDATE users SET last_activity_at = NOW() WHERE user_id = %s;"
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(sql_select, (user_id,))
                result = cur.fetchone()
                email = result[0] if result else None
                if email:
                    cur.execute(sql_update, (user_id,))
        return email
    except Exception:
        return None

def get_total_users() -> int:
    # ... (código existente)
    sql = "SELECT COUNT(*) FROM users;"
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(sql)
                result = cur.fetchone()
        return result[0] if result else 0
    except Exception:
        return 0

# --- FUNCIONES DE MÉTRICAS ---
def save_metric(metric_name: str, user_id: Optional[int], value: int):
    # ... (código existente)
    sql = "INSERT INTO metrics (metric_name, user_id, value) VALUES (%s, %s, %s);"
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, (metric_name, user_id, value))
    except Exception:
        pass

def get_metrics_from_db() -> List[Tuple[Any, ...]]:
    # ... (código existente)
    sql = "SELECT metric_name, user_id, value FROM metrics;"
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(sql)
                return cur.fetchall()
    except Exception:
        return []

def reset_metrics_table() -> bool:
    """Vacía la tabla 'metrics'. Usa TRUNCATE para eficiencia."""
    sql = "TRUNCATE TABLE metrics RESTART IDENTITY;"
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(sql)
        logger.warning("La tabla 'metrics' ha sido vaciada.")
        return True
    except Exception as e:
        logger.error("Fallo al truncar la tabla 'metrics': %s", e)
        return False

# --- NUEVA FUNCIÓN DE AUDITORÍA ---
def log_admin_action(admin_user_id: int, action: str, details: str = "") -> bool:
    """Registra una acción administrativa en la base de datos."""
    sql = "INSERT INTO admin_logs (admin_user_id, action, details) VALUES (%s, %s, %s);"
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, (admin_user_id, action, details))
        return True
    except Exception as e:
        logger.error("Fallo al registrar la acción del admin: %s", e)
        return False
