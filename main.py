# main.py
import logging
from .bot import KindleEmailBot
from .config import settings
from .database import setup_database
from .core.metrics import metrics_collector

# Configurar logging inicial
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main() -> None:
    """Función principal para configurar e iniciar el bot."""
    logger.info("🚀 Iniciando bot...")
    
    try:
        # 1. Configurar la base de datos
        setup_database()
        logger.info("✅ Base de datos configurada.")

        # 2. Cargar métricas históricas
        metrics_collector.load_from_db()
        logger.info("✅ Métricas cargadas.")

        # 3. Crear una instancia del bot
        bot = KindleEmailBot(settings)
        
        # 4. Iniciar el bot en modo polling (esta llamada es bloqueante)
        bot.run()

    except Exception as e:
        logger.critical(f"CRITICAL: El bot ha fallado fatalmente: {e}", exc_info=True)
        raise

if __name__ == "__main__":
    main()