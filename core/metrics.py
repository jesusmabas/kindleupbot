import time
import asyncio
import logging
from collections import defaultdict
from datetime import datetime
from typing import Optional, Dict, Any, List, Tuple

from ..database import get_metrics_from_db, save_metric, get_total_users

logger = logging.getLogger(__name__)

class MetricsCollector:
    def __init__(self):
        self.start_time = time.time()
        self.metrics = defaultdict(int)
        self.user_metrics = defaultdict(lambda: defaultdict(int))
        self.error_log = []
        self.response_times = []
        self.daily_stats = defaultdict(lambda: defaultdict(int))
        self.lock = asyncio.Lock()

    def reset(self):
        logger.warning("Reiniciando el colector de métricas en memoria.")
        self.metrics.clear()
        self.user_metrics.clear()
        self.error_log.clear()
        self.response_times.clear()
        self.daily_stats.clear()

    def load_from_db(self):
        try:
            logger.info("Cargando métricas históricas desde la base de datos...")
            historical_metrics = get_metrics_from_db()
            for metric_name, user_id, value in historical_metrics:
                self.metrics[metric_name] += value
                if user_id:
                    self.user_metrics[user_id][metric_name] += value
            logger.info(f"{len(historical_metrics)} registros de métricas cargados.")
        except Exception as e:
            logger.error(f"Error cargando métricas: {e}")

    async def _save_metric_async(self, metric_name: str, user_id: Optional[int], value: int):
        try:
            await asyncio.to_thread(save_metric, metric_name, user_id, value)
        except Exception as e:
            logger.error(f"Error guardando métrica {metric_name}: {e}")

    async def increment(self, metric_name: str, user_id: Optional[int] = None, value: int = 1):
        async with self.lock:
            self.metrics[metric_name] += value
            if user_id:
                self.user_metrics[user_id][metric_name] += value
            today = datetime.now().strftime('%Y-%m-%d')
            self.daily_stats[today][metric_name] += value
            asyncio.create_task(self._save_metric_async(metric_name, user_id, value))

    def log_error(self, error_type: str, error_message: str, user_id: Optional[int] = None):
        error_data = {
            'timestamp': datetime.now().isoformat(),
            'type': error_type, 'message': error_message, 'user_id': user_id,
            'traceback': None
        }
        self.error_log.insert(0, error_data)
        if len(self.error_log) > 100: self.error_log.pop()
        asyncio.create_task(self.increment('errors_total'))
        asyncio.create_task(self.increment(f'error_{error_type}', user_id))

    def log_response_time(self, duration: float, operation: str):
        self.response_times.insert(0, {'duration': duration, 'operation': operation, 'timestamp': time.time()})
        if len(self.response_times) > 1000: self.response_times.pop()

    async def get_summary(self) -> Dict[str, Any]:
        uptime = time.time() - self.start_time
        avg_response_time = sum(r['duration'] for r in self.response_times) / len(self.response_times) if self.response_times else 0
        total_users_count = await asyncio.to_thread(get_total_users)
        return {
            'uptime_formatted': self._format_uptime(uptime),
            'uptime_seconds': uptime,
            'total_users': total_users_count,
            'total_documents_sent': self.metrics.get('document_sent', 0),
            'total_documents_received': self.metrics.get('document_received', 0),
            'total_errors': self.metrics.get('errors_total', 0),
            'commands_executed': self.metrics.get('commands_total', 0),
            'avg_response_time_ms': round(avg_response_time * 1000, 2),
            'recent_errors': self.error_log[:10],
            'top_formats': self._get_top_formats(),
            'daily_stats': dict(self.daily_stats),
            'success_rate': self._calculate_success_rate(),
        }

    def _calculate_success_rate(self) -> float:
        sent = self.metrics.get('document_sent', 0)
        received = self.metrics.get('document_received', 0)
        return 100.0 if received == 0 else round((sent / received) * 100, 2)

    def get_user_stats(self, user_id: int) -> Dict[str, Any]:
        user_data = self.user_metrics.get(user_id, {})
        return {
            'documents_sent': user_data.get('document_sent', 0),
            'documents_received': user_data.get('document_received', 0),
            'commands_used': user_data.get('commands_total', 0),
            'errors_encountered': sum(v for k, v in user_data.items() if k.startswith('error_')),
            'top_format': self._get_user_top_format(user_id),
            'success_rate': self._calculate_user_success_rate(user_id),
            'last_activity': "N/A", # Este campo podría implementarse en el futuro
        }

    def _calculate_user_success_rate(self, user_id: int) -> float:
        user_data = self.user_metrics.get(user_id, {})
        sent = user_data.get('document_sent', 0)
        received = user_data.get('document_received', 0)
        return 100.0 if received == 0 else round((sent / received) * 100, 2)

    def _format_uptime(self, seconds: float) -> str:
        days, rem = divmod(seconds, 86400)
        hours, rem = divmod(rem, 3600)
        minutes, _ = divmod(rem, 60)
        return f"{int(days)}d {int(hours)}h {int(minutes)}m"

    def _get_top_formats(self) -> List[Tuple[str, int]]:
        formats = {k.replace('format_', ''): v for k, v in self.metrics.items() if k.startswith('format_')}
        return sorted(formats.items(), key=lambda x: x[1], reverse=True)[:10]

    def _get_user_top_format(self, user_id: int) -> str:
        user_data = self.user_metrics.get(user_id, {})
        formats = {k.replace('format_', ''): v for k, v in user_data.items() if k.startswith('format_')}
        return max(formats.items(), key=lambda x: x[1])[0] if formats else "Ninguno"

# Instancia global única que será utilizada por toda la aplicación.
# Cualquier módulo puede importarla con: from kindleupbot.core.metrics import metrics_collector
metrics_collector = MetricsCollector()