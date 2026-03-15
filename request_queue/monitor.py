import asyncio
import structlog
from prometheus_client import Gauge
from request_queue.redis_queue import RedisQueueManager

logger = structlog.get_logger()

REDIS_QUEUE_DEPTH = Gauge(
    "redis_queue_depth",
    "Current depth of Redis request queues",
    ["model_type"]
)

REDIS_QUEUE_WAIT_MS = Gauge(
    "redis_queue_wait_ms_estimate",
    "Estimated wait time in queue based on depth and throughput",
    ["model_type"]
)


class QueueMonitor:
    """
    Background task that polls Redis queue depths
    and exports them to Prometheus.
    """
    def __init__(self, queue_manager: RedisQueueManager, poll_interval: float = 2.0):
        self._queue = queue_manager
        self._interval = poll_interval
        self._task: asyncio.Task | None = None

    def start(self):
        self._task = asyncio.create_task(self._monitor_loop(), name="queue_monitor")
        logger.info("queue_monitor_started")

    def stop(self):
        if self._task:
            self._task.cancel()

    async def _monitor_loop(self):
        while True:
            try:
                depths = await self._queue.get_queue_depths()
                for model_type, depth in depths.items():
                    REDIS_QUEUE_DEPTH.labels(model_type=model_type).set(depth)
                    if depth > 0:
                        logger.info(
                            "queue_depth",
                            model_type=model_type,
                            depth=depth
                        )
            except Exception as e:
                logger.warning("queue_monitor_error", error=str(e))
            await asyncio.sleep(self._interval)
