import json
import time
import asyncio
import redis.asyncio as aioredis
import structlog
from dataclasses import dataclass, field, asdict
from typing import Optional
from config import settings

logger = structlog.get_logger()

QUEUE_KEYS = {
    "small": "llm:queue:small",
    "large": "llm:queue:large",
}
RESULT_PREFIX = "llm:result:"
RESULT_TTL_S = 300      # Results expire after 5 minutes
MAX_QUEUE_SIZE = 500    # Hard cap per queue


@dataclass
class QueuedRequest:
    request_id: str
    prompt: str
    max_tokens: int
    temperature: float
    model_type: str
    enqueue_time: float = field(default_factory=time.time)
    priority: int = 0       # 0 = normal, 1 = high (reserved for future)

    def to_json(self) -> str:
        return json.dumps(asdict(self))

    @classmethod
    def from_json(cls, data: str) -> "QueuedRequest":
        return cls(**json.loads(data))


@dataclass
class QueueResult:
    request_id: str
    status: str             # "success" | "error" | "timeout"
    generated_text: str = ""
    model_used: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_ms: float = 0.0
    error: str = ""

    def to_json(self) -> str:
        return json.dumps(asdict(self))

    @classmethod
    def from_json(cls, data: str) -> "QueueResult":
        return cls(**json.loads(data))


class RedisQueueManager:
    """
    Async Redis-based distributed request queue.

    Enqueue: LPUSH to queue key
    Dequeue: BRPOP from queue key (blocking, atomic)
    Result:  SET with TTL
    """

    def __init__(self):
        self._redis: Optional[aioredis.Redis] = None

    async def connect(self):
        self._redis = await aioredis.from_url(
            f"redis://{settings.redis_host}:{settings.redis_port}/{settings.redis_db}",
            encoding="utf-8",
            decode_responses=True,
            max_connections=20,
        )
        await self._redis.ping()
        logger.info("redis_connected", host=settings.redis_host, port=settings.redis_port)

    async def disconnect(self):
        if self._redis:
            await self._redis.aclose()
            logger.info("redis_disconnected")

    async def enqueue(self, req: QueuedRequest) -> int:
        """Push request to the tail. Returns current queue length."""
        queue_key = QUEUE_KEYS.get(req.model_type)
        if not queue_key:
            raise ValueError(f"No queue for model_type: {req.model_type}")

        # Enforce max queue size
        current_len = await self._redis.llen(queue_key)
        if current_len >= MAX_QUEUE_SIZE:
            raise OverflowError(
                f"Queue '{req.model_type}' is full ({current_len}/{MAX_QUEUE_SIZE})"
            )

        await self._redis.lpush(queue_key, req.to_json())
        new_len = await self._redis.llen(queue_key)

        logger.info(
            "request_enqueued",
            request_id=req.request_id,
            model_type=req.model_type,
            queue_depth=new_len,
        )
        return new_len

    async def dequeue(self, model_type: str, timeout: int = 5) -> Optional[QueuedRequest]:
        """
        Blocking pop from request_queue. Returns None on timeout.
        BRPOP pops from the right (FIFO with LPUSH).
        """
        queue_key = QUEUE_KEYS.get(model_type)
        if not queue_key:
            raise ValueError(f"No queue for model_type: {model_type}")

        result = await self._redis.brpop(queue_key, timeout=timeout)
        if result is None:
            return None

        _, data = result
        req = QueuedRequest.from_json(data)

        wait_ms = round((time.time() - req.enqueue_time) * 1000, 2)
        logger.info(
            "request_dequeued",
            request_id=req.request_id,
            model_type=model_type,
            wait_ms=wait_ms,
        )
        return req

    async def store_result(self, result: QueueResult):
        """Store inference result so the caller can poll for it."""
        key = f"{RESULT_PREFIX}{result.request_id}"
        await self._redis.setex(key, RESULT_TTL_S, result.to_json())

    async def get_result(self, request_id: str) -> Optional[QueueResult]:
        """Poll for a result."""
        key = f"{RESULT_PREFIX}{request_id}"
        data = await self._redis.get(key)
        if data is None:
            return None
        return QueueResult.from_json(data)

    async def poll_result(
        self,
        request_id: str,
        timeout_s: float = 120.0,
        interval_s: float = 0.1
    ) -> Optional[QueueResult]:
        """Poll with timeout until result is ready."""
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            result = await self.get_result(request_id)
            if result is not None:
                return result
            await asyncio.sleep(interval_s)
        return None

    async def get_queue_depths(self) -> dict:
        depths = {}
        for model_type, key in QUEUE_KEYS.items():
            depths[model_type] = await self._redis.llen(key)
        return depths

    async def flush_queues(self):
        """Clear all queues — useful for testing."""
        for key in QUEUE_KEYS.values():
            await self._redis.delete(key)
        logger.warning("queues_flushed")

    async def health_check(self) -> bool:
        try:
            await self._redis.ping()
            return True
        except Exception:
            return False
