import time
import asyncio
from collections import defaultdict
from dataclasses import dataclass, field


@dataclass
class TokenBucket:
    capacity: float
    refill_rate: float
    tokens: float = field(init=False)
    last_refill: float = field(init=False)

    def __post_init__(self):
        self.tokens = self.capacity
        self.last_refill = time.monotonic()

    def _refill(self):
        now = time.monotonic()
        elapsed = now - self.last_refill
        added = elapsed * self.refill_rate
        self.tokens = min(self.capacity, self.tokens + added)
        self.last_refill = now

    def consume(self, tokens: float = 1.0) -> bool:
        self._refill()
        if self.tokens >= tokens:
            self.tokens -= tokens
            return True
        return False


class RateLimiter:
    def __init__(self, capacity: float = 10.0, refill_rate: float = 5.0):
        self.capacity = capacity
        self.refill_rate = refill_rate
        self._buckets: dict[str, TokenBucket] = defaultdict(
            lambda: TokenBucket(capacity=self.capacity, refill_rate=self.refill_rate)
        )
        self._lock = asyncio.Lock()

    async def is_allowed(self, client_id: str) -> bool:
        async with self._lock:
            bucket = self._buckets[client_id]  # defaultdict auto-creates
            return bucket.consume()

    async def cleanup(self):
        """Remove stale buckets in-place — preserves defaultdict type."""
        async with self._lock:
            cutoff = time.monotonic() - 300  # 5 min inactive
            stale_keys = [
                k for k, v in self._buckets.items()
                if v.last_refill < cutoff
            ]
            for k in stale_keys:
                del self._buckets[k]
