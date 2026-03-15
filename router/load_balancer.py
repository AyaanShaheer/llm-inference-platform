import asyncio
import time
from dataclasses import dataclass, field
from typing import Optional
from enum import Enum


class BalancingStrategy(str, Enum):
    ROUND_ROBIN = "round_robin"
    LEAST_CONNECTIONS = "least_connections"


@dataclass
class Worker:
    worker_id: str
    host: str
    port: int
    model_type: str          # "small" or "large"
    active_connections: int = 0
    total_requests: int = 0
    total_latency_ms: float = 0.0
    is_healthy: bool = True
    last_health_check: float = field(default_factory=time.monotonic)

    @property
    def url(self) -> str:
        return f"http://{self.host}:{self.port}"

    @property
    def avg_latency_ms(self) -> float:
        if self.total_requests == 0:
            return 0.0
        return self.total_latency_ms / self.total_requests

    def record_request(self, latency_ms: float):
        self.total_requests += 1
        self.total_latency_ms += latency_ms


class LoadBalancer:
    """
    Supports two strategies:
      - Round Robin: cycle through healthy workers evenly
      - Least Connections: always pick worker with fewest active connections
    """
    def __init__(self, strategy: BalancingStrategy = BalancingStrategy.ROUND_ROBIN):
        self.strategy = strategy
        self._workers: dict[str, list[Worker]] = {
            "small": [],
            "large": []
        }
        self._rr_index: dict[str, int] = {"small": 0, "large": 0}
        self._lock = asyncio.Lock()

    async def register_worker(self, worker: Worker):
        async with self._lock:
            model_type = worker.model_type
            if model_type not in self._workers:
                self._workers[model_type] = []
            # Avoid duplicate registration
            existing_ids = [w.worker_id for w in self._workers[model_type]]
            if worker.worker_id not in existing_ids:
                self._workers[model_type].append(worker)

    async def deregister_worker(self, worker_id: str, model_type: str):
        async with self._lock:
            self._workers[model_type] = [
                w for w in self._workers[model_type]
                if w.worker_id != worker_id
            ]

    async def get_worker(self, model_type: str) -> Optional[Worker]:
        async with self._lock:
            healthy = [
                w for w in self._workers.get(model_type, [])
                if w.is_healthy
            ]
            if not healthy:
                return None

            if self.strategy == BalancingStrategy.ROUND_ROBIN:
                idx = self._rr_index.get(model_type, 0) % len(healthy)
                self._rr_index[model_type] = idx + 1
                return healthy[idx]

            elif self.strategy == BalancingStrategy.LEAST_CONNECTIONS:
                return min(healthy, key=lambda w: w.active_connections)

        return None

    async def get_all_workers(self) -> dict:
        async with self._lock:
            return {
                model_type: [
                    {
                        "worker_id": w.worker_id,
                        "url": w.url,
                        "model_type": w.model_type,
                        "active_connections": w.active_connections,
                        "total_requests": w.total_requests,
                        "avg_latency_ms": round(w.avg_latency_ms, 2),
                        "is_healthy": w.is_healthy,
                    }
                    for w in workers
                ]
                for model_type, workers in self._workers.items()
            }

    async def mark_unhealthy(self, worker_id: str, model_type: str):
        async with self._lock:
            for w in self._workers.get(model_type, []):
                if w.worker_id == worker_id:
                    w.is_healthy = False
                    w.last_health_check = time.monotonic()

    async def mark_healthy(self, worker_id: str, model_type: str):
        async with self._lock:
            for w in self._workers.get(model_type, []):
                if w.worker_id == worker_id:
                    w.is_healthy = True
                    w.last_health_check = time.monotonic()
