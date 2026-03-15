import time
import asyncio
import httpx
import structlog
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional

from router.load_balancer import LoadBalancer, Worker, BalancingStrategy
from router.routing_logic import classify_prompt, explain_routing_decision
from router.metrics import (
    ROUTED_REQUESTS, ROUTING_LATENCY, NO_WORKER_AVAILABLE,
    ACTIVE_WORKERS, WORKER_CONNECTIONS, metrics_endpoint
)
from config import settings

structlog.configure(
    processors=[
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.add_log_level,
        structlog.processors.JSONRenderer()
    ]
)
logger = structlog.get_logger()

# ─────────────────────────────────────────
# Load Balancer (singleton)
# ─────────────────────────────────────────
load_balancer = LoadBalancer(strategy=BalancingStrategy.LEAST_CONNECTIONS)


async def health_check_loop():
    """Background task: ping all workers every 10s, mark healthy/unhealthy."""
    while True:
        await asyncio.sleep(10)
        all_workers = await load_balancer.get_all_workers()
        for model_type, workers in all_workers.items():
            for w_info in workers:
                url = f"http://{w_info['url'].replace('http://', '')}"
                try:
                    async with httpx.AsyncClient(timeout=2.0) as client:
                        r = await client.get(f"{url}/health")
                        if r.status_code == 200:
                            await load_balancer.mark_healthy(w_info["worker_id"], model_type)
                        else:
                            await load_balancer.mark_unhealthy(w_info["worker_id"], model_type)
                except Exception:
                    await load_balancer.mark_unhealthy(w_info["worker_id"], model_type)
                    logger.warning("worker_health_check_failed", worker_id=w_info["worker_id"])


async def register_default_workers():
    """Register the default inference worker on startup."""
    worker = Worker(
        worker_id="worker-1",
        host=settings.worker_host if settings.worker_host != "0.0.0.0" else "localhost",
        port=settings.worker_port,
        model_type="small",
    )
    await load_balancer.register_worker(worker)

    # Register same worker for large (for now — will split in later steps)
    large_worker = Worker(
        worker_id="worker-1-large",
        host=settings.worker_host if settings.worker_host != "0.0.0.0" else "localhost",
        port=settings.worker_port,
        model_type="large",
    )
    await load_balancer.register_worker(large_worker)
    logger.info("default_workers_registered")


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("router_starting", port=settings.router_port)
    await register_default_workers()
    task = asyncio.create_task(health_check_loop())
    yield
    task.cancel()
    logger.info("router_shutdown")


app = FastAPI(
    title="LLM Inference Platform - Model Router",
    version="1.0.0",
    lifespan=lifespan
)


# ─────────────────────────────────────────
# Schemas (local, lightweight)
# ─────────────────────────────────────────
class RouteRequest(BaseModel):
    request_id: str
    prompt: str
    model_preference: Optional[str] = "auto"
    max_tokens: int = 256
    temperature: float = 0.7
    stream: bool = False


class WorkerRegisterRequest(BaseModel):
    worker_id: str
    host: str
    port: int
    model_type: str  # "small" or "large"


# ─────────────────────────────────────────
# Routes
# ─────────────────────────────────────────
@app.get("/health")
async def health():
    workers = await load_balancer.get_all_workers()
    total_healthy = sum(
        1 for wlist in workers.values()
        for w in wlist if w["is_healthy"]
    )
    return {
        "status": "ok",
        "service": "router",
        "healthy_workers": total_healthy,
        "strategy": load_balancer.strategy.value
    }


@app.get("/metrics")
async def metrics():
    return metrics_endpoint()


@app.get("/workers")
async def list_workers():
    return await load_balancer.get_all_workers()


@app.post("/workers/register")
async def register_worker(req: WorkerRegisterRequest):
    worker = Worker(
        worker_id=req.worker_id,
        host=req.host,
        port=req.port,
        model_type=req.model_type
    )
    await load_balancer.register_worker(worker)
    logger.info("worker_registered", worker_id=req.worker_id, model_type=req.model_type)
    return {"status": "registered", "worker_id": req.worker_id}


@app.post("/workers/{worker_id}/deregister")
async def deregister_worker(worker_id: str, model_type: str):
    await load_balancer.deregister_worker(worker_id, model_type)
    return {"status": "deregistered", "worker_id": worker_id}


@app.post("/route")
async def route_request(req: RouteRequest):
    start = time.perf_counter()

    # 1. Decide which model type
    model_type = classify_prompt(req.prompt, req.model_preference, req.max_tokens)

    # 2. Pick a worker
    worker = await load_balancer.get_worker(model_type)

    if not worker:
        NO_WORKER_AVAILABLE.labels(model_type=model_type).inc()
        logger.error(
            "no_worker_available",
            request_id=req.request_id,
            model_type=model_type
        )
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"No healthy worker available for model type: {model_type}"
        )

    # 3. Track connection
    worker.active_connections += 1
    WORKER_CONNECTIONS.labels(worker_id=worker.worker_id).set(worker.active_connections)

    logger.info(
        "routing_decision",
        request_id=req.request_id,
        model_type=model_type,
        worker_id=worker.worker_id,
        strategy=load_balancer.strategy.value
    )

    try:
        # 4. Forward to inference worker
        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(
                f"{worker.url}/infer",
                json={
                    "request_id": req.request_id,
                    "prompt": req.prompt,
                    "max_tokens": req.max_tokens,
                    "temperature": req.temperature,
                    "model_type": model_type
                }
            )

        elapsed_ms = (time.perf_counter() - start) * 1000
        worker.record_request(elapsed_ms)

        ROUTED_REQUESTS.labels(
            model_type=model_type,
            strategy=load_balancer.strategy.value
        ).inc()
        ROUTING_LATENCY.observe(elapsed_ms / 1000)

        if response.status_code != 200:
            await load_balancer.mark_unhealthy(worker.worker_id, model_type)
            raise HTTPException(
                status_code=response.status_code,
                detail=f"Worker error: {response.text}"
            )

        logger.info(
            "request_forwarded",
            request_id=req.request_id,
            worker_id=worker.worker_id,
            latency_ms=round(elapsed_ms, 2)
        )
        return response.json()

    except httpx.ConnectError:
        await load_balancer.mark_unhealthy(worker.worker_id, model_type)
        logger.error("worker_unreachable", worker_id=worker.worker_id)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Worker {worker.worker_id} is unreachable"
        )
    except httpx.TimeoutException:
        logger.error("worker_timeout", worker_id=worker.worker_id)
        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            detail=f"Worker {worker.worker_id} timed out"
        )
    finally:
        worker.active_connections = max(0, worker.active_connections - 1)
        WORKER_CONNECTIONS.labels(worker_id=worker.worker_id).set(worker.active_connections)


@app.get("/debug/routing")
async def debug_routing(prompt: str, model_preference: str = "auto", max_tokens: int = 256):
    """Debug endpoint: see routing decision without making a real request."""
    return explain_routing_decision(prompt, model_preference, max_tokens)
