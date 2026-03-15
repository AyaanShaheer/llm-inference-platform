import time
import asyncio
import structlog
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel
from typing import Optional

from inference_worker.model_manager import ModelManager
from inference_worker.batch_processor import BatchProcessor, InferenceJob
from inference_worker.metrics import (
    INFERENCE_REQUESTS, INFERENCE_LATENCY, TOKENS_GENERATED,
    QUEUE_DEPTH, GPU_MEMORY_ALLOCATED_GB, MODELS_LOADED,
    metrics_endpoint
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

model_manager = ModelManager()
batch_processor = BatchProcessor(model_manager)

# Queue-related globals (initialized via /queue/start)
_queue_manager = None
_queue_workers = []
_queue_monitor = None


async def metrics_updater():
    while True:
        await asyncio.sleep(5)
        gpu_info = model_manager.get_gpu_memory_info()
        if gpu_info.get("available"):
            GPU_MEMORY_ALLOCATED_GB.set(gpu_info["allocated_gb"])
        MODELS_LOADED.set(len(model_manager.list_loaded()))


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("worker_starting", port=settings.worker_port)
    loop = asyncio.get_event_loop()
    for model_type in ["small", "large"]:
        await loop.run_in_executor(None, model_manager.load_model, model_type)
    batch_processor.start()
    metrics_task = asyncio.create_task(metrics_updater())
    logger.info("worker_ready", models=model_manager.list_loaded())
    yield
    batch_processor.stop()
    metrics_task.cancel()
    if _queue_manager is not None:
        for qw in _queue_workers:
            qw.stop()
        _queue_monitor.stop()
        await _queue_manager.disconnect()
    logger.info("worker_shutdown")


app = FastAPI(
    title="LLM Inference Platform - Inference Worker",
    version="1.0.0",
    lifespan=lifespan
)


# ─────────────────────────────────────────
# Schemas
# ─────────────────────────────────────────
class InferRequest(BaseModel):
    request_id: str
    prompt: str
    max_tokens: int = 256
    temperature: float = 0.7
    model_type: str = "small"


class InferResponse(BaseModel):
    request_id: str
    model_used: str
    generated_text: str
    prompt_tokens: int
    completion_tokens: int
    latency_ms: float
    status: str = "success"


class AsyncInferRequest(BaseModel):
    request_id: str
    prompt: str
    max_tokens: int = 256
    temperature: float = 0.7
    model_type: str = "small"


# ─────────────────────────────────────────
# Core Routes
# ─────────────────────────────────────────
@app.get("/health")
async def health():
    gpu = model_manager.get_gpu_memory_info()
    return {
        "status": "ok",
        "service": "inference-worker",
        "models_loaded": model_manager.list_loaded(),
        "gpu": gpu,
        "batches_processed": batch_processor.total_batches,
        "total_requests": batch_processor.total_requests,
    }


@app.get("/metrics")
async def metrics():
    return metrics_endpoint()


@app.get("/models")
async def list_models():
    return {
        "loaded": model_manager.list_loaded(),
        "gpu": model_manager.get_gpu_memory_info(),
    }


@app.post("/models/{model_type}/reload")
async def reload_model(model_type: str):
    if model_type not in ["small", "large"]:
        raise HTTPException(status_code=400, detail="model_type must be 'small' or 'large'")
    loop = asyncio.get_event_loop()
    loaded = await loop.run_in_executor(None, model_manager.hot_reload, model_type)
    return {"status": "reloaded", "model": loaded.model_name, "device": loaded.device}


@app.post("/infer", response_model=InferResponse)
async def infer(req: InferRequest):
    if req.model_type not in ["small", "large"]:
        raise HTTPException(status_code=400, detail="model_type must be 'small' or 'large'")

    q_size = batch_processor._queues[req.model_type].qsize()
    QUEUE_DEPTH.labels(model_type=req.model_type).set(q_size)

    logger.info(
        "infer_request_received",
        request_id=req.request_id,
        model_type=req.model_type,
        prompt_length=len(req.prompt),
        max_tokens=req.max_tokens,
        queue_depth=q_size,
    )

    job = InferenceJob(
        request_id=req.request_id,
        prompt=req.prompt,
        max_tokens=req.max_tokens,
        temperature=req.temperature,
        model_type=req.model_type,
    )

    try:
        result = await asyncio.wait_for(batch_processor.submit(job), timeout=120.0)
        INFERENCE_REQUESTS.labels(model_type=req.model_type, status="success").inc()
        INFERENCE_LATENCY.labels(model_type=req.model_type).observe(result.latency_ms / 1000)
        TOKENS_GENERATED.labels(model_type=req.model_type).inc(result.completion_tokens)
        logger.info(
            "infer_complete",
            request_id=req.request_id,
            latency_ms=result.latency_ms,
            completion_tokens=result.completion_tokens,
        )
        return InferResponse(
            request_id=result.request_id,
            model_used=result.model_used,
            generated_text=result.generated_text,
            prompt_tokens=result.prompt_tokens,
            completion_tokens=result.completion_tokens,
            latency_ms=result.latency_ms,
            status="success",
        )
    except asyncio.TimeoutError:
        INFERENCE_REQUESTS.labels(model_type=req.model_type, status="timeout").inc()
        raise HTTPException(status_code=504, detail="Inference timed out")
    except Exception as e:
        INFERENCE_REQUESTS.labels(model_type=req.model_type, status="error").inc()
        raise HTTPException(status_code=500, detail=str(e))


# ─────────────────────────────────────────
# Redis Queue Routes
# ─────────────────────────────────────────
@app.post("/queue/start")
async def start_queue_workers():
    global _queue_manager, _queue_workers, _queue_monitor

    if _queue_manager is not None:
        return {"status": "already_running"}

    from request_queue.redis_queue import RedisQueueManager
    from inference_worker.queue_worker import QueueWorker
    from request_queue.monitor import QueueMonitor

    _queue_manager = RedisQueueManager()
    await _queue_manager.connect()

    for model_type in ["small", "large"]:
        qw = QueueWorker(_queue_manager, batch_processor, model_type, concurrency=2)
        qw.start()
        _queue_workers.append(qw)

    _queue_monitor = QueueMonitor(_queue_manager)
    _queue_monitor.start()

    return {"status": "queue_workers_started", "model_types": ["small", "large"]}


@app.get("/queue/status")
async def queue_status():
    if _queue_manager is None:
        return {"status": "not_started"}
    depths = await _queue_manager.get_queue_depths()
    healthy = await _queue_manager.health_check()
    return {"status": "running", "redis_healthy": healthy, "queue_depths": depths}


@app.post("/queue/flush")
async def flush_queues():
    if _queue_manager is None:
        raise HTTPException(status_code=400, detail="Queue not started")
    await _queue_manager.flush_queues()
    return {"status": "flushed"}


@app.post("/infer/async")
async def infer_async(req: AsyncInferRequest):
    if _queue_manager is None:
        raise HTTPException(status_code=503, detail="Queue not started. POST /queue/start first.")

    from request_queue.redis_queue import QueuedRequest
    queued = QueuedRequest(
        request_id=req.request_id,
        prompt=req.prompt,
        max_tokens=req.max_tokens,
        temperature=req.temperature,
        model_type=req.model_type,
    )
    try:
        queue_depth = await _queue_manager.enqueue(queued)
    except OverflowError as e:
        raise HTTPException(status_code=429, detail=str(e))

    return {
        "request_id": req.request_id,
        "status": "queued",
        "queue_depth": queue_depth,
        "poll_url": f"/result/{req.request_id}",
    }


@app.get("/result/{request_id}")
async def get_result(request_id: str, wait: bool = False):
    if _queue_manager is None:
        raise HTTPException(status_code=503, detail="Queue not started")
    if wait:
        result = await _queue_manager.poll_result(request_id, timeout_s=120.0)
    else:
        result = await _queue_manager.get_result(request_id)
    if result is None:
        return {"status": "pending", "request_id": request_id}
    return result.__dict__
