import time
import uuid
import httpx
import structlog
import asyncio
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, HTTPException, status
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from api_gateway.schemas import InferenceRequest, InferenceResponse, HealthResponse, ErrorResponse
from api_gateway.rate_limiter import RateLimiter
from api_gateway.metrics import (
    REQUEST_COUNT, REQUEST_LATENCY, ACTIVE_REQUESTS,
    RATE_LIMITED_COUNT, metrics_endpoint
)
from config import settings

# --- Logger ---
structlog.configure(
    processors=[
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.add_log_level,
        structlog.processors.JSONRenderer()
    ]
)
logger = structlog.get_logger()

# --- Rate Limiter ---
rate_limiter = RateLimiter(capacity=10.0, refill_rate=5.0)

# --- Cleanup task ---
async def cleanup_task():
    while True:
        await asyncio.sleep(300)
        await rate_limiter.cleanup()
        logger.info("rate_limiter_cleanup_done")

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("api_gateway_starting", port=settings.app_port)
    task = asyncio.create_task(cleanup_task())
    yield
    task.cancel()
    logger.info("api_gateway_shutdown")

# --- App ---
app = FastAPI(
    title="LLM Inference Platform - API Gateway",
    description="Entry point for the Distributed LLM Inference Platform",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

ROUTER_URL = f"http://{settings.router_host}:{settings.router_port}"


# ─────────────────────────────────────────
# Middleware: request timing + logging
# ─────────────────────────────────────────
@app.middleware("http")
async def request_middleware(request: Request, call_next):
    request_id = str(uuid.uuid4())
    request.state.request_id = request_id
    start = time.perf_counter()
    ACTIVE_REQUESTS.inc()

    try:
        response = await call_next(request)
        latency = (time.perf_counter() - start) * 1000
        REQUEST_LATENCY.observe(latency / 1000)
        REQUEST_COUNT.labels(
            method=request.method,
            endpoint=request.url.path,
            status=response.status_code
        ).inc()
        logger.info(
            "request_completed",
            request_id=request_id,
            method=request.method,
            path=request.url.path,
            status=response.status_code,
            latency_ms=round(latency, 2)
        )
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Latency-Ms"] = str(round(latency, 2))
        return response
    finally:
        ACTIVE_REQUESTS.dec()


# ─────────────────────────────────────────
# Routes
# ─────────────────────────────────────────
@app.get("/health", response_model=HealthResponse)
async def health_check():
    router_reachable = False
    try:
        async with httpx.AsyncClient(timeout=2.0) as client:
            r = await client.get(f"{ROUTER_URL}/health")
            router_reachable = r.status_code == 200
    except Exception:
        pass

    return HealthResponse(
        status="ok",
        service="api-gateway",
        router_reachable=router_reachable
    )


@app.get("/metrics")
async def metrics():
    return metrics_endpoint()


@app.post("/v1/inference", response_model=InferenceResponse)
async def inference(request: Request, body: InferenceRequest):
    # Rate limiting by client IP
    client_ip = request.client.host or "unknown"
    allowed = await rate_limiter.is_allowed(client_ip)

    if not allowed:
        RATE_LIMITED_COUNT.inc()
        logger.warning("rate_limit_exceeded", client_ip=client_ip)
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Rate limit exceeded. Please slow down."
        )

    # Inject request_id from middleware
    body.request_id = getattr(request.state, "request_id", body.request_id)

    logger.info(
        "inference_request",
        request_id=body.request_id,
        model_preference=body.model_preference,
        prompt_length=len(body.prompt),
        max_tokens=body.max_tokens
    )

    # Forward to router
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                f"{ROUTER_URL}/route",
                json=body.model_dump()
            )
            if response.status_code != 200:
                raise HTTPException(
                    status_code=response.status_code,
                    detail=f"Router error: {response.text}"
                )
            return InferenceResponse(**response.json())

    except httpx.ConnectError:
        logger.error("router_unreachable", request_id=body.request_id)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Router service is unreachable"
        )
    except httpx.TimeoutException:
        logger.error("router_timeout", request_id=body.request_id)
        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            detail="Router timed out"
        )


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    request_id = getattr(request.state, "request_id", str(uuid.uuid4()))
    logger.error("unhandled_exception", request_id=request_id, error=str(exc))
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(request_id=request_id, error=str(exc)).model_dump()
    )
