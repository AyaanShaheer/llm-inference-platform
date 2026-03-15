from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
from fastapi import Response

INFERENCE_REQUESTS = Counter(
    "worker_inference_requests_total",
    "Total inference requests processed",
    ["model_type", "status"]
)

INFERENCE_LATENCY = Histogram(
    "worker_inference_latency_seconds",
    "End-to-end inference latency (queue + model)",
    ["model_type"],
    buckets=[0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0]
)

BATCH_SIZE = Histogram(
    "worker_batch_size",
    "Requests per batch",
    buckets=[1, 2, 3, 4, 5, 6, 7, 8]
)

TOKENS_GENERATED = Counter(
    "worker_tokens_generated_total",
    "Total tokens generated",
    ["model_type"]
)

QUEUE_DEPTH = Gauge(
    "worker_queue_depth",
    "Current requests waiting in queue",
    ["model_type"]
)

GPU_MEMORY_ALLOCATED_GB = Gauge(
    "worker_gpu_memory_allocated_gb",
    "GPU memory currently allocated (GB)"
)

MODELS_LOADED = Gauge(
    "worker_models_loaded",
    "Number of models currently loaded in memory"
)


def metrics_endpoint() -> Response:
    return Response(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST
    )
