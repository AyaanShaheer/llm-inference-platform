from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
from fastapi import Response

ROUTED_REQUESTS = Counter(
    "router_routed_requests_total",
    "Total requests routed",
    ["model_type", "strategy"]
)

ROUTING_LATENCY = Histogram(
    "router_routing_latency_seconds",
    "Time to make routing decision + forward",
    buckets=[0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0]
)

NO_WORKER_AVAILABLE = Counter(
    "router_no_worker_available_total",
    "Times no healthy worker was available",
    ["model_type"]
)

ACTIVE_WORKERS = Gauge(
    "router_active_workers",
    "Number of healthy workers",
    ["model_type"]
)

WORKER_CONNECTIONS = Gauge(
    "router_worker_active_connections",
    "Active connections per worker",
    ["worker_id"]
)


def metrics_endpoint() -> Response:
    return Response(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST
    )
