from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
from fastapi import Response

# --- Counters ---
REQUEST_COUNT = Counter(
    "gateway_requests_total",
    "Total requests received by API Gateway",
    ["method", "endpoint", "status"]
)

RATE_LIMITED_COUNT = Counter(
    "gateway_rate_limited_total",
    "Total requests rejected by rate limiter"
)

# --- Histograms ---
REQUEST_LATENCY = Histogram(
    "gateway_request_latency_seconds",
    "Request latency at the API Gateway",
    buckets=[0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0]
)

# --- Gauges ---
ACTIVE_REQUESTS = Gauge(
    "gateway_active_requests",
    "Currently active requests in the gateway"
)


def metrics_endpoint() -> Response:
    return Response(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST
    )
