
# Distributed LLM Inference & Model Delivery Platform

A production-style distributed system for serving optimized LLMs with:
- Model routing (small vs large)
- Continuous batching
- Redis-based distributed queue
- Quantized model serving (FP16, GPTQ, AWQ)
- Prometheus + Grafana monitoring
- Kubernetes deployment

## Architecture
Client → API Gateway → Router → Redis Queue → Inference Workers → Response

## Services
| Service | Port | Description |
|---|---|---|
| API Gateway | 8000 | Request validation + entry point |
| Router | 8001 | Model selection + load balancing |
| Inference Worker | 8002 | Model inference engine |
| Metrics | 8003 | Prometheus metrics exporter |

## Stack
Python, FastAPI, Redis, Docker, Kubernetes, Prometheus, Grafana
