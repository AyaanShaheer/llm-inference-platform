# Distributed LLM Inference & Model Delivery Platform

A production-grade distributed system for serving optimized LLMs with model 
routing, continuous batching, Redis queuing, Prometheus monitoring, and 
full Kubernetes deployment.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         CLIENT                                   │
└─────────────────────────┬───────────────────────────────────────┘
                          │ HTTP
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│                    API GATEWAY :8000                             │
│  -  Token Bucket Rate Limiting (10 req/s per client)             │
│  -  Request ID injection + structured logging                    │
│  -  Prometheus metrics exposure                                  │
│  -  Request forwarding to Router                                 │
└─────────────────────────┬───────────────────────────────────────┘
                          │ HTTP
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│                   MODEL ROUTER :8001                             │
│  -  Dynamic routing heuristic (prompt complexity scoring)        │
│  -  Load balancing: Least Connections + Round Robin              │
│  -  Worker health checks every 10s                               │
│  -  Routes to: small (GPT-2 124M) or large (GPT-2-medium 354M)  │
└──────────┬──────────────────────────────┬───────────────────────┘
           │ Direct (sync)                │ Via Redis (async)
           ▼                              ▼
┌───────────────────────┐   ┌─────────────────────────────────────┐
│   INFERENCE WORKER    │   │         REDIS QUEUE :6379            │
│        :8002          │   │  -  LPUSH / BRPOP (FIFO)             │
│                       │◄──│  -  Max 500 req/queue                │
│  -  Model Manager      │   │  -  Result TTL: 5 minutes            │
│    (lazy load + hot   │   │  -  Burst absorption                 │
│     reload)           │   └─────────────────────────────────────┘
│  -  Continuous         │
│    Batching Engine    │
│    (max batch=8,      │
│     wait=50ms)        │
│  -  GPU Inference      │
│    (CUDA FP16/FP32)   │
└──────────┬────────────┘
           │ Metrics
           ▼
┌─────────────────────────────────────────────────────────────────┐
│              MONITORING STACK                                    │
│  Prometheus :9090  ──scrapes──►  All 3 services (/metrics)     │
│  Grafana :3000     ──reads───►   Prometheus                     │
│                                                                  │
│  Dashboards: Latency p50/p95/p99, Throughput, GPU Memory,      │
│              Queue Depth, Rate Limits, Tokens Generated         │
└─────────────────────────────────────────────────────────────────┘
```

---

## Services

| Service           | Port | Description                                    |
|-------------------|------|------------------------------------------------|
| API Gateway       | 8000 | Entry point, rate limiting, request routing    |
| Model Router      | 8001 | Model selection, load balancing                |
| Inference Worker  | 8002 | GPU inference, continuous batching             |
| Redis             | 6379 | Distributed request queue                     |
| Prometheus        | 9090 | Metrics collection                             |
| Grafana           | 3000 | Monitoring dashboards                          |

---

## Quick Start

### Local Development
```bash
# 1. Setup
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
sudo service redis-server start

# 2. Start services (3 terminals)
python -m api_gateway.run          # Terminal 1
python -m router.run               # Terminal 2
python -m inference_worker.run     # Terminal 3 (loads GPT-2 on startup)

# 3. Test
curl -X POST http://localhost:8000/v1/inference \
  -H "Content-Type: application/json" \
  -d '{"prompt": "The future of AI is", "max_tokens": 100}'
```

### Docker Compose (Full Stack)
```bash
docker compose up -d
# Wait ~2 min for model loading, then:
curl http://localhost:8000/health
```

### Monitoring
```
Grafana:    http://localhost:3000  (admin / admin123)
Prometheus: http://localhost:9090
```

---

## API Reference

### POST /v1/inference
```json
{
  "prompt": "string (required, 1-4096 chars)",
  "model_preference": "auto | small | large  (default: auto)",
  "max_tokens": "integer 1-2048 (default: 256)",
  "temperature": "float 0.0-2.0 (default: 0.7)",
  "stream": "bool (default: false)"
}
```

**Response:**
```json
{
  "request_id": "uuid",
  "model_used": "gpt2 | gpt2-medium",
  "generated_text": "string",
  "prompt_tokens": 5,
  "completion_tokens": 100,
  "latency_ms": 1027.71,
  "status": "success"
}
```

### POST /v1/inference (Async via Redis)
```bash
# Enqueue
curl -X POST http://localhost:8002/infer/async \
  -d '{"request_id": "req-001", "prompt": "...", "model_type": "small"}'

# Poll result (long-poll up to 120s)
curl "http://localhost:8002/result/req-001?wait=true"
```

---

## Key Design Decisions & Tradeoffs

### 1. Continuous Batching over Static Batching
**Decision:** Wait up to 50ms or until 8 requests arrive before running inference.  
**Why:** Static batching (fix batch size=N, wait forever) wastes GPU when 
traffic is low. Continuous batching adapts — at low load it processes 
immediately (batch=1), at high load it groups efficiently.  
**Tradeoff:** Adds up to 50ms queue wait per request. Acceptable for throughput 
gains (7.2x measured improvement).

### 2. Dual Queue Architecture (Sync + Redis)
**Decision:** Keep direct `/infer` (sync, low-latency) AND `/infer/async` (Redis-backed).  
**Why:** Not all use cases need a queue. Interactive chat → sync. 
Batch jobs, burst traffic → async Redis queue.  
**Tradeoff:** Two code paths to maintain. Redis adds 1-5ms overhead.

### 3. Least-Connections Load Balancing
**Decision:** Use Least Connections over Round Robin as default.  
**Why:** Inference requests have wildly different execution times (50-token 
request: 546ms, 400-token: 4085ms). Round Robin would pile long requests 
onto a worker that's already busy. LC routes to least-busy worker.  
**Tradeoff:** Requires tracking active connections per worker (state), 
Round Robin is stateless. At very high worker counts, LC has O(N) selection cost.

### 4. Token Bucket Rate Limiting
**Decision:** 10 req/s capacity, 5 req/s refill per client IP.  
**Why:** Protects inference workers from being overwhelmed. A single client 
cannot monopolize GPU compute.  
**Tradeoff:** Legitimate burst users (e.g., batch testing) get 429 errors. 
Could add per-API-key limits with higher caps in production.

### 5. Model Registry with Hot Reload
**Decision:** Models stay loaded in VRAM; support hot reload without restart.  
**Why:** Model loading takes 2-5 minutes. Restarting a worker for a model 
update is unacceptable in production. Hot reload swaps the model in-place.  
**Tradeoff:** During hot reload, ~2-5 min window where that model type 
serves from remaining workers only.

### 6. FP32 over FP16 on RTX 3050 Laptop
**Decision:** Default to FP32 on consumer Laptop GPUs.  
**Why:** Benchmarks show FP16 is 25% SLOWER on RTX 3050 Laptop due to 
limited tensor core throughput. FP16 benefits require A100/H100-class hardware.  
**Tradeoff:** 2x higher VRAM usage. On 4GB VRAM, limits max batch size.

---

## Benchmark Results

Full report: [`experiments/EXPERIMENT_REPORT.md`](experiments/EXPERIMENT_REPORT.md)

| Optimization        | Improvement |
|---------------------|-------------|
| Batching 1→8        | **7.2x throughput** |
| Batching 1→8        | **24% lower p99** |
| Smart routing       | **~48% lower avg latency** |
| max_tokens 100→50   | **45% lower p95** |

---

## Kubernetes Deployment

```bash
# Apply all manifests
kubectl apply -k deployment/k8s/base/

# Check status
kubectl get pods -n llm-platform
kubectl get hpa -n llm-platform

# Scale workers manually
kubectl scale deployment inference-worker -n llm-platform --replicas=3
```

**HPA config:** Workers auto-scale 1→4 replicas at 70% CPU or 80% memory.

---

## CI/CD Pipeline

GitHub Actions workflows:
- **`ci-cd.yml`** — On push to `main`: test → build → push to Docker Hub → deploy to K8s
- **`pr-check.yml`** — On every PR: run tests + validate K8s YAML

---

## Future Roadmap

| Priority | Feature | Impact |
|----------|---------|--------|
| High | Speculative decoding | 2-3x latency reduction |
| High | GPTQ/AWQ quantization | 4x model compression, enables larger models on 4GB VRAM |
| High | Semantic cache (embedding similarity) | Eliminates redundant inference for similar prompts |
| Medium | Multi-node distributed inference | Scale beyond single GPU memory |
| Medium | GPU memory manager | Dynamic allocation across model pool |
| Medium | Canary deployment system | Safe model version rollouts |
| Low | Fault tolerance + retry | Auto-recover from worker crashes |
| Low | Chaos testing framework | Validate resilience under failures |
| Low | A/B testing framework | Compare model quality/latency tradeoffs |

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| API | FastAPI, Uvicorn, Pydantic v2 |
| Serving | HuggingFace Transformers, PyTorch 2.3 |
| Queue | Redis 7.2 (LPUSH/BRPOP) |
| Monitoring | Prometheus, Grafana |
| Containers | Docker, Docker Compose |
| Orchestration | Kubernetes, Kustomize, HPA |
| CI/CD | GitHub Actions |
| Testing | pytest, pytest-asyncio |
| GPU | NVIDIA CUDA 12.1 |

---
