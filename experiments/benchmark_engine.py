import time
import torch
import numpy as np
import json
from dataclasses import dataclass, field, asdict
from typing import Literal
from transformers import AutoTokenizer, AutoModelForCausalLM

PROMPTS = [
    "The future of artificial intelligence is",
    "In a world where technology advances rapidly",
    "The most important lesson I have learned is",
    "Scientists recently discovered that",
    "The key to building great software is",
    "Once upon a time in a distant galaxy",
    "The economy will change significantly because",
    "To solve climate change we need to",
    "The human brain is fascinating because",
    "Machine learning works by",
    "Distributed systems require careful design of",
    "The best programming language for AI is",
    "Neural networks learn from data by",
    "Cloud computing has transformed the way",
    "The principles of good system design include",
    "Open source software has changed",
]


@dataclass
class BenchmarkConfig:
    model_name: str
    precision: Literal["fp32", "fp16"]
    batch_size: int
    max_new_tokens: int
    num_warmup: int = 3
    num_runs: int = 20


@dataclass
class BenchmarkResult:
    config: BenchmarkConfig
    latencies_ms: list[float] = field(default_factory=list)
    tokens_per_second: float = 0.0

    @property
    def p50(self): return round(float(np.percentile(self.latencies_ms, 50)), 2)
    @property
    def p95(self): return round(float(np.percentile(self.latencies_ms, 95)), 2)
    @property
    def p99(self): return round(float(np.percentile(self.latencies_ms, 99)), 2)
    @property
    def mean(self): return round(float(np.mean(self.latencies_ms)), 2)
    @property
    def throughput(self): return round(self.tokens_per_second, 2)

    def summary(self) -> dict:
        return {
            "model": self.config.model_name,
            "precision": self.config.precision,
            "batch_size": self.config.batch_size,
            "max_new_tokens": self.config.max_new_tokens,
            "p50_ms": self.p50,
            "p95_ms": self.p95,
            "p99_ms": self.p99,
            "mean_ms": self.mean,
            "throughput_tokens_per_sec": self.throughput,
        }


class BenchmarkEngine:

    def __init__(self, model_name: str, precision: str):
        self.model_name = model_name
        self.precision = precision
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        print(f"\n{'='*60}")
        print(f"Loading {model_name} [{precision}] on {self.device}")
        print(f"{'='*60}")

        dtype = torch.float16 if precision == "fp16" and self.device == "cuda" else torch.float32

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype,
        ).to(self.device)
        self.model.eval()

        params_m = sum(p.numel() for p in self.model.parameters()) / 1e6
        print(f"Loaded: {params_m:.1f}M params | dtype={dtype} | device={self.device}")

    def _run_batch(self, prompts: list[str], max_new_tokens: int) -> tuple[float, int]:
        """Run one forward pass. Returns (latency_ms, total_new_tokens)."""
        inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        ).to(self.device)

        if self.device == "cuda":
            torch.cuda.synchronize()

        start = time.perf_counter()
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_new_tokens=max_new_tokens,
                do_sample=False,        # greedy for determinism
                pad_token_id=self.tokenizer.eos_token_id,
            )
        if self.device == "cuda":
            torch.cuda.synchronize()

        elapsed_ms = (time.perf_counter() - start) * 1000
        input_len = inputs["input_ids"].shape[1]
        total_new = sum(len(o) - input_len for o in outputs)
        return elapsed_ms, total_new

    def run(self, config: BenchmarkConfig) -> BenchmarkResult:
        result = BenchmarkResult(config=config)

        # Warmup
        print(f"\nWarmup ({config.num_warmup} runs, batch={config.batch_size})...")
        for i in range(config.num_warmup):
            batch = PROMPTS[:config.batch_size]
            self._run_batch(batch, config.max_new_tokens)
            print(f"  warmup {i+1}/{config.num_warmup} done")

        # Benchmark
        print(f"Benchmarking ({config.num_runs} runs)...")
        total_tokens = 0
        for i in range(config.num_runs):
            # Rotate through prompts for variety
            start_idx = (i * config.batch_size) % len(PROMPTS)
            batch = []
            for j in range(config.batch_size):
                batch.append(PROMPTS[(start_idx + j) % len(PROMPTS)])

            latency_ms, new_tokens = self._run_batch(batch, config.max_new_tokens)
            result.latencies_ms.append(latency_ms)
            total_tokens += new_tokens

            if (i + 1) % 5 == 0:
                print(f"  run {i+1}/{config.num_runs} | {latency_ms:.0f}ms | {new_tokens} tokens")

        total_time_s = sum(result.latencies_ms) / 1000
        result.tokens_per_second = total_tokens / total_time_s if total_time_s > 0 else 0

        print(f"\nResults: p50={result.p50}ms p95={result.p95}ms p99={result.p99}ms tps={result.throughput}")
        return result

    def cleanup(self):
        del self.model
        if self.device == "cuda":
            torch.cuda.empty_cache()
        print(f"Cleaned up {self.model_name}")
