import asyncio
import time
import torch
import structlog
from dataclasses import dataclass, field
from typing import Optional
from inference_worker.model_manager import ModelManager, LoadedModel

logger = structlog.get_logger()

# Batching config
MAX_BATCH_SIZE = 8
MAX_WAIT_MS = 50          # Max wait to fill a batch (milliseconds)
MAX_NEW_TOKENS_CAP = 512  # Hard cap per request


@dataclass
class InferenceJob:
    request_id: str
    prompt: str
    max_tokens: int
    temperature: float
    model_type: str
    future: asyncio.Future = field(default_factory=asyncio.Future)
    enqueue_time: float = field(default_factory=time.perf_counter)


@dataclass
class InferenceResult:
    request_id: str
    generated_text: str
    prompt_tokens: int
    completion_tokens: int
    latency_ms: float
    model_used: str


class BatchProcessor:
    """
    Continuous batching engine.

    Algorithm:
      1. Requests land in _queue (asyncio.Queue)
      2. Background worker wakes up when:
         - batch_size == MAX_BATCH_SIZE, OR
         - MAX_WAIT_MS elapsed since first item arrived
      3. All jobs in the batch run as a single model forward pass
      4. Results are dispatched to individual futures
    """

    def __init__(self, model_manager: ModelManager):
        self._model_manager = model_manager
        self._queues: dict[str, asyncio.Queue] = {
            "small": asyncio.Queue(),
            "large": asyncio.Queue(),
        }
        self._tasks: list[asyncio.Task] = []
        self._running = False
        self.total_batches = 0
        self.total_requests = 0

    def start(self):
        self._running = True
        for model_type in ["small", "large"]:
            task = asyncio.create_task(
                self._batch_loop(model_type),
                name=f"batch_loop_{model_type}"
            )
            self._tasks.append(task)
        logger.info("batch_processor_started")

    def stop(self):
        self._running = False
        for task in self._tasks:
            task.cancel()
        logger.info("batch_processor_stopped")

    async def submit(self, job: InferenceJob) -> InferenceResult:
        """Submit a job and await its result."""
        queue = self._queues.get(job.model_type)
        if queue is None:
            raise ValueError(f"No queue for model_type: {job.model_type}")
        await queue.put(job)
        return await job.future

    async def _batch_loop(self, model_type: str):
        """Continuously collect and process batches for one model type."""
        queue = self._queues[model_type]

        while self._running:
            batch: list[InferenceJob] = []

            # Wait for at least one job
            try:
                first_job = await asyncio.wait_for(queue.get(), timeout=1.0)
                batch.append(first_job)
            except asyncio.TimeoutError:
                continue

            # Collect more jobs up to MAX_BATCH_SIZE within MAX_WAIT_MS
            deadline = time.perf_counter() + (MAX_WAIT_MS / 1000.0)
            while len(batch) < MAX_BATCH_SIZE:
                remaining = deadline - time.perf_counter()
                if remaining <= 0:
                    break
                try:
                    job = await asyncio.wait_for(queue.get(), timeout=remaining)
                    batch.append(job)
                except asyncio.TimeoutError:
                    break

            # Process the batch
            await self._process_batch(model_type, batch)

    async def _process_batch(self, model_type: str, batch: list[InferenceJob]):
        """Run a single batched forward pass."""
        loaded = self._model_manager.get_model(model_type)
        if loaded is None:
            # Model not loaded yet — fail all jobs in batch
            for job in batch:
                if not job.future.done():
                    job.future.set_exception(
                        RuntimeError(f"Model '{model_type}' not loaded")
                    )
            return

        self.total_batches += 1
        self.total_requests += len(batch)
        batch_start = time.perf_counter()

        logger.info(
            "batch_processing",
            model_type=model_type,
            batch_size=len(batch),
            batch_id=self.total_batches,
        )

        try:
            # Run inference in a thread pool to avoid blocking the event loop
            results = await asyncio.get_event_loop().run_in_executor(
                None,
                self._run_inference_sync,
                loaded,
                batch,
            )

            batch_latency = (time.perf_counter() - batch_start) * 1000

            for job, result in zip(batch, results):
                result.latency_ms = round(
                    (time.perf_counter() - job.enqueue_time) * 1000, 2
                )
                if not job.future.done():
                    job.future.set_result(result)

            logger.info(
                "batch_complete",
                model_type=model_type,
                batch_size=len(batch),
                batch_latency_ms=round(batch_latency, 2),
            )

        except Exception as e:
            logger.error("batch_error", model_type=model_type, error=str(e))
            for job in batch:
                if not job.future.done():
                    job.future.set_exception(e)

    def _run_inference_sync(
        self, loaded: LoadedModel, batch: list[InferenceJob]
    ) -> list[InferenceResult]:
        """Synchronous batched inference — runs in thread pool."""
        model = loaded.model
        tokenizer = loaded.tokenizer
        device = loaded.device

        prompts = [job.prompt for job in batch]
        max_new_tokens = min(
            max(job.max_tokens for job in batch),
            MAX_NEW_TOKENS_CAP
        )

        # Tokenize with padding
        inputs = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=1024,
        ).to(device)

        prompt_lengths = inputs["attention_mask"].sum(dim=1).tolist()

        with torch.no_grad():
            outputs = model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=batch[0].temperature,   # use first job's temp
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        results = []
        for i, job in enumerate(batch):
            prompt_len = int(prompt_lengths[i])
            new_tokens = outputs[i][prompt_len:]
            generated = tokenizer.decode(new_tokens, skip_special_tokens=True)
            completion_tokens = len(new_tokens)

            results.append(InferenceResult(
                request_id=job.request_id,
                generated_text=generated.strip(),
                prompt_tokens=prompt_len,
                completion_tokens=completion_tokens,
                latency_ms=0.0,     # set after by caller
                model_used=loaded.model_name,
            ))

        return results
