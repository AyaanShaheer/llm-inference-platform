import asyncio
import structlog
from request_queue.redis_queue import RedisQueueManager, QueuedRequest, QueueResult
from inference_worker.batch_processor import BatchProcessor, InferenceJob

logger = structlog.get_logger()


class QueueWorker:
    """
    Pulls requests from Redis queue and feeds them into the BatchProcessor.
    Runs as a background coroutine per model type.
    """

    def __init__(
        self,
        queue_manager: RedisQueueManager,
        batch_processor: BatchProcessor,
        model_type: str,
        concurrency: int = 4
    ):
        self._queue = queue_manager
        self._batch = batch_processor
        self._model_type = model_type
        self._concurrency = concurrency
        self._tasks: list[asyncio.Task] = []
        self._running = False

    def start(self):
        self._running = True
        for i in range(self._concurrency):
            task = asyncio.create_task(
                self._consume_loop(),
                name=f"queue_worker_{self._model_type}_{i}"
            )
            self._tasks.append(task)
        logger.info(
            "queue_worker_started",
            model_type=self._model_type,
            concurrency=self._concurrency
        )

    def stop(self):
        self._running = False
        for t in self._tasks:
            t.cancel()

    async def _consume_loop(self):
        """Pull → infer → store result, forever."""
        while self._running:
            try:
                queued_req: QueuedRequest = await self._queue.dequeue(
                    self._model_type, timeout=2
                )
                if queued_req is None:
                    continue

                # Submit to batch processor
                job = InferenceJob(
                    request_id=queued_req.request_id,
                    prompt=queued_req.prompt,
                    max_tokens=queued_req.max_tokens,
                    temperature=queued_req.temperature,
                    model_type=queued_req.model_type,
                )

                try:
                    result = await asyncio.wait_for(
                        self._batch_processor_submit(job),
                        timeout=120.0
                    )
                    queue_result = QueueResult(
                        request_id=result.request_id,
                        status="success",
                        generated_text=result.generated_text,
                        model_used=result.model_used,
                        prompt_tokens=result.prompt_tokens,
                        completion_tokens=result.completion_tokens,
                        latency_ms=result.latency_ms,
                    )
                except asyncio.TimeoutError:
                    queue_result = QueueResult(
                        request_id=queued_req.request_id,
                        status="timeout",
                        error="Inference timed out"
                    )
                except Exception as e:
                    queue_result = QueueResult(
                        request_id=queued_req.request_id,
                        status="error",
                        error=str(e)
                    )

                await self._queue.store_result(queue_result)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("queue_worker_error", model_type=self._model_type, error=str(e))
                await asyncio.sleep(1)

    async def _batch_processor_submit(self, job: InferenceJob):
        return await self._batch.submit(job)
