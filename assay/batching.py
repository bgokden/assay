"""Dynamic batching for a prefill-only decision model.

Every request is one forward pass, so there is no decode loop and no scheduler to write: the
batcher collects requests that arrive close together, runs them as one packed batch, and
returns each caller their own answers. Measured on an RTX 5090, 24 questions cost 56 ms in one
batch against 544 ms answered one at a time, so this is where the throughput is.
"""

from __future__ import annotations

import asyncio
import dataclasses
import time
from typing import Any

import torch

from assay.schema import Answer, Question
from assay.serving import runner_for


@dataclasses.dataclass
class Job:
    item: Any  # whatever the runner prepared: a packed batch, or a state
    questions: list[Question]
    future: asyncio.Future
    queued_at: float


class Metrics:
    """Counters and reservoirs a production deployment wants to see, kept dependency-free."""

    def __init__(self, window: int = 1024) -> None:
        self.window = window
        self.requests = 0
        self.errors = 0
        self.rejected = 0
        self.batches = 0
        self.batched_requests = 0
        self.latencies_ms: list[float] = []
        self.waits_ms: list[float] = []
        self.batch_sizes: list[int] = []

    def _record(self, values: list[float], value: float) -> None:
        values.append(value)
        if len(values) > self.window:
            del values[: len(values) - self.window]

    def observe_batch(self, size: int, wait_ms: float) -> None:
        self.batches += 1
        self.batched_requests += size
        self._record(self.batch_sizes, size)
        self._record(self.waits_ms, wait_ms)

    def observe_request(self, latency_ms: float) -> None:
        self.requests += 1
        self._record(self.latencies_ms, latency_ms)

    @staticmethod
    def quantile(values: list[float], q: float) -> float:
        if not values:
            return 0.0
        ordered = sorted(values)
        return float(ordered[min(len(ordered) - 1, int(q * len(ordered)))])

    def snapshot(self, queue_depth: int) -> dict[str, Any]:
        return {
            "requests": self.requests,
            "errors": self.errors,
            "rejected": self.rejected,
            "batches": self.batches,
            "queue_depth": queue_depth,
            "mean_batch_size": (self.batched_requests / self.batches) if self.batches else 0.0,
            "latency_ms": {
                "p50": round(self.quantile(self.latencies_ms, 0.5), 1),
                "p95": round(self.quantile(self.latencies_ms, 0.95), 1),
                "p99": round(self.quantile(self.latencies_ms, 0.99), 1),
            },
            "queue_wait_ms": {
                "p50": round(self.quantile(self.waits_ms, 0.5), 1),
                "p95": round(self.quantile(self.waits_ms, 0.95), 1),
            },
        }

    def prometheus(self, queue_depth: int) -> str:
        s = self.snapshot(queue_depth)
        lines = [
            "# HELP assay_requests_total Decision requests served",
            "# TYPE assay_requests_total counter",
            f"assay_requests_total {s['requests']}",
            "# HELP assay_errors_total Requests that failed during inference",
            "# TYPE assay_errors_total counter",
            f"assay_errors_total {s['errors']}",
            "# HELP assay_rejected_total Requests rejected because the queue was full",
            "# TYPE assay_rejected_total counter",
            f"assay_rejected_total {s['rejected']}",
            "# HELP assay_batches_total Forward passes run",
            "# TYPE assay_batches_total counter",
            f"assay_batches_total {s['batches']}",
            "# HELP assay_queue_depth Requests waiting for a forward pass",
            "# TYPE assay_queue_depth gauge",
            f"assay_queue_depth {s['queue_depth']}",
            "# HELP assay_batch_size_mean Mean requests per forward pass",
            "# TYPE assay_batch_size_mean gauge",
            f"assay_batch_size_mean {s['mean_batch_size']:.3f}",
            "# HELP assay_latency_ms Request latency in milliseconds",
            "# TYPE assay_latency_ms summary",
        ]
        for q in ("p50", "p95", "p99"):
            lines.append(f'assay_latency_ms{{quantile="{q}"}} {s["latency_ms"][q]}')
        return "\n".join(lines) + "\n"


class QueueFull(Exception):
    """Raised when the batcher is saturated; the caller should answer 503 and shed load."""


class Batcher:
    """Collects requests for a short window and runs them as one forward pass. The runner
    knows how its tier turns a request into work; the batcher only sizes and groups."""

    def __init__(
        self,
        model: Any,
        max_batch_size: int = 16,
        max_batch_tokens: int = 16384,
        max_wait_ms: float = 5.0,
        max_queue: int = 256,
        max_state_tokens: int = 4096,
        runner: Any = None,
    ) -> None:
        self.model = model
        self.runner = runner if runner is not None else runner_for(model, max_state_tokens)
        self.max_batch_size = max_batch_size
        self.max_batch_tokens = max_batch_tokens
        self.max_wait_ms = max_wait_ms
        self.max_queue = max_queue
        self.queue: asyncio.Queue[Job] = asyncio.Queue()
        self.metrics = Metrics()
        self._task: asyncio.Task | None = None
        self._loop: asyncio.AbstractEventLoop | None = None

    async def start(self) -> None:
        """Idempotent, and bound to the event loop that calls it. A host that runs requests on
        a fresh loop each time (starlette's TestClient outside a context manager, or a
        restarted server) would otherwise leave the batch loop and its futures on a dead loop,
        so the queue is rebuilt whenever the running loop changes. There is no await between
        the checks and create_task, so no lock is needed."""
        loop = asyncio.get_running_loop()
        if self._loop is not loop:
            self.queue = asyncio.Queue()
            self._loop = loop
            self._task = None
        if self._task is None or self._task.done():
            self._task = asyncio.create_task(self._run())

    async def stop(self) -> None:
        if self._task is not None and self._loop is asyncio.get_running_loop():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None

    async def submit(self, item: Any, questions: list[Question]) -> list[Answer]:
        await self.start()
        if self.queue.qsize() >= self.max_queue:
            self.metrics.rejected += 1
            raise QueueFull(f"{self.queue.qsize()} requests already queued")
        future: asyncio.Future = asyncio.get_running_loop().create_future()
        self.queue.put_nowait(Job(item, questions, future, time.perf_counter()))
        return await future

    async def _collect(self) -> list[Job]:
        """One job, then whatever else arrives within the window and fits the batch budget."""
        jobs = [await self.queue.get()]
        tokens = self.runner.size(jobs[0].item)
        deadline = time.perf_counter() + self.max_wait_ms / 1000.0
        while len(jobs) < self.max_batch_size:
            remaining = deadline - time.perf_counter()
            if remaining <= 0:
                break
            try:
                job = await asyncio.wait_for(self.queue.get(), timeout=remaining)
            except TimeoutError:
                break
            if tokens + self.runner.size(job.item) > self.max_batch_tokens and len(jobs) > 1:
                self.queue.put_nowait(job)  # too big for this batch, leads the next one
                break
            jobs.append(job)
            tokens += self.runner.size(job.item)
        return jobs

    def _infer(self, jobs: list[Job]) -> list[list[Answer]]:
        device = getattr(self.model, "device", None)
        use_cuda = device is not None and getattr(device, "type", "") == "cuda"
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_cuda):
            return self.runner.answer_batch([j.item for j in jobs], [j.questions for j in jobs])

    async def _run(self) -> None:
        loop = asyncio.get_running_loop()
        while True:
            jobs = await self._collect()
            started = time.perf_counter()
            wait_ms = (started - min(j.queued_at for j in jobs)) * 1000.0
            self.metrics.observe_batch(len(jobs), wait_ms)
            # gather captures any failure as a value, so one bad batch cannot kill the loop
            (outcome,) = await asyncio.gather(
                loop.run_in_executor(None, self._infer, jobs), return_exceptions=True
            )
            if isinstance(outcome, BaseException):
                self.metrics.errors += len(jobs)
                for job in jobs:
                    if not job.future.done():
                        job.future.set_exception(outcome)
                continue
            for job, answers in zip(jobs, outcome):
                if not job.future.done():
                    job.future.set_result(answers)
