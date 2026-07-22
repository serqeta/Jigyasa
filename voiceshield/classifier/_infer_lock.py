"""
Single-GPU-thread inference (single-flight, thread-affine).

Concurrent CUDA work from FastAPI's thread pool (`/v2/analyze`, `/v2/report`)
and the live WebSocket runner corrupts the shared CUDA context — every crash
we saw (SIGSEGV / "double free" / "unaligned chunk") was that. A lock around
the *forward* alone is NOT enough: PyTorch frees CUDA tensors by refcount on
whatever thread drops the last reference, so allocations/frees still race.

Fix: confine ALL GPU work to ONE dedicated thread. Every scorer runs its
CUDA path via `run_on_gpu(...)`, which hands the work to a single-worker
executor and blocks for the result. Callers (many devices, the live runner,
upload/report requests) queue on that one thread — CPU-side prep (decode,
resample, VAD, feature extraction) still runs concurrently across requests;
only the GPU forward is single-flighted. This makes multi-device use SAFE.

For higher parallel throughput than one GPU thread can give, the next step is
dynamic batching or additional GPU worker processes (see
docs/ARCHITECTURE_ASCII.md) — not more threads on one context.
"""

import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, TypeVar

_T = TypeVar("_T")

# Coarse process-wide scoring lock. Only ONE chunk is scored at a time across
# the whole process — the live runner and every upload/report request. This is
# what actually stops the concurrency crash: not just the GPU forward but the
# whole path (Silero VAD, feature extraction, fusion) is single-flighted.
# Acquired in PipelineRunner around the scoring, NOT around the source read,
# so an idle live-mic stream can't starve queued uploads.
INFERENCE_LOCK = threading.Lock()

# Exactly one worker → all CUDA allocations, forwards, and frees happen on the
# same thread, so nothing races on the context.
_GPU_EXECUTOR = ThreadPoolExecutor(max_workers=1, thread_name_prefix="gpu-infer")


def run_on_gpu(fn: Callable[..., _T], *args: Any, **kwargs: Any) -> _T:
    """Run `fn` on the single GPU thread and return its result (blocking)."""
    return _GPU_EXECUTOR.submit(fn, *args, **kwargs).result()
