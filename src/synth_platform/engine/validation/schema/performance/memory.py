"""Peak memory tracking with psutil when available."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator, List, Optional


def peak_memory_mb() -> float:
    """Return current peak RSS in MB, best-effort and Windows-safe."""
    try:
        import psutil

        return float(psutil.Process().memory_info().rss) / (1024 * 1024)
    except Exception:
        pass

    try:
        import resource

        usage = resource.getrusage(resource.RUSAGE_SELF)
        rss = float(usage.ru_maxrss)

        if rss > 10_000_000:
            return rss / (1024 * 1024)

        return rss / 1024
    except Exception:
        return 0.0


@contextmanager
def track_peak_memory(*, use_tracemalloc: Optional[bool] = None, target_rows: int = 0) -> Iterator[List[float]]:
    """Track peak RSS memory in MB during a block."""
    peaks: List[float] = [peak_memory_mb()]
    enable_trace = use_tracemalloc

    if enable_trace is None:
        enable_trace = target_rows > 0 and target_rows <= 5_000

    if enable_trace:
        import tracemalloc

        tracemalloc.start()
        try:
            yield peaks
        finally:
            _, traced_peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            peaks[0] = max(peaks[0], traced_peak / (1024 * 1024), peak_memory_mb())
    else:
        try:
            yield peaks
        finally:
            peaks[0] = max(peaks[0], peak_memory_mb())
