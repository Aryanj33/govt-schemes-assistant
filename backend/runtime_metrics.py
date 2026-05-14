"""
Lightweight in-process metrics for operator views and prototype validation.
"""

from __future__ import annotations

import statistics
import time
from collections import defaultdict, deque
from typing import Deque, Dict


class RuntimeMetrics:
    def __init__(self):
        self.started_at = time.time()
        self.request_counts: Dict[str, int] = defaultdict(int)
        self.error_counts: Dict[str, int] = defaultdict(int)
        self.latencies_ms: Dict[str, Deque[float]] = defaultdict(lambda: deque(maxlen=200))

    def record(self, name: str, latency_ms: float, success: bool = True):
        self.request_counts[name] += 1
        self.latencies_ms[name].append(latency_ms)
        if not success:
            self.error_counts[name] += 1

    def increment(self, name: str):
        self.request_counts[name] += 1

    def summary(self) -> Dict[str, object]:
        endpoints = {}
        for name, samples in self.latencies_ms.items():
            ordered = sorted(samples)
            p95_index = max(int(len(ordered) * 0.95) - 1, 0) if ordered else 0
            endpoints[name] = {
                "count": self.request_counts.get(name, 0),
                "errors": self.error_counts.get(name, 0),
                "avg_ms": round(statistics.fmean(samples), 1) if samples else 0.0,
                "p95_ms": round(ordered[p95_index], 1) if ordered else 0.0,
            }

        return {
            "uptime_seconds": round(time.time() - self.started_at, 1),
            "endpoints": endpoints,
        }


_metrics = RuntimeMetrics()


def get_runtime_metrics() -> RuntimeMetrics:
    return _metrics
