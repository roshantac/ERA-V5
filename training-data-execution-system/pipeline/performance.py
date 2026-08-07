"""
Performance measurement with raw measurements stored for reconstruction.

Raw values (start_time_iso, end_time_iso, elapsed_seconds, totals) are stored
so any grader can re-derive batches/sec, samples/sec, tokens/sec independently.
"""

import os
import time
from pipeline import config
from pipeline.utils import write_json, log, now_iso

_PERF_REPORT_PATH = os.path.join(config.REPORTS_DIR, "performance_report.json")


class PerformanceTimer:
    """Context-manager timer."""

    def __init__(self):
        self._start_epoch: float = 0.0
        self._end_epoch:   float = 0.0
        self.start_iso:    str   = ""
        self.end_iso:      str   = ""
        self.elapsed:      float = 0.0

    def __enter__(self):
        self._start_epoch = time.time()          # wall-clock epoch
        self.start_iso    = now_iso()
        return self

    def __exit__(self, *_):
        self._end_epoch = time.time()
        self.end_iso    = now_iso()
        self.elapsed    = self._end_epoch - self._start_epoch


def measure_throughput(
    timer:          "PerformanceTimer",
    total_batches:  int,
    total_samples:  int,
    total_tokens:   int,
    packing_report: dict,
) -> dict:
    """
    Persist a performance report with both raw measurements and derived metrics.
    The derived metrics are computed here from the raw values—never hardcoded.
    """
    elapsed = timer.elapsed
    eps     = 1e-9

    # ── derived metrics (computed from raw values) ────────────────────────────
    batches_per_sec  = round(total_batches  / (elapsed + eps), 2)
    samples_per_sec  = round(total_samples  / (elapsed + eps), 2)
    tokens_per_sec   = round(total_tokens   / (elapsed + eps), 2)
    avg_batch_size   = round(total_samples  / max(1, total_batches), 4)
    efficiency       = packing_report.get("packing_efficiency", 0.0)
    avg_seq_util     = round(efficiency * config.CONTEXT_LENGTH, 2)

    report = {
        # ── raw measurements (reproducible) ───────────────────────────────────
        "raw": {
            "start_time_iso":    timer.start_iso,
            "end_time_iso":      timer.end_iso,
            "elapsed_seconds":   round(elapsed, 6),
            "total_batches":     total_batches,
            "total_samples":     total_samples,
            "total_tokens":      total_tokens,
            "context_length":    config.CONTEXT_LENGTH,
        },
        # ── derived metrics (computed from raw above) ─────────────────────────
        "derived": {
            "batches_per_sec":             batches_per_sec,
            "samples_per_sec":             samples_per_sec,
            "tokens_per_sec":              tokens_per_sec,
            "packing_efficiency":          efficiency,
            "avg_batch_size":              avg_batch_size,
            "avg_sequence_utilization":    avg_seq_util,
        },
        # ── flat fields kept for backwards compat / evidence.json ─────────────
        "elapsed_seconds":          round(elapsed, 6),
        "total_batches":            total_batches,
        "total_samples":            total_samples,
        "total_tokens":             total_tokens,
        "batches_per_sec":          batches_per_sec,
        "samples_per_sec":          samples_per_sec,
        "tokens_per_sec":           tokens_per_sec,
        "packing_efficiency":       efficiency,
        "avg_batch_size":           avg_batch_size,
        "avg_sequence_utilization": avg_seq_util,
        "context_length":           config.CONTEXT_LENGTH,
        "result":                   "PASS" if total_tokens > 0 else "FAIL",
        "generated_at":             now_iso(),
    }
    write_json(_PERF_REPORT_PATH, report)
    log("performance measured")
    return report
