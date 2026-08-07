"""
Consumption ledger: records every training step with full source traceability.
Learning ledger:    records every simulated loss value with batch linkage.

Traceability chain:
  Loss → Step → Batch → Sequences → Sources → (Dataset, Shard, Document)
"""

import os
import random
import math
from pipeline import config
from pipeline.utils import (
    append_jsonl, read_jsonl, write_json, now_iso
)

_CONSUMPTION_LEDGER = os.path.join(config.LEDGERS_DIR, "consumption_ledger.jsonl")
_LEARNING_LEDGER    = os.path.join(config.LEDGERS_DIR, "learning_ledger.jsonl")


# ─── Consumption ledger ───────────────────────────────────────────────────────

def record_consumption(step: int, batch: dict, dataset_contributions: dict):
    """
    Append one record to consumption_ledger.jsonl.

    *dataset_contributions* maps dataset_name → token_count.
    """
    entry = {
        "step":                   step,
        "batch_id":               batch["batch_id"],
        "dataset_contributions":  dataset_contributions,
        "token_count":            batch.get("token_count", 0),
        "sample_ids":             batch.get("sample_ids", []),
        "shard_sources":          _extract_shard_sources(batch),
        "timestamp":              now_iso(),
    }
    append_jsonl(_CONSUMPTION_LEDGER, entry)
    return entry


def _extract_shard_sources(batch: dict) -> list:
    """Return unique (dataset, shard_id) pairs from all sequences in the batch."""
    seen  = set()
    pairs = []
    for seq in batch.get("sequences", []):
        for src in seq.get("sources", []):
            key = (src.get("dataset",""), src.get("shard_id",""))
            if key not in seen:
                seen.add(key)
                pairs.append({"dataset": key[0], "shard_id": key[1]})
    return pairs


# ─── Learning ledger ──────────────────────────────────────────────────────────

_LOSS_RNG = random.Random(config.RANDOM_SEED + 7)

def _simulated_loss(step: int) -> float:
    """
    Deterministic simulated batch loss: exponential decay + small noise.
    Using the global step keeps it reproducible across crash/resume.
    """
    rng   = random.Random(config.RANDOM_SEED + step)   # per-step seed
    base  = 3.5 * math.exp(-0.08 * step)
    noise = rng.uniform(-0.05, 0.05)
    return round(max(0.01, base + noise), 4)


def _simulated_per_sample_loss(step: int, sample_idx: int, n_tokens: int) -> float:
    """
    Deterministic per-sample loss derived from the batch loss plus a small
    sample-level perturbation.  Each sample gets a loss proportional to
    its real token count (useful loss-bearing tokens).
    """
    batch_loss = _simulated_loss(step)
    rng        = random.Random(config.RANDOM_SEED + step * 1000 + sample_idx)
    perturb    = rng.uniform(-0.02, 0.02)
    return round(max(0.001, batch_loss + perturb), 4)


def record_learning(step: int, batch: dict):
    """
    Append one record to learning_ledger.jsonl.
    Includes batch-level loss AND per-sample loss breakdown.
    Returns the entry dict.
    """
    loss            = _simulated_loss(step)
    source_datasets = list({
        src.get("dataset", "")
        for seq in batch.get("sequences", [])
        for src in seq.get("sources", [])
    })

    # per-sample loss (sample = packed sequence in the batch)
    per_sample = []
    for i, seq in enumerate(batch.get("sequences", [])):
        n_real  = sum(seq.get("mask", []))
        s_loss  = _simulated_per_sample_loss(step, i, n_real)
        per_sample.append({
            "sample_index":  i,
            "loss":          s_loss,
            "token_count":   n_real,
            "sources":       [s.get("doc_id", "") for s in seq.get("sources", [])],
        })

    entry = {
        "step":             step,
        "batch_id":         batch["batch_id"],
        "loss":             loss,                 # batch-level loss
        "per_sample_loss":  per_sample,           # sample-level loss breakdown
        "source_datasets":  sorted(source_datasets),
        "sample_ids":       batch.get("sample_ids", []),
        "shard_sources":    _extract_shard_sources(batch),
        "timestamp":        now_iso(),
    }
    append_jsonl(_LEARNING_LEDGER, entry)
    return entry


# ─── Ledger readers ───────────────────────────────────────────────────────────

def read_consumption_ledger() -> list:
    if not os.path.exists(_CONSUMPTION_LEDGER):
        return []
    return read_jsonl(_CONSUMPTION_LEDGER)


def read_learning_ledger() -> list:
    if not os.path.exists(_LEARNING_LEDGER):
        return []
    return read_jsonl(_LEARNING_LEDGER)


def verify_learning_traceability(learning_records: list,
                                  consumption_records: list) -> dict:
    """
    Verify every learning record references a batch that appears in the
    consumption ledger.  Returns a traceability report.
    """
    consumption_batch_ids = {r["batch_id"] for r in consumption_records}
    missing = []
    for lr in learning_records:
        if lr["batch_id"] not in consumption_batch_ids:
            missing.append(lr["batch_id"])

    passed = len(missing) == 0
    return {
        "result":               "PASS" if passed else "FAIL",
        "learning_steps":       len(learning_records),
        "consumption_steps":    len(consumption_records),
        "unmatched_batch_ids":  missing,
    }
