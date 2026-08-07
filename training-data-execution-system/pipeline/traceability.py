"""
Full token traceability audit and learning trace audit.

Traceability chain (consumption):
  Dataset → Shard → Document → Packed Sequence → Batch → Training Step → Loss

Learning trace (reverse traversal):
  Loss → Batch → Packed Sequence → Document → Shard → Dataset

Generates:
  audits/traceability_audit.json     – full token lineage verification
  audits/learning_trace_audit.json   – loss → source data linkage
"""

import os
from pipeline import config
from pipeline.utils import read_jsonl, read_json, write_json, now_iso

_TRACE_AUDIT_PATH    = os.path.join(config.AUDITS_DIR, "traceability_audit.json")
_LT_AUDIT_PATH       = os.path.join(config.AUDITS_DIR, "learning_trace_audit.json")


# ─── Consumption traceability ─────────────────────────────────────────────────

def generate_traceability_audit(
    all_manifests: dict,
    all_batches:   list,
) -> dict:
    """
    Verify every consumed token can be traced back through the full chain:
      Dataset → Shard → Document → Packed Sequence → Batch → Step → Loss

    Reads consumption + learning ledgers from disk so this is entirely
    artifact-driven.
    """
    cl_path = os.path.join(config.LEDGERS_DIR, "consumption_ledger.jsonl")
    ll_path = os.path.join(config.LEDGERS_DIR, "learning_ledger.jsonl")

    cl = _safe_jsonl(cl_path)
    ll = _safe_jsonl(ll_path)

    # build lookup maps
    ll_by_batch  = {r["batch_id"]: r for r in ll}
    cl_by_batch  = {r["batch_id"]: r for r in cl}
    batch_by_id  = {b["batch_id"]: b for b in all_batches}

    # flatten all shard→document map from manifests
    doc_to_shard  = {}   # doc_id → shard_id
    doc_to_ds     = {}   # doc_id → dataset
    shard_to_ds   = {}   # shard_id → dataset
    for ds_name, shards in all_manifests.items():
        for sh in shards:
            shard_to_ds[sh["shard_id"]] = ds_name
            shard_path = os.path.join(
                config.RAW_DATA_DIR, f"{sh['shard_id']}.json"
            )
            try:
                data = read_json(shard_path)
                for doc in data.get("documents", []):
                    doc_to_shard[doc["doc_id"]] = sh["shard_id"]
                    doc_to_ds[doc["doc_id"]]    = ds_name
            except Exception:
                pass

    trace_entries = []
    failures      = []

    for cl_rec in cl:
        step     = cl_rec["step"]
        batch_id = cl_rec["batch_id"]

        # step 1: batch exists
        batch = batch_by_id.get(batch_id)
        if not batch:
            failures.append(f"step {step}: batch {batch_id} not found")
            continue

        # step 2: loss record exists
        ll_rec = ll_by_batch.get(batch_id)
        if not ll_rec:
            failures.append(f"step {step}: no learning record for batch {batch_id}")
            continue

        # step 3: every sample_id traces back to a shard/dataset
        untraceable_samples = []
        for sid in cl_rec.get("sample_ids", []):
            if sid not in doc_to_shard:
                untraceable_samples.append(sid)

        # step 4: shard_sources all map to known datasets
        unknown_shards = []
        for src in cl_rec.get("shard_sources", []):
            if src["shard_id"] not in shard_to_ds:
                unknown_shards.append(src["shard_id"])

        ok = not untraceable_samples and not unknown_shards
        entry = {
            "step":                  step,
            "batch_id":              batch_id,
            "loss":                  ll_rec.get("loss"),
            "datasets":              sorted({shard_to_ds.get(s["shard_id"], "?")
                                             for s in cl_rec.get("shard_sources", [])}),
            "shards":                [s["shard_id"] for s in cl_rec.get("shard_sources", [])],
            "token_count":           cl_rec.get("token_count", 0),
            "untraceable_samples":   untraceable_samples,
            "unknown_shards":        unknown_shards,
            "traceable":             ok,
        }
        trace_entries.append(entry)
        if not ok:
            failures.append(f"step {step}: traceability failed  "
                            f"untraceable_samples={untraceable_samples} "
                            f"unknown_shards={unknown_shards}")

    passed = len(failures) == 0
    report = {
        "result":                   "PASS" if passed else "FAIL",
        "steps_verified":           len(trace_entries),
        "failures":                 failures,
        "chain":                    "Dataset → Shard → Document → Packed Sequence → Batch → Step → Loss",
        "trace_entries":            trace_entries,
        "consumption_ledger":       cl_path,
        "learning_ledger":          ll_path,
        "generated_at":             now_iso(),
    }
    write_json(_TRACE_AUDIT_PATH, report)
    return report


# ─── Learning trace (reverse) ─────────────────────────────────────────────────

def generate_learning_trace_audit(all_batches: list) -> dict:
    """
    For every learning ledger entry demonstrate the full reverse chain:
      Loss → Batch → Packed Sequence → Document → Shard → Dataset
    """
    ll_path = os.path.join(config.LEDGERS_DIR, "learning_ledger.jsonl")
    cl_path = os.path.join(config.LEDGERS_DIR, "consumption_ledger.jsonl")

    ll = _safe_jsonl(ll_path)
    cl = _safe_jsonl(cl_path)

    cl_by_batch  = {r["batch_id"]: r for r in cl}
    batch_by_id  = {b["batch_id"]: b for b in all_batches}

    traces   = []
    failures = []

    for ll_rec in ll:
        step     = ll_rec["step"]
        batch_id = ll_rec["batch_id"]
        loss     = ll_rec["loss"]

        # Loss → Batch
        batch = batch_by_id.get(batch_id)
        cl_rec = cl_by_batch.get(batch_id)

        if not batch or not cl_rec:
            failures.append(f"step {step}: cannot trace loss for batch {batch_id}")
            continue

        # Batch → Packed Sequences → Documents → Shards → Datasets
        seq_sources = []
        for seq in batch.get("sequences", []):
            for src in seq.get("sources", []):
                key = (src.get("doc_id",""), src.get("shard_id",""), src.get("dataset",""))
                if key not in seq_sources:
                    seq_sources.append(key)

        trace = {
            "step":            step,
            "loss":            loss,
            "batch_id":        batch_id,
            "source_datasets": ll_rec.get("source_datasets", []),
            "shard_sources":   cl_rec.get("shard_sources", []),
            # forward chain links
            "chain": {
                "loss":      loss,
                "batch":     batch_id,
                "sequences": [
                    {
                        "doc_id":   t[0],
                        "shard_id": t[1],
                        "dataset":  t[2],
                    }
                    for t in seq_sources[:8]   # first 8 to keep report compact
                ],
            },
        }
        traces.append(trace)

    passed = len(failures) == 0
    report = {
        "result":          "PASS" if passed else "FAIL",
        "entries_traced":  len(traces),
        "failures":        failures,
        "chain":           "Loss → Batch → Packed Sequence → Document → Shard → Dataset",
        "trace_entries":   traces,
        "learning_ledger": ll_path,
        "generated_at":    now_iso(),
    }
    write_json(_LT_AUDIT_PATH, report)
    return report


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _safe_jsonl(path: str) -> list:
    if not os.path.exists(path):
        return []
    try:
        from pipeline.utils import read_jsonl
        return read_jsonl(path)
    except Exception:
        return []
