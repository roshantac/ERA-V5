"""
Enhanced evaluation firewall.

Validates not only data routing but also ledger contents:
  - EVAL data never entered a training batch
  - VALIDATION data never entered a training batch
  - no loss-bearing batch references EVAL data
  - no loss-bearing batch references VALIDATION data

Generates:
  audits/firewall_audit.jsonl         – per-event trail
  audits/firewall_report.json         – routing summary (compat)
  audits/firewall_audit_report.json   – full ledger inspection report
"""

import os
from pipeline import config
from pipeline.utils import (
    write_json, append_jsonl, read_jsonl, log, now_iso
)

_AUDIT_PATH        = os.path.join(config.AUDITS_DIR, "firewall_audit.jsonl")
_REPORT_PATH       = os.path.join(config.AUDITS_DIR, "firewall_report.json")
_FULL_REPORT_PATH  = os.path.join(config.AUDITS_DIR, "firewall_audit_report.json")

_EVAL_DATASETS = {"EVAL", "VALIDATION"}


# ─── Routing firewall ─────────────────────────────────────────────────────────

def is_eval_shard(shard_manifest: dict) -> bool:
    return shard_manifest.get("dataset", "") in _EVAL_DATASETS


def block_if_eval(shard_manifest: dict) -> bool:
    """
    Return True (safe) or False (blocked).
    Blocked shards are recorded in the audit trail.
    """
    if not is_eval_shard(shard_manifest):
        return True

    shard_id = shard_manifest.get("shard_id", "unknown")
    dataset  = shard_manifest.get("dataset",  "unknown")
    entry    = {
        "event":     "FIREWALL_BLOCK",
        "shard_id":  shard_id,
        "dataset":   dataset,
        "used_for":  "train_attempted",
        "reason":    (f"Dataset '{dataset}' is designated eval/validation; "
                      "must not enter loss-bearing training."),
        "timestamp": now_iso(),
        "blocked":   True,
    }
    append_jsonl(_AUDIT_PATH, entry)
    log(f"[PASS] eval_shard_blocked  shard={shard_id}  dataset={dataset}")
    return False


def record_allowed_shard(shard_manifest: dict, used_for: str = "train"):
    entry = {
        "event":     "SHARD_ALLOWED",
        "shard_id":  shard_manifest.get("shard_id", "unknown"),
        "dataset":   shard_manifest.get("dataset",  "unknown"),
        "used_for":  used_for,
        "timestamp": now_iso(),
    }
    append_jsonl(_AUDIT_PATH, entry)


def record_eval_shard_use(shard_manifest: dict, used_for: str = "eval"):
    entry = {
        "event":     "EVAL_SHARD_USED_FOR_EVAL",
        "shard_id":  shard_manifest.get("shard_id", "unknown"),
        "dataset":   shard_manifest.get("dataset",  "unknown"),
        "used_for":  used_for,
        "timestamp": now_iso(),
    }
    append_jsonl(_AUDIT_PATH, entry)


# ─── Post-training ledger inspection ─────────────────────────────────────────

def generate_firewall_report(all_manifests: dict) -> dict:
    """Routing-level summary (backwards compat)."""
    audit_records = []
    if os.path.exists(_AUDIT_PATH):
        audit_records = read_jsonl(_AUDIT_PATH)

    blocked = [r for r in audit_records if r.get("event") == "FIREWALL_BLOCK"]
    allowed = [r for r in audit_records if r.get("event") == "SHARD_ALLOWED"]
    eval_in_train = [
        r for r in allowed
        if r.get("dataset", "") in _EVAL_DATASETS
    ]

    shard_usage = []
    for ds_name, shards in all_manifests.items():
        is_eval = config.DATASETS[ds_name]["is_eval"]
        for sh in shards:
            shard_usage.append({
                "shard_id": sh["shard_id"],
                "dataset":  ds_name,
                "used_for": "eval" if is_eval else "train",
            })

    passed = len(eval_in_train) == 0
    report = {
        "result":           "PASS" if passed else "FAIL",
        "eval_datasets":    sorted(_EVAL_DATASETS),
        "blocked_attempts": len(blocked),
        "allowed_train":    len(allowed),
        "eval_in_train":    eval_in_train,
        "shard_usage":      shard_usage,
        "audit_trail":      _AUDIT_PATH,
        "generated_at":     now_iso(),
    }
    write_json(_REPORT_PATH, report)
    return report


def generate_full_firewall_audit(
    all_manifests: dict,
    consumption_ledger_path: str,
    learning_ledger_path:    str,
) -> dict:
    """
    Inspect both ledgers to confirm eval data never appeared in any
    loss-bearing batch.

    Checks:
      1. Routing audit (already done above).
      2. Consumption ledger: no shard_source entry belongs to an eval dataset.
      3. Learning ledger: no source_dataset entry is an eval dataset.
    """
    from pipeline.utils import read_jsonl as _rjl
    import os as _os

    def _safe_jsonl(path):
        if not _os.path.exists(path):
            return []
        return _rjl(path)

    # ── 1. routing ────────────────────────────────────────────────────────────
    routing = generate_firewall_report(all_manifests)
    routing_pass = routing["result"] == "PASS"

    # ── 2. consumption ledger ─────────────────────────────────────────────────
    cl_records  = _safe_jsonl(consumption_ledger_path)
    cl_violations = []
    for rec in cl_records:
        for src in rec.get("shard_sources", []):
            if src.get("dataset", "") in _EVAL_DATASETS:
                cl_violations.append({
                    "step":     rec.get("step"),
                    "batch_id": rec.get("batch_id"),
                    "dataset":  src.get("dataset"),
                    "shard_id": src.get("shard_id"),
                })
        for ds in rec.get("dataset_contributions", {}).keys():
            if ds in _EVAL_DATASETS:
                cl_violations.append({
                    "step":     rec.get("step"),
                    "batch_id": rec.get("batch_id"),
                    "dataset":  ds,
                    "source":   "dataset_contributions",
                })

    # ── 3. learning ledger ────────────────────────────────────────────────────
    ll_records    = _safe_jsonl(learning_ledger_path)
    ll_violations = []
    for rec in ll_records:
        for ds in rec.get("source_datasets", []):
            if ds in _EVAL_DATASETS:
                ll_violations.append({
                    "step":     rec.get("step"),
                    "batch_id": rec.get("batch_id"),
                    "loss":     rec.get("loss"),
                    "dataset":  ds,
                })
        for src in rec.get("shard_sources", []):
            if src.get("dataset", "") in _EVAL_DATASETS:
                ll_violations.append({
                    "step":     rec.get("step"),
                    "batch_id": rec.get("batch_id"),
                    "dataset":  src.get("dataset"),
                    "source":   "shard_sources",
                })

    full_pass = routing_pass and not cl_violations and not ll_violations

    report = {
        "result":                    "PASS" if full_pass else "FAIL",
        "routing_check":             routing["result"],
        "consumption_ledger_violations": cl_violations,
        "learning_ledger_violations":    ll_violations,
        "eval_datasets_checked":     sorted(_EVAL_DATASETS),
        "consumption_ledger_steps_checked": len(cl_records),
        "learning_ledger_steps_checked":    len(ll_records),
        "audit_trail":               _AUDIT_PATH,
        "consumption_ledger":        consumption_ledger_path,
        "learning_ledger":           learning_ledger_path,
        "generated_at":              now_iso(),
    }
    write_json(_FULL_REPORT_PATH, report)
    return report
