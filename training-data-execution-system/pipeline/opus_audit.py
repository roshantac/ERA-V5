"""
OPUS candidate selection auditor.

Decision logic:
  score >= ACCEPT_THRESHOLD  → ACCEPT
  score >= DEFER_THRESHOLD   → DEFER  (re-evaluated if floor override triggered)
  score <  DEFER_THRESHOLD   → REJECT

Protected-floor override:
  If cumulative OPUS share falls below PROTECTED_FLOORS["OPUS"], all deferred
  candidates are force-accepted to restore compliance.

Generates:
  audits/opus_audit_trail.jsonl   – per-candidate decision records
  audits/opus_audit_summary.json  – compat summary
  audits/opus_audit_report.json   – spec-required detailed report
"""

import os
import random
from pipeline import config
from pipeline.utils import write_json, append_jsonl, read_jsonl, log, now_iso, sha256_str

_AUDIT_TRAIL_PATH = os.path.join(config.AUDITS_DIR, "opus_audit_trail.jsonl")
_SUMMARY_PATH     = os.path.join(config.AUDITS_DIR, "opus_audit_summary.json")
_REPORT_PATH      = os.path.join(config.AUDITS_DIR, "opus_audit_report.json")


def _score_document(doc: dict) -> float:
    """Deterministic score derived from doc content hash + per-doc perturbation."""
    text_hash  = sha256_str(doc.get("text", ""))
    base_score = (int(text_hash[:8], 16) % 1000) / 1000.0
    doc_rng    = random.Random(doc["doc_id"])
    perturb    = doc_rng.uniform(-0.05, 0.05)
    return round(min(1.0, max(0.0, base_score + perturb)), 4)


def _decide(score: float) -> str:
    """Return decision string for a given score."""
    if score >= config.OPUS_SCORE_THRESHOLD:
        return "ACCEPT"
    if score >= config.OPUS_DEFER_THRESHOLD:
        return "DEFER"
    return "REJECT"


def audit_opus_candidates(documents: list) -> dict:
    """
    Score every OPUS document.  Apply accept/defer/reject logic.
    Apply protected-floor override if needed.
    Record every decision in the audit trail.
    """
    accepted_ids = []
    deferred_ids = []
    rejected_ids = []
    all_records  = []
    override_applied = False

    # ── first pass: score and decide ─────────────────────────────────────────
    for doc in documents:
        score    = _score_document(doc)
        decision = _decide(score)
        reason   = (
            f"score {score} >= accept_threshold {config.OPUS_SCORE_THRESHOLD}"
            if decision == "ACCEPT" else
            f"score {score} in defer_range [{config.OPUS_DEFER_THRESHOLD}, {config.OPUS_SCORE_THRESHOLD})"
            if decision == "DEFER" else
            f"score {score} < defer_threshold {config.OPUS_DEFER_THRESHOLD}"
        )
        record = {
            "candidate_id": doc["doc_id"],
            "score":        score,
            "decision":     decision,
            "accepted":     decision == "ACCEPT",
            "reason":       reason,
            "override":     False,
            "timestamp":    now_iso(),
        }
        all_records.append(record)
        if decision == "ACCEPT":
            accepted_ids.append(doc["doc_id"])
        elif decision == "DEFER":
            deferred_ids.append(doc["doc_id"])
        else:
            rejected_ids.append(doc["doc_id"])

    # ── second pass: protected-floor override ─────────────────────────────────
    # If OPUS acceptance ratio < floor, force-accept deferred docs
    total         = len(documents)
    opus_floor    = config.PROTECTED_FLOORS.get("OPUS", 0.05)
    current_ratio = len(accepted_ids) / total if total else 0.0

    if config.OPUS_FLOOR_OVERRIDE and current_ratio < opus_floor and deferred_ids:
        override_applied = True
        log(f"OPUS floor override triggered: ratio={current_ratio:.3f} < floor={opus_floor}  "
            f"force-accepting {len(deferred_ids)} deferred docs")
        for rec in all_records:
            if rec["decision"] == "DEFER":
                rec["decision"] = "ACCEPT"
                rec["accepted"] = True
                rec["override"] = True
                rec["reason"]  += f" → floor override (ratio {current_ratio:.3f} < floor {opus_floor})"
                accepted_ids.append(rec["candidate_id"])
        deferred_ids.clear()

    # ── write audit trail ─────────────────────────────────────────────────────
    for rec in all_records:
        append_jsonl(_AUDIT_TRAIL_PATH, rec)

    acceptance_ratio = round(len(accepted_ids) / total, 4) if total else 0.0
    log("OPUS decisions recorded")
    log(f"OPUS audit: {total} candidates, {len(accepted_ids)} accepted, "
        f"{len(deferred_ids)} deferred, {len(rejected_ids)} rejected  "
        f"ratio={acceptance_ratio}  override={override_applied}")

    summary = {
        "candidates_seen":   total,
        "accepted":          len(accepted_ids),
        "deferred":          len(deferred_ids),
        "rejected":          len(rejected_ids),
        "acceptance_ratio":  acceptance_ratio,
        "floor_override":    override_applied,
        "threshold":         config.OPUS_SCORE_THRESHOLD,
        "defer_threshold":   config.OPUS_DEFER_THRESHOLD,
        "audit_trail":       _AUDIT_TRAIL_PATH,
        "result":            "PASS" if total > 0 else "FAIL",
        "generated_at":      now_iso(),
    }
    write_json(_SUMMARY_PATH, summary)

    report = {
        "total_candidates":  total,
        "accepted_count":    len(accepted_ids),
        "deferred_count":    len(deferred_ids),
        "rejected_count":    len(rejected_ids),
        "acceptance_ratio":  acceptance_ratio,
        "floor_override_applied": override_applied,
        "threshold":         config.OPUS_SCORE_THRESHOLD,
        "defer_threshold":   config.OPUS_DEFER_THRESHOLD,
        "result":            "PASS" if total > 0 else "FAIL",
        "candidate_decision_refs": {
            "trail_file":   _AUDIT_TRAIL_PATH,
            "record_count": total,
        },
        "sample_decisions":  all_records[:5],
        "generated_at":      now_iso(),
    }
    write_json(_REPORT_PATH, report)
    return summary


def load_audit_trail() -> list:
    if not os.path.exists(_AUDIT_TRAIL_PATH):
        return []
    return read_jsonl(_AUDIT_TRAIL_PATH)
