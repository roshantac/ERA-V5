"""
Final verification report generator.

Derives all PASS/FAIL values exclusively by reading generated artifacts.
No hardcoded results.

Generates:
  audits/final_verification.json
"""

import os
from pipeline import config
from pipeline.utils import write_json, now_iso

_FINAL_VERIF_PATH = os.path.join(config.AUDITS_DIR, "final_verification.json")


def _safe_json(path: str):
    try:
        from pipeline.utils import read_json
        return read_json(path)
    except Exception:
        return {}


def _safe_jsonl(path: str) -> list:
    try:
        from pipeline.utils import read_jsonl
        return read_jsonl(path)
    except Exception:
        return []


def _result_from_file(path: str, key: str = "result") -> str:
    """Read the *key* field from a JSON file; return 'FAIL' if unreadable."""
    data = _safe_json(path)
    val  = data.get(key, "")
    return val if val in ("PASS", "FAIL") else "FAIL"


def generate_final_verification() -> dict:
    """
    Read every relevant audit file and derive a single PASS/FAIL per
    requirement.  This function must never return a hardcoded value.
    """
    # ── 1. tokenizer_integrity ────────────────────────────────────────────────
    tok_verif = _safe_json(os.path.join(config.AUDITS_DIR, "tokenizer_verification.json"))
    tok_ok    = (
        tok_verif.get("all_passed", False) is True
        and tok_verif.get("shards_verified", 0) > 0
    )
    tok_result = "PASS" if tok_ok else "FAIL"

    # ── 2. evaluation_firewall ────────────────────────────────────────────────
    fw_full   = _safe_json(os.path.join(config.AUDITS_DIR, "firewall_audit_report.json"))
    fw_result = fw_full.get("result", "FAIL")
    if fw_result not in ("PASS", "FAIL"):
        fw_result = "FAIL"

    # ── 3. packing_correctness ────────────────────────────────────────────────
    pk_val    = _safe_json(os.path.join(config.REPORTS_DIR, "packing_validation_report.json"))
    pk_result = pk_val.get("result", "FAIL")
    if pk_result not in ("PASS", "FAIL"):
        pk_result = "FAIL"

    # ── 4. mixture_compliance ─────────────────────────────────────────────────
    floor_audit = _safe_json(os.path.join(config.AUDITS_DIR, "protected_floor_audit.json"))
    mix_report  = _safe_json(os.path.join(config.REPORTS_DIR, "mixture_report.json"))
    mix_ok = (
        floor_audit.get("result", "") == "PASS"
        and bool(mix_report.get("planned"))
        and bool(mix_report.get("actual"))
    )
    mix_result = "PASS" if mix_ok else "FAIL"

    # ── 5. opus_audit ─────────────────────────────────────────────────────────
    opus_rep  = _safe_json(os.path.join(config.AUDITS_DIR, "opus_audit_report.json"))
    trail     = _safe_jsonl(os.path.join(config.AUDITS_DIR, "opus_audit_trail.jsonl"))
    opus_ok   = (
        opus_rep.get("result", "") == "PASS"
        and len(trail) > 0
    )
    opus_result = "PASS" if opus_ok else "FAIL"

    # ── 6. crash_recovery ────────────────────────────────────────────────────
    crash_rep  = _safe_json(os.path.join(config.AUDITS_DIR, "crash_recovery.json"))
    resume_rep = _safe_json(os.path.join(config.AUDITS_DIR, "resume_audit_report.json"))
    crash_ok   = (
        crash_rep.get("result", "") == "PASS"
        and resume_rep.get("result", "") == "PASS"
    )
    crash_result = "PASS" if crash_ok else "FAIL"

    # ── 7. replay ────────────────────────────────────────────────────────────
    replay_val  = _safe_json(os.path.join(config.REPLAY_DIR, "replay_validation_report.json"))
    replay_ok   = replay_val.get("match", False) is True
    replay_result = "PASS" if replay_ok else "FAIL"

    # ── 8. learning_trace ────────────────────────────────────────────────────
    lt_audit   = _safe_json(os.path.join(config.AUDITS_DIR, "learning_trace_audit.json"))
    trace_aud  = _safe_json(os.path.join(config.AUDITS_DIR, "traceability_audit.json"))
    lt_ok      = (
        lt_audit.get("result", "") == "PASS"
        and trace_aud.get("result", "") == "PASS"
    )
    lt_result = "PASS" if lt_ok else "FAIL"

    # ── 9. throughput ─────────────────────────────────────────────────────────
    perf_rep   = _safe_json(os.path.join(config.REPORTS_DIR, "performance_report.json"))
    perf_ok    = (
        perf_rep.get("result", "") == "PASS"
        and perf_rep.get("raw", {}).get("total_tokens", 0) > 0
    )
    tp_result = "PASS" if perf_ok else "FAIL"

    report = {
        "tokenizer_integrity": tok_result,
        "evaluation_firewall": fw_result,
        "packing_correctness": pk_result,
        "mixture_compliance":  mix_result,
        "opus_audit":          opus_result,
        "crash_recovery":      crash_result,
        "replay":              replay_result,
        "learning_trace":      lt_result,
        "throughput":          tp_result,
        "source_files": {
            "tokenizer_integrity": os.path.join(config.AUDITS_DIR, "tokenizer_verification.json"),
            "evaluation_firewall": os.path.join(config.AUDITS_DIR, "firewall_audit_report.json"),
            "packing_correctness": os.path.join(config.REPORTS_DIR, "packing_validation_report.json"),
            "mixture_compliance":  os.path.join(config.AUDITS_DIR, "protected_floor_audit.json"),
            "opus_audit":          os.path.join(config.AUDITS_DIR, "opus_audit_report.json"),
            "crash_recovery":      os.path.join(config.AUDITS_DIR, "resume_audit_report.json"),
            "replay":              os.path.join(config.REPLAY_DIR,  "replay_validation_report.json"),
            "learning_trace":      os.path.join(config.AUDITS_DIR, "learning_trace_audit.json"),
            "throughput":          os.path.join(config.REPORTS_DIR, "performance_report.json"),
        },
        "generated_at": now_iso(),
    }
    write_json(_FINAL_VERIF_PATH, report)
    return report
