"""
Evidence bundle generator — fully artifact-driven, zero hardcoded values.

Reads every report/audit/ledger from disk and assembles evidence.json + evidence.md.
"""

import os
import json
from pipeline import config
from pipeline.utils import read_json, read_jsonl, write_json, now_iso


def _safe_read_json(path: str):
    try:
        return read_json(path)
    except Exception:
        return {}


def _safe_read_jsonl(path: str) -> list:
    try:
        return read_jsonl(path)
    except Exception:
        return []


# ─── Individual evidence collectors ───────────────────────────────────────────

def _ev_tokenizer_integrity() -> dict:
    tok_manifest = _safe_read_json(config.TOKENIZER_MANIFEST)
    verif_path   = os.path.join(config.AUDITS_DIR, "tokenizer_verification.json")
    verif        = _safe_read_json(verif_path)
    manifest_verif_path = os.path.join(config.AUDITS_DIR, "manifest_verification_report.json")
    manifest_verif = _safe_read_json(manifest_verif_path)

    # PASS only if all shards verified AND manifest verification passed
    passed = (
        verif.get("all_passed", False) is True
        and verif.get("shards_verified", 0) > 0
        and manifest_verif.get("result", "") == "PASS"
    )
    return {
        "result": "PASS" if passed else "FAIL",
        "evidence": {
            "tokenizer_hash":          tok_manifest.get("tokenizer_hash", ""),
            "vocab_size":              tok_manifest.get("vocab_size", 0),
            "shards_verified":         verif.get("shards_verified", 0),
            "all_hashes_matched":      verif.get("all_passed", False),
            "manifest_verification":   manifest_verif.get("result", "N/A"),
            "tokenizer_manifest_path": config.TOKENIZER_MANIFEST,
            "verification_path":       verif_path,
            "manifest_verif_path":     manifest_verif_path,
        },
    }


def _ev_evaluation_firewall() -> dict:
    fw_full_path = os.path.join(config.AUDITS_DIR, "firewall_audit_report.json")
    fw_full      = _safe_read_json(fw_full_path)
    audit_path   = os.path.join(config.AUDITS_DIR, "firewall_audit.jsonl")
    audit        = _safe_read_jsonl(audit_path)
    blocked      = [r for r in audit if r.get("event") == "FIREWALL_BLOCK"]

    passed = fw_full.get("result", "") == "PASS"
    return {
        "result": "PASS" if passed else "FAIL",
        "evidence": {
            "blocked_attempts":              len(blocked),
            "blocked_datasets":              list({r["dataset"] for r in blocked}),
            "consumption_ledger_violations": fw_full.get("consumption_ledger_violations", []),
            "learning_ledger_violations":    fw_full.get("learning_ledger_violations", []),
            "routing_check":                 fw_full.get("routing_check", "N/A"),
            "firewall_audit_report_path":    fw_full_path,
            "audit_trail_path":              audit_path,
        },
    }


def _ev_packing_correctness() -> dict:
    val_path  = os.path.join(config.REPORTS_DIR, "packing_validation_report.json")
    val       = _safe_read_json(val_path)
    audit_path = os.path.join(config.REPORTS_DIR, "packed_batch_audit.json")

    passed = (
        val.get("result", "") == "PASS"
        and val.get("tokens_used", 0) > 0
        and val.get("no_token_overlap", False)
        and val.get("no_token_loss", False)
        and val.get("mask_correct", False)
    )
    return {
        "result": "PASS" if passed else "FAIL",
        "evidence": {
            "tokens_used":        val.get("tokens_used", 0),
            "tokens_possible":    val.get("tokens_possible", 0),
            "padding_tokens":     val.get("padding_tokens", 0),
            "packing_efficiency": val.get("packing_efficiency", 0),
            "no_token_overlap":   val.get("no_token_overlap", False),
            "no_token_loss":      val.get("no_token_loss", False),
            "mask_correct":       val.get("mask_correct", False),
            "errors":             val.get("errors", []),
            "packing_validation_path": val_path,
            "packed_batch_audit_path": audit_path,
        },
    }


def _ev_mixture_compliance() -> dict:
    mix_path   = os.path.join(config.REPORTS_DIR, "mixture_report.json")
    floor_path = os.path.join(config.AUDITS_DIR,  "protected_floor_audit.json")
    mix        = _safe_read_json(mix_path)
    floor      = _safe_read_json(floor_path)

    mix_ok   = bool(mix.get("planned")) and bool(mix.get("actual"))
    floor_ok = floor.get("result", "") == "PASS"
    passed   = mix_ok and floor_ok
    return {
        "result": "PASS" if passed else "FAIL",
        "evidence": {
            "planned":             mix.get("planned", {}),
            "actual":              mix.get("actual", {}),
            "deviation":           mix.get("deviation", {}),
            "floor_result":        floor.get("result", "N/A"),
            "floor_details":       floor.get("floors", []),
            "mixture_report_path": mix_path,
            "floor_audit_path":    floor_path,
        },
    }


def _ev_opus_audit() -> dict:
    report_path = os.path.join(config.AUDITS_DIR, "opus_audit_report.json")
    trail_path  = os.path.join(config.AUDITS_DIR, "opus_audit_trail.jsonl")
    report      = _safe_read_json(report_path)
    trail       = _safe_read_jsonl(trail_path)

    passed = report.get("result", "") == "PASS" and len(trail) > 0
    return {
        "result": "PASS" if passed else "FAIL",
        "evidence": {
            "total_candidates":  report.get("total_candidates", 0),
            "accepted":          report.get("accepted_count", 0),
            "rejected":          report.get("rejected_count", 0),
            "acceptance_ratio":  report.get("acceptance_ratio", 0),
            "trail_records":     len(trail),
            "opus_report_path":  report_path,
            "trail_path":        trail_path,
        },
    }


def _ev_crash_recovery() -> dict:
    crash_path  = os.path.join(config.AUDITS_DIR, "crash_recovery.json")
    resume_path = os.path.join(config.AUDITS_DIR, "resume_audit_report.json")
    crash       = _safe_read_json(crash_path)
    resume      = _safe_read_json(resume_path)

    passed = (
        crash.get("result", "") == "PASS"
        and resume.get("result", "") == "PASS"
        and resume.get("duplicate_detected", True) is False
        and resume.get("skipped_batch_detected", True) is False
    )
    return {
        "result": "PASS" if passed else "FAIL",
        "evidence": {
            "checkpoint_batch":         crash.get("checkpoint_batch", ""),
            "expected_next":            crash.get("expected_next", ""),
            "resumed_batch":            crash.get("resumed_batch", ""),
            "match":                    crash.get("match", False),
            "duplicate_detected":       resume.get("duplicate_detected", None),
            "skipped_batch_detected":   resume.get("skipped_batch_detected", None),
            "crash_recovery_path":      crash_path,
            "resume_audit_path":        resume_path,
        },
    }


def _ev_replay() -> dict:
    val_path = os.path.join(config.REPLAY_DIR, "replay_validation_report.json")
    rep_path = os.path.join(config.REPLAY_DIR, "replay_report.json")
    val      = _safe_read_json(val_path)

    passed = val.get("match", False) is True
    return {
        "result": "PASS" if passed else "FAIL",
        "evidence": {
            "original_hash":     val.get("original_hash", ""),
            "replay_hash":       val.get("replay_hash", ""),
            "match":             val.get("match", False),
            "batches_replayed":  val.get("batches_replayed", 0),
            "mismatches":        val.get("per_batch_mismatches", []),
            "replay_validation_path": val_path,
            "replay_report_path":     rep_path,
        },
    }


def _ev_learning_trace() -> dict:
    lt_path    = os.path.join(config.AUDITS_DIR, "learning_trace_audit.json")
    trace_path = os.path.join(config.AUDITS_DIR, "traceability_audit.json")
    lt         = _safe_read_json(lt_path)
    trace      = _safe_read_json(trace_path)

    passed = lt.get("result", "") == "PASS" and trace.get("result", "") == "PASS"
    return {
        "result": "PASS" if passed else "FAIL",
        "evidence": {
            "learning_entries_traced": lt.get("entries_traced", 0),
            "consumption_steps":       trace.get("steps_verified", 0),
            "lt_failures":             lt.get("failures", []),
            "trace_failures":          trace.get("failures", []),
            "learning_trace_path":     lt_path,
            "traceability_path":       trace_path,
            "learning_ledger":         os.path.join(config.LEDGERS_DIR, "learning_ledger.jsonl"),
            "consumption_ledger":      os.path.join(config.LEDGERS_DIR, "consumption_ledger.jsonl"),
        },
    }


def _ev_throughput() -> dict:
    perf_path = os.path.join(config.REPORTS_DIR, "performance_report.json")
    perf      = _safe_read_json(perf_path)

    raw     = perf.get("raw", {})
    derived = perf.get("derived", {})
    passed  = (
        perf.get("result", "") == "PASS"
        and raw.get("total_tokens", 0) > 0
        and raw.get("elapsed_seconds", 0) > 0
    )
    return {
        "result": "PASS" if passed else "FAIL",
        "evidence": {
            "raw_measurements":          raw,
            "derived_metrics":           derived,
            "batches_per_sec":           perf.get("batches_per_sec", 0),
            "samples_per_sec":           perf.get("samples_per_sec", 0),
            "tokens_per_sec":            perf.get("tokens_per_sec", 0),
            "packing_efficiency":        perf.get("packing_efficiency", 0),
            "avg_sequence_utilization":  perf.get("avg_sequence_utilization", 0),
            "performance_report_path":   perf_path,
        },
    }


# ─── Bundle ───────────────────────────────────────────────────────────────────

def generate_evidence_bundle() -> dict:
    evidence = {
        "generated_at":         now_iso(),
        "tokenizer_integrity":  _ev_tokenizer_integrity(),
        "evaluation_firewall":  _ev_evaluation_firewall(),
        "packing_correctness":  _ev_packing_correctness(),
        "mixture_compliance":   _ev_mixture_compliance(),
        "opus_audit":           _ev_opus_audit(),
        "crash_recovery":       _ev_crash_recovery(),
        "replay":               _ev_replay(),
        "learning_trace":       _ev_learning_trace(),
        "throughput":           _ev_throughput(),
    }
    write_json(config.EVIDENCE_JSON, evidence)
    return evidence


# ─── Markdown report ──────────────────────────────────────────────────────────

def generate_evidence_md(evidence: dict):
    def r(key):
        return evidence.get(key, {}).get("result", "N/A")

    def e(key):
        ev = evidence.get(key, {}).get("evidence", {})
        parts = []
        for k, v in ev.items():
            if isinstance(v, (str, int, float, bool)) and len(str(v)) < 80:
                parts.append(f"{k}={v}")
        return "; ".join(parts[:3])

    def detail(key):
        return "```json\n" + json.dumps(evidence.get(key, {}), indent=2) + "\n```"

    lines = [
        "# LLM Pretraining Simulation — Evidence Report",
        "",
        f"Generated: {evidence.get('generated_at', '')}",
        "",
        "| Requirement | Result | Evidence |",
        "|---|---|---|",
        f"| Tokenizer integrity | **{r('tokenizer_integrity')}** | {e('tokenizer_integrity')} |",
        f"| Evaluation firewall | **{r('evaluation_firewall')}** | {e('evaluation_firewall')} |",
        f"| Packing correctness | **{r('packing_correctness')}** | {e('packing_correctness')} |",
        f"| Mixture compliance  | **{r('mixture_compliance')}** | {e('mixture_compliance')} |",
        f"| OPUS audit trail    | **{r('opus_audit')}** | {e('opus_audit')} |",
        f"| Crash recovery      | **{r('crash_recovery')}** | {e('crash_recovery')} |",
        f"| Replay              | **{r('replay')}** | {e('replay')} |",
        f"| Learning trace      | **{r('learning_trace')}** | {e('learning_trace')} |",
        f"| Throughput          | **{r('throughput')}** | {e('throughput')} |",
        "",
        "---",
        "",
    ]
    for key in [
        "tokenizer_integrity", "evaluation_firewall", "packing_correctness",
        "mixture_compliance", "opus_audit", "crash_recovery",
        "replay", "learning_trace", "throughput",
    ]:
        title = key.replace("_", " ").title()
        lines += [f"## {title}", "", detail(key), ""]

    with open(config.EVIDENCE_MD, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))
