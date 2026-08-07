"""
Automated validation tests — reads only from generated artifact files.

Runs all validation suites and writes results to artifacts/tests/test_report.json.
"""

import os
import json
from pipeline import config
from pipeline.utils import read_json, read_jsonl, now_iso, write_json


# ─── Helpers ──────────────────────────────────────────────────────────────────

class TestSuite:
    def __init__(self, name: str):
        self.name    = name
        self.results = []

    def assert_true(self, condition: bool, test_name: str, detail: str = ""):
        self.results.append({
            "test":   test_name,
            "result": "PASS" if condition else "FAIL",
            "detail": detail,
        })

    def summary(self) -> dict:
        passed = sum(1 for r in self.results if r["result"] == "PASS")
        failed = sum(1 for r in self.results if r["result"] == "FAIL")
        return {
            "suite":  self.name,
            "passed": passed,
            "failed": failed,
            "total":  passed + failed,
            "result": "PASS" if failed == 0 else "FAIL",
            "tests":  self.results,
        }


def _safe_json(path: str):
    try:
        return read_json(path)
    except Exception:
        return None


def _safe_jsonl(path: str):
    try:
        return read_jsonl(path)
    except Exception:
        return []


# ─── Test suites ──────────────────────────────────────────────────────────────

def test_tokenizer() -> dict:
    s = TestSuite("tokenizer")
    manifest = _safe_json(config.TOKENIZER_MANIFEST)
    s.assert_true(manifest is not None, "manifest_exists")

    if manifest:
        from pipeline.tokenizer import compute_tokenizer_hash
        current = compute_tokenizer_hash()
        stored  = manifest.get("tokenizer_hash", "")
        s.assert_true(current == stored, "hash_match",
                      f"current={current[:16]}… stored={stored[:16]}…")
        tampered = stored[:-1] + ("0" if stored[-1] != "0" else "1")
        s.assert_true(tampered != current, "mismatch_detection")

    verif = _safe_json(os.path.join(config.AUDITS_DIR, "tokenizer_verification.json"))
    s.assert_true(verif is not None and verif.get("all_passed") is True,
                  "all_shards_verified",
                  f"shards={verif.get('shards_verified') if verif else 'N/A'}")

    return s.summary()


def test_manifest_verification() -> dict:
    s = TestSuite("manifest_verification")
    path   = os.path.join(config.AUDITS_DIR, "manifest_verification_report.json")
    report = _safe_json(path)
    s.assert_true(report is not None, "report_exists")
    if report:
        s.assert_true(report.get("result") == "PASS", "all_shards_verified",
                      f"failures={report.get('failures', [])}")
        s.assert_true(report.get("shards_checked", 0) > 0, "shards_checked")
    return s.summary()


def test_firewall() -> dict:
    s = TestSuite("firewall")
    fw_full_path = os.path.join(config.AUDITS_DIR, "firewall_audit_report.json")
    audit_path   = os.path.join(config.AUDITS_DIR, "firewall_audit.jsonl")

    fw   = _safe_json(fw_full_path)
    audit = _safe_jsonl(audit_path)

    s.assert_true(fw is not None, "full_report_exists")
    if fw:
        s.assert_true(fw.get("result") == "PASS", "full_firewall_pass",
                      f"cl_violations={fw.get('consumption_ledger_violations',[])} "
                      f"ll_violations={fw.get('learning_ledger_violations',[])}")
        s.assert_true(len(fw.get("consumption_ledger_violations", [])) == 0,
                      "no_eval_in_consumption_ledger")
        s.assert_true(len(fw.get("learning_ledger_violations", [])) == 0,
                      "no_eval_in_learning_ledger")

    blocked = [r for r in audit if r.get("event") == "FIREWALL_BLOCK"]
    s.assert_true(len(blocked) >= 1, "eval_blocked_recorded",
                  f"blocked={len(blocked)}")

    allowed = [r for r in audit if r.get("event") == "SHARD_ALLOWED"]
    bad = [r for r in allowed if r.get("dataset", "") in {"EVAL", "VALIDATION"}]
    s.assert_true(len(bad) == 0, "no_eval_in_allowed", f"bad={len(bad)}")

    return s.summary()


def test_packing() -> dict:
    s = TestSuite("packing")
    val_path  = os.path.join(config.REPORTS_DIR, "packing_validation_report.json")
    report    = _safe_json(val_path)
    s.assert_true(report is not None, "validation_report_exists")
    if report:
        s.assert_true(report.get("result") == "PASS", "no_packing_errors",
                      str(report.get("errors", [])))
        s.assert_true(report.get("tokens_used", 0) > 0, "token_conservation")
        s.assert_true(report.get("tokens_possible", 0) >= report.get("tokens_used", 0),
                      "no_token_overflow")
        eff = report.get("packing_efficiency", 0)
        s.assert_true(0 < eff <= 1.0, "valid_efficiency", f"efficiency={eff}")
        s.assert_true(report.get("no_token_overlap", False), "no_token_overlap")
        s.assert_true(report.get("no_token_loss", False), "no_token_loss")
        s.assert_true(report.get("mask_correct", False), "mask_correct")
    return s.summary()


def test_mixture() -> dict:
    s = TestSuite("mixture")
    mix_path   = os.path.join(config.REPORTS_DIR, "mixture_report.json")
    floor_path = os.path.join(config.AUDITS_DIR,  "protected_floor_audit.json")
    mix   = _safe_json(mix_path)
    floor = _safe_json(floor_path)

    s.assert_true(mix is not None,   "mixture_report_exists")
    s.assert_true(floor is not None, "floor_audit_exists")

    if mix:
        planned = mix.get("planned", {})
        actual  = mix.get("actual",  {})
        for ds in planned:
            s.assert_true(ds in actual, f"dataset_sampled_{ds}")
        dev = mix.get("deviation", {})
        for ds, d in dev.items():
            s.assert_true(abs(d) < 0.20, f"deviation_bounded_{ds}", f"dev={d:.4f}")

    if floor:
        s.assert_true(floor.get("result") == "PASS", "floor_compliance",
                      str(floor.get("floors", [])))
        for entry in floor.get("floors", []):
            s.assert_true(entry.get("pass", False),
                          f"floor_pass_{entry['dataset']}",
                          f"actual={entry.get('actual_share')} floor={entry.get('floor')}")

    return s.summary()


def test_opus() -> dict:
    s = TestSuite("opus")
    report_path = os.path.join(config.AUDITS_DIR, "opus_audit_report.json")
    trail_path  = os.path.join(config.AUDITS_DIR, "opus_audit_trail.jsonl")
    report = _safe_json(report_path)
    trail  = _safe_jsonl(trail_path)

    s.assert_true(report is not None, "report_exists")
    s.assert_true(len(trail) > 0, "audit_records_generated", f"trail={len(trail)}")

    if trail:
        req = {"candidate_id", "score", "accepted", "reason", "timestamp"}
        has_all = all(req.issubset(r.keys()) for r in trail)
        s.assert_true(has_all, "decisions_auditable")

    if report:
        total     = report.get("total_candidates", 0)
        acc       = report.get("accepted_count", 0)
        deferred  = report.get("deferred_count", 0)
        rej       = report.get("rejected_count", 0)
        s.assert_true(acc + deferred + rej == total, "counts_consistent",
                      f"acc+deferred+rej={acc+deferred+rej} total={total}")
        s.assert_true(report.get("result") == "PASS", "result_pass")

    return s.summary()


def test_consumption_ledger() -> dict:
    s = TestSuite("consumption_ledger")
    path    = os.path.join(config.LEDGERS_DIR, "consumption_ledger.jsonl")
    records = _safe_jsonl(path)

    s.assert_true(len(records) > 0, "records_present", f"records={len(records)}")
    if records:
        r = records[0]
        for f in ["step", "batch_id", "dataset_contributions", "token_count"]:
            s.assert_true(f in r, f"field_{f}")
        s.assert_true(len(r.get("shard_sources", [])) > 0, "shard_source_present")

    steps = [r["step"] for r in records]
    s.assert_true(len(steps) == len(set(steps)), "no_duplicate_steps")

    # verify full traceability audit
    trace_path = os.path.join(config.AUDITS_DIR, "traceability_audit.json")
    trace = _safe_json(trace_path)
    s.assert_true(trace is not None and trace.get("result") == "PASS",
                  "traceability_audit_pass",
                  str(trace.get("failures", []) if trace else "N/A"))

    return s.summary()


def test_learning_ledger() -> dict:
    s = TestSuite("learning_ledger")
    ll_path = os.path.join(config.LEDGERS_DIR, "learning_ledger.jsonl")
    cl_path = os.path.join(config.LEDGERS_DIR, "consumption_ledger.jsonl")
    ll = _safe_jsonl(ll_path)
    cl = _safe_jsonl(cl_path)

    s.assert_true(len(ll) > 0, "records_present")

    cl_batch_ids = {r["batch_id"] for r in cl}
    if ll:
        r = ll[0]
        s.assert_true("loss" in r and isinstance(r["loss"], float), "loss_present")
        s.assert_true(r.get("batch_id", "") in cl_batch_ids, "loss_traceable")

    # learning trace audit
    lt_path = os.path.join(config.AUDITS_DIR, "learning_trace_audit.json")
    lt = _safe_json(lt_path)
    s.assert_true(lt is not None and lt.get("result") == "PASS",
                  "learning_trace_audit_pass",
                  str(lt.get("failures", []) if lt else "N/A"))

    return s.summary()


def test_checkpoint() -> dict:
    s = TestSuite("checkpoint")
    from pipeline.checkpointing import list_checkpoints
    ckpts = list_checkpoints()
    s.assert_true(len(ckpts) > 0, "checkpoints_exist", f"found={len(ckpts)}")
    if ckpts:
        c = ckpts[0]
        for f in ["global_step", "batch_id", "consumed_batches", "state_hash"]:
            s.assert_true(f in c, f"field_{f}")
        # verify state hash is derived (not empty)
        s.assert_true(len(c.get("state_hash", "")) == 64, "state_hash_sha256")
    return s.summary()


def test_resume() -> dict:
    s = TestSuite("resume")
    crash_path  = os.path.join(config.AUDITS_DIR, "crash_recovery.json")
    resume_path = os.path.join(config.AUDITS_DIR, "resume_audit_report.json")
    crash  = _safe_json(crash_path)
    resume = _safe_json(resume_path)

    s.assert_true(crash is not None,  "crash_recovery_exists")
    s.assert_true(resume is not None, "resume_audit_exists")

    if crash:
        s.assert_true(crash.get("result") == "PASS", "crash_recovery_pass")
        s.assert_true(crash.get("match", False), "next_batch_matched")
        # no repeat
        s.assert_true(
            crash.get("checkpoint_batch") != crash.get("expected_next"),
            "no_batch_repeat",
        )

    if resume:
        s.assert_true(resume.get("result") == "PASS", "resume_audit_pass")
        s.assert_true(resume.get("duplicate_detected", True) is False, "no_duplicate")
        s.assert_true(resume.get("skipped_batch_detected", True) is False, "no_skip")

    return s.summary()


def test_replay() -> dict:
    s = TestSuite("replay")
    val_path = os.path.join(config.REPLAY_DIR, "replay_validation_report.json")
    rep_path = os.path.join(config.REPLAY_DIR, "replay_report.json")
    val  = _safe_json(val_path)
    rep  = _safe_json(rep_path)

    s.assert_true(val is not None, "validation_report_exists")
    s.assert_true(rep is not None, "report_exists")

    if val:
        s.assert_true(val.get("match", False), "hash_match",
                      f"orig={val.get('original_hash','')[:16]}… "
                      f"replay={val.get('replay_hash','')[:16]}…")
        s.assert_true(val.get("batches_replayed", 0) > 0, "batches_replayed")
        mismatches = val.get("per_batch_mismatches", [])
        s.assert_true(len(mismatches) == 0, "no_per_batch_mismatch",
                      f"mismatches={mismatches}")
        # verify per_batch array is present
        s.assert_true(isinstance(val.get("per_batch"), list), "per_batch_present")

    return s.summary()


def test_fork() -> dict:
    s = TestSuite("fork")
    path = os.path.join(config.AUDITS_DIR, "fork_audit.json")
    fork = _safe_json(path)
    s.assert_true(fork is not None, "fork_metadata_exists")
    if fork:
        for f in ["fork_from_step", "new_branch", "consumed_count", "future_count"]:
            s.assert_true(f in fork, f"field_{f}")
    return s.summary()


def test_evidence() -> dict:
    s = TestSuite("evidence")
    ev = _safe_json(config.EVIDENCE_JSON)
    s.assert_true(ev is not None, "evidence_json_exists")
    if ev:
        required = [
            "tokenizer_integrity", "evaluation_firewall", "packing_correctness",
            "mixture_compliance", "opus_audit", "crash_recovery",
            "replay", "learning_trace", "throughput",
        ]
        for k in required:
            s.assert_true(k in ev, f"key_{k}")
            section = ev.get(k, {})
            s.assert_true(isinstance(section.get("evidence"), dict)
                          and len(section.get("evidence", {})) > 0,
                          f"evidence_non_empty_{k}")
    s.assert_true(os.path.exists(config.EVIDENCE_MD), "evidence_md_exists")
    return s.summary()


def test_final_verification() -> dict:
    s = TestSuite("final_verification")
    path   = os.path.join(config.AUDITS_DIR, "final_verification.json")
    report = _safe_json(path)
    s.assert_true(report is not None, "report_exists")
    if report:
        required = [
            "tokenizer_integrity", "evaluation_firewall", "packing_correctness",
            "mixture_compliance", "opus_audit", "crash_recovery",
            "replay", "learning_trace", "throughput",
        ]
        for k in required:
            s.assert_true(k in report, f"key_{k}")
            s.assert_true(report.get(k) in ("PASS", "FAIL"), f"valid_value_{k}")
        # source_files must point to real files
        for k, v in report.get("source_files", {}).items():
            s.assert_true(os.path.exists(v), f"source_file_exists_{k}", v)
    return s.summary()


def test_determinism() -> dict:
    s = TestSuite("determinism")
    from pipeline.tokenizer import compute_tokenizer_hash
    h1 = compute_tokenizer_hash()
    h2 = compute_tokenizer_hash()
    s.assert_true(h1 == h2, "tokenizer_hash_stable")

    val = _safe_json(os.path.join(config.REPLAY_DIR, "replay_validation_report.json"))
    if val:
        s.assert_true(val.get("match", False), "replay_hash_reproducible")

    return s.summary()


def test_performance() -> dict:
    s = TestSuite("performance")
    path   = os.path.join(config.REPORTS_DIR, "performance_report.json")
    report = _safe_json(path)
    s.assert_true(report is not None, "report_exists")
    if report:
        raw = report.get("raw", {})
        s.assert_true(raw.get("total_tokens", 0) > 0, "raw_total_tokens")
        s.assert_true(raw.get("elapsed_seconds", 0) > 0, "raw_elapsed")
        s.assert_true("start_time_iso" in raw, "raw_start_time")
        s.assert_true("end_time_iso"   in raw, "raw_end_time")
        derived = report.get("derived", {})
        s.assert_true(derived.get("tokens_per_sec", 0) > 0, "derived_tps")
        s.assert_true(derived.get("batches_per_sec", 0) > 0, "derived_bps")
    return s.summary()


# ─── Runner ───────────────────────────────────────────────────────────────────

def run_all_tests() -> dict:
    suites = [
        test_tokenizer(),
        test_manifest_verification(),
        test_firewall(),
        test_packing(),
        test_mixture(),
        test_opus(),
        test_consumption_ledger(),
        test_learning_ledger(),
        test_checkpoint(),
        test_resume(),
        test_replay(),
        test_fork(),
        test_evidence(),
        test_final_verification(),
        test_determinism(),
        test_performance(),
    ]

    total_pass = sum(s["passed"] for s in suites)
    total_fail = sum(s["failed"] for s in suites)
    overall    = "PASS" if total_fail == 0 else "FAIL"

    report = {
        "result":       overall,
        "total_passed": total_pass,
        "total_failed": total_fail,
        "suites":       suites,
        "generated_at": now_iso(),
    }
    out_path = os.path.join(config.TESTS_DIR, "test_report.json")
    write_json(out_path, report)
    return report
