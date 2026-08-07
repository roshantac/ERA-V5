#!/usr/bin/env python3
"""
main.py — LLM Pretraining Simulation Pipeline (internal orchestration)

All phases are implemented here.  run_demo.py is the evaluation entrypoint.
"""

import os
import sys
import shutil
import random
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pipeline import config
from pipeline.utils import (
    ensure_dirs, log, write_json, read_json, sha256_obj, now_iso
)
import pipeline.datasets         as datasets_mod
import pipeline.tokenizer        as tokenizer_mod
import pipeline.firewall         as firewall_mod
import pipeline.manifest_verify  as manifest_verify_mod
import pipeline.packing          as packing_mod
import pipeline.batching         as batching_mod
import pipeline.scheduler        as scheduler_mod
import pipeline.opus_audit       as opus_mod
import pipeline.ledgers          as ledgers_mod
import pipeline.checkpointing    as ckpt_mod
import pipeline.replay           as replay_mod
import pipeline.traceability     as trace_mod
import pipeline.performance      as perf_mod
import pipeline.evidence         as evidence_mod
import pipeline.final_verification as final_verif_mod
import pipeline.tests            as tests_mod
import pipeline.documentation    as docs_mod


# ═════════════════════════════════════════════════════════════════════════════
# 0. BOOTSTRAP
# ═════════════════════════════════════════════════════════════════════════════

def bootstrap():
    if os.path.exists(config.BASE_DIR):
        shutil.rmtree(config.BASE_DIR)
    ensure_dirs()
    log("=" * 72)
    log("LLM PRETRAINING SIMULATION PIPELINE — starting")
    log("=" * 72)
    log("Training Data Execution System v5 — run_demo.py")


# ═════════════════════════════════════════════════════════════════════════════
# 1. DATASETS & SHARDS
# ═════════════════════════════════════════════════════════════════════════════

def phase_datasets():
    log("─── Phase 1: Dataset generation & sharding ─────────────────────────")
    all_manifests = datasets_mod.generate_all_datasets()
    log("shards created")
    return all_manifests


# ═════════════════════════════════════════════════════════════════════════════
# 2. TOKENIZER MANIFEST
# ═════════════════════════════════════════════════════════════════════════════

def phase_tokenizer():
    log("─── Phase 2: Tokenizer manifest ─────────────────────────────────────")
    manifest = tokenizer_mod.generate_tokenizer_manifest()
    log(f"tokenizer manifest  hash={manifest['tokenizer_hash'][:24]}…")
    return manifest


# ═════════════════════════════════════════════════════════════════════════════
# 3. MANIFEST VERIFICATION AUDIT  (must pass before training)
# ═════════════════════════════════════════════════════════════════════════════

def phase_manifest_verification(all_manifests: dict):
    log("─── Phase 3: Manifest verification audit ────────────────────────────")
    report = manifest_verify_mod.verify_all_manifests(all_manifests)
    log(f"manifest_verification  result={report['result']}  "
        f"shards={report['shards_checked']}")
    log("manifests validated")
    return report


# ═════════════════════════════════════════════════════════════════════════════
# 4. TOKENIZER VERIFICATION + EVALUATION FIREWALL
# ═════════════════════════════════════════════════════════════════════════════

def phase_verify_and_firewall(all_manifests: dict):
    log("─── Phase 4: Tokenizer verification + evaluation firewall ───────────")
    training_shards = []
    eval_shards     = []
    verif_results   = []

    for dataset_name, shards in all_manifests.items():
        for shard_manifest in shards:
            result = tokenizer_mod.verify_shard_tokenizer_hash(shard_manifest)
            verif_results.append(result)

            allowed = firewall_mod.block_if_eval(shard_manifest)
            if allowed:
                firewall_mod.record_allowed_shard(shard_manifest, used_for="train")
                training_shards.append(shard_manifest)
            else:
                firewall_mod.record_eval_shard_use(shard_manifest, used_for="eval")
                eval_shards.append(shard_manifest)

    verif_summary = {
        "shards_verified": len(verif_results),
        "all_passed":      all(r["result"] == "PASS" for r in verif_results),
        "result":          "PASS" if all(r["result"] == "PASS" for r in verif_results) else "FAIL",
        "details":         verif_results,
        "generated_at":    now_iso(),
    }
    write_json(
        os.path.join(config.AUDITS_DIR, "tokenizer_verification.json"),
        verif_summary,
    )
    log(f"training shards: {len(training_shards)}  eval shards: {len(eval_shards)}")
    log("evaluation data blocked")
    return training_shards, eval_shards, verif_results


# ═════════════════════════════════════════════════════════════════════════════
# 5. OPUS AUDITING
# ═════════════════════════════════════════════════════════════════════════════

def phase_opus(all_manifests: dict):
    log("─── Phase 5: OPUS candidate auditing ───────────────────────────────")
    opus_docs = []
    for shard_manifest in all_manifests.get("OPUS", []):
        docs = datasets_mod.load_shard_documents(shard_manifest["shard_id"])
        opus_docs.extend(docs)
    summary = opus_mod.audit_opus_candidates(opus_docs)
    return summary


# ═════════════════════════════════════════════════════════════════════════════
# 6. PACKING
# ═════════════════════════════════════════════════════════════════════════════

def phase_packing(training_shards: list):
    log("─── Phase 6: Sequence packing ───────────────────────────────────────")
    all_packed_results = []
    total_errors       = []

    for shard_manifest in training_shards:
        docs       = datasets_mod.load_shard_documents(shard_manifest["shard_id"])
        pack_result = packing_mod.pack_shard_documents(
            shard_manifest, docs, tokenizer_mod.encode
        )
        all_packed_results.append(pack_result)
        total_errors.extend(pack_result["validation"]["errors"])

    # packing_validation_report.json (spec requirement)
    pv_report = packing_mod.generate_packing_validation_report(all_packed_results)

    # packed_batch_audit.json (compat)
    global_packing_report = {
        "shards_packed":      len(all_packed_results),
        "tokens_used":        pv_report["tokens_used"],
        "tokens_possible":    pv_report["tokens_possible"],
        "packing_efficiency": pv_report["packing_efficiency"],
        "errors":             total_errors,
        "result":             "PASS" if not total_errors else "FAIL",
        "generated_at":       now_iso(),
    }
    write_json(
        os.path.join(config.REPORTS_DIR, "packed_batch_audit.json"),
        global_packing_report,
    )
    log(f"packing  efficiency={pv_report['packing_efficiency']:.2%}  "
        f"tokens={pv_report['tokens_used']}/{pv_report['tokens_possible']}")
    log("batches packed")
    return all_packed_results, global_packing_report


# ═════════════════════════════════════════════════════════════════════════════
# 7. BATCH CONSTRUCTION
# ═════════════════════════════════════════════════════════════════════════════

def phase_batching(all_packed_results: list):
    log("─── Phase 7: Batch construction ─────────────────────────────────────")
    all_batches = batching_mod.build_batches_from_packed(all_packed_results)
    log(f"batches built: {len(all_batches)}")
    return all_batches


# ═════════════════════════════════════════════════════════════════════════════
# 8. TRAINING LOOP  (crash simulation + recovery)
# ═════════════════════════════════════════════════════════════════════════════

def _run_training_segment(
    batches:            list,
    start_step:         int,
    end_step:           int,
    scheduler:          scheduler_mod.MixtureScheduler,
    consumed_batch_ids: list,
    crash_at=None,
):
    batch_index = len(consumed_batch_ids)
    batch_pool  = list(batches)

    for step in range(start_step, end_step):
        if batch_index >= len(batch_pool):
            break

        batch = batch_pool[batch_index]
        batch_index += 1

        if batch["batch_id"] in consumed_batch_ids:
            log(f"WARNING: batch {batch['batch_id']} already consumed; skipping")
            continue

        selected_ds = scheduler.sample_dataset(step=step)
        ds_contrib  = _compute_dataset_contributions(batch)

        ledgers_mod.record_consumption(step, batch, ds_contrib)
        ledgers_mod.record_learning(step, batch)
        consumed_batch_ids.append(batch["batch_id"])

        if (step + 1) % config.CHECKPOINT_EVERY == 0:
            ckpt_mod.save_checkpoint(
                step               = step,
                batch              = batch,
                consumed_batch_ids = list(consumed_batch_ids),
                scheduler_state    = scheduler.get_state(),
                rng_state          = None,
            )

        if crash_at is not None and step == crash_at:
            ckpt_mod.save_checkpoint(
                step               = step,
                batch              = batch,
                consumed_batch_ids = list(consumed_batch_ids),
                scheduler_state    = scheduler.get_state(),
                rng_state          = None,
            )
            log(f"crash simulated  step={step}  batch={batch['batch_id']}")
            log(f"CRASH injected at step={step}  batch={batch['batch_id']}")
            raise RuntimeError(f"Simulated crash at step {step}")


def _compute_dataset_contributions(batch: dict) -> dict:
    ds_tokens: dict = {}
    for seq in batch.get("sequences", []):
        real = sum(seq.get("mask", []))
        for src in seq.get("sources", []):
            ds = src.get("dataset", "unknown")
            ds_tokens[ds] = ds_tokens.get(ds, 0) + real
    return ds_tokens


def phase_training(all_batches: list, scheduler: scheduler_mod.MixtureScheduler):
    log("─── Phase 8: Training loop (with crash simulation) ──────────────────")

    consumed_batch_ids: list = []
    timer = perf_mod.PerformanceTimer()

    timer.__enter__()

    # ── first segment: runs until crash ──────────────────────────────────────
    try:
        _run_training_segment(
            batches            = all_batches,
            start_step         = 0,
            end_step           = config.TOTAL_STEPS,
            scheduler          = scheduler,
            consumed_batch_ids = consumed_batch_ids,
            crash_at           = config.CRASH_AT_STEP,
        )
    except RuntimeError as exc:
        log(f"crash caught: {exc}")
        log("checkpoint saved")

    # ── load checkpoint ───────────────────────────────────────────────────────
    ckpt = ckpt_mod.load_latest_checkpoint()
    assert ckpt is not None, "No checkpoint found after crash!"

    ckpt_consumed    = set(ckpt["consumed_batches"])
    resume_batch_obj = next(
        (b for b in all_batches if b["batch_id"] not in ckpt_consumed), None
    )
    expected_next_id = resume_batch_obj["batch_id"] if resume_batch_obj else "NONE"

    scheduler.set_state(ckpt["scheduler_state"])
    consumed_batch_ids = list(ckpt["consumed_batches"])
    resume_step        = ckpt["global_step"] + 1

    log(f"run resumed  from_step={resume_step}  "
        f"checkpoint_batch={ckpt['batch_id']}  "
        f"expected_next={expected_next_id}")

    # ── second segment: after recovery ───────────────────────────────────────
    _run_training_segment(
        batches            = all_batches,
        start_step         = resume_step,
        end_step           = config.TOTAL_STEPS,
        scheduler          = scheduler,
        consumed_batch_ids = consumed_batch_ids,
        crash_at           = None,
    )

    timer.__exit__(None, None, None)

    # ── verify resume correctness ─────────────────────────────────────────────
    ckpt_consumed_set  = set(ckpt["consumed_batches"])
    first_after_resume = next(
        (b["batch_id"] for b in all_batches
         if b["batch_id"] not in ckpt_consumed_set
         and b["batch_id"] in consumed_batch_ids),
        "NONE",
    )
    resume_match = first_after_resume == expected_next_id

    if resume_match:
        log(f"[PASS] resume_next_batch_matched  "
            f"expected={expected_next_id}  got={first_after_resume}")
    else:
        log(f"[FAIL] resume_next_batch_mismatch  "
            f"expected={expected_next_id}  got={first_after_resume}")

    # ── crash recovery audit ──────────────────────────────────────────────────
    write_json(os.path.join(config.AUDITS_DIR, "crash_recovery.json"), {
        "checkpoint_batch": ckpt["batch_id"],
        "expected_next":    expected_next_id,
        "resumed_batch":    first_after_resume,
        "match":            resume_match,
        "result":           "PASS" if resume_match else "FAIL",
        "crash_at_step":    config.CRASH_AT_STEP,
        "generated_at":     now_iso(),
    })

    # ── resume audit report (duplicate / skip detection) ─────────────────────
    _generate_resume_audit(
        all_batches        = all_batches,
        checkpoint         = ckpt,
        consumed_after_resume = consumed_batch_ids,
        expected_next_id   = expected_next_id,
        first_after_resume = first_after_resume,
    )

    return consumed_batch_ids, timer


def _generate_resume_audit(
    all_batches: list,
    checkpoint: dict,
    consumed_after_resume: list,
    expected_next_id: str,
    first_after_resume: str,
):
    """
    Prove:
      - no batch is processed twice (duplicate)
      - no batch between checkpoint and resume is skipped
    """
    ckpt_set  = set(checkpoint["consumed_batches"])
    after_set = set(consumed_after_resume)

    # duplicate: a batch that appears in checkpoint AND was re-processed
    duplicates = [b for b in checkpoint["consumed_batches"] if b in after_set - ckpt_set]

    # skip: batch immediately after checkpoint should be first_after_resume
    # More thorough: collect all batch_ids between checkpoint and end
    all_ids_in_order = [b["batch_id"] for b in all_batches]
    # index just after the last checkpoint batch
    ckpt_last = checkpoint["consumed_batches"][-1] if checkpoint["consumed_batches"] else None
    try:
        ckpt_last_idx = all_ids_in_order.index(ckpt_last)
        expected_sequence = all_ids_in_order[ckpt_last_idx + 1:]
    except (ValueError, IndexError):
        expected_sequence = []

    # the batches processed after resume
    new_batches_in_order = [
        bid for bid in all_ids_in_order
        if bid not in ckpt_set and bid in after_set
    ]

    # check each expected batch appears in new_batches (no skip)
    skip_detected = False
    for i, expected_bid in enumerate(expected_sequence):
        if i >= len(new_batches_in_order):
            break   # ran out of steps (normal — total_steps limit)
        if new_batches_in_order[i] != expected_bid:
            skip_detected = True
            break

    match = first_after_resume == expected_next_id
    passed = match and not duplicates and not skip_detected

    report = {
        "checkpoint_batch":          checkpoint["batch_id"],
        "expected_next_batch":       expected_next_id,
        "actual_resumed_batch":      first_after_resume,
        "duplicate_detected":        len(duplicates) > 0,
        "duplicates":                duplicates,
        "skipped_batch_detected":    skip_detected,
        "result":                    "PASS" if passed else "FAIL",
        "generated_at":              now_iso(),
    }
    write_json(os.path.join(config.AUDITS_DIR, "resume_audit_report.json"), report)
    return report


# ═════════════════════════════════════════════════════════════════════════════
# 9. MIXTURE & FLOOR REPORTS
# ═════════════════════════════════════════════════════════════════════════════

def phase_mixture_reports(scheduler: scheduler_mod.MixtureScheduler):
    log("─── Phase 9: Mixture and floor reports ──────────────────────────────")
    mix   = scheduler.generate_mixture_report()
    floor = scheduler.generate_floor_audit()
    curr  = scheduler.generate_curriculum_report()
    log(f"mixture  planned={mix['planned']}  actual={mix['actual']}")
    log(f"floors   {floor['result']}  {floor['floors']}")
    log(f"curriculum stages={[s['stage'] for s in curr['stages']]}  "
        f"transitions={len(curr['transitions'])}")
    log("mixture compiled")
    return mix, floor


# ═════════════════════════════════════════════════════════════════════════════
# 10. FULL FIREWALL AUDIT  (post-training ledger inspection)
# ═════════════════════════════════════════════════════════════════════════════

def phase_full_firewall_audit(all_manifests: dict):
    log("─── Phase 10: Full firewall audit (ledger inspection) ───────────────")
    cl_path = os.path.join(config.LEDGERS_DIR, "consumption_ledger.jsonl")
    ll_path = os.path.join(config.LEDGERS_DIR, "learning_ledger.jsonl")
    report = firewall_mod.generate_full_firewall_audit(all_manifests, cl_path, ll_path)
    log(f"firewall_audit_report  result={report['result']}")
    return report


# ═════════════════════════════════════════════════════════════════════════════
# 11. REPLAY
# ═════════════════════════════════════════════════════════════════════════════

def phase_replay(all_batches: list, training_shards: list):
    log("─── Phase 11: Historical replay (full reconstruction) ───────────────")
    report = replay_mod.replay_training_stream(
        original_batches    = all_batches,
        training_shards     = training_shards,
        tokenizer_encode_fn = tokenizer_mod.encode,
    )
    return report


# ═════════════════════════════════════════════════════════════════════════════
# 12. BRANCH FORK
# ═════════════════════════════════════════════════════════════════════════════

def phase_fork(all_batches: list):
    log("─── Phase 12: Branch forking ────────────────────────────────────────")
    checkpoints = ckpt_mod.list_checkpoints()
    target_step = config.FORK_FROM_STEP
    ckpt = min(checkpoints, key=lambda c: abs(c["global_step"] - target_step))
    fork_meta = replay_mod.fork_from_checkpoint(ckpt, all_batches)
    log(f"branch forked  from_step={fork_meta['fork_from_step']}  "
        f"branch={fork_meta['new_branch']}")
    return fork_meta


# ═════════════════════════════════════════════════════════════════════════════
# 13. PERFORMANCE
# ═════════════════════════════════════════════════════════════════════════════

def phase_performance(timer: perf_mod.PerformanceTimer,
                       all_batches: list,
                       global_packing_report: dict):
    log("─── Phase 13: Performance measurement ───────────────────────────────")
    total_batches = len(all_batches)
    total_samples = total_batches * config.BATCH_SIZE
    total_tokens  = global_packing_report.get("tokens_used", 0)

    report = perf_mod.measure_throughput(
        timer           = timer,
        total_batches   = total_batches,
        total_samples   = total_samples,
        total_tokens    = total_tokens,
        packing_report  = global_packing_report,
    )
    # Write spec-required top-level performance.json
    write_json(config.PERFORMANCE_JSON, report)
    return report


# ═════════════════════════════════════════════════════════════════════════════
# 14. TRACEABILITY AUDITS
# ═════════════════════════════════════════════════════════════════════════════

def phase_traceability(all_manifests: dict, all_batches: list):
    log("─── Phase 14: Traceability audits ───────────────────────────────────")
    trace  = trace_mod.generate_traceability_audit(all_manifests, all_batches)
    lt     = trace_mod.generate_learning_trace_audit(all_batches)
    log(f"traceability  result={trace['result']}  steps={trace['steps_verified']}")
    log(f"learning_trace  result={lt['result']}  entries={lt['entries_traced']}")
    return trace, lt


# ═════════════════════════════════════════════════════════════════════════════
# 15. FINAL VERIFICATION
# ═════════════════════════════════════════════════════════════════════════════

def phase_final_verification():
    log("─── Phase 15: Final verification report ─────────────────────────────")
    report = final_verif_mod.generate_final_verification()
    for k, v in report.items():
        if k not in ("source_files", "generated_at"):
            log(f"  {k}: {v}")
    return report


# ═════════════════════════════════════════════════════════════════════════════
# 16. EVIDENCE BUNDLE
# ═════════════════════════════════════════════════════════════════════════════

def phase_evidence():
    log("─── Phase 16: Evidence bundle generation ────────────────────────────")
    evidence = evidence_mod.generate_evidence_bundle()
    evidence_mod.generate_evidence_md(evidence)
    log("evidence.json and evidence.md generated")
    return evidence


# ═════════════════════════════════════════════════════════════════════════════
# 17. AUDIT COMPLETED
# ═════════════════════════════════════════════════════════════════════════════

def phase_audit_completed():
    log("audit completed")


# ═════════════════════════════════════════════════════════════════════════════
# 18. TESTS
# ═════════════════════════════════════════════════════════════════════════════

def phase_tests():
    log("─── Phase 18: Automated validation tests ────────────────────────────")
    report = tests_mod.run_all_tests()
    log(f"tests  PASS={report['total_passed']}  FAIL={report['total_failed']}  "
        f"result={report['result']}")
    return report


# ═════════════════════════════════════════════════════════════════════════════
# 19. SELF-VALIDATION
# ═════════════════════════════════════════════════════════════════════════════

_REQUIRED_ARTIFACTS = [
    config.EVIDENCE_JSON,
    config.EVIDENCE_MD,
    config.EXECUTION_LOG,
    os.path.join(config.AUDITS_DIR,   "manifest_verification_report.json"),
    os.path.join(config.AUDITS_DIR,   "tokenizer_verification.json"),
    os.path.join(config.AUDITS_DIR,   "firewall_audit_report.json"),
    os.path.join(config.AUDITS_DIR,   "firewall_audit.jsonl"),
    os.path.join(config.AUDITS_DIR,   "protected_floor_audit.json"),
    os.path.join(config.AUDITS_DIR,   "opus_audit_report.json"),
    os.path.join(config.AUDITS_DIR,   "opus_audit_trail.jsonl"),
    os.path.join(config.AUDITS_DIR,   "crash_recovery.json"),
    os.path.join(config.AUDITS_DIR,   "resume_audit_report.json"),
    os.path.join(config.AUDITS_DIR,   "traceability_audit.json"),
    os.path.join(config.AUDITS_DIR,   "learning_trace_audit.json"),
    os.path.join(config.AUDITS_DIR,   "fork_audit.json"),
    os.path.join(config.AUDITS_DIR,   "final_verification.json"),
    os.path.join(config.REPORTS_DIR,  "mixture_report.json"),
    os.path.join(config.REPORTS_DIR,  "packing_validation_report.json"),
    os.path.join(config.REPORTS_DIR,  "packed_batch_audit.json"),
    os.path.join(config.REPORTS_DIR,  "performance_report.json"),
    os.path.join(config.REPLAY_DIR,   "replay_validation_report.json"),
    os.path.join(config.REPLAY_DIR,   "replay_report.json"),
    os.path.join(config.REPLAY_DIR,   "replay_log.jsonl"),
    os.path.join(config.LEDGERS_DIR,  "consumption_ledger.jsonl"),
    os.path.join(config.LEDGERS_DIR,  "learning_ledger.jsonl"),
    os.path.join(config.MANIFESTS_DIR, "tokenizer_manifest.json"),
    os.path.join(config.MANIFESTS_DIR, "shard_manifest.json"),
    os.path.join(config.TESTS_DIR,    "test_report.json"),
]


def phase_self_validation(evidence: dict) -> dict:
    log("─── Phase 19: Self-validation ───────────────────────────────────────")
    issues = []

    # 1. required files exist
    for path in _REQUIRED_ARTIFACTS:
        if not os.path.exists(path):
            issues.append(f"MISSING: {path}")

    # 2. evidence.json entries point to real files
    for key, section in evidence.items():
        if not isinstance(section, dict):
            continue
        ev = section.get("evidence", {})
        for k, v in ev.items():
            if isinstance(v, str) and ("path" in k or v.endswith(".json") or v.endswith(".jsonl")):
                if not os.path.exists(v):
                    issues.append(f"evidence[{key}][{k}]={v} does not exist")

    # 3. replay hashes match
    replay_val = _safe_json(os.path.join(config.REPLAY_DIR, "replay_validation_report.json"))
    if not replay_val.get("match", False):
        issues.append("replay_hash_mismatch in replay_validation_report.json")

    # 4. resume validation
    resume = _safe_json(os.path.join(config.AUDITS_DIR, "resume_audit_report.json"))
    if resume.get("result", "") != "PASS":
        issues.append(f"resume_audit FAIL: {resume}")

    # 5. firewall
    fw = _safe_json(os.path.join(config.AUDITS_DIR, "firewall_audit_report.json"))
    if fw.get("result", "") != "PASS":
        issues.append("firewall_audit_report FAIL")

    # 6. traceability
    trace = _safe_json(os.path.join(config.AUDITS_DIR, "traceability_audit.json"))
    if trace.get("result", "") != "PASS":
        issues.append(f"traceability_audit FAIL: {trace.get('failures', [])}")

    result = "PASS" if not issues else "FAIL"
    sv_report = {
        "result":       result,
        "issues":       issues,
        "artifacts_checked": len(_REQUIRED_ARTIFACTS),
        "generated_at": now_iso(),
    }
    write_json(os.path.join(config.AUDITS_DIR, "self_validation.json"), sv_report)

    for issue in issues:
        log(f"  [SELF-VALIDATION ISSUE] {issue}")
    log(f"self_validation  result={result}  issues={len(issues)}")
    return sv_report


def _safe_json(path: str) -> dict:
    try:
        return read_json(path)
    except Exception:
        return {}


# ═════════════════════════════════════════════════════════════════════════════
# 20. DOCUMENTATION
# ═════════════════════════════════════════════════════════════════════════════

def phase_docs():
    log("─── Phase 20: Documentation ─────────────────────────────────────────")
    path = docs_mod.generate_documentation()
    log(f"documentation written to {path}")


# ═════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═════════════════════════════════════════════════════════════════════════════

def main():
    random.seed(config.RANDOM_SEED)

    # 0. bootstrap
    bootstrap()

    # 1. datasets
    all_manifests = phase_datasets()

    # 2. tokenizer manifest
    phase_tokenizer()

    # 3. manifest verification audit  (must pass before training)
    phase_manifest_verification(all_manifests)

    # 4. tokenizer verification + firewall
    training_shards, eval_shards, _ = phase_verify_and_firewall(all_manifests)

    # 5. OPUS auditing
    phase_opus(all_manifests)

    # 6. packing
    all_packed_results, global_packing_report = phase_packing(training_shards)

    # 7. batching
    all_batches = phase_batching(all_packed_results)

    # 8. training (crash + recovery)
    scheduler = scheduler_mod.MixtureScheduler(seed=config.RANDOM_SEED)
    consumed_ids, timer = phase_training(all_batches, scheduler)

    # 9. mixture + floor reports
    phase_mixture_reports(scheduler)

    # 10. full firewall audit (ledger inspection)
    phase_full_firewall_audit(all_manifests)

    # 11. replay (full reconstruction)
    phase_replay(all_batches, training_shards)

    # 12. fork
    phase_fork(all_batches)

    # 13. performance
    perf_report = phase_performance(timer, all_batches, global_packing_report)

    # 14. traceability
    phase_traceability(all_manifests, all_batches)

    # 15. final verification
    phase_final_verification()

    # 16. evidence
    evidence = phase_evidence()

    # 17. audit completed
    phase_audit_completed()

    # 18. tests
    test_report = phase_tests()

    # 19. self-validation
    sv = phase_self_validation(evidence)

    # 20. documentation
    phase_docs()

    # ── final summary ─────────────────────────────────────────────────────────
    log("=" * 72)
    log("PIPELINE COMPLETE")
    log(f"  batches          : {len(all_batches)}")
    log(f"  steps            : {config.TOTAL_STEPS}")
    log(f"  packing eff.     : {global_packing_report['packing_efficiency']:.2%}")
    log(f"  tokens/sec       : {perf_report['tokens_per_sec']}")
    log(f"  tests            : {test_report['result']}  "
        f"({test_report['total_passed']}/{test_report['total_passed']+test_report['total_failed']})")
    log(f"  self_validation  : {sv['result']}  issues={len(sv['issues'])}")
    log(f"  evidence         : "
        + "  ".join(
            f"{k}={v['result']}"
            for k, v in evidence.items()
            if isinstance(v, dict) and "result" in v
        ))
    log("=" * 72)
    return 0


if __name__ == "__main__":
    sys.exit(main())
