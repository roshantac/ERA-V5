"""
Full-reconstruction replay.

Replay must reconstruct the entire training stream from stored artifacts:
  - shard order
  - document order within shards
  - packed sequence contents (re-derived from documents + tokenizer)
  - batch groupings (re-derived from packed sequences)
  - batch hashes

Every original_batch_hash is compared to the corresponding replay_batch_hash.
A single mismatch causes replay failure.

Generates:
  replay/replay_validation_report.json   – per-batch comparison
  replay/replay_report.json              – aggregate (kept for backwards compat)
  replay/replay_log.jsonl               – per-batch audit trail
"""

import os
from pipeline import config
from pipeline.utils import sha256_obj, write_json, append_jsonl, log, now_iso

_REPLAY_LOG_PATH        = os.path.join(config.REPLAY_DIR, "replay_log.jsonl")
_REPLAY_REPORT          = os.path.join(config.REPLAY_DIR, "replay_report.json")
_REPLAY_VALIDATION_PATH = os.path.join(config.REPLAY_DIR, "replay_validation_report.json")
_FORK_AUDIT_PATH        = os.path.join(config.AUDITS_DIR, "fork_audit.json")


# ─── Batch fingerprint ────────────────────────────────────────────────────────

def _batch_fingerprint(batch: dict) -> str:
    """Stable hash (excludes mutable fields like created_at)."""
    return sha256_obj({
        "batch_id":   batch["batch_id"],
        "batch_index": batch["batch_index"],
        "sample_ids": batch["sample_ids"],
        "input_hash": batch["input_hash"],
        "seq_count":  batch["seq_count"],
    })


# ─── Full reconstruction replay ───────────────────────────────────────────────

def replay_training_stream(
    original_batches: list,
    training_shards: list,
    tokenizer_encode_fn,
) -> dict:
    """
    Reconstruct the full training stream from first principles and compare
    every batch hash to the original.

    Steps:
      1. Re-tokenise and re-pack every training shard (same order as original).
      2. Re-assemble batches from packed sequences.
      3. Re-derive batch fingerprints.
      4. Compare original fingerprints vs replay fingerprints.

    Parameters
    ----------
    original_batches    : list of batch dicts produced during training
    training_shards     : list of shard manifest dicts (training set only,
                          same order as used during packing)
    tokenizer_encode_fn : tokenizer.encode callable
    """
    import pipeline.datasets as ds_mod
    import pipeline.packing  as pack_mod
    import pipeline.batching as batch_mod

    # ── step 1: re-pack all training shards ──────────────────────────────────
    replay_packed_results = []
    for shard_manifest in training_shards:
        docs = ds_mod.load_shard_documents(shard_manifest["shard_id"])
        result = pack_mod.pack_shard_documents(
            shard_manifest, docs, tokenizer_encode_fn
        )
        replay_packed_results.append(result)

    # ── step 2: re-assemble batches ───────────────────────────────────────────
    replay_batches = batch_mod.build_batches_from_packed(replay_packed_results)

    # ── step 3: compare per-batch ─────────────────────────────────────────────
    per_batch = []
    all_match = True

    # Align by batch_index; both lists should be identical in length
    max_len = max(len(original_batches), len(replay_batches))
    for i in range(max_len):
        if i >= len(original_batches):
            all_match = False
            per_batch.append({
                "batch_index":    i,
                "batch_id":       "MISSING_IN_ORIGINAL",
                "original_hash":  "",
                "replay_hash":    _batch_fingerprint(replay_batches[i]),
                "match":          False,
                "note":           "extra batch in replay",
            })
            continue
        if i >= len(replay_batches):
            all_match = False
            per_batch.append({
                "batch_index":    i,
                "batch_id":       original_batches[i]["batch_id"],
                "original_hash":  _batch_fingerprint(original_batches[i]),
                "replay_hash":    "",
                "match":          False,
                "note":           "missing batch in replay",
            })
            continue

        orig  = original_batches[i]
        rep   = replay_batches[i]
        oh    = _batch_fingerprint(orig)
        rh    = _batch_fingerprint(rep)
        match = (oh == rh)
        if not match:
            all_match = False

        record = {
            "batch_index":    i,
            "batch_id":       orig["batch_id"],
            "original_hash":  oh,
            "replay_hash":    rh,
            "match":          match,
            # full content for independent verification
            "original_sample_ids": orig["sample_ids"],
            "replay_sample_ids":   rep["sample_ids"],
            "original_input_hash": orig["input_hash"],
            "replay_input_hash":   rep["input_hash"],
        }
        per_batch.append(record)
        append_jsonl(_REPLAY_LOG_PATH, record)

    # ── step 4: aggregate hash ────────────────────────────────────────────────
    original_agg = sha256_obj([b["original_hash"] for b in per_batch if b["original_hash"]])
    replay_agg   = sha256_obj([b["replay_hash"]   for b in per_batch if b["replay_hash"]])
    agg_match    = (original_agg == replay_agg) and all_match

    # ── log mandatory events ──────────────────────────────────────────────────
    if agg_match:
        log("[PASS] replay_hash_matched")
    else:
        mismatches = [b["batch_id"] for b in per_batch if not b["match"]]
        log(f"[FAIL] replay_hash_mismatch  mismatches={mismatches}")

    log("historical stream replayed")

    # ── write validation report ───────────────────────────────────────────────
    validation_report = {
        "result":              "PASS" if agg_match else "FAIL",
        "original_hash":       original_agg,
        "replay_hash":         replay_agg,
        "match":               agg_match,
        "batches_original":    len(original_batches),
        "batches_replayed":    len(replay_batches),
        "per_batch_mismatches": [b["batch_id"] for b in per_batch if not b["match"]],
        "per_batch":           per_batch,
        "generated_at":        now_iso(),
    }
    write_json(_REPLAY_VALIDATION_PATH, validation_report)

    # backwards-compat report (same aggregate fields)
    write_json(_REPLAY_REPORT, {
        "original_hash":    original_agg,
        "replay_hash":      replay_agg,
        "match":            agg_match,
        "result":           "PASS" if agg_match else "FAIL",
        "batches_replayed": len(per_batch),
        "generated_at":     now_iso(),
    })

    return validation_report


# ─── Branch forking ───────────────────────────────────────────────────────────

def fork_from_checkpoint(checkpoint: dict, original_batches: list) -> dict:
    """
    Create a branch fork from the state captured in *checkpoint*.
    """
    fork_step   = checkpoint["global_step"]
    branch_name = f"branch_from_step_{fork_step:06d}"

    consumed_ids   = set(checkpoint["consumed_batches"])
    future_batches = [
        b["batch_id"] for b in original_batches
        if b["batch_id"] not in consumed_ids
    ]

    fork_meta = {
        "fork_from_step":   fork_step,
        "new_branch":       branch_name,
        "checkpoint_hash":  checkpoint.get("state_hash", ""),
        "consumed_batches": checkpoint["consumed_batches"],
        "future_batches":   future_batches,
        "consumed_count":   len(checkpoint["consumed_batches"]),
        "future_count":     len(future_batches),
        "forked_at":        now_iso(),
    }
    write_json(_FORK_AUDIT_PATH, fork_meta)
    log("branch forked")
    return fork_meta
