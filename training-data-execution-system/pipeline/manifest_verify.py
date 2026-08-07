"""
Manifest verification audit.

Runs before training starts. Verifies:
  - every declared shard file exists on disk
  - shard document counts match manifest
  - shard content hashes match re-computed values
  - tokenizer hashes match current tokenizer
  - manifest metadata is internally consistent

Training must not begin if verification fails.
"""

import os
from pipeline import config
from pipeline.utils import (
    sha256_obj, read_json, write_json, log, now_iso
)

_REPORT_PATH = os.path.join(config.AUDITS_DIR, "manifest_verification_report.json")


def verify_all_manifests(all_manifests: dict) -> dict:
    """
    Verify every shard declared in *all_manifests*.
    Returns the verification report and raises RuntimeError on any failure.
    """
    from pipeline.tokenizer import compute_tokenizer_hash
    current_tok_hash = compute_tokenizer_hash()

    results  = []
    failures = []

    for dataset_name, shards in all_manifests.items():
        for shard_manifest in shards:
            shard_id  = shard_manifest["shard_id"]
            shard_path = os.path.join(config.RAW_DATA_DIR, f"{shard_id}.json")

            entry = {
                "shard_id":            shard_id,
                "dataset":             dataset_name,
                "file_exists":         False,
                "doc_count_match":     False,
                "content_hash_match":  False,
                "tokenizer_hash_match": False,
                "metadata_consistent": False,
                "errors":              [],
            }

            # ── 1. file exists ────────────────────────────────────────────────
            if not os.path.exists(shard_path):
                entry["errors"].append(f"Shard file not found: {shard_path}")
                results.append(entry)
                failures.append(shard_id)
                continue
            entry["file_exists"] = True

            # ── 2. load and parse ─────────────────────────────────────────────
            try:
                data = read_json(shard_path)
            except Exception as exc:
                entry["errors"].append(f"JSON parse error: {exc}")
                results.append(entry)
                failures.append(shard_id)
                continue

            documents = data.get("documents", [])

            # ── 3. document count ─────────────────────────────────────────────
            expected_count = shard_manifest.get("document_count", -1)
            actual_count   = len(documents)
            if expected_count == actual_count:
                entry["doc_count_match"] = True
            else:
                entry["errors"].append(
                    f"doc_count mismatch: manifest={expected_count} actual={actual_count}"
                )

            # ── 4. content hash ───────────────────────────────────────────────
            recomputed_hash = sha256_obj(documents)
            manifest_hash   = shard_manifest.get("content_hash", "")
            if recomputed_hash == manifest_hash:
                entry["content_hash_match"] = True
            else:
                entry["errors"].append(
                    f"content_hash mismatch: manifest={manifest_hash[:16]}… "
                    f"recomputed={recomputed_hash[:16]}…"
                )

            # ── 5. tokenizer hash ─────────────────────────────────────────────
            manifest_tok = shard_manifest.get("tokenizer_hash", "")
            if manifest_tok == current_tok_hash:
                entry["tokenizer_hash_match"] = True
            else:
                entry["errors"].append(
                    f"tokenizer_hash mismatch: manifest={manifest_tok[:16]}… "
                    f"current={current_tok_hash[:16]}…"
                )

            # ── 6. metadata consistency ───────────────────────────────────────
            stored_manifest = data.get("manifest", {})
            consistent = (
                stored_manifest.get("shard_id")    == shard_id
                and stored_manifest.get("dataset") == dataset_name
                and stored_manifest.get("document_count") == actual_count
            )
            if consistent:
                entry["metadata_consistent"] = True
            else:
                entry["errors"].append(
                    f"metadata inconsistency: stored_manifest={stored_manifest.get('shard_id')} "
                    f"expected={shard_id}"
                )

            entry["passed"] = not entry["errors"]
            if entry["errors"]:
                failures.append(shard_id)

            results.append(entry)

    overall_pass = len(failures) == 0
    report = {
        "result":         "PASS" if overall_pass else "FAIL",
        "shards_checked": len(results),
        "failures":       failures,
        "details":        results,
        "tokenizer_hash": current_tok_hash,
        "generated_at":   now_iso(),
    }
    write_json(_REPORT_PATH, report)

    if not overall_pass:
        msg = (f"Manifest verification FAILED for shards: {failures}. "
               "Training cannot begin.")
        log(f"[FAIL] manifest_verification  failures={failures}")
        raise RuntimeError(msg)

    log(f"manifest_verification PASS  shards_checked={len(results)}")
    return report
