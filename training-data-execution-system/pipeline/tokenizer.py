"""
Deterministic tokenizer.

Vocabulary: integers 1..VOCAB_SIZE-1; 0 is reserved for PAD.
Encoding: each word maps deterministically to a token id via SHA-256 mod.
The tokenizer hash is derived from the full vocab definition so it is
stable across runs and tamper-evident.
"""

import json
import hashlib
import os
from pipeline import config
from pipeline.utils import sha256_str, write_json, read_json, log, now_iso


# ─── Core tokenizer ───────────────────────────────────────────────────────────

def _word_to_id(word: str) -> int:
    """Map a word to a vocab id deterministically (1..VOCAB_SIZE-1)."""
    raw = int(hashlib.sha256(word.encode("utf-8")).hexdigest(), 16)
    # +1 so 0 stays reserved for PAD
    return (raw % (config.VOCAB_SIZE - 1)) + 1


def encode(text: str) -> list:
    """Tokenize *text* into a list of integer token ids."""
    tokens = []
    for word in text.split():
        tokens.append(_word_to_id(word))
    return tokens


def decode(token_ids: list) -> str:
    """Reverse lookup is lossy (hash is one-way); returns hex representations."""
    return " ".join(f"<{tid}>" for tid in token_ids)


# ─── Hash / manifest ──────────────────────────────────────────────────────────

def _vocab_definition() -> dict:
    """
    Canonical representation of the full vocabulary used to derive the hash.
    Reproducible across runs because it only depends on VOCAB_SIZE.
    """
    return {
        "vocab_size": config.VOCAB_SIZE,
        "pad_token_id": config.PAD_TOKEN_ID,
        "encoding": "sha256_mod",
        "version": "v1",
    }


def compute_tokenizer_hash() -> str:
    """Return the deterministic hash of this tokenizer's vocab definition."""
    defn = json.dumps(_vocab_definition(), sort_keys=True)
    return hashlib.sha256(defn.encode("utf-8")).hexdigest()


# ─── Manifest generation & verification ───────────────────────────────────────

def generate_tokenizer_manifest() -> dict:
    """Write tokenizer_manifest.json and return its contents."""
    tok_hash = compute_tokenizer_hash()
    manifest = {
        "tokenizer_hash": tok_hash,
        "vocab_size":     config.VOCAB_SIZE,
        "pad_token_id":   config.PAD_TOKEN_ID,
        "encoding":       "sha256_mod",
        "version":        "v1",
        "created_at":     now_iso(),
    }
    write_json(config.TOKENIZER_MANIFEST, manifest)
    return manifest


def verify_shard_tokenizer_hash(shard_manifest: dict) -> dict:
    """
    Compare the tokenizer_hash stored in a shard manifest against the
    current tokenizer hash.

    Returns a verification result dict and raises RuntimeError on mismatch.
    """
    current_hash  = compute_tokenizer_hash()
    manifest_hash = shard_manifest.get("tokenizer_hash", "")
    shard_id      = shard_manifest.get("shard_id", "unknown")

    passed = current_hash == manifest_hash
    result = {
        "shard_id":        shard_id,
        "tokenizer_hash":  current_hash,
        "manifest_hash":   manifest_hash,
        "match":           passed,
        "result":          "PASS" if passed else "FAIL",
        "source_manifest": os.path.join(
            config.MANIFESTS_DIR,
            f"{shard_manifest.get('dataset','unknown')}_manifest.json",
        ),
        "verified_at":     now_iso(),
    }

    if passed:
        log(f"[PASS] tokenizer_hash_verified  shard={shard_id}")
    else:
        log(f"[FAIL] tokenizer_hash_mismatch  shard={shard_id}  "
            f"current={current_hash}  manifest={manifest_hash}")
        raise RuntimeError(
            f"Tokenizer hash mismatch on shard {shard_id}: "
            f"expected {manifest_hash}, got {current_hash}"
        )

    return result
