"""
Synthetic dataset generation and shard manifest creation.

Each dataset produces deterministic documents.
Documents are split into shards with SHA-256 content hashes.
"""

import os
import json
import random
from pipeline import config
from pipeline.utils import (
    sha256_obj, sha256_str, write_json, read_json, log, ensure_dirs, now_iso
)


# ─── Deterministic document generator ─────────────────────────────────────────

_WORD_POOL = [
    "the", "quick", "brown", "fox", "jumps", "over", "lazy", "dog",
    "model", "training", "data", "token", "sequence", "batch", "layer",
    "gradient", "learning", "rate", "loss", "epoch", "dataset", "corpus",
    "neural", "network", "transformer", "attention", "head", "embedding",
    "pretraining", "fine", "tuning", "optimizer", "weight", "parameter",
    "language", "code", "book", "web", "crawl", "text", "document", "shard",
    "manifest", "hash", "verify", "checkpoint", "replay", "deterministic",
    "pipeline", "schedule", "mixture", "floor", "protected", "audit",
]

def _make_document(dataset: str, doc_idx: int, rng: random.Random) -> dict:
    """Create a synthetic document for *dataset* at position *doc_idx*."""
    length = rng.randint(30, 80)           # words
    words  = [rng.choice(_WORD_POOL) for _ in range(length)]
    text   = " ".join(words)
    return {
        "doc_id":  f"{dataset}_{doc_idx:05d}",
        "dataset": dataset,
        "text":    text,
        "meta": {
            "source":    dataset,
            "index":     doc_idx,
            "word_count": length,
        },
    }


# ─── Public API ────────────────────────────────────────────────────────────────

def generate_all_datasets() -> dict:
    """
    Generate every synthetic dataset, split into shards, write shard files
    and per-shard manifest entries.

    Returns a dict:  dataset_name → list[shard_manifest_dict]
    """
    ensure_dirs()
    rng = random.Random(config.RANDOM_SEED)          # deterministic

    all_manifests = {}   # dataset → [shard_manifest, ...]

    for dataset_name, cfg in config.DATASETS.items():
        doc_count   = cfg["doc_count"]
        num_shards  = cfg["shards"]

        # ── build all documents for this dataset ──────────────────────────────
        documents = [_make_document(dataset_name, i, rng) for i in range(doc_count)]

        # ── split into shards ─────────────────────────────────────────────────
        shards = _split_into_shards(documents, num_shards)

        dataset_shard_manifests = []
        for shard_idx, shard_docs in enumerate(shards):
            shard_id       = f"{dataset_name}_shard_{shard_idx:03d}"
            content_hash   = sha256_obj(shard_docs)
            # tokenizer hash is constant; verified later against actual tokenizer
            tok_hash       = _tokenizer_hash_placeholder()

            manifest_entry = {
                "shard_id":       shard_id,
                "dataset":        dataset_name,
                "document_count": len(shard_docs),
                "content_hash":   content_hash,
                "tokenizer_hash": tok_hash,
                "shard_index":    shard_idx,
                "is_eval":        cfg["is_eval"],
                "created_at":     now_iso(),
            }

            # write shard data file
            shard_path = os.path.join(
                config.RAW_DATA_DIR, f"{shard_id}.json"
            )
            write_json(shard_path, {
                "manifest": manifest_entry,
                "documents": shard_docs,
            })

            dataset_shard_manifests.append(manifest_entry)

        all_manifests[dataset_name] = dataset_shard_manifests

        # ── dataset-level manifest ────────────────────────────────────────────
        dataset_manifest = {
            "dataset":       dataset_name,
            "doc_count":     doc_count,
            "num_shards":    num_shards,
            "is_eval":       cfg["is_eval"],
            "dataset_hash":  sha256_obj(dataset_shard_manifests),
            "shards":        dataset_shard_manifests,
        }
        write_json(
            os.path.join(config.MANIFESTS_DIR, f"{dataset_name}_manifest.json"),
            dataset_manifest,
        )

    # ── master shard manifest ────────────────────────────────────────────────
    flat_shards = []
    for shards in all_manifests.values():
        flat_shards.extend(shards)

    write_json(
        os.path.join(config.MANIFESTS_DIR, "shard_manifest.json"),
        {"shards": flat_shards, "total_shards": len(flat_shards)},
    )

    log(f"datasets generated: {len(config.DATASETS)} datasets, "
        f"{len(flat_shards)} shards total")
    return all_manifests


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _split_into_shards(documents: list, num_shards: int) -> list:
    """Split *documents* into *num_shards* roughly-equal lists."""
    size   = max(1, len(documents) // num_shards)
    shards = []
    for i in range(num_shards):
        start = i * size
        end   = start + size if i < num_shards - 1 else len(documents)
        shards.append(documents[start:end])
    return shards


def _tokenizer_hash_placeholder() -> str:
    """
    Return the *expected* tokenizer hash so manifest entries are pre-filled.
    The actual tokenizer module derives the same value independently.
    """
    from pipeline.tokenizer import compute_tokenizer_hash
    return compute_tokenizer_hash()


def load_shard_documents(shard_id: str) -> list:
    """Load documents for a given shard_id from disk."""
    path = os.path.join(config.RAW_DATA_DIR, f"{shard_id}.json")
    data = read_json(path)
    return data["documents"]


def get_all_training_shards(all_manifests: dict) -> list:
    """Return shard manifest entries for non-eval datasets only."""
    result = []
    for dataset_name, shards in all_manifests.items():
        if not config.DATASETS[dataset_name]["is_eval"]:
            result.extend(shards)
    return result


def get_all_eval_shards(all_manifests: dict) -> list:
    """Return shard manifest entries for eval/validation datasets."""
    result = []
    for dataset_name, shards in all_manifests.items():
        if config.DATASETS[dataset_name]["is_eval"]:
            result.extend(shards)
    return result
