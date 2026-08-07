"""
Central configuration for the pretraining simulation pipeline.
All parameters are fixed-seed deterministic.
"""

import os

# ─── Reproducibility ───────────────────────────────────────────────────────────
RANDOM_SEED = 42

# ─── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR          = "submission_artifacts"
MANIFESTS_DIR     = os.path.join(BASE_DIR, "manifests")
LEDGERS_DIR       = os.path.join(BASE_DIR, "ledgers")
CHECKPOINTS_DIR   = os.path.join(BASE_DIR, "checkpoints")
REPORTS_DIR       = os.path.join(BASE_DIR, "reports")
AUDITS_DIR        = os.path.join(BASE_DIR, "audits")
REPLAY_DIR        = os.path.join(BASE_DIR, "replay")
TESTS_DIR         = os.path.join(BASE_DIR, "tests")
RAW_DATA_DIR      = os.path.join(BASE_DIR, "raw_data")

EVIDENCE_JSON     = os.path.join(BASE_DIR, "evidence.json")
EVIDENCE_MD       = os.path.join(BASE_DIR, "evidence.md")
# Spec requires run.log (not execution.log); keep both names pointing same place
RUN_LOG           = os.path.join(BASE_DIR, "run.log")
EXECUTION_LOG     = RUN_LOG          # alias so existing code still works

# Top-level performance.json (spec requirement)
PERFORMANCE_JSON  = os.path.join(BASE_DIR, "performance.json")

# ─── Tokenizer ─────────────────────────────────────────────────────────────────
VOCAB_SIZE        = 512   # small synthetic vocab
TOKENIZER_MANIFEST= os.path.join(MANIFESTS_DIR, "tokenizer_manifest.json")

# ─── Dataset definitions ───────────────────────────────────────────────────────
# name → (doc_count, shards, is_eval)
DATASETS = {
    "COMMON_CRAWL": {"doc_count": 120, "shards": 3, "is_eval": False},
    "BOOKS":        {"doc_count":  60, "shards": 2, "is_eval": False},
    "CODE":         {"doc_count":  90, "shards": 3, "is_eval": False},
    "OPUS":         {"doc_count":  40, "shards": 2, "is_eval": False},
    "EVAL":         {"doc_count":  20, "shards": 1, "is_eval": True},
    "VALIDATION":   {"doc_count":  20, "shards": 1, "is_eval": True},
}

# ─── Curriculum stages ─────────────────────────────────────────────────────────
# Each stage defines lane weights (mixture) and which step range it covers.
# Stages are processed in order; the scheduler transitions between them.
CURRICULUM_STAGES = [
    {
        "stage":      "warmup",
        "step_range": (0, 15),
        "lane_weights": {
            "COMMON_CRAWL": 0.60,
            "BOOKS":        0.10,
            "CODE":         0.20,
            "OPUS":         0.10,
        },
    },
    {
        "stage":      "main",
        "step_range": (15, 35),
        "lane_weights": {
            "COMMON_CRAWL": 0.50,
            "BOOKS":        0.15,
            "CODE":         0.25,
            "OPUS":         0.10,
        },
    },
    {
        "stage":      "cooldown",
        "step_range": (35, 9999),
        "lane_weights": {
            "COMMON_CRAWL": 0.40,
            "BOOKS":        0.20,
            "CODE":         0.25,
            "OPUS":         0.15,
        },
    },
]

# ─── Packing ───────────────────────────────────────────────────────────────────
CONTEXT_LENGTH    = 128    # tokens per packed sequence
PAD_TOKEN_ID      = 0

# ─── Training ──────────────────────────────────────────────────────────────────
TOTAL_STEPS       = 40
BATCH_SIZE        = 4      # packed sequences per batch
CHECKPOINT_EVERY  = 10     # save checkpoint every N steps
CRASH_AT_STEP     = 22     # inject crash here

# ─── Mixture targets (training datasets only) — used as fallback ──────────────
TARGET_MIXTURE = {
    "COMMON_CRAWL": 0.50,
    "BOOKS":        0.15,
    "CODE":         0.25,
    "OPUS":         0.10,
}

# ─── Protected floors ──────────────────────────────────────────────────────────
PROTECTED_FLOORS = {
    "BOOKS": 0.10,
    "OPUS":  0.05,
}

# ─── OPUS selection ────────────────────────────────────────────────────────────
OPUS_SCORE_THRESHOLD  = 0.50   # accept candidates with score >= threshold
OPUS_DEFER_THRESHOLD  = 0.35   # defer (re-evaluate later) if score in [defer, accept)
# protected-floor override: if OPUS share is below floor, force-accept deferred docs
OPUS_FLOOR_OVERRIDE   = True

# ─── Branch fork ───────────────────────────────────────────────────────────────
FORK_FROM_STEP    = 10
