# LLM Pretraining Simulation — Architecture Documentation

## Overview

This project implements an end-to-end miniature LLM pretraining simulation pipeline
covering dataset sharding, tokenization, packing, mixture scheduling, checkpointing,
crash recovery, replay, forking, and evidence generation.

---

## Data Flow

```
Synthetic Documents
        │
        ▼
Shard Files  ──►  Shard Manifests (content_hash, tokenizer_hash)
        │
        ▼ tokenizer verification
Packed Sequences (context_length=128)
        │
        ▼
Batch Builder  ──►  Batch Manifests (batch_id, sample_ids, input_hash)
        │
        ├──────────────────────────────────────────►  Firewall check
        │                                                    │
        │                                              blocked? → audit
        │
        ▼ (training datasets only)
Mixture Scheduler  ──►  Dataset selection per step
        │
        ▼
Training Loop  ──►  Consumption Ledger
                ──►  Learning Ledger
                ──►  Checkpoints (every N steps)
                ──►  Crash simulation & recovery
        │
        ▼
Replay & Fork
        │
        ▼
Evidence Bundle  ──►  evidence.json + evidence.md
```

---

## Components

### 1. Datasets & Shards (`pipeline/datasets.py`)

Six synthetic datasets are generated deterministically from a fixed seed.
Each dataset is split into shards.  Every shard file stores:
- The shard manifest (shard_id, dataset, document_count, content_hash, tokenizer_hash)
- The raw documents with doc_id, text, and metadata

The `content_hash` is SHA-256 of the JSON-serialised document list.
The `tokenizer_hash` is pre-filled from `compute_tokenizer_hash()` so that
later verification can compare manifest value vs live value.

### 2. Tokenizer (`pipeline/tokenizer.py`)

A deterministic word-level tokenizer using SHA-256 mod VOCAB_SIZE to map words
to integer ids.  The vocab is described by a canonical dict; the tokenizer hash
is SHA-256 of that dict.  Any change to the vocab definition changes the hash,
making tampering detectable.

Verification: before any shard is processed, `verify_shard_tokenizer_hash()`
compares the manifest hash to the live hash.  A mismatch raises `RuntimeError`.

### 3. Evaluation Firewall (`pipeline/firewall.py`)

Datasets EVAL and VALIDATION are designated eval-only.  `block_if_eval()` is
called for every shard before it may enter a training batch.  Blocked shards
are recorded in `firewall_audit.jsonl`.  The firewall report proves the set of
shards allowed for training contains no eval dataset.

### 4. Sequence Packing (`pipeline/packing.py`)

`SequencePacker` maintains a buffer of tokens.  When the buffer reaches
`CONTEXT_LENGTH` (128) it is flushed as a `PackedSequence`.  The final window
is padded.  Every position stores a segment label (doc_id) and every sequence
stores its source list.  Attention masks have 1 for real tokens and 0 for pad.

Validation (`validate_packed_sequences`) checks:
- Token/mask/segment length == CONTEXT_LENGTH
- Mask-0 positions hold PAD_TOKEN_ID

### 5. Batch Builder (`pipeline/batching.py`)

`BatchBuilder` collects packed sequences into batches of `BATCH_SIZE`.  Each
batch receives a deterministic `input_hash` (SHA-256 of the sequences dict)
enabling replay and deduplication.

### 6. Mixture Scheduler (`pipeline/scheduler.py`)

`MixtureScheduler.sample_dataset()` draws a dataset name on every step.
Weights are derived from `TARGET_MIXTURE` with a floor-boost for any dataset
falling below its `PROTECTED_FLOORS` value.  The full `get_state()` /
`set_state()` interface enables checkpointing and exact restoration.

### 7. OPUS Candidate Auditing (`pipeline/opus_audit.py`)

Every OPUS document receives a deterministic score derived from its content
hash plus a reproducible per-document perturbation.  Candidates above
`OPUS_SCORE_THRESHOLD` are accepted.  Every decision is appended to
`opus_audit_trail.jsonl` with candidate_id, score, accepted, and reason.

### 8. Ledgers (`pipeline/ledgers.py`)

**Consumption ledger** (`consumption_ledger.jsonl`): one record per training
step containing batch_id, dataset_contributions, token_count, sample_ids, and
shard_sources.  Full traceability: Loss → Step → Batch → Shard → Document.

**Learning ledger** (`learning_ledger.jsonl`): one record per step containing
simulated loss, source_datasets, and batch_id linking back to the consumption
ledger.

### 9. Checkpointing (`pipeline/checkpointing.py`)

Every `CHECKPOINT_EVERY` steps the pipeline saves a checkpoint JSON containing:
- global_step, batch_id
- consumed_batches list
- scheduler_state (rng state + counts)
- rng_state for the data-order rng
- state_hash (SHA-256 of step + batch_id + consumed_batches)

### 10. Crash Recovery

A crash is injected at step `CRASH_AT_STEP`.  On restart, the latest checkpoint
is loaded, state is restored, and training continues from the first unconsumed
batch.  A `crash_recovery.json` audit records checkpoint_batch, expected_next,
resumed_batch, and match.

### 11. Replay (`pipeline/replay.py`)

After training, every batch's fingerprint (SHA-256 of batch_id + sample_ids +
input_hash) is re-derived.  An aggregate hash of all fingerprints is compared
to the original aggregate.  Because all inputs are deterministic, both hashes
are identical.

### 12. Branch Forking

`fork_from_checkpoint()` selects the checkpoint at `FORK_FROM_STEP`, lists its
consumed batches, identifies future batches not yet consumed, and writes a
`fork_audit.json` with the new branch name and metadata.

### 13. Evidence Generation (`pipeline/evidence.py`)

Every section of `evidence.json` is assembled by reading generated artifact
files (JSON reports, JSONL ledgers).  No value is hardcoded — all come from
`_safe_read_json` / `_safe_read_jsonl` calls on known output paths.

---

## How Each Evidence Entry Is Derived

| Evidence key | Source file(s) |
|---|---|
| tokenizer_integrity | `manifests/tokenizer_manifest.json`, `audits/tokenizer_verification.json` |
| evaluation_firewall | `audits/firewall_report.json`, `audits/firewall_audit.jsonl` |
| packing_correctness | `reports/packed_batch_audit.json` |
| mixture_compliance  | `reports/mixture_report.json`, `audits/protected_floor_audit.json` |
| opus_audit          | `audits/opus_audit_summary.json`, `audits/opus_audit_trail.jsonl` |
| crash_recovery      | `audits/crash_recovery.json` |
| replay              | `replay/replay_report.json` |
| learning_trace      | `audits/learning_traceability.json` |
| throughput          | `reports/performance_report.json` |

---

## Determinism

All randomness passes through `random.Random` instances seeded from
`RANDOM_SEED = 42`.  Hashes are SHA-256 of deterministic JSON (keys sorted).
Re-running `python main.py` produces bit-identical manifests, batch hashes,
and evidence results.

---

## Failure Guards

| Failure | Guard |
|---|---|
| Resume repeating a batch | consumed_batches set checked before processing |
| Resume skipping a batch | next_batch pointer derived from checkpoint |
| Replay different hashes | deterministic fingerprint re-derivation |
| Eval data in training | `block_if_eval()` called before every shard |
| Packing report mismatch | `validate_packed_sequences()` compared to produced batches |
| Hardcoded evidence | all values read from disk artifacts |

Generated: 2026-08-07T14:36:12.096833Z
