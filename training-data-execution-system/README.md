# Training Data Execution System — V5

A complete miniature LLM pretraining data pipeline that is **correct, reproducible, auditable and efficient**.

## Quick Start

```bash
python run_demo.py
```

This single command regenerates `submission_artifacts/` from scratch, runs every phase, and exits with a full evidence bundle.

---

## Architecture

```
Documents
  → Tokenised shards (SHA-256 content-hashed, tokenizer-locked)
  → Shard manifests (verified before training begins)
  → Evaluation firewall (EVAL/VALIDATION shards blocked at routing + ledger level)
  → Sequence packing (greedy, CONTEXT_LENGTH=128, produces tokens/mask/position_ids/segments)
  → Batches (deterministic input_hash per batch)
  → Curriculum-staged mixture scheduler (warmup → main → cooldown, lane weights + protected floors)
  → OPUS candidate auditing (ACCEPT / DEFER / REJECT + protected-floor override)
  → Training loop (40 steps, crash at step 22)
  → Consumption ledger (token-level source traceability per step)
  → Learning ledger (batch-level + per-sample loss per step)
  → Checkpoint (saved every 10 steps + at crash point)
  → Crash + resume (no repeated or skipped batches, proved by resume_audit_report.json)
  → Historical replay (full stream reconstructed from raw shards, every batch hash verified)
  → Branch fork (from step-9 checkpoint)
  → Evidence bundle (evidence.json + evidence.md, all values derived from artifacts)
```

---

## Design Decisions

### Determinism
All randomness flows through `random.Random` instances seeded from `RANDOM_SEED = 42`. Hashes are SHA-256 of deterministically sorted JSON. Repeated runs produce bit-identical manifests, batch hashes, and evidence results.

### Tokenizer integrity
A word-level tokenizer maps each word to a vocab id via `SHA-256 mod VOCAB_SIZE`. The tokenizer hash is derived from a canonical vocab definition dict. Every shard manifest stores this hash; verification runs before any shard enters a training batch.

### Packing
The greedy `SequencePacker` fills `CONTEXT_LENGTH` windows before flushing. Each `PackedSequence` stores:
- `tokens` — token ids
- `mask` — attention mask (1=real, 0=pad)
- `position_ids` — sequential within real tokens, 0 for padding
- `segments` — per-position doc_id label
- `sources` — list of contributing documents

### Curriculum stages
Three stages (`warmup`, `main`, `cooldown`) each define their own lane weights. The scheduler transitions automatically when the current step crosses a stage boundary. Protected floors override weights to guarantee minimum BOOKS/OPUS representation.

### OPUS auditing
Every OPUS document receives a deterministic score. The decision is:
- **ACCEPT** if score ≥ 0.50
- **DEFER** if score ∈ [0.35, 0.50)
- **REJECT** if score < 0.35

If the OPUS share drops below its protected floor, all deferred documents are force-accepted (floor override).

### Crash recovery
Training crashes at step 22. The latest checkpoint is loaded, state is fully restored, and training resumes at step 23. `resume_audit_report.json` proves:
- `actual_resumed_batch == expected_next_batch`
- `duplicate_detected = false`
- `skipped_batch_detected = false`

### Replay
The full historical stream is reconstructed by re-tokenising and re-packing all training shards from raw documents, then re-assembling batches. Every batch fingerprint is compared. A single hash mismatch would fail the replay.

### Evidence generation
`evidence.json` and `evidence.md` are assembled entirely by reading generated artifact files from disk. No values are hardcoded. `final_verification.json` derives each PASS/FAIL from its corresponding source file.

---

## Output Structure

```
submission_artifacts/
  run.log                        ← complete execution log
  evidence.json                  ← machine-readable evidence bundle
  evidence.md                    ← human-readable evidence table
  performance.json               ← throughput with raw measurements
  manifests/
    tokenizer_manifest.json
    shard_manifest.json
    {DATASET}_manifest.json      ← per-dataset
  ledgers/
    consumption_ledger.jsonl     ← per-step token consumption
    learning_ledger.jsonl        ← per-step + per-sample loss
  checkpoints/
    checkpoint_step_*.json       ← every 10 steps + crash point
  audits/
    manifest_verification_report.json
    tokenizer_verification.json
    firewall_audit.jsonl
    firewall_audit_report.json
    opus_audit_trail.jsonl
    opus_audit_report.json
    crash_recovery.json
    resume_audit_report.json
    protected_floor_audit.json
    traceability_audit.json
    learning_trace_audit.json
    fork_audit.json
    final_verification.json
    self_validation.json
  reports/
    packed_batch_audit.json
    packing_validation_report.json
    mixture_report.json
    curriculum_report.json
    performance_report.json
    ARCHITECTURE.md
  replay/
    replay_validation_report.json
    replay_report.json
    replay_log.jsonl
  tests/
    test_report.json
```

---

## Automated Tests

Tests read only from generated artifacts. Run via `run_demo.py` (embedded at the end of the pipeline). Results in `submission_artifacts/tests/test_report.json`.

Suites: tokenizer, manifest_verification, firewall, packing, mixture, opus, consumption_ledger, learning_ledger, checkpoint, resume, replay, fork, evidence, final_verification, determinism, performance.

---

## Key Invariants Proved

| Invariant | Proof artifact |
|---|---|
| Tokenizer hash locked | `manifests/tokenizer_manifest.json` + `audits/tokenizer_verification.json` |
| Eval data never in training | `audits/firewall_audit_report.json` |
| No token loss in packing | `reports/packing_validation_report.json` |
| Masks correct | `reports/packing_validation_report.json` → `mask_correct=true` |
| Position ids present | Stored in every packed sequence |
| Protected floors respected | `audits/protected_floor_audit.json` |
| OPUS deferral + override | `audits/opus_audit_trail.jsonl` → `decision` field |
| No duplicate batch on resume | `audits/resume_audit_report.json` → `duplicate_detected=false` |
| No skipped batch on resume | `audits/resume_audit_report.json` → `skipped_batch_detected=false` |
| Replay hashes match | `replay/replay_validation_report.json` → `match=true` |
| Loss traceable to source | `audits/learning_trace_audit.json` + `audits/traceability_audit.json` |
