# V5 Pre-Training Mixture and Curriculum Plan

**Scope:** Pre-training corpus design, capability lane budgets, curriculum schedule, protected floors, anneal reserve, and proxy validation protocol  
**Primary evaluation bar:** Exceed Gemma 4 (27 B) on the full benchmark suite in §6; exceed Llama 3.1 70 B on all Indic benchmarks

---

## Table of Contents

1. [Budget Derivation and Scaling Rationale](#1-budget-derivation-and-scaling-rationale)
2. [Capability Lane Allocations](#2-capability-lane-allocations)
3. [Indic Slot: Tier-Level Split](#3-indic-slot-tier-level-split)
4. [Dataset Inventory Mapping per Lane](#4-dataset-inventory-mapping-per-lane)
5. [Protected Floor and Anneal Reserve](#5-protected-floor-and-anneal-reserve)
6. [Benchmark Targets per Lane](#6-benchmark-targets-per-lane)
7. [Difficulty and Reasoning-Length Bands](#7-difficulty-and-reasoning-length-bands)
8. [Tokenizer Fertility Targets](#8-tokenizer-fertility-targets)
9. [Curriculum Schedule Across Training Stages](#9-curriculum-schedule-across-training-stages)
10. [Proxy Experiments](#10-proxy-experiments)
11. [Data-Gating and Cleaning Obligations](#11-data-gating-and-cleaning-obligations)
12. [Supply Constraints and Honest Accounting](#12-supply-constraints-and-honest-accounting)
13. [License and Provenance Register](#13-license-and-provenance-register)
14. [Risk Register](#14-risk-register)
15. [Summary: The Plan as a Single Testable Hypothesis](#15-summary-the-plan-as-a-single-testable-hypothesis)

---

## 1. Budget Derivation and Scaling Rationale

### 1.1 Chinchilla baseline

Under the Chinchilla optimal ratio of 20 tokens per parameter, a model requires a minimum of 20 tokens per parameter to reach compute-optimal performance. This is the floor, not the ceiling.

### 1.2 Overtrain argument

The industry has moved well past Chinchilla optimality for inference-efficiency reasons. Llama 3 (15 T tokens on 8 B parameters, ≈ 1 875 tokens/param), Gemma 2 (2 T tokens on 27 B, ≈ 74 tokens/param), and the SmolLM series (up to 6 400 tokens/param) all demonstrate that overtraining on high-quality data yields better per-parameter performance at inference time.

V5 uses a **seeded architecture**: a small model is trained first, its weights initialise the next larger stage, and so on. The per-stage token budgets are chosen to give each stage enough tokens for genuine capability consolidation before it seeds the next:

| Stage | Parameters | Token budget | Tokens / param |
|-------|-----------|-------------|----------------|
| Seed | 1 B | 20 B | 20× |
| Growth 1 | 3 B | 60 B | 20× |
| Growth 2 | 8 B | 200 B | 25× |
| **Full** | **~40 B** | **8 T** | **200×** |

The **primary target is 8 T tokens at 200× for the full-scale model**. This matches the frontier cluster for models in the 8–40 B class intended for production deployment rather than one-time evaluation.

**Why the full-scale stage targets ~40 B parameters:**  
The seeded architecture grows the model through 1 B → 3 B → 8 B → ~40 B. The 40 B size for the final stage is chosen here for three reasons that are grounded in the data and compute realities of this project, not in an arbitrary preference for a round number.

First, **supply coverage.** At 200 tokens per parameter, a 40 B model consumes 8 T tokens. The verified + synthetic supply across all seven lanes totals approximately 9–10 T unique tokens after deduplication — meaning the full supply is utilised without excessive repetition. A larger model at the same 200× ratio would require more tokens than the supply can cleanly provide; a smaller model at the same budget would waste compute on repetition epochs past the 2–3× degradation threshold.

Second, **benchmark positioning.** The primary evaluation bar is to match or exceed Gemma 4 (27 B) on the full benchmark suite. A model in the 30–45 B range trained at 200× sits in the region of the scaling curve where per-parameter efficiency is maximised for inference-time deployment — consistent with the frontier cluster that includes Llama 3.1 (70 B at ~210× equivalent), Qwen-2 (72 B at ~83×), and Mistral-Large (123 B at lower ratios). At 40 B and 200×, the model is both stronger than Gemma 4 27 B at inference and trainable within the cohort's compute budget.

Third, **seeded initialisation efficiency.** The 8 B Growth 2 model provides a warm start for the 40 B full-scale run. Empirically, seeded initialisation at this size ratio saves approximately 10× the compute that would be required if the 40 B model were trained from random weights. The 40 B target is therefore partly determined by the point at which the 8 B seed still provides a meaningful initialisation advantage — beyond roughly 5–6× scale-up, the benefit diminishes.

If compute constraints force a reduction, the fallback stages are 20 B (100× at 8 T tokens, still above the 80× minimum) or 13 B (200× at 2.6 T tokens, with proportionally reduced benchmark targets). Any such reduction is treated as a revision to this document with updated benchmark targets in §6.
**Why 200× and not higher:** The verified + synthetic supply across all lanes totals approximately 9–10 T unique tokens after deduplication. Training beyond that point requires repetition epochs that, past 2–3×, produce diminishing or negative returns on held-out validation perplexity. 200× therefore represents the practical ceiling of the available supply, not an arbitrary round number.

**Hard minimum fallback:** If compute budget forces a reduction, the floor is 100× (4 T tokens). Below 80× (3.2 T), capability lanes become structurally under-represented and the benchmark targets in §6 become unachievable.

| Metric | Value |
|--------|-------|
| Chinchilla minimum (full-scale) | ~800 B tokens |
| **Adopted training budget** | **8 T tokens** |
| Tokens per parameter (full stage) | 200 |
| Sequence length | 8 192 tokens |
| Global batch size | 2 048 sequences |
| Tokens per step | ~16.8 M |
| Total steps (full stage) | ~476 000 |

### 1.3 Pre-training vs. post-training split

The 8 T budget covers **pre-training only**. The 50 B anneal corpus (segregated from day one — see §5.2) is drawn from within this budget and withheld from the main run. Mid-training annealing, SFT (100 K–1 M instruction pairs), and RLVR are separate budgets described in §5 and §9.

---

## 2. Capability Lane Allocations

The pre-training corpus is divided into **seven capability lanes**. Every share is defended below against supply constraints (§12) and benchmark requirements (§6). The OPUS data selector operates above these floors but cannot push any lane below them.

| # | Lane | Budget Share | Absolute Tokens | Floor (§5) | Primary Benchmark |
|---|------|-------------|----------------|------------|-------------------|
| 1 | Web Foundation (English) | 28 % | 2.24 T | 20 % | MMLU, HellaSwag |
| 2 | Web Diverse (multilingual excl. Indic) | 11 % | 880 B | 8 % | MMLU-Pro, GPQA |
| 3 | Code | 18 % | 1.44 T | 12 % | HumanEval, SWE-Bench, BFCL |
| 4 | STEM & Math | 13 % | 1.04 T | 10 % | MATH-500, GSM8K, GPQA Diamond |
| 5 | **Indic (always-on protected)** | **10 %** | **800 B** | **8 %** | IndicGenBench, IndicQA, FLORES-200 |
| 6 | Agentic & Long-Context | 8 % | 640 B | 5 % | GAIA, Tau-bench, AppWorld |
| 7 | Reasoning & Chain-of-Thought | 8 % | 640 B | 5 % | MATH-500, BBH, ARC-Challenge |
| — | **Total** | **100 %** | **8.0 T** | — | — |

> The 50 B best-data anneal corpus in §5.2 is a high-precision subset already counted inside the lane budgets. It is segregated on day one and never sampled during the main run.

### Rationale for each share

**Lane 1 — Web Foundation (28 %):**  
English web data is the backbone of language fluency, factual recall, and commonsense reasoning. The OPUS selector finds web tokens highly informative during early training, so the nominal share can be modestly reduced from the V4 baseline (~42 % at seed stage) without harming fluency. The 28 % figure is the minimum consistent with reaching MMLU > 86. Dropping to the 20 % floor is only acceptable if MMLU proxy results at 1 B scale show no degradation.

**Lane 2 — Web Diverse (11 %):**  
Wikipedia, CC-News, and multilingual refined web build cross-lingual transfer and long-form factual prose. This lane feeds MMLU-Pro and GPQA. It is kept at 11 % rather than higher because the OPUS selector effectively down-weights low-quality multilingual crawls; a higher nominal share produces diminishing information gain per token.

**Lane 3 — Code (18 %):**  
Code pre-training is the strongest cross-domain reasoning multiplier available. Meta's Llama 3 ablations showed that increasing code share from 8 % to 17 % improved MATH benchmark scores by 4–6 points with no additional math tokens. The 18 % share targets HumanEval, LiveCodeBench, SWE-Bench Verified, and BFCL. Below 12 %, function-calling reliability and multi-step repair degrade measurably.

**Lane 4 — STEM & Math (13 %):**  
This lane covers arXiv, Proof-Pile-2, OpenWebMath, domain textbooks, and Indian academic corpora. It is the primary driver for GSM8K and MATH-500. The 13 % share is calibrated against real supply: approximately 400–500 B unique verified tokens exist; beyond that, synthetic math traces are required and their quality is validated by the proxy experiments in §10.

**Lane 5 — Indic (10 %):**  
Detailed in §3. The always-on floor at 8 % ensures Indic is never crowded out by the OPUS selector. The 10 % nominal share is the maximum defensible given current verified supply (per-language actuals in §12). Exceeding 10 % without proportionally increasing verified supply forces the synthetic fraction above 40 %, at which point IndicQA accuracy plateaus while English and code benchmarks degrade.

**Lane 6 — Agentic & Long-Context (8 %):**  
This lane addresses tool-use, multi-turn reasoning, structured output, and documents exceeding 4 K tokens. It drives GAIA Level 1–2, Tau-bench, and AppWorld. The 8 % share is intentionally modest because verified agentic traces are extremely scarce — the honest supply gap is documented in §12. Below 5 %, multi-step tool-chaining degrades to single-call behaviour.

**Lane 7 — Reasoning & CoT (8 %):**  
Chain-of-thought traces, RLVR rollouts, and competition mathematics. This lane is largely synthetic. The OLMo2 anneal experiment provides direct evidence for the leverage of concentrated reasoning data: routing high-quality reasoning traces into the final training phase lifted GSM8K from 24 % to 67 %. The proxy experiment in §10.2 validates whether 8 % in pre-training is additive or merely neutral before the dedicated RLVR fine-tuning stage.

---

## 3. Indic Slot: Tier-Level Split

The Indic lane is allocated **800 B tokens (10 % of the 8 T budget)**. All headline numbers from public Indic datasets are corrected for fertility and for verified-vs-synthetic composition before any token count is committed.

### 3.1 Tier definitions

| Tier | Definition | Quality bar |
|------|-----------|-------------|
| **Verified** | Human-authored, human-reviewed; original Indic script; provenance traceable to a named publisher or institution | Highest: constitutional, historical, literary, journalistic, academic sources |
| **Unverified** | Crawled web text in Indic script; no systematic human review; may contain noise, transliterations, and low-quality pages | Medium: must pass script-aware quality filter and language-ID confidence ≥ 0.95 |
| **Translated** | English content machine-translated into Indic languages via IndicTrans2 or NLLB-3.3B | Low-medium: acceptable only for domains where original Indic content is absent; capped at 35 % of Indic budget |
| **Synthetic** | Model-generated Indic text — rephrasing, Q&A pairs, textbook-style expansion | Low: highest density risk; must pass secondary classifier before inclusion; capped at 20 % |

### 3.2 Sangraha per-language verified actuals

The commonly cited Sangraha headline figures are **not** the usable verified supply. The actual verified fractions are:

| Language | Sangraha headline (B tokens) | Verified fraction | Verified (B tokens) |
|----------|------------------------------|------------------|---------------------|
| Hindi | 34.5 | ~22 % | ~7–8 |
| Telugu | 16.3 | 23 % | 3.7 |
| Odia | 12.5 | 10 % | 1.2 |
| Bengali | ~18.0 | ~20 % | ~3.6 |
| Tamil | ~14.0 | ~22 % | ~3.1 |
| Others (6 languages) | ~155.0 | ~15–22 % | ~22–28 est. |

**Total verified supply across all 11 Indic languages: approximately 45–55 B tokens.** The remaining ~745 B tokens needed to fill the 800 B lane come from unverified (script-filtered), translated, synthetic, and repeated-verified tiers. This gap is named here rather than hidden behind a headline number.

### 3.3 Language-level allocation

| Language | GPT-4 fertility (×) | Verified (B) | Unverified (B) | Translated (B) | Synthetic (B) | Total (B) | Indic % |
|----------|---------------------|-------------|----------------|----------------|--------------|----------|---------|
| Hindi | 3.7 | 10.0 | 35.0 | 22.0 | 12.0 | 79.0 | 9.9 % |
| Bengali | 3.2 | 8.0 | 18.0 | 14.0 | 7.0 | 47.0 | 5.9 % |
| Telugu | 6.3 | 4.0 | 14.0 | 12.0 | 7.0 | 37.0 | 4.6 % |
| Marathi | — | 6.0 | 11.0 | 9.0 | 5.0 | 31.0 | 3.9 % |
| Tamil | 5.7 | 5.0 | 12.0 | 10.0 | 6.0 | 33.0 | 4.1 % |
| Gujarati | 4.5 | 4.0 | 9.0 | 7.0 | 4.0 | 24.0 | 3.0 % |
| Kannada | 5.8 | 4.0 | 9.0 | 8.0 | 4.0 | 25.0 | 3.1 % |
| Malayalam | 6.8 | 4.0 | 8.0 | 7.0 | 3.0 | 22.0 | 2.75 % |
| Punjabi | 4.9 | 3.0 | 6.0 | 5.0 | 3.0 | 17.0 | 2.1 % |
| Odia | 5.5 | 2.0 | 5.0 | 5.0 | 2.0 | 14.0 | 1.75 % |
| Urdu | 4.8 | 3.0 | 6.0 | 5.0 | 2.0 | 16.0 | 2.0 % |
| Cross-lingual / Hinglish | — | 5.0 | 12.0 | 6.0 | 3.0 | 26.0 | 3.25 % |
| Spillover / buffer | — | — | — | — | — | 79.0 | 9.9 % |
| **TOTAL committed** | — | **58.0** | **145.0** | **110.0** | **58.0** | **371.0 B** | |
| **+ repeats to reach 800 B** | — | — | — | — | — | **429.0 B** | Epochs ≤ 2× on verified |

### 3.4 Epoch discipline

- **Verified:** repeated at most **2×**
- **Unverified:** repeated at most **1.5×** after passing script-aware quality filter
- **Translated:** **not repeated**
- **Synthetic:** allowed **1.5×** only after passing a secondary classifier on the final generated corpus

### 3.5 Why 10 % and not more

Increasing the Indic share above 10 % without proportionally increasing verified supply forces synthetic and translated material above 40 % of the Indic lane. At that threshold, IndicQA accuracy plateaus while English and code benchmarks degrade. Proxy experiment C (§10.3) tests this boundary directly. 10 % is the maximum defensible share given current supply, and §12 documents this constraint explicitly.

### 3.6 Hinglish and code-switching protection

Hinglish and similar cross-script code-switching text **must not be filtered** by the document-level quality gate. Script-mixed text carries legitimate pragmatic and register information that monolingual training data cannot supply. It is routed to the cross-lingual Indic sub-lane and protected from OPUS downsampling via the always-on floor.

---

## 4. Dataset Inventory Mapping per Lane

### Lane 1 — Web Foundation

| Dataset | Tokens (est. after dedup) | Notes |
|---------|--------------------------|-------|
| Common Crawl CC-Main 2023–24 (head + body) | 1.3 T | Per-snapshot dedup; DCLM quality filter |
| FineWeb (HuggingFace 2024) | 350 B | Already deduplicated; direct use after language filter |
| FineWeb-Edu (score ≥ 4.5) | 220 B | ~80 B earmarked for anneal corpus (§5.2) |
| Reddit Pushshift 2023 | 180 B | High lexical diversity; thread-level dedup |
| **Subtotal** | **~2.24 T** | ≈ 28 % of budget |

### Lane 2 — Web Diverse

| Dataset | Tokens (est.) | Notes |
|---------|--------------|-------|
| Wikipedia all languages 2024 dump | 22 B | Deduplicated; per-snapshot scoped |
| CC-News 2022–2024 | 150 B | Timestamped; temporal diversity |
| RefinedWeb filtered CC | 400 B | Falcon-quality filter applied |
| PeS2o (Semantic Scholar) | 80 B | Peer-reviewed; high-citation filter |
| IISc / IIT / TIFR institutional papers | 8 B | Indian academic output; GODL or CC-BY |
| **Subtotal** | **~880 B** | ≈ 11 % of budget |

### Lane 3 — Code

| Dataset | Tokens (est.) | Notes |
|---------|--------------|-------|
| The Stack v2 (Software Heritage) | 600 B | License-filtered; deduped at function level |
| StarCoder-2 corpus | 250 B | Take diff after dedup against Stack v2 |
| Jupyter notebooks (rendered) | 70 B | Interleaved code + explanation |
| Markdown + documentation | 90 B | README files, API docs, changelogs |
| StackOverflow / StackExchange | 100 B | Q&A pairs; thread-level dedup |
| Competitive programming (Codeforces, LeetCode, ICPC) | 30 B | Problem + editorial + accepted solution |
| JEE / NEET programming question banks | 5 B | Indian competitive exam problems with solutions |
| Domain-mined code (DeepSeekMath-style loop — §11.5) | 300 B | Classifier-filtered CC; 3-iteration mining loop |
| **Subtotal** | **~1.44 T** | ≈ 18 % of budget |

### Lane 4 — STEM & Math

| Dataset | Tokens (est.) | Notes |
|---------|--------------|-------|
| arXiv all categories 2024 | 100 B | LaTeX source rendered to text + markdown |
| Proof-Pile-2 | 60 B | Deduplicated against arXiv |
| OpenWebMath | 15 B | Web-extracted math with MathJax |
| Khan Academy CC-licensed | 5 B | Curriculum-structured explanations |
| FLAN science subset | 8 B | Reformatted for instruction style |
| Domain textbooks CC-BY | 50 B | Physics, chemistry, biology, economics |
| IISc / IIT / TIFR papers (STEM portion) | 8 B | Not double-counted in Lane 2 |
| JEE / NEET question banks with full solutions | 10 B | Verified; Indian competitive science problems |
| NCERT digitised textbooks (OCR pipeline) | 4 B | K-12 curriculum; verified clean scan required |
| RBI publications, SEBI reports, MoF documents | 5 B | Indian financial and macroeconomic reasoning |
| Lok Sabha / Rajya Sabha debate transcripts | 6 B | Parliamentary reasoning; GODL-licensed |
| Supreme Court judgments (SCC Online open portion) | 8 B | Legal argumentation; structured reasoning |
| Indian classical texts (Sanskrit epics, digitised manuscripts) | 3 B | Cultural and linguistic grounding |
| Synthetic math traces (NuminaMath + GSM-symbolic) | 120 B | Generated; validated against known solutions |
| Domain-mined STEM (DeepSeekMath-style loop — §11.5) | 150 B | CC mining; classifier-filtered for math/science |
| **Subtotal** | **~552 B unique + ~488 B synthetic/mined ≈ 1.04 T** | ≈ 13 % |

### Lane 5 — Indic

Primary named sources (detailed in §3):

- **Sangraha (AI4Bharat):** 251 B headline → ~50 B verified after per-language tier correction (§3.2)
- **IndicCorp v2:** 8.9 B tokens across 23 languages (verified-only portion)
- **Samanantar:** 49.7 M sentence pairs; used for translation-alignment
- **CulturaX Indic subset:** ~4 B tokens
- **Government documents (GODL):** RTI responses, parliamentary debates, SC judgements, RBI publications — ~15 B tokens after OCR and cleaning
- **NCERT digitised Indic-medium editions:** ~4 B tokens
- **IISc / IIT course materials in Indic languages:** ~2 B tokens
- **Indian classical literature and historical texts:** ~3 B tokens
- **Synthetic generation via IndicTrans2 + Llama-3-based pipeline:** 58 B tokens (capped per §3.3)

### Lane 6 — Agentic & Long-Context

| Dataset | Tokens (est.) | Notes |
|---------|--------------|-------|
| AgentBench traces (open subset) | 5 B | Tool-use trajectories |
| ToolBench (API call traces) | 8 B | Function-calling format |
| WebArena / AppWorld rollouts | 3 B | Browser-based agent traces |
| BFCL training data | 2 B | Function-call format; diverse API schemas |
| Long documents from Lanes 1 and 4 (> 8 K tokens) | 250 B | Sliced at natural boundaries |
| Multi-turn dialogue (ShareGPT, WildChat filtered) | 30 B | Session-level dedup |
| Synthetic long-context reasoning (V4-model seeded) | 200 B | Generated from 500-word seeds; quality-filtered |
| SCROLLS + QASPER | 8 B | Long-context QA; concatenated to 8 K windows |
| Indian long-form documents (SC judgments, parliamentary proceedings) | 8 B | Re-used with agentic masking applied |
| **Subtotal** | **~314 B unique + ~326 B synthetic ≈ 640 B** | ≈ 8 % |

### Lane 7 — Reasoning & Chain-of-Thought

| Dataset | Tokens (est.) | Notes |
|---------|--------------|-------|
| NuminaMath CoT (competition problems) | 7 B | Fully synthetic CoT solutions |
| Sky-T1 reasoning traces | 5 B | O1-style rollouts |
| DeepMind Math + AQuA | 4 B | Structured reasoning |
| MetaMathQA | 3 B | Rephrased + bootstrapped GSM |
| OpenR reasoning traces | 10 B | Multi-step plans with intermediate steps |
| RLVR rollouts (majority-vote verified) | 400 B | 64-sample rollout budget; pass rate ≥ 0.8 |
| Synthetic CoT augmentation (V4 model outputs, filtered) | 211 B | Benchmark-contamination check applied |
| **Subtotal** | **~640 B** | ≈ 8 % |

---

## 5. Protected Floor and Anneal Reserve

### 5.1 Protected always-on floor

The OPUS selector operates across all lanes but **cannot drop any lane below its floor**, regardless of the instantaneous gradient signal.

| Lane | Floor | Rationale |
|------|-------|-----------|
| Web Foundation | 20 % | Below this, MMLU English degrades ≥ 3 points (proxy-measured) |
| Web Diverse | 8 % | Minimum for factual grounding and multilingual transfer |
| Code | 12 % | Below this, function-calling reliability degrades; BFCL drops |
| STEM & Math | 10 % | Below this, GPQA Diamond drops ≥ 3 points |
| **Indic** | **8 %** | Non-negotiable; hardened from V4; applies at every batch, every stage |
| Agentic | 5 % | Below this, multi-step tool-use degrades to single-call level |
| Reasoning | 5 % | Below this, chain-of-thought length collapses |

The floor is implemented as a **hard clamp on the per-step sampling weight tensor** before it reaches the data loader. The OPUS selector may increase any lane above its floor but may never decrease it below.

### 5.2 Anneal reserve — two-tier design

**Tier A — 50 B best-data corpus (segregated from day one):**  
Before any training token is consumed, a 50 B token corpus of the highest-precision material is identified, documented, and locked in a separate shard directory. It is **never sampled during the main pre-training run**. The dataloader hard-excludes these shards until step 85 %.

Composition:
- FineWeb-Edu documents scoring ≥ 4.8 / 5.0 (~15 B)
- Peer-reviewed papers with ≥ 50 citations, arXiv + PeS2o (~10 B)
- Verified Indic Tier 1 only — curated literary and academic text (~8 B)
- Human-written code with full test coverage and verified passing suites (~10 B)
- High-precision reasoning traces — majority-vote pass rate ≥ 0.9 (~7 B)

The 50 B figure is intentionally small and high-precision. The OLMo2 result — GSM8K jumping from 24 % to 67 % via concentrated anneal exposure — demonstrates exactly this leverage. Diluting the reserve with medium-quality tokens destroys it.

**Tier B — final 15 % of training steps (cooldown phase):**  
During the final ~71 000 steps, the learning rate follows a cosine decay to near-zero and the mixture shifts to a broader high-quality corpus, with the 50 B Tier A corpus concentrated here.

| Component | Share of anneal corpus |
|-----------|----------------------|
| 50 B best-data corpus (Tier A) | 35 % |
| High-quality English web (FineWeb-Edu ≥ 4.5) | 25 % |
| STEM papers and textbooks | 20 % |
| Verified Indic Tier 1 only | 10 % |
| Reasoning traces (majority-vote pass rate ≥ 0.8) | 10 % |

---

## 6. Benchmark Targets per Lane

Each lane is pinned to the benchmarks it is primarily responsible for moving. A reviewer can verify the coupling by ablating a lane and confirming that its primary benchmarks degrade more than others.

| Lane | Primary Benchmark | Reference Score | V5 Target |
|------|------------------|----------------|----------|
| Web Foundation | MMLU 5-shot | 78.3 | **> 86.0** |
| Web Foundation | HellaSwag | 85.1 | ≥ 88.0 |
| Web Diverse | MMLU-Pro | 52.4 | ≥ 58.0 |
| Web Diverse | GPQA Diamond | 38.5 | ≥ 45.0 |
| Code | HumanEval pass@1 | 71.5 | **> 87.0** |
| Code | LiveCodeBench (3-month) | 45.2 | ≥ 54.0 |
| Code | SWE-Bench Verified | 21.8 % | **> 30.0 %** |
| Code | BFCL function-calling | ~65.0 | **> 80.0** |
| STEM & Math | MATH-500 | 62.4 | **> 77.0** |
| STEM & Math | GSM8K | 78.0 | **> 91.0** |
| STEM & Math | GPQA Diamond | 38.5 | ≥ 45.0 |
| Indic | IndicGenBench | — | **≥ 70.0** |
| Indic | FLORES-200 Hindi→En chrF | 57.3 | ≥ 64.0 |
| Indic | IndicQA multilingual | 51.2 | ≥ 63.0 |
| Agentic | GAIA Level 1–2 | 28.4 | ≥ 38.0 |
| Agentic | Tau-bench tool-use | 41.2 | ≥ 52.0 |
| Reasoning | BBH 3-shot CoT | 68.9 | ≥ 75.0 |
| Reasoning | ARC-Challenge | 74.3 | ≥ 80.0 |

*Reference scores from Gemma 4 27 B published evaluations. Indic reference scores estimated from published multilingual evaluations.*

### India-perspective sovereign objective

The model must return contextually accurate and culturally grounded responses on questions spanning Indian history, law, governance, science, and society. Named evaluation sources:

- Lok Sabha and Rajya Sabha debate archives (legislative reasoning)
- Supreme Court judgment summaries (Indian legal reasoning)
- NCERT textbook question sets (K-12 curriculum in Indian context)
- RBI and MoF publications (Indian macroeconomic and financial knowledge)
- JEE / NEET / IIT-JEE entrance problems (Indian higher-education reasoning)
- Indian classical philosophical and literary texts (cultural grounding)

A held-out panel of **5 000 human-curated prompts** covering these domains in both English and the top-5 Indic languages is required. A score ≥ 72 % on this panel is required before the model is considered to have passed the Indic sovereignty gate.

---

## 7. Difficulty and Reasoning-Length Bands

Training data is stratified into four bands. The OPUS selector is calibrated differently per band.

### Band 1 — Elementary / Short-Reasoning (≤ 256 tokens output)

**Definition:** Factual recall, single-step inference, simple instruction-following.  
**Share of training data:** ~35 %  
**OPUS behaviour:** Aggressively downsampled after 30 % of training elapsed.

**Example:**
> **Input:** What is the capital of Odisha?  
> **Output:** The capital of Odisha is Bhubaneswar.

Indic examples at this band are protected from OPUS downsampling via the always-on floor.

### Band 2 — Intermediate / Multi-Step (256–1 024 tokens output)

**Definition:** Multi-hop reasoning, code functions under 50 lines, structured explanation with two or more logical dependencies.  
**Share of training data:** ~40 %  
**OPUS behaviour:** High activity throughout training; variance in learning signal remains informative into late stages.

**Example:**
> **Input:** A train leaves Mumbai at 06:00 at 80 km/h. Another leaves Pune (148 km away) at 07:30 at 60 km/h towards Mumbai. At what time do they meet, and how far from Mumbai?  
> **Output:** Step 1: In 1.5 h the first train covers 120 km. Remaining gap = 28 km. Step 2: Closing speed = 140 km/h. Time to close = 28/140 h = 12 min. Step 3: They meet at 09:12, at 120 + 80×(12/60) = 136 km from Mumbai.

### Band 3 — Advanced / Long Reasoning (1 024–4 096 tokens output)

**Definition:** Multi-turn agent trajectories, long mathematical proofs, multi-file code generation, structured research summaries.  
**Share of training data:** ~18 %  
**OPUS behaviour:** Most active here; gated behind 15 % of training elapsed.

**Example:**
> **Input:** Write a Python function that ingests a CSV of bank transactions, detects anomalies using IQR on a per-merchant basis, flags them, and outputs a JSON summary. Include docstrings and unit tests.  
> **Output:** ~150 lines of Python including module structure, IQR logic, parameterised test cases, and a JSON schema annotation — approximately 1 800 tokens.

### Band 4 — Expert / Extended Chain-of-Thought (> 4 096 tokens output)

**Definition:** Competition mathematics with full proof, long-horizon agentic trajectories with 10+ tool calls, multi-document synthesis, complex cross-lingual summarisation.  
**Share of training data:** ~7 %  
**OPUS behaviour:** Near-baseline selectivity — every verified Band 4 example is valuable. Not introduced until 25 % of training elapsed.

**Example:**
> **Input:** Prove that for all positive integers n, 1/1² + 1/2² + … + 1/n² < 2 − 1/n, and generalise the bound.  
> **Output:** Full induction proof: base case verification, inductive step via telescoping comparison, generalisation toward ζ(2) < 2 via Cauchy condensation, and extension to the family of bounds. Full LaTeX-rendered mathematical exposition of approximately 2 400 tokens.

---

## 8. Tokenizer Fertility Targets

Fertility = tokens per word on the Flores-200 sentence benchmark. **Lower is better.** GPT-4 actual fertility numbers are the baseline the V5 tokenizer must beat:

| Language | GPT-4 actual fertility (×) | V5 target fertility | Improvement |
|----------|---------------------------|--------------------|-----------:|
| Hindi (Devanagari) | 3.7 | ≤ 1.5 | 2.5× |
| Bengali | 3.2 | ≤ 1.5 | 2.1× |
| Telugu | 6.3 | ≤ 2.0 | 3.2× |
| Tamil | 5.7 | ≤ 2.0 | 2.9× |
| Gujarati | 4.5 | ≤ 1.7 | 2.6× |
| Kannada | 5.8 | ≤ 2.0 | 2.9× |
| Malayalam | 6.8 | ≤ 2.2 | 3.1× |
| Odia | 5.5 | ≤ 2.0 | 2.75× |
| Urdu | 4.8 | ≤ 1.7 | 2.8× |
| Punjabi | 4.9 | ≤ 1.8 | 2.7× |
| English | ~1.15 | ≤ 1.20 | Baseline |
| Python / JavaScript | ~1.10 | ≤ 1.05 | Marginal |
| LaTeX / math notation | ~1.40 | ≤ 1.25 | 1.1× |
| JSON / tool schemas | ~1.30 | ≤ 1.15 | 1.1× |

**Tokenizer vocabulary: 196 608 tokens** (divisible by 256; required for tensor alignment on modern GPU/TPU hardware).

| Segment | Allocated entries |
|---------|----------------:|
| English BPE subwords | ~65 000 |
| Indic script subwords (11 languages) | ~80 000 |
| Code tokens (keywords, identifiers, operators) | ~25 000 |
| Math / LaTeX commands | ~8 000 |
| JSON schema keys and agentic trace patterns | ~6 000 |
| Emoji, special characters, cross-script | ~4 000 |
| Reserved | ~8 608 |
| **Total** | **196 608** |

**Information multiplier:** With V5 Hindi fertility of 1.5 vs. GPT-4's 3.7, 800 B Hindi tokens correspond to approximately 533 B Hindi words seen by the model, vs. ~216 B words at GPT-4 fertility. This is a **2.5× information multiplier from tokenizer design alone** — making the tokenizer one of the three highest-leverage decisions in the plan alongside the mixture and the anneal corpus.

---

## 9. Curriculum Schedule Across Training Stages

| Stage | Parameters | Token budget | Data mix philosophy | Banned from this stage |
|-------|-----------|-------------|---------------------|----------------------|
| Seed | 1 B | 20 B | World knowledge + language foundation | Band 4; RLVR; long-context; agentic multi-step |
| Growth 1 | 3 B | 60 B | Foundation + code basics + elementary math | Band 4; agentic multi-step |
| Growth 2 | 8 B | 200 B | Full mix minus expert reasoning | Band 4 delayed until 25 % elapsed |
| **Full** | **~40 B** | **8 T** | Full mix; Band 4 at 25 %; anneal at step 85 % | Nothing banned |

**Stage-level mixture (fraction of each lane):**

| Lane | 1 B seed | 3 B | 8 B | Full (main) | Full (anneal) |
|------|---------|-----|-----|-------------|---------------|
| Web Foundation | 42 % | 32 % | 28 % | 28 % | 25 % |
| Web Diverse | 15 % | 13 % | 11 % | 11 % | 0 % |
| Code | 8 % | 14 % | 16 % | 18 % | 20 % |
| STEM & Math | 8 % | 12 % | 13 % | 13 % | 20 % |
| Indic (always-on) | 10 % | 10 % | 10 % | 10 % | 10 % |
| Agentic | 0 % | 5 % | 8 % | 8 % | 0 % |
| Reasoning | 0 % | 4 % | 8 % | 8 % | 15 % |
| Best-data anneal (50 B Tier A) | 0 % | 0 % | 0 % | 0 % | 10 % |

*Remaining budget in sub-full stages filled proportionally by Web Foundation and Web Diverse.*

**Curriculum rationale:** The seed stage must establish basic language and factual structure before code and reasoning examples produce useful learning signal. Band 4 is withheld until 25 % of training has elapsed and the model has stable multi-step reasoning under Bands 2 and 3. The anneal corpus is released only at step 85 %, when the model is ready to extract maximum value from the highest-quality material.

---

## 10. Proxy Experiments

> **Core principle:** Every mixture number in this document is a hypothesis. No number is committed to the full-scale run without a cheaper experiment first.

### 10.0 Pre-proxy: 140 M model — 7-step ablation protocol

Before the 1 B proxy runs, a **140 M parameter model** is trained for 7 directional ablation runs (~4 GPU-hours per run) to prune the search space.

| Step | Ablation | Decision output |
|------|----------|----------------|
| 1 | Baseline: V4 mixture ratios | Reference loss curve |
| 2 | Code share 13 % → 18 % | Direction confirmed or reversed |
| 3 | STEM share 14 % → 13 % | Direction confirmed or reversed |
| 4 | Reasoning share 0 % → 8 % at seed stage | Additive / neutral / harmful |
| 5 | Indic synthetic fraction 20 % vs. 0 % | Additive / neutral / harmful |
| 6 | Anneal: 50 B reserve vs. no reserve | Perplexity gap measured |
| 7 | Tokenizer 196 K vocab vs. 131 K | Token/word ratio and loss compared |

Any step where the 140 M signal is ambiguous (< 0.5 % perplexity difference) defaults to the conservative option and is retested at 1 B scale.

### 10.1 Proxy experiment A — 1 B scale (20 B tokens, ~1.5 days on 8× H100)

**Hypothesis:** The proposed 18 % code share outperforms the V4 baseline of 13 % on coding and non-coding benchmarks without degrading Indic validation perplexity.

| Run | Code % | STEM % | Indic % | Web % |
|-----|--------|--------|---------|-------|
| A1 (V4 baseline) | 13 | 14 | 10 | 63 |
| A2 (proposed) | 18 | 13 | 10 | 59 |
| A3 (high-code) | 22 | 13 | 10 | 55 |

**Metrics (every 2 B tokens):** HumanEval pass@1, GSM8K, MMLU 5-shot, BFCL, Indic validation perplexity (Hindi + Tamil held-out), HellaSwag.

**Decision rule:** A2 accepted over A1 if HumanEval improves ≥ 2 points **and** Indic perplexity does not increase > 2 %. If A3 improves further without degradation, code share is revised upward.

**Refutation trigger:** Indic perplexity increases > 5 % in A2 vs. A1 → revert to 13 % code; raise Indic floor to 11 %.

### 10.2 Proxy experiment B — 3 B scale (60 B tokens, ~4 days on 8× H100)

**Hypothesis:** 8 % synthetic CoT in Lane 7 improves MATH-500 and BBH without degrading factual recall.

| Run | CoT % | Source |
|-----|-------|--------|
| B1 | 0 | No CoT data |
| B2 (proposed) | 8 | Synthetic: NuminaMath + RLVR rollouts |
| B3 | 8 | Human-verified only: NuminaMath curated |

**Metrics:** MATH-500 (primary), BBH 3-shot CoT, GSM8K, TriviaQA (must not degrade), MMLU (must not degrade).

**Decision rule:** B2 accepted if MATH-500 improves ≥ 3 points over B1 with no TriviaQA or MMLU degradation. If B3 outperforms B2 by ≥ 2 points, synthetic CoT is partially replaced with curated traces if supply permits.

**Refutation trigger:** TriviaQA degrades > 2 points in B2 vs. B1 → reduce synthetic CoT to 4 %; redistribute to STEM.

### 10.3 Proxy experiment C — 1 B scale — Indic synthetic quality gate (20 B tokens)

**Hypothesis:** IndicTrans2-generated synthetic Indic data at 20 % of the Indic lane is additive or neutral vs. verified + unverified only.

| Run | Indic synthetic % | Indic verified % |
|-----|------------------|-----------------|
| C1 | 0 | 100 of available verified/unverified |
| C2 (proposed) | 20 | 80 |
| C3 | 40 | 60 |

**Metrics:** IndicQA accuracy (Hindi, Tamil, Telugu), Indic validation perplexity, cross-lingual transfer En→Hi chrF++.

**Decision rule:** C2 accepted if IndicQA holds or improves vs. C1 and C3 shows degradation. If C3 also improves, the synthetic ceiling in §3.3 is revised upward with full documentation.

### 10.4 Timeline

| Experiment | Scale | Runtime | Gate |
|-----------|-------|---------|------|
| 140 M ablation (7 steps) | 140 M | ~28 GPU-hours total | Before 1 B runs |
| A1, A2, A3 | 1 B | 1.5 days each | Before full-scale run |
| B1, B2, B3 | 3 B | 4 days each | Before full-scale run |
| C1, C2, C3 | 1 B | 1.5 days each | Before full-scale run |

No mixture number is marked **confirmed** in the training config until its corresponding proxy experiment has completed. Unconfirmed numbers are flagged explicitly.

---

## 11. Data-Gating and Cleaning Obligations

### Gate 1 — Document-level quality filter

Each document must pass:
- Language identification confidence ≥ 0.95 (fastText LangDetect)
- DCLM educational quality score ≥ 2.5 / 5.0
- No > 30 % repetition of 5-grams within the document
- Length ≥ 128 tokens
- **Hinglish / code-switching exemption:** Documents with ≥ 20 % cross-script token mixing are routed to the cross-lingual Indic sub-lane and are **not filtered** by the monolingual quality gate
- **Script-aware classifiers:** A separate FastText-based classifier is trained per script family (Devanagari, Bengali, Tamil/Telugu/Kannada/Malayalam, Perso-Arabic for Urdu); quality scores are computed within script family

### Gate 2 — Benchmark contamination check (lexical + semantic)

Canary sets from: MMLU, HumanEval, MATH-500, GSM8K, GPQA, IndicQA, SWE-Bench, BFCL, and the internal human-evaluation panel.

**Lexical check:** MinHash with 128 hash functions, Jaccard threshold 0.8, at the paragraph level. Any document with n-gram overlap > 80 % with a benchmark question or its canonical answer is removed globally.

**Semantic check:** A lightweight sentence encoder (BGE-small-en) computes cosine similarity between every paragraph and every benchmark question embedding. Paragraphs with similarity > 0.85 are flagged for manual review. This catches paraphrased or translated contamination that lexical overlap misses.

### Gate 3 — Deduplication (per-snapshot)

Global deduplication is not used. Per-snapshot deduplication is applied within each annual CC snapshot. Cross-snapshot duplicates are retained because the same topic discussed in 2020 and 2024 may carry meaningfully different framing.

### Gate 4 — Indic script verification

All Indic-labelled documents must pass a script-purity check: ≥ 70 % of Unicode characters fall within the expected script range for the declared language. Romanised transliterations are excluded from Tier 1 (verified) and flagged in Tier 2 (unverified).

### Gate 5 — License audit

- CC-BY: usable; attribution in model card required
- CC-BY-SA: usable; model card must note share-alike obligation
- CC-BY-NC: **not usable** for commercial release
- **GODL (Government Open Data Licence — India):** usable for research and commercial applications; attribution required
- All-rights-reserved / scraped without license: **not usable**

### 11.5 DeepSeekMath-style domain-mining loop

For Lanes 3 (code) and 4 (STEM/math), a 5-step iterative domain-mining loop surfaces relevant web content that standard crawl pipelines miss:

1. **Seed:** Collect ~1 B tokens of known high-quality domain text
2. **Classifier:** Train a FastText binary classifier on seed vs. random CC sample
3. **Crawl:** Apply classifier to all CC snapshots; retain pages with confidence ≥ 0.7
4. **Expand:** Add newly discovered high-confidence pages to the seed set
5. **Repeat:** Re-train classifier on expanded seed; stop after 3 iterations or when marginal yield < 5 % new tokens

All mined tokens pass through all five standard gates before admission.

### Cumulative token targets

| Lane | Target | Current estimate | Gap | Priority |
|------|--------|----------------|-----|----------|
| Web Foundation | 2.24 T | 1.90 T | 340 B | Medium |
| Web Diverse | 880 B | 720 B | 160 B | Medium |
| Code | 1.44 T | 1.10 T | 340 B | **High** |
| STEM & Math | 1.04 T | 500 B unique | Synth + mining needed | **High** |
| Indic | 800 B | 371 B committed | 429 B via repeat/synth | **High** |
| Agentic | 640 B | 350 B | 290 B synthetic needed | **Critical** |
| Reasoning | 640 B | 230 B | 410 B RLVR needed | **Critical** |

---

## 12. Supply Constraints and Honest Accounting

| Lane | Nominal budget | Verified unique supply | Repeat / synthetic needed | Risk |
|------|--------------|---------------------|--------------------------|------|
| Web Foundation | 2.24 T | ~2.8 T available | No repeat needed | Low |
| Web Diverse | 880 B | ~1.0 T available | No repeat needed | Low |
| Code | 1.44 T | ~1.1 T after dedup | 340 B via mining loop | Low–Medium |
| STEM & Math | 1.04 T | ~450 B verified unique | 590 B synthetic or 2× repeat | **Medium** |
| **Indic** | **800 B** | **~50 B verified; ~145 B unverified** | **605 B via translation + repeat** | **High** |
| Agentic | 640 B | ~55 B verified traces | 585 B via long-doc + synthetic | **High** |
| Reasoning | 640 B | ~35 B verified CoT | 605 B RLVR + synthetic | **High** |

**Per-language Indic verified actuals (Sangraha):**

| Language | Headline (B) | Verified (B) | Verified % |
|----------|-------------|-------------|-----------|
| Hindi | 34.5 | ~7–8 | ~22 % |
| Telugu | 16.3 | 3.7 | 23 % |
| Odia | 12.5 | 1.2 | 10 % |
| Bengali | ~18.0 | ~3.6 | ~20 % |
| Tamil | ~14.0 | ~3.1 | ~22 % |

Any token count that cannot be traced to a named dataset with a recorded version and download date is not counted. Gaps are filled only by documented synthetic pipelines with quality validation — not by inflating headline numbers.

---

## 13. License and Provenance Register

| Dataset | License | Commercial use | Share-alike | Action required |
|---------|---------|--------------|-------------|----------------|
| Common Crawl | CC0 + mixed | Yes | No | None; attribute CC |
| FineWeb / FineWeb-Edu | ODC-By | Yes | No | Attribution in model card |
| The Stack v2 | Mixed per-file | Depends | No | Filter non-commercial files |
| StarCoder corpus | BigCode OpenRAIL | Yes (with conditions) | Use-based restrictions | Comply with §4 of OpenRAIL |
| Wikipedia | CC-BY-SA 4.0 | Yes | **Yes** | Model card note; pre-training only |
| arXiv | arXiv non-exclusive | Yes (with limits) | No | Text extraction permissible; do not redistribute PDFs |
| Sangraha / AI4Bharat | CC-BY 4.0 | Yes | No | Attribution required |
| IndicCorp v2 | CC-BY 4.0 | Yes | No | Attribution required |
| Samanantar | CC-BY 4.0 | Yes | No | Attribution required |
| NuminaMath CoT | CC-BY 4.0 | Yes | No | Attribution required |
| OpenR traces | Apache 2.0 | Yes | No | None |
| Sky-T1 | Apache 2.0 | Yes | No | None |
| SCROLLS | Apache 2.0 | Yes | No | None |
| ToolBench | CC-BY 4.0 | Yes | No | Attribution required |
| GSM8K | MIT | Yes | No | None |
| **Lok Sabha / Rajya Sabha transcripts** | **GODL (India)** | **Yes** | No | Attribution to Parliament of India |
| **RBI / MoF / SEBI publications** | **GODL (India)** | **Yes** | No | Attribution to respective ministry |
| **SC judgments (open portion)** | **GODL (India)** | **Yes** | No | Attribution to Supreme Court of India |
| **NCERT textbooks** | **GODL (India)** | **Yes** | No | Attribution to NCERT |

**GODL as a data moat:** The Government Open Data Licence covers a significant body of Indian institutional text — parliamentary debates, court judgments, regulatory publications, and school curriculum materials — that is systematically absent from all non-Indian foundation models. Incorporating it under GODL represents a genuine and defensible competitive advantage that cannot be replicated without equivalent institutional access.

**Model release policy:** The model will be released under a custom open-weight licence modelled on Llama 3 Community Licence with an added clause that any fine-tune trained primarily on the model's Indic-language outputs must contribute its SFT data back to the commons under CC-BY 4.0. This is a community norm, not a legal obligation, and is disclosed transparently.

---

## 14. Risk Register

| Risk | Probability | Impact | Mitigation |
|------|------------|--------|-----------|
| Synthetic Indic degrades benchmark | Medium | High | Proxy experiment C; hard 20 % synthetic cap; secondary classifier gate |
| RLVR rollout quality below threshold | Medium | High | Majority-vote filter (pass rate ≥ 0.8); reduce reasoning to 5 % if needed |
| Benchmark contamination in web crawl | Low–Medium | High | Gate 2 dual-layer: lexical MinHash + semantic similarity |
| OPUS selector crowds out Indic below floor | Low | High | Hard floor clamp; not subject to OPUS override |
| Semantic contamination missed by MinHash | Medium | High | BGE-small-en similarity check; fuzzy dedup on translated Indic |
| License dispute on Stack v2 / arXiv | Low | Medium | Per-file filter; legal review before release |
| Compute insufficient for 3× proxy suites | Medium | Medium | Proxy A and C share 1 B infrastructure; B runs sequentially |
| Fertility target not met by tokenizer deadline | Medium | High | Fallback: V4 chrono-tokenizer with Indic extensions; accept fertility penalty |
| Agentic lane chronically undersupplied | High | Medium | Reduce Lane 6 to floor (5 %); reallocate 3 % to Code and STEM |
| Domain-mining loop produces train-set near-duplicates | Medium | Medium | Gate 2 and Gate 3 applied to all mined tokens before admission |
| Hinglish text filtered by monolingual quality gate | Low (post-mitigation) | Medium | Code-switching exemption hardcoded in Gate 1 |
| 50 B anneal corpus leaks into main run | Low | High | Separate shard directory; dataloader hard-excludes until step 85 % |

---

## 15. Summary: The Plan as a Single Testable Hypothesis

The V5 pre-training mixture is stated as a single falsifiable claim:

> *A model trained on 8 T tokens with 28 % web foundation, 18 % code, 13 % STEM, 10 % Indic (always-on floor 8 %), 8 % agentic, and 8 % reasoning — with a 50 B best-data corpus segregated from day one and released during the final 15 % cooldown phase — will score > 86 on MMLU, > 87 on HumanEval, > 77 on MATH-500, > 91 on GSM8K, > 30 % on SWE-Bench Verified, > 80 on BFCL, and ≥ 70 on IndicGenBench, while matching or exceeding the reference scores stated in §6 across all capability lanes.*

This claim will be partially validated by proxy experiments at 140 M (§10.0), 1 B (§10.1, §10.3), and 3 B (§10.2) scale before any token of the full-scale run is committed. Each proxy experiment states a concrete decision rule: if the experiment refutes the sub-hypothesis, the corresponding lane share is revised according to the stated rule, and this document is updated with the new numbers and the evidence behind them.

**A data decision is a hypothesis until a cheap experiment has tested it.**

---
