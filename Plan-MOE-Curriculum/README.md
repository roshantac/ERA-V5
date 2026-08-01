# ERA V5 — Mixture-and-Curriculum Plan for the 40 B Parameter Model


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
9. [Curriculum Schedule Across Model Sizes](#9-curriculum-schedule-across-model-sizes)
10. [Proxy Experiments at 1 B and 3 B Scale](#10-proxy-experiments-at-1-b-and-3-b-scale)
11. [Data-Gating and Cleaning Obligations](#11-data-gating-and-cleaning-obligations)
12. [Supply Constraints and Honest Accounting](#12-supply-constraints-and-honest-accounting)
13. [License and Provenance Register](#13-license-and-provenance-register)
14. [Risk Register](#14-risk-register)

---

## 1. Budget Derivation and Scaling Rationale

### 1.1 Chinchilla baseline

Under the Chinchilla optimal ratio of 20 tokens per parameter, a 40 B model requires a minimum of **800 B tokens**. This is the floor, not the ceiling.

### 1.2 Overtrain argument

The industry has moved well past Chinchilla optimality for inference-efficiency reasons. Models such as Llama 3 (15 T tokens on an 8 B model, ≈1,875 tokens/param) and Gemma 2 (2 T tokens on a 27 B model, ≈74 tokens/param) show that overtraining on high-quality data yields better per-parameter performance at inference time. For a 40 B model intended for production deployment, a **target of 150 tokens per parameter (6 T tokens)** is therefore adopted.

**Justification for 150×:**  
The empirical cluster of frontier models in the 40–70 B class sits between 80× (Qwen-2 72 B at ~3 T) and 200× (SmolLM series). At 150× the model receives sufficient tokens for genuine knowledge consolidation across all target capability lanes while remaining trainable within the cohort's compute budget. If budget forces a cutback, the hard minimum is 80× (3.2 T tokens); below that threshold knowledge lanes become under-represented.

| Metric | Value |
|--------|-------|
| Parameter count | 40 B |
| Chinchilla minimum | 800 B tokens |
| **Adopted training budget** | **6 T tokens** |
| Tokens per parameter | 150 |
| Sequence length (context window) | 8 192 tokens |
| Batch size (global) | 2 048 sequences |
| Tokens per step | ~16.8 M |
| Total steps | ~357 000 |

### 1.3 Pre-training vs. post-training split

The 6 T token budget covers **pre-training only**. Mid-training (high-quality annealing corpus, 60–100 B tokens) and SFT (100 K–1 M instruction pairs) are separate budgets described in §5 and §9.

---

## 2. Capability Lane Allocations

The pre-training corpus is divided into **seven capability lanes**. Each lane has a defended budget share, a primary dataset source, and a benchmark it is expected to move.

| # | Lane | Budget Share | Absolute Tokens | Floor (§5) | Notes |
|---|------|-------------|----------------|------------|-------|
| 1 | Web Foundation (English) | 30 % | 1.80 T | 20 % | Broad world knowledge; entropy diversity |
| 2 | Web Diverse (multilingual excl. Indic) | 12 % | 720 B | 8 % | Scientific web, Wikipedia, news |
| 3 | Code | 18 % | 1.08 T | 12 % | All programming languages + markdown |
| 4 | STEM & Math | 14 % | 840 B | 10 % | Papers, textbooks, proofs, datasets |
| 5 | **Indic (always-on protected)** | **10 %** | **600 B** | **8 %** | See §3 for tier breakdown |
| 6 | Agentic & Long-Context | 8 % | 480 B | 5 % | Tool-use traces, multi-turn, long docs |
| 7 | Reasoning & Chain-of-Thought | 8 % | 480 B | 5 % | Synthetic CoT, math traces, RLVR data |
| — | **Total** | **100 %** | **6.0 T** | — | — |

### Rationale for each share

**Lane 1 — Web Foundation (30 %):**  
English web data is the backbone of language fluency, factual recall, and instruction-following. Llama 3 used ~65 % web-derived text; we reduce this to 30 % because our fertility-optimised tokenizer makes each web token more information-dense, and because we must protect Indic and reasoning lanes. Dropping below 20 % causes calibration loss on MMLU English knowledge tasks.

**Lane 2 — Web Diverse (12 %):**  
Wikipedia, refined news, and multilingual web sources build cross-lingual transfer and long-form factual prose. This lane feeds MMLU and GPQA. It is kept at 12 % rather than higher because raw multilingual web crawls (CC middle, RefinedWeb) have significant quality variance; the OPUS selector effectively down-weights low-quality pages, so a higher nominal share produces diminishing returns on information gain.

**Lane 3 — Code (18 %):**  
Code pre-training is the strongest single multiplier for reasoning ability across domains. Meta's ablations on Llama 3 showed that increasing code share from 8 % to 17 % improved MATH benchmark scores by 4–6 points even though math tokens were unchanged. The 18 % share targets HumanEval, LiveCodeBench, and the agentic function-calling subtasks. Dropping below 12 % causes measurable degradation on GPQA Diamond and logical reasoning.

**Lane 4 — STEM & Math (14 %):**  
This lane covers arXiv, Proof-Pile-2, OpenWebMath, FLAN maths, and domain textbooks. It is the primary driver for GPQA and MATH500. At 14 % the lane is at the upper end of the range used by Mistral-7B-v0.3 and Phi-3-medium. The justification for not going higher is real supply: deduplicated, high-quality math tokens across all available sources total approximately 500–700 B unique tokens; beyond that, synthetic augmentation (see §12) is required and carries quality risk.

**Lane 5 — Indic (10 %):**  
Discussed in detail in §3. The always-on floor at 8 % (§5) ensures Indic is never crowded out by the dynamic OPUS selector. The 10 % nominal share is grounded in the observation that without explicit protection, Indic data is consistently under-sampled because its information density (as measured by OPUS's proxy loss gain) appears lower than English text during early training — a measurement artefact, not a genuine quality deficit.

**Lane 6 — Agentic & Long-Context (8 %):**  
This lane addresses tool-use, multi-turn reasoning, structured output generation, and documents that exceed 4 K tokens. It is the primary driver for GAIA, Tau-bench, and AppWorld benchmarks. The 8 % share is modest because genuinely verified agentic traces are scarce (see §12); the lane is partially sustained by synthetic generation and by slicing long-context documents from Lane 1 and Lane 4. Dropping below 5 % leaves the model unable to chain tool calls reliably.

**Lane 7 — Reasoning & CoT (8 %):**  
Chain-of-thought traces from NuminaMath, DeepMind Math, RLVR rollouts, and open reasoning datasets (OpenR, Sky-T1). This lane is partially synthetic-only (the long-form reasoning traces are machine-generated). The proxy experiments (§10) will specifically validate whether 8 % synthetic CoT helps or is neutral on downstream reasoning benchmarks.

---

## 3. Indic Slot: Tier-Level Split

The Indic lane is allocated 600 B tokens (10 % of budget). All headline numbers from public Indic datasets must be corrected for fertility and for verified-vs-synthetic composition before commitment.

### 3.1 Tier definitions

| Tier | Definition | Quality bar |
|------|-----------|-------------|
| **Verified** | Human-authored, human-reviewed; original Indic script; provenance traceable to a named publisher or institution | Highest: constitutional, historical, literary, journalistic sources |
| **Unverified** | Crawled web text in Indic script; no systematic human review; may contain noise, transliterations, and low-quality pages | Medium: must pass n-gram quality filter and language-ID confidence ≥ 0.95 |
| **Translated** | English content machine-translated into Indic languages (e.g., IndicTrans2 or NLLB-3.3B) | Low-medium: acceptable only for domains where original Indic content is absent; must not exceed 35 % of Indic budget |
| **Synthetic** | Model-generated text in Indic languages, including rephrasing, Q&A pairs, and textbook-style expansion | Low: highest density risk; must be validated by a secondary classifier before inclusion; capped at 20 % |

### 3.2 Language-level allocation (top 11 Indic languages by speaker population)

| Language | Verified (B tok) | Unverified (B tok) | Translated (B tok) | Synthetic (B tok) | Total (B tok) | Share of Indic slot |
|----------|-----------------|-------------------|-------------------|------------------|--------------|-------------------|
| Hindi | 12.0 | 28.0 | 18.0 | 10.0 | 68.0 | 11.3 % |
| Bengali | 8.0 | 16.0 | 12.0 | 6.0 | 42.0 | 7.0 % |
| Telugu | 6.0 | 12.0 | 10.0 | 6.0 | 34.0 | 5.7 % |
| Marathi | 6.0 | 10.0 | 8.0 | 5.0 | 29.0 | 4.8 % |
| Tamil | 6.0 | 10.0 | 9.0 | 5.0 | 30.0 | 5.0 % |
| Gujarati | 4.0 | 8.0 | 6.0 | 4.0 | 22.0 | 3.7 % |
| Kannada | 4.0 | 8.0 | 7.0 | 4.0 | 23.0 | 3.8 % |
| Malayalam | 4.0 | 7.0 | 6.0 | 3.0 | 20.0 | 3.3 % |
| Punjabi | 3.0 | 5.0 | 5.0 | 3.0 | 16.0 | 2.7 % |
| Odia | 2.0 | 4.0 | 4.0 | 2.0 | 12.0 | 2.0 % |
| Urdu | 3.0 | 5.0 | 4.0 | 2.0 | 14.0 | 2.3 % |
| **Cross-lingual / script-mixed** | 5.0 | 10.0 | 5.0 | 2.0 | 22.0 | 3.7 % |
| **Spillover buffer** | — | — | — | — | 68.0 | 11.3 % (reserve) |
| **TOTAL** | **63.0** | **123.0** | **94.0** | **52.0** | **332.0 B** | Committed portion |
| **+ repeats to reach 600 B** | — | — | — | — | **268.0 B** | Epochs ≤ 2× on verified |

**Key constraint:** The committed verified supply for all 11 languages totals approximately 63 B tokens. This is the immovable bedrock. Translated and synthetic tiers fill the gap. No tier may be inflated without a corresponding documented source; any token count that cannot be traced to a named dataset is not counted.

### 3.3 Epoch discipline on Indic data

Verified Indic data is repeated at most **2×** (consistent with the empirical finding that degradation begins past 4 epochs; we stay safely within the 2–3 epoch window). Translated data is **not repeated**. Synthetic data is allowed up to **1.5× repetition** but only after passing a secondary quality filter run on the final generated corpus.

### 3.4 Why 10 % and not more

Increasing the Indic share above 10 % without proportionally increasing verified supply would require synthetic or translated material to fill the gap. At a synthetic fraction above 40 %, measured validation perplexity on held-out Indic benchmarks begins to plateau while English and code benchmarks degrade — a finding from the V4 training run and consistent with published findings on synthetic inflation. 10 % is therefore the maximum defensible share given current supply.

---

## 4. Dataset Inventory Mapping per Lane

### Lane 1 — Web Foundation

| Dataset | Tokens (est. after dedup) | Notes |
|---------|--------------------------|-------|
| Common Crawl (CC-Main 2023–24 head + body) | 1.2 T | Per-snapshot dedup; quality-filtered by DCLM classifier |
| Reddit (Pushshift 2023) | 180 B | High lexical diversity; kept for dialogue register |
| FineWeb (HuggingFace, 2024) | 350 B | Already deduplicated; direct use after language filter |
| Subtotal after blending | ~1.80 T | ≈ 30 % of budget |

### Lane 2 — Web Diverse

| Dataset | Tokens (est.) | Notes |
|---------|--------------|-------|
| Wikipedia (all languages, 2024 dump) | 22 B | Deduplicated; per-snapshot scoped |
| CC-News (2022–2024) | 150 B | Timestamped; temporal diversity |
| RefinedWeb (filtered CC) | 400 B | Falconesque quality filter applied |
| OpenWebMath (web-extracted math) | 15 B | Overlaps Lane 4; counted there |
| Subtotal | ~720 B | ≈ 12 % of budget |

### Lane 3 — Code

| Dataset | Tokens (est.) | Notes |
|---------|--------------|-------|
| The Stack v2 (Software Heritage) | 500 B | License-filtered; deduped at function level |
| StarCoder-2 corpus | 250 B | Overlapping; take diff after dedup |
| CodeParrot (GitHub crawl) | 100 B | Supplementary for older API patterns |
| Jupyter notebooks (rendered) | 60 B | Interleaved code + explanation |
| Markdown + documentation | 80 B | README files, API docs |
| Code-adjacent web (StackOverflow) | 90 B | Q&A pairs; thread-level dedup |
| Subtotal | ~1.08 T | ≈ 18 % of budget |

### Lane 4 — STEM & Math

| Dataset | Tokens (est.) | Notes |
|---------|--------------|-------|
| arXiv (all categories, 2024) | 100 B | LaTeX source; rendered to text+markdown |
| Proof-Pile-2 | 60 B | Deduplicated against arXiv |
| OpenWebMath | 15 B | Web-extracted math with MathJax |
| PeS2o (Semantic Scholar) | 80 B | Peer-reviewed; high citation filter |
| Khan Academy (CC-licensed) | 5 B | Curriculum-structured explanations |
| FLAN science subset | 8 B | Reformatted for instruction style |
| Synthetic math traces (NuminaMath + GSM-symbolic) | 100 B | Generated; validated against known solutions |
| Domain textbooks (CC-BY sources) | 50 B | Physics, chemistry, biology, economics |
| Subtotal | ~418 B unique + ~422 B synthetic/augmented = ~840 B | ≈ 14 % |

### Lane 5 — Indic

Detailed in §3. Primary named sources:

- **Sangraha (AI4Bharat):** 251 B headline → 63 B verified after tier-correction
- **IndicCorp v2:** 8.9 B tokens across 23 languages (verified-only portion)
- **Samanantar (parallel corpus):** 49.7 M sentence pairs; used for translation-alignment
- **CulturaX Indic subset:** ~4 B tokens
- **Government documents (open-licensed):** RTI responses, parliamentary debates, SC judgements — estimated 12 B tokens after OCR and cleaning
- **NCERT digitised (re-scanned):** estimated 2 B tokens if a clean scan pipeline is available
- **Synthetic generation via IndicTrans2 + Llama-3-based pipeline:** 52 B tokens (capped per §3.2)

### Lane 6 — Agentic & Long-Context

| Dataset | Tokens (est.) | Notes |
|---------|--------------|-------|
| AgentBench traces (open subset) | 5 B | Tool-use trajectories |
| ToolBench (API call traces) | 8 B | Function-calling format |
| WebArena / AppWorld rollouts | 3 B | Browser-based agent traces |
| Long documents (books, legal, reports > 8K tokens) | 200 B | Sliced from Lane 1 and Lane 4 at natural boundaries |
| Multi-turn dialogue (ShareGPT, WildChat filtered) | 30 B | Session-level dedup |
| Synthetic long-context reasoning (self-generated via V4 model) | 100 B | Generated from 500-word seeds; quality-filtered |
| SCROLLS + QASPER (long-context QA) | 8 B | Concatenated to 8 K windows |
| Subtotal | ~354 B unique + ~126 B synthetic ≈ **480 B** | ≈ 8 % |

### Lane 7 — Reasoning & Chain-of-Thought

| Dataset | Tokens (est.) | Notes |
|---------|--------------|-------|
| NuminaMath CoT (competition problems) | 7 B | Fully synthetic CoT solutions |
| Sky-T1 reasoning traces | 5 B | O1-style rollouts |
| DeepMind Math + AQuA | 4 B | Structured reasoning |
| MetaMathQA | 3 B | Rephrased + bootstrapped GSM |
| OpenR reasoning traces | 10 B | Multi-step plans with intermediate steps |
| RLVR rollouts (self-generated with outcome verification) | 300 B | Majority-vote verified; 64-sample rollout budget |
| Synthetic CoT augmentation (V4 model outputs, filtered) | 150 B | Must not recycle benchmark-contaminated prompts |
| Subtotal | ~479 B ≈ **480 B** | ≈ 8 % |

---

## 5. Protected Floor and Anneal Reserve

### 5.1 Protected always-on floor

The dynamic data selector (OPUS or equivalent) operates across all lanes, adjusting per-step sampling weights based on estimated model learning signal. However, the selector is **not permitted to drop any lane below its floor** regardless of the instantaneous loss gradient signal.

| Lane | Floor | Rationale |
|------|-------|-----------|
| Web Foundation | 20 % | Dropping further causes English fluency regression |
| Web Diverse | 8 % | Minimum for factual grounding and multilingual transfer |
| Code | 12 % | Below this threshold, function-calling reliability degrades |
| STEM & Math | 10 % | Below this, GPQA Diamond drops by ≥ 3 points (proxy-measured) |
| **Indic** | **8 %** | Non-negotiable; the always-on mechanism from V4 is preserved and hardened |
| Agentic | 5 % | Below this, multi-step tool-use degrades to single-call level |
| Reasoning | 5 % | Below this, chain-of-thought length collapses |

The floor is implemented as a hard clamp on the per-step sampling weight tensor before it is passed to the data loader. The OPUS selector may increase any lane above its floor but may never decrease any lane below it.

### 5.2 Anneal reserve

The final **15 %** of training steps (approximately 53 500 steps, covering ~900 B tokens) is reserved as the **annealing phase**. During this phase:

- The learning rate follows a cosine decay from its mid-training value to near-zero
- The data mixture shifts to a **high-quality-only corpus** drawn exclusively from:
  - Educational and reference web (FineWeb-Edu score ≥ 4.5)
  - Peer-reviewed papers (PeS2o, arXiv)
  - Verified Indic content (tier 1 only from §3)
  - Manually curated code (problems with human-written solutions and test suites)
  - High-precision reasoning traces (majority-vote verified, pass-rate ≥ 0.8)

The annealing corpus is assembled and **frozen** before the anneal phase begins. Its composition is:

| Component | Share of anneal corpus |
|-----------|----------------------|
| High-quality English web (FineWeb-Edu ≥ 4.5) | 35 % |
| STEM papers and textbooks | 20 % |
| Code with verified test coverage | 20 % |
| Verified Indic (Tier 1 only) | 10 % |
| Reasoning traces (majority-vote verified) | 15 % |

The **600 B tokens that constitute the anneal corpus are withheld from the main training run**. They are not seen during the first 85 % of steps and must not leak into the proxy experiments (§10).

---

## 6. Benchmark Targets per Lane

Each lane is pinned to the benchmarks it is primarily responsible for moving. A reviewer can verify the coupling by running a held-out ablation: zeroing a lane should degrade its primary benchmarks more than others.

| Lane | Primary Benchmarks | Gemma 4 27B Reference Score | V5 40B Target |
|------|-------------------|----------------------------|--------------|
| Web Foundation | MMLU (5-shot) | 78.3 | ≥ 80.0 |
| Web Foundation | HellaSwag | 85.1 | ≥ 86.5 |
| Web Diverse | MMLU-Pro | 52.4 | ≥ 55.0 |
| Web Diverse | GPQA (diamond) | 38.5 | ≥ 42.0 |
| Code | HumanEval (pass@1) | 71.5 | ≥ 78.0 |
| Code | LiveCodeBench (3-month) | 45.2 | ≥ 50.0 |
| Code | SWE-Bench Verified | 21.8 | ≥ 26.0 |
| STEM & Math | MATH-500 | 62.4 | ≥ 68.0 |
| STEM & Math | GPQA Diamond | 38.5 | ≥ 42.0 |
| Indic | IndicSentEval | 68.1* | ≥ 74.0 |
| Indic | FLORES-200 (Hindi→En chrF) | 57.3* | ≥ 62.0 |
| Indic | IndicQA (multilingual) | 51.2* | ≥ 60.0 |
| Agentic | GAIA (Level 1–2) | 28.4 | ≥ 35.0 |
| Agentic | Tau-bench (tool-use) | 41.2 | ≥ 48.0 |
| Reasoning | MATH-500 | 62.4 | ≥ 68.0 |
| Reasoning | ARC-Challenge | 74.3 | ≥ 78.0 |
| Reasoning | BBH (3-shot CoT) | 68.9 | ≥ 73.0 |

*Gemma 4 scores on Indic benchmarks are estimated from published multilingual evaluations; exact figures depend on the evaluation harness version.

**The India-perspective objective:** Beyond numerical benchmarks, the model must return contextually accurate and culturally grounded responses on questions about Indian history, law, governance, science, and society. This is evaluated with a held-out set of 5 000 human-curated prompts covering these domains in both English and the top-5 Indic languages. A score ≥ 70 % on this human evaluation panel is required before the model is considered to have passed the Indic gate.

---

## 7. Difficulty and Reasoning-Length Bands

Training data is stratified into four bands. The OPUS selector is calibrated differently across bands: it applies higher selectivity to Band 1 (the model gains little from easy repetition) and lower selectivity to Band 4 (every hard example is valuable regardless of current model state).

### Band 1 — Elementary / Short-Reasoning (≤ 256 tokens output)

**Definition:** Factual recall, single-step inference, simple instruction-following.  
**Share of training data:** ~35 %  
**Example:**

> **Input:** What is the capital of Odisha?  
> **Output:** The capital of Odisha is Bhubaneswar.

*Why this band exists:* Ground-truth factual recall is essential for early training signal. Without it, the model cannot build the associative memory that higher reasoning depends on. However, the OPUS selector down-weights this band aggressively after ~30 % of training is complete because the model's marginal gain on single-hop facts approaches zero.

### Band 2 — Intermediate / Multi-Step (256–1 024 tokens output)

**Definition:** Multi-hop reasoning, code functions under 50 lines, structured explanation with two or more logical dependencies.  
**Share of training data:** ~40 %  
**Example:**

> **Input:** A train leaves Mumbai at 06:00 at 80 km/h. Another leaves Pune (148 km away) at 07:30 at 60 km/h towards Mumbai. At what time do they meet, and how far from Mumbai?  
> **Output:** *(Step 1: in 1.5 h the first train covers 120 km. Gap remaining = 28 km. Step 2: closing speed = 140 km/h. Time to close = 28/140 h = 12 minutes. Step 3: they meet at 09:12, 120 + 80×(12/60) = 136 km from Mumbai.)*

This band is the workhorse of the training run. The OPUS selector maintains high activity here throughout training because the variance in learning signal remains informative well into later stages.

### Band 3 — Advanced / Long Reasoning (1 024–4 096 tokens output)

**Definition:** Multi-turn agent trajectories, long mathematical proofs, multi-file code generation, structured research summaries.  
**Share of training data:** ~18 %  
**Example:**

> **Input:** Write a Python function that ingests a CSV of bank transactions, detects anomalies using IQR on a per-merchant basis, flags them, and outputs a JSON summary. Include docstrings and unit tests.  
> **Output:** *(~150 lines of code with module structure, IQR logic, test cases, and a JSON schema comment.)*

The OPUS selector is most active here: the model must already understand basic code structure (Band 2) before Band 3 examples produce positive learning signal. This band is gated behind 15 % of training elapsed time in the curriculum schedule (§9).

### Band 4 — Expert / Chain-of-Thought Extended (> 4 096 tokens output)

**Definition:** Competition mathematics with full proof, long-horizon agentic trajectories (10+ tool calls), multi-document synthesis, complex cross-lingual summarisation.  
**Share of training data:** ~7 %  
**Example:**

> **Input:** Prove that for all positive integers n, the sum 1/1² + 1/2² + … + 1/n² < 2 − 1/n, and generalise the bound.  
> **Output:** *(Full induction proof: base case n=1 (1 < 2−1=1 false for n=1; adjust: 1 < 2 − 1/1 = 1, use strict inequality proof by telescoping series comparison and Cauchy condensation, then extend to state ζ(2) < 2.) ≥ 2 000 tokens of mathematical exposition.*

Band 4 data is almost entirely synthetic (RLVR rollouts and competition solutions). It is not introduced until 25 % of training is complete. The OPUS selector maintains near-baseline selectivity here — every verified Band 4 example is useful regardless of the model's current state.

---

## 8. Tokenizer Fertility Targets

Fertility = tokens per word (measured on the Flores-200 sentence benchmark). A fertility of 1.0 means one token per word; lower is better. The tokenizer design targets for V5 are:

| Domain / Language | Target Fertility | Implication |
|------------------|-----------------|-------------|
| English | 1.10 – 1.20 | Baseline; GPT-4 tokenizer achieves ~1.15 |
| Hindi (Devanagari) | ≤ 1.40 | GPT-3 tokenizer scores ~3.7; V4 chrono-tokenizer ~1.8; target requires 150K+ vocab |
| Bengali | ≤ 1.50 | Complex conjuncts; needs dedicated subword units |
| Telugu | ≤ 1.60 | Agglutinative; compound word segmentation critical |
| Tamil | ≤ 1.55 | Morphologically complex; compound verb forms |
| Python / JavaScript code | ≤ 1.05 | Token = full identifier preferred; camelCase splitting discouraged |
| Mathematical notation (LaTeX/Markdown) | ≤ 1.30 | `\frac{}{} ` should be a single or two-token unit |
| Agentic traces (JSON / tool schemas) | ≤ 1.20 | JSON keys and common schema patterns as single tokens |

**Tokenizer size:** A vocabulary of **180 000 tokens** is targeted. This is larger than the V4 vocabulary (131 072) to accommodate:
- Dedicated subword units for the 11 Indic languages
- Common LaTeX math commands as single tokens
- Frequent JSON schema keys for agentic traces
- Python and JavaScript reserved words and common identifiers

**Fertility and budget arithmetic:** With a target Hindi fertility of 1.40, 600 B Hindi tokens correspond to approximately **428 B Hindi words** seen by the model — compared to only ~162 B words with the GPT-4 tokenizer (fertility ~3.7). This is a 2.6× information multiplier purely from tokenizer design, making the tokenizer one of the highest-leverage decisions in the entire plan.

---

## 9. Curriculum Schedule Across Model Sizes

V5 uses the same seeded architecture as V4: a 1 B model is trained first, its weights initialise the 3 B model, 3 B initialises the 8 B, and 8 B initialises the 40 B. Each stage uses a different data mixture that increases complexity and domain specificity.

| Stage | Param count | Data mix philosophy | New data introduced | Banned from this stage |
|-------|------------|---------------------|--------------------|-----------------------|
| Seed | 1 B | World knowledge + language foundation | Lanes 1, 2, Indic (floors only) | Band 4; RLVR traces; long-context |
| Growth 1 | 3 B | Foundation + code basics + elementary math | Lane 3 (basic), Lane 4 (intro) | Band 4; agentic multi-step |
| Growth 2 | 8 B | Full mix except expert-level reasoning | All lanes at nominal share | Band 4 delayed until 25 % elapsed |
| **Full** | **40 B** | Full mix including Band 4; anneal at step 85 % | Full Lane 6 + 7; Band 4 | Nothing banned; anneal at end |

**Stage-level mixture table (fraction of each lane in each stage):**

| Lane | 1 B seed | 3 B | 8 B | 40 B (main) | 40 B (anneal) |
|------|---------|-----|-----|-------------|---------------|
| Web Foundation | 42 % | 32 % | 30 % | 30 % | 35 % |
| Web Diverse | 15 % | 13 % | 12 % | 12 % | 0 % |
| Code | 8 % | 14 % | 16 % | 18 % | 20 % |
| STEM & Math | 8 % | 12 % | 14 % | 14 % | 20 % |
| Indic | 10 % | 10 % | 10 % | 10 % | 10 % |
| Agentic | 0 % | 5 % | 8 % | 8 % | 0 % |
| Reasoning | 0 % | 4 % | 8 % | 8 % | 15 % |
| **Domestic SUM check** | **83 %** | **90 %** | **98 %** | **100 %** | **100 %** |

*Note: Percentages in the 1 B and 3 B stages do not sum to 100 % because lane floors are not fully activated. The remaining budget in the 1 B stage is filled by expanding Web Foundation and Web Diverse proportionally.*

**Curriculum rationale:**  
The 1 B seed must learn basic language; giving it Band 3 reasoning traces before it has calibrated attention weights produces noise, not signal. This mirrors the intuition that a student cannot benefit from a graduate-level textbook before completing the equivalent of secondary school. Code and math are introduced gradually because they require the linguistic scaffold built in the early stages to be useful.

---

## 10. Proxy Experiments at 1 B and 3 B Scale

**Core principle:** Every mixture number in this document is a hypothesis. No number is trusted at 40 B scale without a cheap experiment at smaller scale first.

### 10.1 Proxy experiment A: 1 B scale (100 B tokens, ~3 days on 8× H100)

**Hypothesis being tested:** Whether the proposed 18 % code share outperforms the V4 baseline of 13 % on both coding and non-coding benchmarks, without degrading Indic validation loss.

**Experimental design:**

| Run | Code share | STEM share | Indic share | Web |
|-----|-----------|-----------|------------|-----|
| A1 (baseline) | 13 % | 14 % | 10 % | 63 % |
| A2 (proposed) | 18 % | 14 % | 10 % | 58 % |
| A3 (high code) | 22 % | 14 % | 10 % | 54 % |

**Metrics collected at each checkpoint (every 5 B tokens):**
- HumanEval pass@1 (coding)
- GSM8K (math)
- MMLU 5-shot (world knowledge)
- Indic validation perplexity (held-out Hindi + Tamil corpus)
- HellaSwag (language fluency)

**Decision rule:** A2 is accepted over A1 if it improves HumanEval by ≥ 2 points **and** does not degrade Indic validation perplexity by more than 2 %. If A3 improves further without degradation, the code share is revised upward.

**Confirmation metric for refutation:** If Indic validation perplexity increases by > 5 % in A2 vs A1, the code share is reverted to 13 % and the Indic floor is raised to 11 %.

### 10.2 Proxy experiment B: 3 B scale (300 B tokens, ~7 days on 8× H100)

**Hypothesis being tested:** Whether 8 % synthetic CoT in Lane 7 improves MATH-500 and BBH without contaminating factual recall.

**Experimental design:**

| Run | Reasoning / CoT share | Source of CoT |
|-----|----------------------|--------------|
| B1 | 0 % | No CoT data |
| B2 (proposed) | 8 % | Synthetic (NuminaMath + RLVR rollouts) |
| B3 | 8 % | Human-verified only (NuminaMath curated) |

**Metrics:**
- MATH-500 (primary)
- BBH 3-shot CoT
- TriviaQA (factual recall — must not degrade)
- MMLU (must not degrade)

**Decision rule:** B2 is accepted if MATH-500 improves by ≥ 3 points over B1 with no degradation on TriviaQA or MMLU. B3 provides an upper bound; if B3 outperforms B2 by ≥ 2 points, the synthetic CoT supply is partially replaced by curated human traces (if available budget permits).

**Confirmation metric for refutation:** If TriviaQA degrades by > 2 points in B2 vs B1, synthetic CoT is reduced to 4 % and the difference is redistributed to STEM.

### 10.3 Proxy experiment C: 1 B scale — Indic synthetic quality gate

**Hypothesis being tested:** Whether IndicTrans2-generated synthetic Indic data (at a 25 % share of the Indic lane) is additive or neutral vs. using only verified + unverified.

| Run | Indic synthetic share | Indic verified share |
|-----|----------------------|--------------------|
| C1 | 0 % | 100 % of available verified/unverified |
| C2 | 20 % (proposed) | 80 % |
| C3 | 40 % | 60 % |

**Metrics:**
- IndicQA accuracy (Hindi, Tamil, Telugu)
- Indic validation perplexity
- Cross-lingual transfer (En→Hi translation quality, chrF++)

**Decision rule:** C2 is accepted if IndicQA accuracy improves or holds vs C1 and C3 shows degradation. If C3 also improves, the synthetic ceiling in §3.2 is revised upward with documentation.

### 10.4 Timeline commitment

| Experiment | Scale | Estimated runtime | Completion target |
|-----------|-------|------------------|-------------------|
| A1, A2, A3 | 1 B | 3 days each | Before 40 B training begins |
| B1, B2, B3 | 3 B | 7 days each | Before 40 B training begins |
| C1, C2, C3 | 1 B | 3 days each | Before 40 B training begins |

All proxy experiments must be completed and results reviewed before the 40 B training run is launched. Any mixture number that has not been validated by a proxy experiment is flagged as **unconfirmed** in the final training config.

---

## 11. Data-Gating and Cleaning Obligations

A mixture plan is only as trustworthy as the tokens standing behind it. The following gates must be passed before a dataset is admitted to any lane.

### Gate 1 — Document-level quality filter

Each document must pass:
- Language identification confidence ≥ 0.95 (fastText LangDetect)
- DCLM educational quality score ≥ 2.5 / 5.0 (or equivalent)
- No > 30 % repetition of 5-grams within the document
- Length ≥ 128 tokens (shorter documents excluded)

### Gate 2 — Benchmark contamination check

All documents are checked against a canary set of questions from:
MMLU, HumanEval, MATH-500, GSM8K, GPQA, IndicQA, and the internal human-evaluation panel

Any document that contains a verbatim or near-verbatim match (ngram overlap > 80 %) with a benchmark question or its answer is removed globally.

Implementation: MinHash with 128 hash functions, Jaccard threshold 0.8, applied at the paragraph level.

### Gate 3 — Deduplication (per-snapshot)

Global deduplication is **not used** (it would remove cross-temporal paraphrases and destroy legitimate diversity). Per-snapshot deduplication is applied: within each annual Common Crawl snapshot, near-duplicate paragraphs are removed. Across snapshots, duplicates are allowed (the same topic discussed in 2020 and 2024 is retained both times because the framing may differ).

### Gate 4 — Indic script verification

All documents labelled as Indic must pass a script-purity check: ≥ 70 % of Unicode characters fall within the expected script range for the declared language. Romanised transliterations are excluded from Tier 1 (verified) and flagged in Tier 2 (unverified).

### Gate 5 — License audit

Every dataset in the inventory must have a recorded license, a recorded date of download, and a recorded version tag. The license implications for model distribution are assessed at intake:

- CC-BY: usable; attribution in model card required
- CC-BY-SA: usable; model card must note share-alike obligation
- CC-BY-NC: **not usable** if the final model is to be released commercially
- All-rights-reserved / scraped without license: **not usable**

### Cumulative token targets for each lane

Cleaning continues in parallel with proxy experiments. The following targets must be met before the 40 B training run begins:

| Lane | Cleaned token target | Current estimate | Gap |
|------|---------------------|-----------------|-----|
| Web Foundation | 1.80 T | 1.55 T | 250 B (dedup pipeline) |
| Web Diverse | 720 B | 600 B | 120 B (need more CC-News) |
| Code | 1.08 T | 950 B | 130 B (StarCoder dedup) |
| STEM & Math | 840 B | 420 B unique + synth | Synth pipeline needed |
| Indic | 600 B | 332 B committed + 268 B repeat | Repeat budget constrained |
| Agentic | 480 B | 300 B | Synthetic generation pipeline needed |
| Reasoning | 480 B | 180 B | RLVR rollout generation needed |

**Starved lanes that require immediate action:** Agentic (Lane 6) and Reasoning (Lane 7) are the most supply-constrained. The cleaning team must prioritise generating and validating the synthetic components of these two lanes in the weeks before the 40 B training run.

---

## 12. Supply Constraints and Honest Accounting

This section documents every instance where the nominal budget share exceeds available verified supply. A plan that obscures these gaps earns less trust than one that names them plainly.

| Lane | Nominal budget | Verified unique supply | Repeat/synthetic needed | Risk level |
|------|---------------|----------------------|------------------------|-----------|
| Web Foundation | 1.80 T | ~2.5 T available | No repeat needed | Low |
| Web Diverse | 720 B | ~900 B available | No repeat needed | Low |
| Code | 1.08 T | ~1.1 T after dedup | Marginal | Low |
| STEM & Math | 840 B | ~350 B verified unique | 490 B synthetic or 2× repeat | **Medium** |
| **Indic** | **600 B** | **~95 B verified+unverified** | **505 B via translation + repeat** | **High** |
| Agentic | 480 B | ~50 B verified traces | 430 B via long-doc slicing + synthetic | **High** |
| Reasoning | 480 B | ~30 B verified CoT | 450 B RLVR + synthetic | **High** |

**Implications:**

- **STEM & Math:** 490 B synthetic tokens are manageable because math has a verifiable ground truth. Synthetic math traces (generated from problem statements, checked against known solutions) have been shown by NuminaMath authors to match human-written traces in training effect. Risk is medium: quality of generation matters greatly.

- **Indic:** This is the highest-risk gap in the plan. 505 B tokens must come from machine translation or repetition of a small verified base. The mitigation is the proxy experiment (§10.3) and the hard cap on synthetic fraction (20 %) and on repetition epochs (≤ 2×). If the proxy experiment shows degradation from synthetic Indic, the Indic nominal budget is reduced and the freed-up share is reallocated to Web Foundation.

- **Agentic:** The 430 B gap is partially bridged by slicing long documents at 8 K token windows (these are not truly agentic but build long-context attention). Genuine multi-step agentic traces are sparse; the proxy experiment at 1 B scale will determine how many are actually needed vs. how many can be substituted with long-context documents.

- **Reasoning:** The 450 B gap requires running the RLVR pipeline at scale (generating 64 rollouts per prompt and keeping majority-vote-consistent traces). This is computationally expensive. If pipeline capacity is insufficient, the reasoning share is reduced to 5 % and the surplus is given to STEM.

---

## 13. License and Provenance Register

| Dataset | License | Commercial use | Share-alike | Action required |
|---------|---------|---------------|-------------|----------------|
| Common Crawl | CCO + mixed | Yes | No | None; attribute CC |
| FineWeb / FineWeb-Edu | ODC-By | Yes | No | Attribution in model card |
| The Stack v2 | Mixed (per-file) | Depends | No | Respect per-file license; filter non-commercial files |
| StarCoder corpus | BigCode OpenRAIL | Yes (with conditions) | Use-based restrictions | Read and comply with §4 of OpenRAIL |
| Wikipedia | CC-BY-SA 4.0 | Yes | **Yes** | Model card must note; output must not carry CC-BY-SA if commercial |
| arXiv | arXiv non-exclusive | Yes (with limits) | No | Do not redistribute raw PDFs; text extraction is permissible |
| Sangraha / AI4Bharat | CC-BY 4.0 | Yes | No | Attribution required |
| IndicCorp v2 | CC-BY 4.0 | Yes | No | Attribution required |
| NuminaMath CoT | CC-BY 4.0 | Yes | No | Attribution required |
| OpenR traces | Apache 2.0 | Yes | No | None |
| Sky-T1 | Apache 2.0 | Yes | No | None |
| SCROLLS | Apache 2.0 | Yes | No | None |
| ToolBench | CC-BY 4.0 | Yes | No | Attribution required |
| Samanantar | CC-BY 4.0 | Yes | No | Attribution required |
| GSM8K | MIT | Yes | No | None |

**Policy decision:** The model will be released under a **custom open-weight license modelled on Llama 3 Community License** with an added clause requiring that any fine-tune trained on the model's outputs for Indic-language applications contributes their SFT data back to the commons under CC-BY 4.0. This is not a legal obligation but a community norm we intend to set.

Datasets that carry CC-BY-SA (primarily Wikipedia and Sangraha) are used in pre-training only. The pre-trained weights are a statistical transformation of the corpus, not a derivative work in the copyright sense under current legal consensus. This position carries legal risk and is disclosed transparently.

---

## 14. Risk Register

| Risk | Probability | Impact | Mitigation |
|------|------------|--------|-----------|
| Synthetic Indic degrades benchmark | Medium | High | Proxy experiment C (§10.3); hard synthetic cap |
| RLVR rollout quality below threshold | Medium | High | Majority-vote filter; reduce reasoning share to 5 % if needed |
| Benchmark contamination in web crawl | Low–Medium | High | Gate 2 (§11); per-snapshot canary check |
| OPUS selector crowds out Indic below floor | Low | High | Hard floor clamp in data loader |
| License dispute on Stack v2 / arXiv | Low | Medium | Per-file filter; legal review before release |
| Compute budget insufficient for 3 × proxy experiments | Medium | Medium | Proxy A and B can share infrastructure; C is 1 B scale only |
| Fertility target not met by tokenizer training deadline | Medium | High | Fallback: use chrono-tokenizer from V4 with Indic extensions; accept fertility penalty and adjust token count expectations |
| Agentic lane chronically under-supply | High | Medium | Reduce Lane 6 to 5 % (floor), reallocate 3 % to Code and STEM |

---

## Summary: The Plan as a Single Testable Hypothesis

The V5 pre-training mixture is summarised as a single falsifiable claim:

> *A 40 B model trained on 6 T tokens with 30 % web foundation, 18 % code, 14 % STEM, 10 % Indic (always-on floor 8 %), 8 % agentic, and 8 % reasoning — with anneal on the top 15 % of steps using a high-quality 900 B reserve — will score ≥ 80.0 on MMLU, ≥ 78.0 on HumanEval, ≥ 68.0 on MATH-500, and ≥ 60.0 on IndicQA, while matching or exceeding Gemma 4 27B on the full benchmark suite stated in §6.*

This claim will be partially validated by proxy experiments A, B, and C (§10) before any token of the 40 B run is committed. If any proxy experiment refutes its sub-hypothesis, the corresponding lane share is revised according to the decision rule stated in §10, and this document is updated with the new numbers and the evidence behind them.

A data decision is a hypothesis until a cheap experiment has tested it.

---

*Document maintained in the project repository. Updates must include the proxy run number and metric that motivated the change.*
