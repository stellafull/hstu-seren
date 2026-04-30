# HSTU-SerenFree Implementation Plan

This document is the repository-tracked engineering plan for HSTU-SerenFree. The implementation branch starts from a clean checkpoint of the remote EDA work and targets CPU-verifiable smoke tests before any full GPU run.

## Goal

Implement a label-free, end-to-end, serendipity-aware generative recommender using:

- existing HSTU backbone
- existing RQ-kMeans Semantic IDs
- full-SID item-level input composer
- imminent vs acceptable dual-horizon semantic heads
- AIG (Acceptability-Imminence Gap) bias inside SID decoding
- teacher-mined pseudo-ser positives

Main claim: collaborative-semantic sequence modeling over full SID actions plus decoder-level label-free serendipity bias.

## Frozen V1 Decisions

- Full-SID-in / full-SID-out.
- Item-level HSTU encoder; do not flatten SID tokens in history.
- Shared prefix decoder with mode embeddings for R/I/A heads.
- AIG bias only on semantic levels 1-2 first; level 3 and dedup stay relevance-only.
- Teacher-mined pseudo-ser positives; no online self-mining in first implementation.
- No external reranker in the main path; reranker only as a baseline.
- No differentiable SID refresh in V1.

## Data Assumptions

- Semantic IDs are dataset-specific full SID rows shaped
  `(q1, ..., qL, d)`.
- The final SID column is always the dedup slot; all preceding columns are
  semantic codebook levels.
- Codebook token `0` is valid. Training lookup tables reserve internal `0` for
  padding/missing items by shifting valid SID tokens by `+1`.
- Current HSTU implementation already operates on item sequences.
- Existing datasets and splits from the thesis pipeline remain authoritative.

## Engineering Sequence

1. Stage 1 relevance-only SID generation.
2. Stage 2 dual-horizon heads and collapse monitors.
3. Stage 3 inference-only AIG baseline.
4. Stage 4 teacher-mined pseudo-ser training.

## Stage 1: Relevance-Only SID Generation

Objective: train `q1 -> ... -> qL -> d` generation from the current HSTU hidden state.

Initial implementation slices:

1. SID composer module.
   - Input: tensors shaped `[..., L + 1]` containing `(q1, ..., qL, d)`.
   - Output: item-level embedding shaped `[..., H]`.
   - Verify: CPU unit test for shape, padding behavior, gradients, and dedup conditioning effect.
2. HSTU wrapper.
   - Wrap current HSTU forward path to expose hidden states `[B, T, H]`, recent pooled `h_I`, and full-history pooled `h_A`.
   - Verify: CPU shape test with a tiny synthetic HSTU config.
3. Shared prefix decoder in relevance mode.
   - One decoder conditioned on mode embedding and previous SID prefix tokens.
   - Verify: logits for variable semantic levels and dedup, plus CE loss over synthetic targets.
4. Constrained SID trie and beam search.
   - Build valid SID trie from item SID table.
   - Verify: generated beams always map to known item IDs.
5. CPU smoke script/config.
   - Tiny synthetic batch exercises composer -> HSTU wrapper -> decoder -> loss -> beam.
   - Verify: command exits zero on CPU without requiring datasets or GPU.

Current Stage 1 checkpoint:

- Implemented isolated `models/serenfree/` building blocks.
- Added a CPU smoke script at `src/generative_recommenders_pl/scripts/serenfree_stage1_smoke.py`.
- The smoke path intentionally uses a tiny identity encoder first; replacing it with configured HSTU is the next integration step after pure module tests are stable.

Stage 1 success checks:

- Training path is stable on a tiny CPU subset or synthetic smoke.
- Beam returns valid SIDs only.
- Relevance metrics can be computed against existing baselines once real data is wired.

## Stage 2: Dual-Horizon Heads

Add imminent and acceptable heads, `L_I`, `L_A`, future-window target builder, and monitoring of `JS(p_A || p_I)`.

Success checks:

- `p_I` fits immediate next prefix.
- `p_A` predicts broader future prefix region.
- Mean JS divergence stays above a floor to catch collapse.

## Stage 3: Inference-Only AIG

Add AIG computation at inference and inject semantic-logit bias for levels 1-2 only.

Success checks:

- Compare relevance-only decoder vs AIG-biased decoder.
- Validate some serendipity gain without major relevance collapse.

## Stage 4: Teacher-Mined Pseudo-Ser Training

Freeze the stage-2 checkpoint as teacher, mine pseudo-ser positives from future windows, and fine-tune with a gap/ranking loss.

Success checks:

- Mined parquet schema is stable.
- Stage-4 improves serendipity metrics over stage-3 inference-only AIG without relevance collapse.

## Losses

- Relevance: `L_rel = sum_l CE(q_l) + lambda_d * CE(d)`.
- Imminent: predict next-step semantic prefixes.
- Acceptable: predict future-window semantic distribution or sampled future positives.
- Gap: sampled-softmax / InfoNCE-style ranking over teacher-mined candidates.
- Optional anti-collapse: margin on JS divergence between `p_A` and `p_I`.

## Recommended Defaults

- AIG-biased levels: semantic levels 1 and 2 only when present.
- Later semantic levels and dedup: relevance-only.
- Short window `w`: 10.
- Future window `H`: 20.
- `lambda_I`: 0.2.
- `lambda_A`: 0.2.
- `lambda_G`: 0.05.

## Required Monitoring

- relevance metrics
- serendipity metrics where labels exist
- valid SID rate
- mean JS(p_A, p_I)
- mean and std of AIG
- fraction of non-zero AIG tokens
- generated item popularity
- per-level entropy

## Required Ablations

1. relevance-only decoder
2. inference-only AIG
3. decoder-integrated AIG
4. no acceptable head
5. no imminent head
6. no acceptability gate
7. no popularity penalty
8. level-1 only vs 1+2 vs 1+2+3
9. semantic-only input vs full-SID input
10. full-SID input + atomic-ID residual

## Current CPU-First Rule

Do not request a GPU run until the repository has:

- committed Stage 1 scaffolding
- unit tests for the new pure-PyTorch modules
- a CPU smoke path that exercises the end-to-end Stage 1 module chain
- documented remaining GPU-only validation gaps
