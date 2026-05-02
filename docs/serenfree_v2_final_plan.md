# V2 Final Plan: Label-Free Serendipity with Generative SID Decoding

Status: active execution plan. This supersedes V1 Stage2/3/4 for new work.

## Non-negotiables

- Evaluation protocol: `LOO_FULL_CATALOG` only. No GTS, sampled split, or 1-positive+100-negative eval.
- Serendipity labels: final evaluation only (`metrics.ser_targets_only`). Never train, early-stop, mine, or tune hyperparameters on ser labels.
- Main claim: label-free serendipity mechanism, not HSTU/SID strength.
- S1 remains frozen as `HSTU-GenSID-R` relevance baseline.
- S2/S3/S4 must be rewritten around future-window A/I, geometry, and context-level ranking.
- Full catalog: all models rank/decode against the same item universe.
- Verify first: audit/sanity before large experiments.

## V1 issues found

- `src/generative_recommenders_pl/models/hstu_serenfree.py` currently uses `_acceptable_semantic_targets` from a recent history-window proxy. V2 must replace this with future-window targets from `loo_train.parquet`.
- `configs/experiment/serenfree_stage3_*` and `serenfree_stage4_*` hard-code `aig_levels: [1, 2]`. V2 must make AIG/geometry levels config-driven/adaptive and explicitly exclude dedup.
- `src/generative_recommenders_pl/scripts/mine_serenfree_pseudo_ser.py` mines item-level pseudo labels by teacher A-I gap. V2 must replace this with context-level label-free candidates.
- Existing eval can compute HR/NDCG but does not yet bind every run to frozen LOO manifest hashes/item universe/trie hashes.

## Required V2 artifacts

### Data / audit

- `tools/build_loo_manifest.py`
- `tools/audit_loo_manifest.py`
- `tools/build_future_window_targets.py`
- `tools/audit_ser_target_reachability.py`
- tests for LOO, future windows, no ser-label leakage

### Model / loss

- trie marginal multi-positive SID loss
- A/I future-window loss paths
- A/I collapse diagnostics: JS per semantic level, entropy, A-I target separation
- LF-rank loss with stopgrad relevance
- tests for trie marginal NLL

### Geometry / inference / eval

- `src/generative_recommenders_pl/serenfree/geometry.py`
- `src/generative_recommenders_pl/serenfree/collision_resolver.py`
- late-fusion full-catalog LOO inference: R beam top-M -> teacher-force R/A/I -> geometry -> relevance-safe score
- context-level `tools/mine_label_free_ser_candidates.py`
- tests for geometry, AIG/no-dedup, LF miner, collision resolver

## Default configs

```yaml
protocol:
  name: LOO_FULL_CATALOG
  k_values: [10, 50, 100, 200]
  forbid_sampled_eval: true
  forbid_gts_eval: true

future_targets:
  imminent_window: 3
  acceptable_min_gap: 2
  acceptable_window: 50
  rating_positive_threshold: 4.0
  max_i_targets: 3
  max_a_targets: 32
  exclude_seen: true
  cap_policy_A: rating_then_recency
  cap_policy_I: earliest

geometry:
  enabled: true
  features: [prefix_surprise, centroid_distance, qwen_distance]
  prefix_levels: adaptive_semantic_non_dedup
  centroid_levels: adaptive_semantic_non_dedup
  history_len: 20
  recency_decay: 0.85
  ring_low_quantile: 0.60
  ring_high_quantile: 0.95
  use_dedup: false

loss:
  lambda_R: 1.0
  lambda_I: 0.2
  lambda_A: 0.2
  lambda_rank: 0.05
  lambda_js_margin: 0.0
  rank_temperature: 0.1
  stopgrad_R_in_rank: true

inference:
  constrained_trie: true
  candidate_M: 1000
  relevance_floor_rank: 500
  alpha_grid: [0.0, 0.02, 0.05, 0.1, 0.2, 0.5]
  beta_grid: [0.0, 0.1, 0.25, 0.5, 1.0]
  aig_levels: adaptive_semantic_non_dedup
  geometry_levels: adaptive_semantic_non_dedup
  touch_dedup: false
  history_filter: true
```

## Execution order

1. Freeze V1 S1 outputs/checkpoints/configs/audits.
2. Build and freeze unified LOO manifests.
3. Add eval JSON manifest and universe hashes.
4. Run reachability audits, especially SerenLens Movies.
5. Implement true future-window A/I targets.
6. Implement multi-positive trie marginal loss.
7. MovieLens R-only -> future A/I -> AIG+geometry eval -> LF-rank.
8. Extend to Books; run Movies only after reachability audit passes.
