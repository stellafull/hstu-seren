# V2 Final Plan: Label-Free Serendipity with Generative SID Decoding

Status: active execution plan. This supersedes V1 Stage2/3/4 for new work.

## Non-negotiables

- Default relevance protocol: `LOO_FULL_CATALOG`. It remains the main
  relevance/full-catalog report and protects next-item relevance.
- Additional serendipity protocol: `SER_EVENT_FULL_CATALOG` may be reported
  alongside LOO for prefix-qualified ser-positive target events. Its denominator
  is `num_ser_queries`, not all LOO users.
- Sampled-candidate or 1-positive+N-negative eval is a separate protocol and
  must not be mixed with full-catalog metrics.
- SerenFree main method remains label-free: serendipity labels may not train,
  mine targets, early-stop, or tune alpha/beta/temperature. Any run that uses
  ser labels for HPO is a supervised or label-assisted comparison, not the
  main SerenFree method.
- Small serendipity datasets are still used for target-domain adaptation, but
  only through ordinary interaction sequences and label-free future-session
  acceptability targets.
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
- `tools/build_ser_event_manifest.py`
- `tools/audit_loo_manifest.py`
- `tools/build_future_window_targets.py`
- `tools/audit_ser_target_reachability.py`
- tests for LOO, ser-event manifests, future windows, no ser-label leakage

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
  relevance_name: LOO_FULL_CATALOG
  ser_event_name: SER_EVENT_FULL_CATALOG
  k_values: [10, 20, 50, 100, 200]
  default_report: LOO_FULL_CATALOG
  compare_protocols: [LOO_FULL_CATALOG, SER_EVENT_FULL_CATALOG]
  forbid_mixing_sampled_with_full_catalog: true
  forbid_gts_eval: true

ser_event_validation:
  min_prefix: 1
  rating_positive_threshold: 4.0
  split_policy: latest_positive_test_second_latest_positive_val
  single_positive_policy: deterministic_user_hash_val_or_test
  train_policy: strict_truncate_before_first_heldout_ser_event
  denominator: num_ser_queries

future_targets:
  imminent_window: 3
  acceptable_min_gap: 4
  acceptable_window: 50
  rating_positive_threshold: 4.0
  max_i_targets: 3
  max_a_targets: 32
  exclude_seen: true
  cap_policy_A: rating_then_recency
  cap_policy_I: earliest

geometry:
  enabled: true
  features: [prefix_surprise]
  prefix_levels: adaptive_semantic_non_dedup
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

main_method:
  use_serendipity_labels_for_loss: false
  use_serendipity_labels_for_early_stopping: false
  use_serendipity_labels_for_hpo: false
  use_serendipity_labels_for_eval: true
```

## Execution order

1. Freeze V1 S1 outputs/checkpoints/configs/audits.
2. Build and freeze unified LOO manifests.
3. Build optional ser-event full-catalog manifests for datasets with ser labels.
4. Add eval JSON manifest and universe hashes.
5. Run reachability audits, especially SerenLens Movies.
6. Implement true future-window A/I targets.
7. Implement multi-positive trie marginal loss.
8. MovieLens R-only -> future A/I -> AIG+geometry eval -> LF-rank.
9. Extend to Books; run Movies only after reachability audit passes.

## Current implementation notes

- Future-window targets are used for V2 training only. Relevance validation and
  test use frozen LOO manifests. Serendipity labels can additionally be reported
  through frozen `SER_EVENT_FULL_CATALOG` manifests, but those metrics must not
  choose main-method checkpoints or hyperparameters.
- Stage 2 on SerenLens / Serendipity-2018 is label-free domain adaptation:
  use the dataset's ordinary interaction sequences to build near-session `I_t`
  and future-session acceptable `A_t`; do not use `sequence_ser_label` or answer
  ser labels in the loss.
- `ser_targets_only` is a final reporting bucket in
  `evaluate_serenfree_retrieval.py`; it must not drive training, early stopping,
  checkpoint selection, or hyperparameter selection.
- In LOO, `ser_targets_only` is sparse because only last-item ser positives
  count. In ser-event evaluation, every row is a ser-positive query, so
  `ser_targets_only` equals the protocol denominator `num_ser_queries`.
- Any run that uses ser labels for loss, alpha/beta selection, early stopping,
  or HPO must be reported separately as supervised or label-assisted baseline,
  not as the SerenFree main method.
- Default geometry currently enables only `prefix_surprise`. Additional
  geometry sources such as centroid or Qwen/content distance should stay out of
  default configs until they are wired into the train/eval scoring path.
