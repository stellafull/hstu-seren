# SerenFree V2 task split

Master owns coordination, plan docs, configs, final integration, and verification.

## A: data/audit
Write scope:
- `tools/build_loo_manifest.py`
- `tools/audit_loo_manifest.py`
- `tools/build_future_window_targets.py`
- `tools/audit_ser_target_reachability.py`
- `tests/test_loo_manifest.py`
- `tests/test_future_window_targets.py`
- `tests/test_no_ser_label_leakage.py`

Acceptance:
- LOO manifest creates eval/train parquet + meta hashes.
- train manifest cannot expose ser labels.
- future targets come only from LOO train prefix future windows.
- Movies reachability audit reports mapping/trie/filter/beam fields.

## B: model/loss
Write scope:
- `src/generative_recommenders_pl/serenfree/losses/`
- minimal necessary model hooks
- `tests/test_trie_multipositive_loss.py`

Acceptance:
- multi-positive trie marginal NLL matches brute force on tiny vocab.
- I/A train only semantic non-dedup levels by default.
- LF-rank stopgrads R by default.
- A/I collapse diagnostics expose JS/entropy/separation metrics.

## C: geometry/inference/eval
Write scope:
- `src/generative_recommenders_pl/serenfree/geometry.py`
- `src/generative_recommenders_pl/serenfree/collision_resolver.py`
- late fusion inference/eval minimal hooks
- `tools/mine_label_free_ser_candidates.py`
- geometry/inference/miner/collision tests

Acceptance:
- AIG/geometry levels are adaptive/config-driven semantic-only and never dedup.
- late fusion uses relevance top-M/floor guard.
- ring constraint prevents low-R high-G promotion.
- context-level miner, not item-level pseudo labels.
