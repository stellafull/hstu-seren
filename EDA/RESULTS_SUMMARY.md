# EDA Results Summary

## Where to check the generated artifacts
- Combined summary: `outputs/local_gate/combined/pass_fail_summary.md`
- Per-domain summaries:
  - `outputs/local_gate/books/summary.json`
  - `outputs/local_gate/movielens/summary.json`
  - `outputs/local_gate/movies/summary.json`
- Main detailed artifacts per domain:
  - `input_audit/`
  - `leakage/`
  - `cell_validation/`
  - `ring/`
  - `priors/`
  - `auc/`
  - `controls/`
  - `quadrant/`

## Current result snapshot

| Domain | Cell source | Ring | AUC(L0) | Logistic gate | Quadrant | Alignment median rank pct | Decision |
| --- | --- | --- | ---: | --- | --- | ---: | --- |
| Books | metadata fallback | fail | 0.4648 | fail | fail | 0.5000 | STOP_OR_PIVOT |
| MovieLens | metadata fallback | pass | 0.5230 | fail | pass | 0.0143 | REVISE_CELL_OR_RING |
| Movies & TV | metadata fallback | fail | 0.4925 | fail | fail | 0.5000 | STOP_OR_PIVOT |

## Main observations
- **Leakage audit passed in all three domains.**
- **Cell validation passed in all three domains**, but all runs used **metadata fallback cells** instead of semantic-ID cells.
- **MovieLens** is still the only domain that shows partial mechanism evidence:
  - ring coverage passed
  - quadrant passed
  - but `AUC(L0)` and the logistic gate still failed
- **Books** and **Movies & TV** do not currently support the mechanism under the fallback-cell setup.
- The shuffled negative control is near chance in all three domains, which is a useful sanity check.
- The alignment diagnostic is now fully populated, but under fallback cells it is not strong evidence of the intended semantic-frontier mechanism.

## What is still not complete / not final

### 1. SID-backed EDA is still not done
This remains the biggest missing piece.

All three summaries currently report:
- `"cell_source": "metadata_fallback"`
- `"sid_coverage": 0.0`

So the current results are **not the final intended EDA gate** from the research plan. They are a fallback version that proves the pipeline runs end-to-end, but not yet the true semantic-cell experiment.

### 2. Ring analysis should be re-interpreted after SID cells are available
The ring robustness outputs are generated for all domains, but the fallback-cell geometry is still weak:
- Books ring bounds collapse around `0.0`
- MovieLens ring bounds collapse around `1.0`

So the **ring analysis is implemented and complete at the fallback level**, but it should be rerun and reinterpreted after SID-backed cells are available.

### 3. Combined decision is still a research interpretation step
The pipeline now produces all planned fallback-level artifacts, but the final paper-level interpretation is still manual.
Given the current results, the honest overall status is:

> **Do not proceed to neural `F` yet. First revise cell construction / ring definition, ideally with SID-backed cells.**

## What is complete now
At the current fallback-cell stage, the following EDA modules are implemented and have produced artifacts for all 3 domains:
- input audit
- leakage audit
- semantic cell validation
- ring robustness (tight/default/loose)
- prior scoring (`F_trans`, `E0`, `L0`)
- prior-only AUC
- decile lift
- quadrant heatmap
- rare-cell / popularity controls
- logistic primary gate
- randomized-transition negative control
- `F_trans` alignment spot check

## Recommended next EDA steps
1. Generate or sync real semantic-ID artifacts so `sid_coverage > 0`.
2. Rerun the full EDA gate with SID-backed cells.
3. Recheck the four key diagnostics first:
   - ring robustness
   - `AUC(L0)`
   - logistic primary gate
   - quadrant heatmap
4. Compare SID-backed results against the current fallback results instead of replacing them blindly.
5. Only after SID-backed reruns should we decide whether the result is really `REVISE_CELL_OR_RING` or `STOP_OR_PIVOT`.
