# Freight Rate Prediction Assessment

Run ID: submission-c27c7409c45454ca

Canonical generated selections:

```text
MAIN|model=hgb_compact|family=HistGradientBoostingRegressor|feature_set=basic_plus_geographic|mae=129.511954241217
DECEMBER|model=december_hgb__december_basic|family=HistGradientBoostingRegressor|feature_set=december_basic|mae=108.7836047385515
```

## Objective

Train a leakage-safe freight-rate regressor on the supplied labeled data, predict all 12,000 future validation loads, and create the fixed Lexington-to-Fort-Wayne December series required by the supplied scorer.

## Dataset overview

- Development: 48,000 rows, 2025-01-01 through 2025-10-31; target `posted_rate`.
- Final validation: 12,000 rows, 2025-11-01 through 2025-12-31.
- December chart input: 31 fixed rows for 2025-12-01 through 2025-12-31.
- Missing development values occur in weight and market index. Impossible non-positive weights are treated as missing and flagged. Valid target extremes are retained.

## Repository structure

```text
run_pipeline.py                  End-to-end entry point
src/                             Features, training, evaluation, reporting
artifacts/                       Metrics, comparisons, audit and error tables
models/                          Refit selected main and December models
reports/assessment_report.pdf    Concise assessment report
reports/loom_script.md            2-3 minute walkthrough script
reports/figures/                  Decision-relevant figures
validation_predictions.csv        Final 12,000-row submission
december-chart-inputs.csv          Completed fixed December input
scorer_results/candidate_december.png
```

## Setup and reproduction

Python 3.11+ is recommended.

```bash
python -m venv .venv
# Windows PowerShell: .venv\Scripts\Activate.ps1
# macOS/Linux: source .venv/bin/activate
python -m pip install -r requirements.txt
python run_pipeline.py
```

The final command locates the supplied root-level files (with tolerant filename aliases), audits them, creates the chronological split, screens feature groups with both model families, compares all eligible HGB and CatBoost candidates, refits the selected families, predicts, validates both outputs, runs the unmodified scorer, and generates the report.

## Validation and model

The primary holdout mirrors the two-month future horizon: train 2025-01-01 through 2025-08-31 (38,477 rows), validate 2025-09-01 through 2025-10-31 (9,523 rows). Feature groups are screened before that comparison on a separate training-only split: fit through 2025-07-01 and validate from 2025-07-02 through 2025-08-31. Fixed HGB and CatBoost proxies score every logical feature set on the same fold-pure matrices, and the lowest equal-family mean MAE chooses one common list. CatBoost's best iteration for that list comes from the same training-only screen. Every outer candidate configuration is then frozen before scoring; outer labels are used only for the required final candidate selection and post-selection diagnostics, never to refit or adapt a candidate before its MAE is recorded.

Selection rule: Lowest finite MAE on the common chronological holdout among all eligible candidates; exact-equality ties are broken deterministically by candidate name.

Selected model: **HistGradientBoostingRegressor** (`hgb_compact`), using **basic_plus_geographic**. Histogram gradient boosting uses an ordinal category encoder fitted within each training fit; previously unseen categories map to a reserved value. MAE is the primary metric because dollars of absolute error are directly interpretable and robust to sparse label anomalies.

- Selected parameters: `{"l2_regularization": 3.0, "learning_rate": 0.06, "max_iter": 250, "max_leaf_nodes": 15, "min_samples_leaf": 30}`
- Saved final model: `models\final_rate_model.joblib`

| Metric | Chronological holdout |
|---|---:|
| MAE | $129.51 |
| RMSE | $640.32 |
| R-squared | 0.8239 |
| sMAPE | 4.527% |

Feature sets that lowered equal-family mean MAE relative to the basic supplied-feature model on the training-only screen: basic_plus_geographic, basic_plus_route. The winning set was frozen before any outer-holdout scoring; groups that did not improve that inner validation window were not carried into final candidate comparison.

## December method

The reduced-schema selection is intentionally separate from the main validation model. Eligible December-compatible HGB and CatBoost candidates use the same chronological outer rows, rate-per-mile target, and posted-rate MAE; within each feature-set comparison, both families receive the same compatible columns. CatBoost early stopping remains internal to the outer training rows.

Selected December model: **HistGradientBoostingRegressor** (`december_hgb__december_basic`), using **december_basic**, with holdout MAE **$108.78**. It uses only fields available in the fixed input plus known date features; coordinate enrichment, when selected, is learned exclusively from development city mappings. It does not fabricate market or quote signals.

- December parameters: `{"l2_regularization": 3.0, "learning_rate": 0.06, "max_iter": 250, "max_leaf_nodes": 15, "min_samples_leaf": 30}`
- Saved December model: `models\december_rate_model.joblib`

## Outputs and scorer

```bash
python score.py --predictions validation_predictions.csv --december-predictions december-chart-inputs.csv
```

Successful scorer output is captured in `artifacts/scorer_output.txt`; the chart is `scorer_results/candidate_december.png`. Local metrics are development holdout metrics; the supplied scorer validates structure only and Spotter evaluates final validation accuracy after submission.

## Reproducibility, assumptions, and limitations

- Fixed random seed: 42; CPU-based eligible HGB and CatBoost comparisons; no external data or credentials.
- The selected main family is HistGradientBoostingRegressor; the separately selected December family is HistGradientBoostingRegressor.
- The common feature list is chosen from one inner chronological window using equal weight for fixed HGB and CatBoost proxy scores; a different window or proxy configuration could choose differently.
- Validation includes eight cities absent from development, so unseen-city performance is inherently uncertain.
- The main shift is market index; chronological validation reduces, but cannot eliminate, future-regime risk.
- Coordinates are internally consistent but appear obfuscated, so supplied distance is treated as authoritative.
- Sparse extreme labels dominate RMSE; they are retained rather than silently removed.
- Historical target encodings were deliberately omitted because sparse routes and chronological leakage risk outweighed their probe value.
- The reported outer-holdout MAE is the required model-selection estimate, not a post-selection unbiased test score.

Loom walkthrough: [ADD FINAL LOOM LINK HERE]
