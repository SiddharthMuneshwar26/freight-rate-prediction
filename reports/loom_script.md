# Loom walkthrough script (about 2-3 minutes)

Run ID: submission-c27c7409c45454ca

```text
MAIN|model=hgb_compact|family=HistGradientBoostingRegressor|feature_set=basic_plus_geographic|mae=129.511954241217
DECEMBER|model=december_hgb__december_basic|family=HistGradientBoostingRegressor|feature_set=december_basic|mae=108.7836047385515
```

Hi, this project predicts future posted freight rates and produces both the 12,000-row validation submission and the fixed December chart requested by the assessment.

The labeled dataset has 48,000 rows from 2025-01-01 through 2025-10-31. The final prediction data starts the next day and runs through December, so I avoided a random split. I trained on 2025-01-01 through 2025-08-31, or 38,477 rows, and held out 2025-09-01 through 2025-10-31, or 9,523 rows. That two-month holdout best matches the final horizon and its lower market-index regime.

The main quality issues were missing weight and market-index values, plus impossible negative weight signs. I treated non-positive weights as missing, retained explicit missing and problem flags, and fitted every learned preprocessing step only on the applicable training partition. I retained valid extreme rates because removing them would be hard to defend. The target is right-skewed, with a median of $2,031, and distance is the dominant raw signal. Final validation also contains eight unseen cities and 1,461 unseen-route rows.

The global-median baseline had an MAE of $1,148.92. Before the outer comparison, I scored every feature group with fixed HGB and CatBoost proxies on a training-only chronological split that fit through 2025-07-01 and validated through 2025-08-31. The lowest equal-family mean MAE froze one common logical feature list without reading outer-holdout labels. CatBoost's iteration count for that list came from those fold-pure matrices, using distance-weighted rate-per-mile MAE so the stopping metric is proportional to posted-rate MAE. I then compared every eligible HGB and CatBoost finalist on the same outer chronological rows, frozen columns, rate-per-mile target, and posted-rate MAE. Every candidate was frozen before outer scoring; the outer holdout made the required final candidate choice and then supported diagnostics, without refitting or adapting candidates before their MAEs were recorded. The rule was: Lowest finite MAE on the common chronological holdout among all eligible candidates; exact-equality ties are broken deterministically by candidate name. The winner was HistGradientBoostingRegressor, candidate hgb_compact, using basic_plus_geographic. Its holdout metrics are MAE $129.51, RMSE $640.32, R-squared 0.8239, and sMAPE 4.527 percent. The leading features were distance, equipment, haversine_distance, quote_signal, weight. Histogram gradient boosting uses an ordinal category encoder fitted within each training fit; previously unseen categories map to a reserved value. The refit model is saved as models\final_rate_model.joblib.

The December file lacks market and quote signals, so I did not invent them. I compared compatible HGB and CatBoost candidates separately on the same outer rows, rate-per-mile target, posted-rate MAE, and identical columns within each feature-set alternative; CatBoost early stopping stayed inside the outer training rows. The winner was HistGradientBoostingRegressor, candidate december_hgb__december_basic, using december_basic, with holdout MAE $108.78. I refit it on all labeled rows and saved it as models\december_rate_model.joblib. Its date features are known in advance, and any coordinates are mapped only from development data.

The key files are `run_pipeline.py`, the small modules under `src`, the machine-readable tables under `artifacts`, `validation_predictions.csv`, and this report. From a fresh environment, run `python -m pip install -r requirements.txt`, then `python run_pipeline.py`. That also executes the supplied scorer and creates `scorer_results/candidate_december.png`.

The main limitations are future-regime and unseen-city uncertainty, plus reliance on two proxy families and one inner chronological window for common-feature screening. The outer-holdout MAE is the required selection estimate, not a post-selection unbiased test score or a hidden-test score.
