Today 1:08 PM
run_pipeline(2).py
Python
run_pipeline(3).py
Python
Freight_Rate_Prediction.ipynb
File
all these files run and there are for a hiring/assesing task which one is best for getting hired
Worked for 4m 41s
Verdict 🏆

Use run_pipeline(3).py as the official submission pipeline, and include Freight_Rate_Prediction.ipynb as the reviewer-friendly walkthrough.

Do not present all three as separate competing solutions. That makes the project look indecisive and AI-generated rather than deliberately engineered.

Ranking
Rank	File	Hiring value
1	run_pipeline(3).py	Best engineering and submission structure—after one important fix
2	Freight_Rate_Prediction.ipynb	Best file for quickly impressing the reviewer
3	run_pipeline(2).py	Strong modeling pipeline, but mostly redundant
Why run_pipeline(3).py should be the main solution

The assessment requires more than prediction accuracy: a reproducible repository, exact prediction output, report, December chart, and Loom walkthrough.

Pipeline 3 demonstrates the strongest Machine Learning Engineer mindset:

Robust schema and input validation.
A chronological holdout designed to match the final 61-day forecasting horizon, rather than an arbitrary random split.
Feature ablation plus rolling monthly temporal-stability evaluation.
Baselines, multiple HGB candidates, a CatBoost benchmark, permutation importance, residual analysis, and December-specific reduced-feature validation.
It automatically generates the README, PDF report, predictions, chart, metrics, and verifies that every required output exists.
It checks that the supplied score.py was not modified, which is a very professional assessment-integrity detail.

That makes it look closest to something an actual MLE would deliver.

Why the notebook should also be included

The notebook is the easiest file for a busy reviewer to understand:

It is self-contained and tells the reviewer exactly how to run it.
Cleaning, EDA, feature engineering, model comparison, error analysis, predictions, and scoring are visible with executed outputs.
Its selected CatBoost model achieved approximately MAE 123.24, R² 0.819 and sMAPE 5.16%, substantially beating Ridge and the naive baseline.
The supplied scorer successfully validated all 12,000 final predictions and all 31 December predictions.

So the ideal repository structure is:

run_pipeline.py              # renamed run_pipeline(3).py — official implementation
Freight_Rate_Prediction.ipynb # visual walkthrough
src/
README.md
requirements.txt
validation_predictions.csv
reports/assessment_report.pdf
Important fix before submission 🚨

Pipeline 3 benchmarks CatBoost, but then chooses the winner using:

selected_name, selected_candidate = min(
    hgb_candidates.items(),
    key=lambda item: item[1].metrics["mae"]
)

That means CatBoost can never be selected, even if its MAE is lower. The script benchmarks CatBoost immediately before selecting only from the HGB dictionary.

This is the biggest weakness across the files. Either:

select the lowest-MAE model across HGB and CatBoost, or
clearly justify why HGB is mandatory despite potentially weaker validation performance.

For a hiring assessment, option 1 is better.

Two notebook cleanups

Remove the output exposing your full local Windows username and folder paths before committing.

Also correct the historical-route feature description. The notebook calls it an “expanding” leakage-safe statistic, but the implementation uses ordinary group medians containing each training row’s own target. It was rejected and therefore does not affect the final selected CatBoost model, but a strong reviewer could notice the inconsistency.

Final recommendation

Submit Pipeline 3 as the canonical implementation after fixing model selection. Keep the notebook as the polished walkthrough. Remove Pipeline 2 from the final repository—or place it in an clearly named experiments/ folder.

Forced to submit one unchanged file right now, I would choose Freight_Rate_Prediction.ipynb, because its reasoning, executed performance and successful scorer output are immediately visible.

freight-rate-ml-assessment.pdf
PDF
this was the task so can you check again
Worked for 1m 34s
Re-checked against the actual task ✅

I’d revise my earlier answer:

Do not submit Freight_Rate_Prediction.ipynb alone.
For this specific assessment, run_pipeline(3).py is the best foundation for the final submission, provided you fix or verify its model-selection logic.

The brief is judging a complete hiring submission, not merely the most readable model file. It requires:

GitHub repository with code, dependencies and run instructions
Exactly formatted validation_predictions.csv
A PDF/DOCX report with the validation strategy and scorer-generated December chart
A 2–3 minute Loom covering EDA, data quality, model choice, validation and code
Updated ranking
Rank	Solution	Verdict
1	run_pipeline(3).py	Best assessment/submission engineering
2	run_pipeline(2).py	Best clear CatBoost modeling solution
3	Freight_Rate_Prediction.ipynb	Excellent supporting walkthrough, not enough alone
Why Pipeline 3 now ranks first 🏆
1. It directly builds the required deliverables

Pipeline 3:

Creates validation_predictions.csv
Fills and validates the December file
Executes the supplied scorer
Confirms the chart exists
Generates assessment_report.pdf
Checks that all required outputs are non-empty

This maps extremely closely to the assessment. It looks like an MLE submission pipeline, not just experimentation code.

2. Its validation methodology is stronger

Pipeline 3 uses:

A blocked chronological holdout based on the final prediction horizon
Feature ablation
Multiple model configurations
Error analysis
Permutation importance
Monthly temporal-stability checks across August, September and October

That temporal-stability section is especially valuable in an interview because it answers:

“How do you know your result wasn’t only good during one lucky month?”

Pipeline 2 uses a sensible October holdout, but Pipeline 3 communicates more mature validation thinking.

3. It follows the December instructions more literally

The supplied README says to fill the predicted_rate column in data/december_chart_inputs.csv, then pass that completed file to score.py.

Pipeline 3 completes that supplied file and passes it to the scorer.

The standalone notebook instead writes a separate december_predictions.csv. It passes scoring, but it is slightly less literal than the stated workflow.

The one issue to resolve before choosing Pipeline 3 🚨

Pipeline 3 adds CatBoost to the comparison table, but the final selection is restricted to:

selected_name, selected_candidate = min(
    hgb_candidates.items(),
    key=lambda item: item[1].metrics["mae"]
)

So the final model is always an HGB candidate; CatBoost cannot win the selection even when it appears in model_comparison.csv.

This is not automatically a bug—my previous answer was too definite about that.

Use this rule:

If the best HGB model has lower MAE than CatBoost, the selection is acceptable, but explain that CatBoost was only a benchmark.
If CatBoost has lower MAE, change the code so all candidate families compete fairly.
If their MAEs are extremely close, choosing HGB for speed, dependency simplicity or stability is defensible—but state that explicitly.

The hiring reviewer should never see a comparison table where one model is numerically best but another is chosen without explanation.

Where Pipeline 2 is stronger

Pipeline 2 and the notebook have a very clean, understandable CatBoost story:

Basic + date features won the ablation
CatBoost achieved MAE 123.24
Ridge achieved MAE 231.94
The naive baseline achieved MAE 1140.16
CatBoost reached R² 0.819 and sMAPE 5.16%

The notebook also visibly proves that:

All 12,000 predictions passed
All 31 December predictions passed
The scorer created candidate_december.png

So if Pipeline 3’s HGB result is meaningfully worse than 123.24 MAE, use Pipeline 2’s CatBoost model instead.

Best final submission structure

Use one canonical solution, not three competing versions:

README.md
requirements.txt
run_pipeline.py
src/
notebooks/
    Freight_Rate_Prediction.ipynb
data/
    train_test.csv
    validation.csv
    validation_predictions_template.csv
    december_chart_inputs.csv
reports/
    assessment_report.pdf
scorer_results/
    candidate_december.png
validation_predictions.csv
My final recommendation

Use Pipeline 3’s overall architecture and reporting system.
Then choose the actual final model by comparing Pipeline 3’s HGB MAE against the CatBoost MAE of 123.24.

Keep Freight_Rate_Prediction.ipynb only as the executed, visual walkthrough. Also record the real Loom and place its link prominently in the README—a generated loom_script.md does not itself satisfy the required Loom submission.

Today 2:36 PM
assessment_report.pdf
PDF
run_pipeline(4).py
Python
Strict review complete. I fixed all material issues, regenerated the repository from a clean output state, and independently reran the supplied scorer.
The biggest additional finding was material: HGB-only inner feature screening chose basic_plus_route, while a family-neutral HGB+CatBoost screen chose basic_plus_geographic. CatBoost now participates upstream as well as in final selection.
Fixes
Feature selection now uses a training-only chronological split:Fit: 2025-01-01 through 2025-07-01
Validate: 2025-07-02 through 2025-08-31

Every feature set is scored by fixed HGB and CatBoost proxies. Equal-family mean MAE selects the common feature set.
CatBoost iteration selection uses fold-pure preprocessing and distance-weighted rate-per-mile MAE, proportional to posted-rate MAE.
All feature columns, parameters, and iterations are frozen before September–October outer scoring.
The outer holdout is used only for the required one-shot final candidate selection and diagnostics—never to adapt and rescore candidates.
Exact expected candidate-set assertions prevent CatBoost from being silently omitted.
Canonical family labels are used throughout: HistGradientBoostingRegressor and CatBoostRegressor.
Saved wrapper and underlying estimator classes are both checked against metadata.
Submission schemas remain unchanged. Only the internal feature_ablation.csv schema was expanded to report both screening families truthfully.
Main comparison
Candidate	Family	Features	MAE	RMSE	Eligible	Selected
hgb_smoother	HistGradientBoostingRegressor	basic_plus_geographic	129.508	640.246	Yes	Yes
hgb_compact	HistGradientBoostingRegressor	basic_plus_geographic	129.535	640.230	Yes	No
catboost_rate_per_mile	CatBoostRegressor	basic_plus_geographic	141.192	643.080	Yes	No
equipment_median_rpm	Baseline	distance + equipment	229.097	670.660	No	No
global_median	Baseline	target_only	1,148.924	1,569.424	No	No


The true minimum eligible MAE is 129.50819581944845, exactly matching the selected row. Full results: [model_comparison.csv](/C:/Users/Siddharth/OneDrive/Desktop/spotter/codex/artifacts/model_comparison.csv).
December comparison
Candidate	Family	Features	MAE	Eligible	Selected
december_hgb__december_basic	HistGradientBoostingRegressor	december_basic	109.244	Yes	Yes
december_hgb__december_enriched_geo	HistGradientBoostingRegressor	december_enriched_geo	109.985	Yes	No
december_catboost__december_basic	CatBoostRegressor	december_basic	120.387	Yes	No
december_catboost__december_enriched_geo	CatBoostRegressor	december_enriched_geo	120.602	Yes	No


Full results: [december_model_comparison.csv](/C:/Users/Siddharth/OneDrive/Desktop/spotter/codex/artifacts/december_model_comparison.csv).
Verification
Clean pipeline: passed in 149.2 seconds.
Independent scorer: passed.
validation_predictions.csv: exactly load_id,predicted_rate, 12,000 unique template-ordered IDs, finite positive predictions.
December output: exact expected columns, 31 rows, finite positive predictions.
Both persisted joblib estimators match their declared families.
candidate_december.png exists and is non-empty.
score.py unchanged; SHA-256 remains f73af4f34acf0c43cdd6e2cd6b2c0ec8b9da4420fd6fdcd69bd86e14d4868634.
README, Loom script, JSON, CSVs, and extracted PDF text agree on models, families, metrics, and selection semantics.
Key source changes: [run_pipeline.py (line 311)](/C:/Users/Siddharth/OneDrive/Desktop/spotter/codex/run_pipeline.py:311), [train.py (line 236)](/C:/Users/Siddharth/OneDrive/Desktop/spotter/codex/src/train.py:236), [evaluate.py (line 31)](/C:/Users/Siddharth/OneDrive/Desktop/spotter/codex/src/evaluate.py:31), [reporting.py (line 315)](/C:/Users/Siddharth/OneDrive/Desktop/spotter/codex/src/reporting.py:315), and [features.py (line 157)](/C:/Users/Siddharth/OneDrive/Desktop/spotter/codex/src/features.py:157).
Changed/generated files:
Root: README.md, validation_predictions.csv, december-chart-inputs.csv
artifacts/: category_shift.csv, data_audit.json, december_model_comparison.csv, distribution_shift.csv, error_analysis.csv, feature_ablation.csv, feature_importance.csv, holdout_predictions.csv, input_manifest.json, missing_values.csv, model_comparison.csv, monthly_target_summary.csv, numeric_summary.csv, pipeline.log, scorer_output.txt, temporal_stability.csv, validation_metrics.json
models/: final_rate_model.joblib, december_rate_model.joblib
reports/: assessment_report.pdf, loom_script.md
reports/figures/: all five generated figures
scorer_results/: candidate_december.png
Remaining limitation: the outer MAE is a model-selection estimate, not an unbiased post-selection test score. The family-neutral feature screen also relies on two proxy families and one chronological inner window.

Worked for 2m 38s
Verdict: the model-selection fix is good, but don’t submit this exact bundle yet ⚠️

The uploaded pipeline now does the important part correctly:

HGB and CatBoost both participate in training-only feature screening.
Both families are included as eligible outer-holdout candidates.
The common feature set and CatBoost iteration count are frozen before outer evaluation.
The final candidate is selected using the lowest finite holdout MAE.
The selected model family is used during the final refit and persisted-model verification.

So the original issue—CatBoost being evaluated but unable to win—has genuinely been fixed.

🚨 The uploaded artifacts contradict the Codex summary

Your pasted Codex result says:

hgb_smoother selected
MAE 129.508
hgb_compact MAE 129.535
December MAE 109.244

But the uploaded PDF says:

hgb_compact selected
hgb_compact MAE 129.51
hgb_smoother MAE 130.26
December MAE 108.78

That is not a rounding difference. It is a different run, different code state, stale report, or nondeterministic result.

A reviewer could notice that your:

PDF
model_comparison.csv
validation_metrics.json
README
Loom narration

do not describe the same final model.

Do not manually change the PDF to hgb_smoother

First establish which run is actually canonical. The code selects dynamically from the measured candidates; it does not prove either hgb_compact or hgb_smoother should win without the corresponding generated artifacts.

PDF presentation also needs fixing 📄

The report contains the required validation explanation and December chart, so its content direction matches the assessment. The task specifically requires the validation/split approach and the chart created by score.py.

However:

Page 4 is almost completely empty, with only a small permutation-importance chart near the top.
On page 5, the December comparison table has severe text overlap between the model, family and feature columns.
The long internal model identifiers make the report look like an unpolished debug artifact rather than a final hiring document.

The underlying information is good, but the layout needs to look deliberate.

One claim I cannot fully verify

Codex says there are “exact expected candidate-set assertions.”

In the uploaded run_pipeline(4).py, I can verify an assertion that both HGB and CatBoost families must be present.

I cannot confirm an assertion for the exact expected candidate names because src/train.py and src/evaluate.py were not included in the latest upload. The visible pipeline builds the list from HGB_FINAL_CANDIDATES and adds CatBoost, but does not itself hardcode an expected set such as:

{
    "hgb_compact",
    "hgb_smoother",
    "catboost_rate_per_mile",
}

That is not a blocker, but Codex’s claim should be verified in the complete repository.

Send Codex this final reconciliation prompt
Perform a final submission-consistency and reproducibility pass.

There is currently a material mismatch between generated artifacts:

- One reported run selects hgb_smoother with MAE 129.50819581944845.
- The current assessment_report.pdf selects hgb_compact with MAE approximately 129.51 and reports hgb_smoother at approximately 130.26.
- One reported December MAE is 109.244, while the current PDF reports 108.78.

Do not manually choose or hardcode either result.

1. Start from the exact repository state intended for submission.
2. Record the current Git commit hash or generate a unique run ID.
3. Delete all generated output directories and files:
   - artifacts/
   - models/
   - reports/
   - scorer_results/
   - validation_predictions.csv
   - the filled December output
4. Restore the original supplied December input before running.
5. Run the complete pipeline once.
6. Save the resulting:
   - model_comparison.csv
   - december_model_comparison.csv
   - validation_metrics.json
   - pipeline.log
   - assessment_report.pdf
   - README.md
   - loom_script.md
   - validation_predictions.csv
7. Run the supplied scorer independently.
8. Run the full pipeline a second time from another clean output state.

Reproducibility requirements:
- Both runs must select the same main candidate and December candidate.
- All model metrics must match within a strict numerical tolerance.
- validation_predictions.csv must match exactly or within an explicitly justified floating-point tolerance.
- If the winning model changes between runs, identify and fix the nondeterminism.
- Ensure random_state/random_seed is explicitly configured everywhere applicable, including preprocessing, HGB, CatBoost, permutation importance, sampling, and any iteration-learning split.
- Ensure dictionary or filesystem iteration order cannot alter candidate ordering or tie resolution.

Artifact consistency requirements:
- Generate README, PDF report, Loom script and validation_metrics.json directly from the same in-memory selected candidate objects used to create the predictions.
- Do not copy model names or metrics manually.
- Add the run ID or Git commit hash to validation_metrics.json and the report footer.
- Add assertions that the selected main model, family, feature set and MAE appear identically in:
  - model_comparison.csv
  - validation_metrics.json
  - README.md
  - loom_script.md
  - extracted assessment_report.pdf text
- Add equivalent checks for the December model.
- Fail the pipeline if any artifact disagrees.

Candidate-set verification:
- Assert that the main eligible candidate-name set exactly equals:
  {
      "hgb_compact",
      "hgb_smoother",
      "catboost_rate_per_mile"
  }
  unless the repository deliberately defines a different documented set.
- Assert that every expected candidate appears exactly once.
- Assert that no unexpected candidate is marked eligible.
- Keep baselines explicitly ineligible.

PDF formatting:
- Remove the mostly empty standalone permutation-importance page.
- Place the importance chart beneath the main model discussion or beside error analysis.
- Fix the December model-comparison table so no text overlaps or clips.
- Use shorter display labels in the PDF while preserving full internal names in CSV/JSON:
  - HGB compact
  - HGB smoother
  - CatBoost
  - HGB basic
  - HGB geographic
  - CatBoost basic
  - CatBoost geographic
- Widen the model/family columns, wrap text safely, and verify all rendered PDF pages.
- Keep the scorer-generated December chart unchanged.

After completing the work, report:
1. The canonical selected main model and exact MAE.
2. The canonical December model and exact MAE.
3. Whether two clean runs produced identical selections and metrics.
4. The run ID or Git commit.
5. Every changed file.
6. Checksums for the final PDF, validation_predictions.csv,
   model_comparison.csv and validation_metrics.json.
7. Any remaining limitation.

Do not state that the repository is submission-ready unless the two clean runs
are consistent and every generated artifact agrees.

Once that mismatch and PDF formatting are fixed, this becomes a strong, hiring-ready MLE submission rather than merely a working model.

run_pipeline(5).py
Python

Yep, that’s the issue — your remote has Muneshar, but your username is Muneshwar.

Run:

git remote set-url origin https://github.com/SiddharthMuneshwar26/freight-rate-ml-assessment.git

Verify:

git remote -v

It should show:

origin  https://github.com/SiddharthMuneshwar26/freight-rate-ml-assessment.git (fetch)
origin  https://github.com/SiddharthMuneshwar26/freight-rate-ml-assessment.git (push)

Then make sure the repository exists at:

https://github.com/SiddharthMuneshwar26/freight-rate-ml-assessment

If that page is still 404, create the empty repository under the correct account first.

Then push:

git push -u origin main
ok before making the repo i think instead of assesment i wan it neutral so i can use it as my project too later

Yes—use a neutral project name so it works both as the hiring submission and as a portfolio project later. 👍

Recommended name
freight-rate-prediction

GitHub URL:

https://github.com/SiddharthMuneshwar26/freight-rate-prediction

Project title inside the README:

# Freight Rate Prediction Pipeline

GitHub description:

Leakage-safe freight rate prediction using chronological validation, feature engineering, HGB and CatBoost model comparison, and reproducible ML artifacts.
Rename your local folder

Because you are currently inside the folder, first move to its parent:

cd "C:\Users\Siddharth\OneDrive\Desktop\spotter"

Rename it:

Rename-Item `
  ".\freight-rate-ml-assessment" `
  "freight-rate-prediction"

Enter the renamed folder:

cd ".\freight-rate-prediction"

Confirm Git still works:

git status

Renaming the outer folder does not damage the Git repository.

Change the remote URL

Your current remote still points to the assessment-style name. Update it:

git remote set-url origin https://github.com/SiddharthMuneshwar26/freight-rate-prediction.git

Verify:

git remote -v

You should see:

origin  https://github.com/SiddharthMuneshwar26/freight-rate-prediction.git (fetch)
origin  https://github.com/SiddharthMuneshwar26/freight-rate-prediction.git (push)
Create the GitHub repository

On GitHub, create a new empty repository with:

freight-rate-prediction

Choose Public unless the company requested private.

Do not add:

README
.gitignore
license

Those already exist locally.

Then push:

git push -u origin main
What should remain assessment-specific?

It is okay to keep these filenames because they are part of the original task and generated submission:

freight-rate-ml-assessment.pdf
readme spotter.md
reports\assessment_report.pdf

They do not make the entire repository assessment-only. They show the original problem and formal report.

Do not rename them right now because your pipeline currently looks for the supplied assessment PDF by its existing filename.

Make the README neutral

Open it:

notepad README.md

Change the top heading from something like:

# Freight Rate Prediction — Machine Learning Engineer Assessment

to:

# Freight Rate Prediction Pipeline

Then add a small context section:

## Project context

This project implements an end-to-end machine-learning pipeline for predicting
freight rates. It was originally developed from a time-constrained modelling
brief and has been structured as a reproducible portfolio project.

Use neutral section names:

Instead of	Use
Assessment objective	Project objective
Candidate submission	Prediction output
Assessment report	Technical report
Assessment artifacts	Generated artifacts
Hiring task	Modelling brief

Do not remove the required submission links or files.

Commit the naming update

After changing the README:

git add README.md
git commit -m "Use portfolio-neutral project naming"

Then, after creating the GitHub repository:

git push -u origin main

The final clean identity should be:

Folder:      freight-rate-prediction
Repository:  freight-rate-prediction
README:      Freight Rate Prediction Pipeline
Entry point: run_pipeline.py

That presents it as a proper ML engineering project while still satisfying the hiring submission.

# Freight Rate Prediction — Machine Learning Engineer Assessment

Run ID: submission-c27c7409c45454ca

Canonical generated selections:


text
MAIN|model=hgb_compact|family=HistGradientBoostingRegressor|feature_set=basic_plus_geographic|mae=129.511954241217
DECEMBER|model=december_hgb__december_basic|family=HistGradientBoostingRegressor|feature_set=december_basic|mae=108.7836047385515


## Objective

Train a leakage-safe freight-rate regressor on the supplied labeled data, predict all 12,000 future validation loads, and create the fixed Lexington-to-Fort-Wayne December series required by the supplied scorer.

## Dataset overview

- Development: 48,000 rows, 2025-01-01 through 2025-10-31; target posted_rate.
- Final validation: 12,000 rows, 2025-11-01 through 2025-12-31.
- December chart input: 31 fixed rows for 2025-12-01 through 2025-12-31.
- Missing development values occur in weight and market index. Impossible non-positive weights are treated as missing and flagged. Valid target extremes are retained.

## Repository structure


text
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


## Setup and reproduction

Python 3.11+ is recommended.


bash
python -m venv .venv
# Windows PowerShell: .venv\Scripts\Activate.ps1
# macOS/Linux: source .venv/bin/activate
python -m pip install -r requirements.txt
python run_pipeline.py


The final command locates the supplied root-level files (with tolerant filename aliases), audits them, creates the chronological split, screens feature groups with both model families, compares all eligible HGB and CatBoost candidates, refits the selected families, predicts, validates both outputs, runs the unmodified scorer, and generates the report.

## Validation and model

The primary holdout mirrors the two-month future horizon: train 2025-01-01 through 2025-08-31 (38,477 rows), validate 2025-09-01 through 2025-10-31 (9,523 rows). Feature groups are screened before that comparison on a separate training-only split: fit through 2025-07-01 and validate from 2025-07-02 through 2025-08-31. Fixed HGB and CatBoost proxies score every logical feature set on the same fold-pure matrices, and the lowest equal-family mean MAE chooses one common list. CatBoost's best iteration for that list comes from the same training-only screen. Every outer candidate configuration is then frozen before scoring; outer labels are used only for the required final candidate selection and post-selection diagnostics, never to refit or adapt a candidate before its MAE is recorded.

Selection rule: Lowest finite MAE on the common chronological holdout among all eligible candidates; exact-equality ties are broken deterministically by candidate name.

Selected model: **HistGradientBoostingRegressor** (hgb_compact), using **basic_plus_geographic**. Histogram gradient boosting uses an ordinal category encoder fitted within each training fit; previously unseen categories map to a reserved value. MAE is the primary metric because dollars of absolute error are directly interpretable and robust to sparse label anomalies.

- Selected parameters: {"l2_regularization": 3.0, "learning_rate": 0.06, "max_iter": 250, "max_leaf_nodes": 15, "min_samples_leaf": 30}
- Saved final model: models\final_rate_model.joblib

| Metric | Chronological holdout |
|---|---:|
| MAE | $129.51 |
| RMSE | $640.32 |
| R-squared | 0.8239 |
| sMAPE | 4.527% |

Feature sets that lowered equal-family mean MAE relative to the basic supplied-feature model on the training-only screen: basic_plus_geographic, basic_plus_route. The winning set was frozen before any outer-holdout scoring; groups that did not improve that inner validation window were not carried into final candidate comparison.

## December method

The reduced-schema selection is intentionally separate from the main validation model. Eligible December-compatible HGB and CatBoost candidates use the same chronological outer rows, rate-per-mile target, and posted-rate MAE; within each feature-set comparison, both families receive the same compatible columns. CatBoost early stopping remains internal to the outer training rows.

Selected December model: **HistGradientBoostingRegressor** (december_hgb__december_basic), using **december_basic**, with holdout MAE **$108.78**. It uses only fields available in the fixed input plus known date features; coordinate enrichment, when selected, is learned exclusively from development city mappings. It does not fabricate market or quote signals.

- December parameters: {"l2_regularization": 3.0, "learning_rate": 0.06, "max_iter": 250, "max_leaf_nodes": 15, "min_samples_leaf": 30}
- Saved December model: models\december_rate_model.joblib

## Outputs and scorer


bash
python score.py --predictions validation_predictions.csv --december-predictions december-chart-inputs.csv


Successful scorer output is captured in artifacts/scorer_output.txt; the chart is scorer_results/candidate_december.png. Local metrics are development holdout metrics; the supplied scorer validates structure only and Spotter evaluates final validation accuracy after submission.

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
can you do the changes and add figures and pics too 


Done — I rewrote it as a portfolio-neutral project README and added:

neutral project branding
result summary table
all existing project figures
scorer-generated December chart
Mermaid workflow diagram
model comparison table
cleaner setup and reproduction instructions
artifact links
limitations and reproducibility sections
original assessment files kept only as project context

README.md

Replace the root README at:

C:\Users\Siddharth\OneDrive\Desktop\spotter\freight-rate-prediction\README.md

The images are referenced using these existing repository paths:

reports/figures/monthly_target.png
reports/figures/distribution_shift.png
reports/figures/target_distribution.png
reports/figures/feature_importance.png
reports/figures/holdout_error_over_time.png
scorer_results/candidate_december.png

They will display automatically on GitHub once the README and image files are committed.

Important before committing

Add your real Loom URL at the bottom:

**Video walkthrough:** https://www.loom.com/share/YOUR_VIDEO_ID

Your pipeline regenerates README.md whenever python run_pipeline.py runs. Therefore, either:

replace the README after your final pipeline run, or
later update write_readme() inside src/reporting.py to generate this neutral version.

For the current submission, replacing it after the final run is simplest.

Then commit:

git add README.md
git commit -m "Polish portfolio README and add project visuals"

Use this neutral GitHub repository name:

freight-rate-prediction

And set the remote:

git remote set-url origin https://github.com/SiddharthMuneshwar26/freight-rate-prediction.git

README.md
Document

Library
/
README.md
Freight Rate Prediction Pipeline







An end-to-end machine-learning pipeline for forecasting freight rates on future loads. The project emphasizes chronological validation, leakage-safe feature engineering, model-family-neutral selection, reproducible execution, and submission-grade output validation.

Originally developed from a time-constrained ML engineering brief, the repository is structured as a reusable portfolio project rather than a one-off notebook.

Project highlights
48,000 labeled development rows and 12,000 future validation rows
A blocked chronological holdout that mirrors the final two-month forecast horizon
Training-only feature screening across both HistGradientBoostingRegressor and CatBoostRegressor
Separate reduced-schema model selection for the fixed December route series
Automated data audits, comparison tables, error analysis, reporting, and scorer execution
Deterministic candidate ordering, stable row sorting, fixed seeds, and generated-artifact consistency checks
Canonical result
Output	Selected model	Feature set	Holdout MAE
Main validation	hgb_compact	basic_plus_geographic	$129.51
December series	december_hgb__december_basic	december_basic	$108.78

Run ID: submission-c27c7409c45454ca

MAIN|model=hgb_compact|family=HistGradientBoostingRegressor|feature_set=basic_plus_geographic|mae=129.511954241217
DECEMBER|model=december_hgb__december_basic|family=HistGradientBoostingRegressor|feature_set=december_basic|mae=108.7836047385515
Visual results
Target behaviour over time




Development-to-validation distribution shift




Target distribution




Selected-model feature importance




Holdout error over time




Fixed December prediction series




Modelling workflow
flowchart LR
    A[Raw development and validation data] --> B[Schema and data-quality audit]
    B --> C[Chronological outer split]
    C --> D[Training-only inner split]
    D --> E[Feature screening with HGB and CatBoost]
    E --> F[Freeze common feature set and CatBoost iterations]
    F --> G[Outer candidate comparison]
    G --> H[Select minimum-MAE eligible model]
    H --> I[Refit on all labeled development rows]
    I --> J[Generate 12,000 validation predictions]
    C --> K[Reduced-schema December comparison]
    K --> L[Refit December-compatible model]
    L --> M[Generate 31 fixed December predictions]
    J --> N[Validate schemas and run supplied scorer]
    M --> N
    N --> O[Generate reports, metrics, figures, and consistency checks]
Objective

Train a leakage-safe freight-rate regressor on the supplied labeled data, predict all 12,000 future validation loads, and generate the fixed Lexington-to-Fort-Wayne December series required by the supplied scorer.

Dataset overview
Development: 48,000 rows from 2025-01-01 through 2025-10-31
Target: posted_rate
Final validation: 12,000 rows from 2025-11-01 through 2025-12-31
December chart input: 31 fixed rows from 2025-12-01 through 2025-12-31
Missing development values occur in weight and market_index
Impossible non-positive weights are treated as missing and explicitly flagged
Valid target extremes are retained rather than silently removed
Validation strategy

The primary holdout mirrors the future forecasting horizon:

Split	Date range	Rows
Outer training	2025-01-01 to 2025-08-31	38,477
Outer holdout	2025-09-01 to 2025-10-31	9,523
Final validation	2025-11-01 to 2025-12-31	12,000

Feature groups and CatBoost iteration counts are decided on a separate training-only chronological window:

Inner split	Date range
Inner training	through 2025-07-01
Inner validation	2025-07-02 to 2025-08-31

Fixed HGB and CatBoost proxies evaluate every logical feature set on the same fold-pure matrices. The feature list with the lowest equal-family mean MAE is frozen before outer-holdout scoring.

The outer holdout is used only for:

final candidate selection,
model diagnostics,
error analysis.

It is not used to adapt a candidate and then rescore it.

Selection rule

Select the eligible candidate with the lowest finite MAE on the common chronological holdout. Exact-equality ties are broken deterministically by candidate name.

Features

The selected basic_plus_geographic feature set combines supplied operational variables with compact, leakage-safe geographic features.

Feature groups evaluated include:

supplied distance, equipment, weight, market, and quote fields
cyclic and calendar date features
route combinations and training-only frequency signals
unseen-route and unseen-city indicators
compact geographic calculations
defensible distance, weight, and market interactions

load_id, posted_rate, and predicted_rate are never model features.

Historical target encodings were deliberately excluded because sparse routes and chronological leakage risk outweighed their potential value.

Main model

The selected main model is:

HistGradientBoostingRegressor
Candidate: hgb_compact
Feature set: basic_plus_geographic

Histogram gradient boosting uses an ordinal categorical encoder fitted only within each training fit. Previously unseen categories map to a reserved value.

Parameters
{
  "l2_regularization": 3.0,
  "learning_rate": 0.06,
  "max_iter": 250,
  "max_leaf_nodes": 15,
  "min_samples_leaf": 30
}
Chronological holdout performance
Metric	Value
MAE	$129.51
RMSE	$640.32
R²	0.8239
sMAPE	4.527%

The generated model is saved locally to:

models/final_rate_model.joblib

The models/ directory may be excluded from Git because the estimators are reproducible from the tracked code and data.

December-compatible model

The fixed December input has a reduced schema, so it receives a separate model-selection process.

All eligible December-compatible HGB and CatBoost candidates use:

the same chronological outer rows,
rate-per-mile modelling,
posted-rate MAE,
identical compatible columns within each feature-set comparison.

The selected model is:

HistGradientBoostingRegressor
Candidate: december_hgb__december_basic
Feature set: december_basic
Holdout MAE: $108.78

It uses only fields available in the fixed input plus known date features. Market and quote signals are not fabricated.

Generated model path:

models/december_rate_model.joblib
Model comparison
Candidate	Family	Feature set	MAE	Eligible	Selected
hgb_compact	HistGradientBoostingRegressor	basic_plus_geographic	129.51	Yes	Yes
hgb_smoother	HistGradientBoostingRegressor	basic_plus_geographic	130.26	Yes	No
catboost_rate_per_mile	CatBoostRegressor	basic_plus_geographic	141.19	Yes	No
equipment_median_rpm	Baseline	distance + equipment	229.10	No	No
global_median	Baseline	target only	1,148.92	No	No

Full results:

artifacts/model_comparison.csv
artifacts/december_model_comparison.csv
artifacts/feature_ablation.csv
Repository structure
.
├── README.md
├── requirements.txt
├── run_pipeline.py
├── score.py
├── train-test.csv
├── validation.csv
├── validation-predictions-template.csv
├── validation_predictions.csv
├── december-chart-inputs-template.csv
├── december-chart-inputs.csv
├── src/
│   ├── __init__.py
│   ├── evaluate.py
│   ├── features.py
│   ├── reporting.py
│   └── train.py
├── artifacts/
│   ├── data_audit.json
│   ├── model_comparison.csv
│   ├── december_model_comparison.csv
│   ├── feature_ablation.csv
│   ├── feature_importance.csv
│   ├── error_analysis.csv
│   ├── temporal_stability.csv
│   ├── validation_metrics.json
│   └── scorer_output.txt
├── reports/
│   ├── assessment_report.pdf
│   ├── loom_script.md
│   └── figures/
│       ├── distribution_shift.png
│       ├── feature_importance.png
│       ├── holdout_error_over_time.png
│       ├── monthly_target.png
│       └── target_distribution.png
└── scorer_results/
    └── candidate_december.png
Setup

Python 3.11+ is recommended.

Windows PowerShell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
$env:PYTHONHASHSEED = "42"
python run_pipeline.py
macOS/Linux
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
PYTHONHASHSEED=42 python run_pipeline.py

The pipeline:

locates and validates all supplied files,
performs data-quality and distribution-shift audits,
creates inner and outer chronological splits,
screens feature groups using both model families,
compares all eligible final candidates,
refits the selected main and December models,
generates both required prediction outputs,
validates their schemas,
executes the unmodified scorer,
generates reports, figures, and consistency checks.
Run the scorer independently
python score.py \
  --predictions validation_predictions.csv \
  --december-predictions december-chart-inputs.csv

On PowerShell:

python score.py `
  --predictions validation_predictions.csv `
  --december-predictions december-chart-inputs.csv

Successful scorer output is captured in artifacts/scorer_output.txt.

The scorer validates the required output structure and generates:

scorer_results/candidate_december.png

Local performance values in this repository come from the chronological development holdout.

Generated outputs
Artifact	Purpose
validation_predictions.csv	Final 12,000-row prediction output
december-chart-inputs.csv	Completed fixed December input
reports/assessment_report.pdf	Technical report
reports/loom_script.md	2–3 minute walkthrough outline
artifacts/validation_metrics.json	Canonical model and run metadata
artifacts/model_comparison.csv	Main candidate comparison
artifacts/error_analysis.csv	Segment-level error analysis
scorer_results/candidate_december.png	Fixed December prediction chart
Original project brief
freight-rate-ml-assessment.pdf
readme spotter.md
Reproducibility
Fixed random seed: 42
Stable date, load_id row ordering
Sorted feature-set and candidate iteration
Deterministic tie-breaking by candidate name
CPU-constrained numerical execution
Candidate-pool assertions for both main and December models
Saved-model family checks
Supplied scorer checksum verification
Pristine December-template checksum verification
Generated README, report, metrics, and comparison-table consistency checks
Content-addressed run identity

The current canonical run is:

submission-c27c7409c45454ca
Assumptions and limitations
The selected feature list depends on one inner chronological window and two fixed proxy families.
The reported outer-holdout MAE is a model-selection estimate, not an unbiased post-selection test score.
Final validation includes eight cities absent from development, so unseen-city performance remains uncertain.
Market index shows the largest development-to-validation shift.
Chronological validation reduces future-regime risk but cannot remove it.
Coordinates appear obfuscated, so supplied distance is treated as authoritative.
Sparse extreme rates dominate RMSE and are retained rather than silently removed.
No external data, credentials, or private APIs are used.
Documentation
Technical report
Walkthrough script
Scorer output

Video walkthrough: [Add final Loom link here]

Library
/
README.md
Freight Rate Prediction Pipeline







An end-to-end machine-learning pipeline for forecasting freight rates on future loads. The project emphasizes chronological validation, leakage-safe feature engineering, model-family-neutral selection, reproducible execution, and submission-grade output validation.

Originally developed from a time-constrained ML engineering brief, the repository is structured as a reusable portfolio project rather than a one-off notebook.

Project highlights
48,000 labeled development rows and 12,000 future validation rows
A blocked chronological holdout that mirrors the final two-month forecast horizon
Training-only feature screening across both HistGradientBoostingRegressor and CatBoostRegressor
Separate reduced-schema model selection for the fixed December route series
Automated data audits, comparison tables, error analysis, reporting, and scorer execution
Deterministic candidate ordering, stable row sorting, fixed seeds, and generated-artifact consistency checks
Canonical result
Output	Selected model	Feature set	Holdout MAE
Main validation	hgb_compact	basic_plus_geographic	$129.51
December series	december_hgb__december_basic	december_basic	$108.78

Run ID: submission-c27c7409c45454ca

MAIN|model=hgb_compact|family=HistGradientBoostingRegressor|feature_set=basic_plus_geographic|mae=129.511954241217
DECEMBER|model=december_hgb__december_basic|family=HistGradientBoostingRegressor|feature_set=december_basic|mae=108.7836047385515
Visual results
Target behaviour over time




Development-to-validation distribution shift




Target distribution




Selected-model feature importance




Holdout error over time




Fixed December prediction series




Modelling workflow
flowchart LR
    A[Raw development and validation data] --> B[Schema and data-quality audit]
    B --> C[Chronological outer split]
    C --> D[Training-only inner split]
    D --> E[Feature screening with HGB and CatBoost]
    E --> F[Freeze common feature set and CatBoost iterations]
    F --> G[Outer candidate comparison]
    G --> H[Select minimum-MAE eligible model]
    H --> I[Refit on all labeled development rows]
    I --> J[Generate 12,000 validation predictions]
    C --> K[Reduced-schema December comparison]
    K --> L[Refit December-compatible model]
    L --> M[Generate 31 fixed December predictions]
    J --> N[Validate schemas and run supplied scorer]
    M --> N
    N --> O[Generate reports, metrics, figures, and consistency checks]
Objective

Train a leakage-safe freight-rate regressor on the supplied labeled data, predict all 12,000 future validation loads, and generate the fixed Lexington-to-Fort-Wayne December series required by the supplied scorer.

Dataset overview
Development: 48,000 rows from 2025-01-01 through 2025-10-31
Target: posted_rate
Final validation: 12,000 rows from 2025-11-01 through 2025-12-31
December chart input: 31 fixed rows from 2025-12-01 through 2025-12-31
Missing development values occur in weight and market_index
Impossible non-positive weights are treated as missing and explicitly flagged
Valid target extremes are retained rather than silently removed
Validation strategy

The primary holdout mirrors the future forecasting horizon:

Split	Date range	Rows
Outer training	2025-01-01 to 2025-08-31	38,477
Outer holdout	2025-09-01 to 2025-10-31	9,523
Final validation	2025-11-01 to 2025-12-31	12,000

Feature groups and CatBoost iteration counts are decided on a separate training-only chronological window:

Inner split	Date range
Inner training	through 2025-07-01
Inner validation	2025-07-02 to 2025-08-31

Fixed HGB and CatBoost proxies evaluate every logical feature set on the same fold-pure matrices. The feature list with the lowest equal-family mean MAE is frozen before outer-holdout scoring.

The outer holdout is used only for:

final candidate selection,
model diagnostics,
error analysis.

It is not used to adapt a candidate and then rescore it.

Selection rule

Select the eligible candidate with the lowest finite MAE on the common chronological holdout. Exact-equality ties are broken deterministically by candidate name.

Features

The selected basic_plus_geographic feature set combines supplied operational variables with compact, leakage-safe geographic features.

Feature groups evaluated include:

supplied distance, equipment, weight, market, and quote fields
cyclic and calendar date features
route combinations and training-only frequency signals
unseen-route and unseen-city indicators
compact geographic calculations
defensible distance, weight, and market interactions

load_id, posted_rate, and predicted_rate are never model features.

Historical target encodings were deliberately excluded because sparse routes and chronological leakage risk outweighed their potential value.

Main model

The selected main model is:

HistGradientBoostingRegressor
Candidate: hgb_compact
Feature set: basic_plus_geographic

Histogram gradient boosting uses an ordinal categorical encoder fitted only within each training fit. Previously unseen categories map to a reserved value.

Parameters
{
  "l2_regularization": 3.0,
  "learning_rate": 0.06,
  "max_iter": 250,
  "max_leaf_nodes": 15,
  "min_samples_leaf": 30
}
Chronological holdout performance
Metric	Value
MAE	$129.51
RMSE	$640.32
R²	0.8239
sMAPE	4.527%

The generated model is saved locally to:

models/final_rate_model.joblib

The models/ directory may be excluded from Git because the estimators are reproducible from the tracked code and data.

December-compatible model

The fixed December input has a reduced schema, so it receives a separate model-selection process.

All eligible December-compatible HGB and CatBoost candidates use:

the same chronological outer rows,
rate-per-mile modelling,
posted-rate MAE,
identical compatible columns within each feature-set comparison.

The selected model is:

HistGradientBoostingRegressor
Candidate: december_hgb__december_basic
Feature set: december_basic
Holdout MAE: $108.78

It uses only fields available in the fixed input plus known date features. Market and quote signals are not fabricated.

Generated model path:

models/december_rate_model.joblib
Model comparison
Candidate	Family	Feature set	MAE	Eligible	Selected
hgb_compact	HistGradientBoostingRegressor	basic_plus_geographic	129.51	Yes	Yes
hgb_smoother	HistGradientBoostingRegressor	basic_plus_geographic	130.26	Yes	No
catboost_rate_per_mile	CatBoostRegressor	basic_plus_geographic	141.19	Yes	No
equipment_median_rpm	Baseline	distance + equipment	229.10	No	No
global_median	Baseline	target only	1,148.92	No	No

Full results:

artifacts/model_comparison.csv
artifacts/december_model_comparison.csv
artifacts/feature_ablation.csv
Repository structure
.
├── README.md
├── requirements.txt
├── run_pipeline.py
├── score.py
├── train-test.csv
├── validation.csv
├── validation-predictions-template.csv
├── validation_predictions.csv
├── december-chart-inputs-template.csv
├── december-chart-inputs.csv
├── src/
│   ├── __init__.py
│   ├── evaluate.py
│   ├── features.py
│   ├── reporting.py
│   └── train.py
├── artifacts/
│   ├── data_audit.json
│   ├── model_comparison.csv
│   ├── december_model_comparison.csv
│   ├── feature_ablation.csv
│   ├── feature_importance.csv
│   ├── error_analysis.csv
│   ├── temporal_stability.csv
│   ├── validation_metrics.json
│   └── scorer_output.txt
├── reports/
│   ├── assessment_report.pdf
│   ├── loom_script.md
│   └── figures/
│       ├── distribution_shift.png
│       ├── feature_importance.png
│       ├── holdout_error_over_time.png
│       ├── monthly_target.png
│       └── target_distribution.png
└── scorer_results/
    └── candidate_december.png
Setup

Python 3.11+ is recommended.

Windows PowerShell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
$env:PYTHONHASHSEED = "42"
python run_pipeline.py
macOS/Linux
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
PYTHONHASHSEED=42 python run_pipeline.py

The pipeline:

locates and validates all supplied files,
performs data-quality and distribution-shift audits,
creates inner and outer chronological splits,
screens feature groups using both model families,
compares all eligible final candidates,
refits the selected main and December models,
generates both required prediction outputs,
validates their schemas,
executes the unmodified scorer,
generates reports, figures, and consistency checks.
Run the scorer independently
python score.py \
  --predictions validation_predictions.csv \
  --december-predictions december-chart-inputs.csv

On PowerShell:

python score.py `
  --predictions validation_predictions.csv `
  --december-predictions december-chart-inputs.csv

Successful scorer output is captured in artifacts/scorer_output.txt.

The scorer validates the required output structure and generates:

scorer_results/candidate_december.png

Local performance values in this repository come from the chronological development holdout.

Generated outputs
Artifact	Purpose
validation_predictions.csv	Final 12,000-row prediction output
december-chart-inputs.csv	Completed fixed December input
reports/assessment_report.pdf	Technical report
reports/loom_script.md	2–3 minute walkthrough outline
artifacts/validation_metrics.json	Canonical model and run metadata
artifacts/model_comparison.csv	Main candidate comparison
artifacts/error_analysis.csv	Segment-level error analysis
scorer_results/candidate_december.png	Fixed December prediction chart
Original project brief
freight-rate-ml-assessment.pdf
readme spotter.md
Reproducibility
Fixed random seed: 42
Stable date, load_id row ordering
Sorted feature-set and candidate iteration
Deterministic tie-breaking by candidate name
CPU-constrained numerical execution
Candidate-pool assertions for both main and December models
Saved-model family checks
Supplied scorer checksum verification
Pristine December-template checksum verification
Generated README, report, metrics, and comparison-table consistency checks
Content-addressed run identity

The current canonical run is:

submission-c27c7409c45454ca
Assumptions and limitations
The selected feature list depends on one inner chronological window and two fixed proxy families.
The reported outer-holdout MAE is a model-selection estimate, not an unbiased post-selection test score.
Final validation includes eight cities absent from development, so unseen-city performance remains uncertain.
Market index shows the largest development-to-validation shift.
Chronological validation reduces future-regime risk but cannot remove it.
Coordinates appear obfuscated, so supplied distance is treated as authoritative.
Sparse extreme rates dominate RMSE and are retained rather than silently removed.
No external data, credentials, or private APIs are used.
Documentation
Technical report
Walkthrough script
Scorer output

Video walkthrough: [Add final Loom link here]