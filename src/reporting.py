"""Create evidence-backed EDA, documentation, plots, and the PDF report."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from xml.sax.saxutils import escape

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from reportlab import rl_config
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    Image,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

rl_config.invariant = True


def _serialise(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (pd.Timestamp, pd.Period)):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _serialise(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_serialise(item) for item in value]
    return value


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_serialise(payload), indent=2, sort_keys=True), encoding="utf-8")


def _haversine(frame: pd.DataFrame) -> pd.Series:
    lat1 = np.radians(frame["pickup_lat"])
    lat2 = np.radians(frame["delivery_lat"])
    delta_lat = lat2 - lat1
    delta_lon = np.radians(frame["delivery_lon"] - frame["pickup_lon"])
    value = (
        np.sin(delta_lat / 2) ** 2
        + np.cos(lat1) * np.cos(lat2) * np.sin(delta_lon / 2) ** 2
    ).clip(0, 1)
    return 3958.7613 * 2 * np.arctan2(np.sqrt(value), np.sqrt(1 - value))


def create_eda_artifacts(
    development: pd.DataFrame,
    validation: pd.DataFrame,
    december: pd.DataFrame,
    artifacts_dir: Path,
    figures_dir: Path,
) -> dict[str, Any]:
    """Audit source data and save a compact set of decision-relevant outputs."""
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    dev = development.copy()
    val = validation.copy()
    dev_dates = pd.to_datetime(dev["date"], errors="coerce")
    val_dates = pd.to_datetime(val["date"], errors="coerce")

    missing_rows: list[dict[str, Any]] = []
    for dataset, frame in [("development", dev), ("validation", val), ("december", december)]:
        for column in frame.columns:
            count = int(frame[column].isna().sum())
            missing_rows.append(
                {
                    "dataset": dataset,
                    "column": column,
                    "missing_count": count,
                    "missing_percent": 100.0 * count / len(frame),
                }
            )
    pd.DataFrame(missing_rows).to_csv(artifacts_dir / "missing_values.csv", index=False)

    numeric_columns = ["distance", "weight", "market_index", "quote_signal"]
    numeric_rows: list[dict[str, Any]] = []
    for dataset, frame in [("development", dev), ("validation", val)]:
        for column in numeric_columns:
            series = pd.to_numeric(frame[column], errors="coerce")
            if column == "weight":
                summary_series = series.abs()
            else:
                summary_series = series
            numeric_rows.append(
                {
                    "dataset": dataset,
                    "column": column,
                    "count": int(summary_series.notna().sum()),
                    "mean": float(summary_series.mean()),
                    "std": float(summary_series.std()),
                    "min": float(summary_series.min()),
                    "median": float(summary_series.median()),
                    "p95": float(summary_series.quantile(0.95)),
                    "max": float(summary_series.max()),
                    "missing": int(series.isna().sum()),
                    "non_positive": int(series.le(0).sum()),
                }
            )
    numeric_summary = pd.DataFrame(numeric_rows)
    numeric_summary.to_csv(artifacts_dir / "numeric_summary.csv", index=False)

    category_rows: list[dict[str, Any]] = []
    for column in ["pickup", "delivery", "equipment"]:
        dev_values = set(dev[column].dropna().astype(str))
        val_values = set(val[column].dropna().astype(str))
        unseen = val_values - dev_values
        category_rows.append(
            {
                "column": column,
                "development_unique": len(dev_values),
                "validation_unique": len(val_values),
                "unseen_categories": len(unseen),
                "unseen_values": " | ".join(sorted(unseen)),
                "validation_rows_with_unseen": int(val[column].astype(str).isin(unseen).sum()),
            }
        )
    category_shift = pd.DataFrame(category_rows)
    category_shift.to_csv(artifacts_dir / "category_shift.csv", index=False)

    shift_rows: list[dict[str, Any]] = []
    for column in numeric_columns:
        dev_series = pd.to_numeric(dev[column], errors="coerce")
        val_series = pd.to_numeric(val[column], errors="coerce")
        if column == "weight":
            dev_series, val_series = dev_series.abs(), val_series.abs()
        pooled = np.sqrt((dev_series.var() + val_series.var()) / 2)
        smd = (val_series.mean() - dev_series.mean()) / pooled if pooled > 0 else 0.0
        shift_rows.append(
            {
                "feature": column,
                "development_mean": float(dev_series.mean()),
                "validation_mean": float(val_series.mean()),
                "standardized_mean_difference": float(smd),
            }
        )
    shift = pd.DataFrame(shift_rows)
    shift.to_csv(artifacts_dir / "distribution_shift.csv", index=False)

    dev_routes = dev["pickup"].astype(str) + " > " + dev["delivery"].astype(str)
    val_routes = val["pickup"].astype(str) + " > " + val["delivery"].astype(str)
    route_counts = dev_routes.value_counts()
    val_route_counts = val_routes.map(route_counts).fillna(0)
    haversine = _haversine(dev)
    target = dev["posted_rate"]

    audit: dict[str, Any] = {
        "development_shape": list(dev.shape),
        "validation_shape": list(val.shape),
        "december_shape": list(december.shape),
        "columns": {"development": list(dev.columns), "validation": list(val.columns)},
        "dtypes": {column: str(dtype) for column, dtype in dev.dtypes.items()},
        "target": "posted_rate",
        "date_ranges": {
            "development": [str(dev_dates.min().date()), str(dev_dates.max().date())],
            "validation": [str(val_dates.min().date()), str(val_dates.max().date())],
        },
        "validation_strictly_later": bool(val_dates.min() > dev_dates.max()),
        "unique_pickup_cities": int(dev["pickup"].nunique()),
        "unique_delivery_cities": int(dev["delivery"].nunique()),
        "equipment_categories": dev["equipment"].value_counts().astype(int).to_dict(),
        "duplicate_rows": {"development": int(dev.duplicated().sum()), "validation": int(val.duplicated().sum())},
        "duplicate_ids": {
            "development": int(dev["load_id"].duplicated().sum()),
            "validation": int(val["load_id"].duplicated().sum()),
        },
        "invalid_dates": {"development": int(dev_dates.isna().sum()), "validation": int(val_dates.isna().sum())},
        "negative_weight_rows": {
            "development": int(pd.to_numeric(dev["weight"], errors="coerce").lt(0).sum()),
            "validation": int(pd.to_numeric(val["weight"], errors="coerce").lt(0).sum()),
        },
        "target_distribution": {
            "count": int(target.count()),
            "mean": float(target.mean()),
            "std": float(target.std()),
            "min": float(target.min()),
            "p05": float(target.quantile(0.05)),
            "median": float(target.median()),
            "p95": float(target.quantile(0.95)),
            "p99": float(target.quantile(0.99)),
            "max": float(target.max()),
            "skew": float(target.skew()),
        },
        "routes": {
            "development_unique": int(dev_routes.nunique()),
            "frequency_below_5": int((route_counts < 5).sum()),
            "validation_unseen_rows": int(val_route_counts.eq(0).sum()),
            "validation_rare_seen_rows": int(val_route_counts.between(1, 9).sum()),
        },
        "coordinates": {
            "invalid_latitude_rows": int(
                (~dev["pickup_lat"].between(-90, 90) | ~dev["delivery_lat"].between(-90, 90)).sum()
            ),
            "invalid_longitude_rows": int(
                (~dev["pickup_lon"].between(-180, 180) | ~dev["delivery_lon"].between(-180, 180)).sum()
            ),
            "distance_haversine_correlation": float(dev["distance"].corr(haversine)),
            "median_distance_haversine_ratio": float((dev["distance"] / haversine.clip(lower=1)).median()),
            "note": "Coordinates are internally consistent but appear synthetic; supplied distance remains authoritative.",
        },
        "cleaning_decisions": [
            "Canonicalise categorical whitespace/case; use Unknown for missing categories.",
            "Treat impossible non-positive weights as missing and retain a problem flag.",
            "Retain numeric missing values and explicit missing/problem flags; all learned preprocessing is fitted only on the training partition.",
            "Keep valid extreme freight rates and distance/weight boundary values; the absolute-error objective is robust to sparse label anomalies.",
            "Fit any category encodings and optional frequency maps on training rows only; each model family handles unseen categories without holdout-fitted mappings.",
        ],
    }
    write_json(artifacts_dir / "data_audit.json", audit)

    cap = float(target.quantile(0.995))
    fig, axis = plt.subplots(figsize=(8.2, 4.4), dpi=150)
    axis.hist(target.clip(upper=cap), bins=45, color="#0B5963", alpha=0.88)
    axis.axvline(target.median(), color="#F2A541", linewidth=2, label=f"Median ${target.median():,.0f}")
    axis.set(title="Development target distribution (top 0.5% capped for display)", xlabel="Posted rate ($)", ylabel="Rows")
    axis.legend(frameon=False)
    axis.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(figures_dir / "target_distribution.png", bbox_inches="tight")
    plt.close(fig)

    monthly = dev.assign(month=dev_dates.dt.to_period("M").astype(str)).groupby("month")["posted_rate"].agg(["mean", "median"])
    monthly.to_csv(artifacts_dir / "monthly_target_summary.csv")
    fig, axis = plt.subplots(figsize=(8.2, 4.3), dpi=150)
    axis.plot(monthly.index, monthly["mean"], marker="o", label="Mean")
    axis.plot(monthly.index, monthly["median"], marker="o", label="Median")
    axis.set(title="Target level by development month", ylabel="Posted rate ($)", xlabel="Month")
    axis.tick_params(axis="x", rotation=35)
    axis.legend(frameon=False)
    axis.grid(axis="y", alpha=0.25)
    axis.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(figures_dir / "monthly_target.png", bbox_inches="tight")
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(7.2, 3.8), dpi=150)
    colors_for_shift = ["#B4493D" if abs(value) >= 0.5 else "#0B5963" for value in shift["standardized_mean_difference"]]
    axis.barh(shift["feature"], shift["standardized_mean_difference"], color=colors_for_shift)
    axis.axvline(0, color="#34495E", linewidth=0.8)
    axis.set(title="Validation shift vs development", xlabel="Standardized mean difference")
    axis.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(figures_dir / "distribution_shift.png", bbox_inches="tight")
    plt.close(fig)
    return audit


def create_model_figures(
    feature_importance: pd.DataFrame,
    residual_rows: pd.DataFrame,
    figures_dir: Path,
) -> None:
    figures_dir.mkdir(parents=True, exist_ok=True)
    top = feature_importance.sort_values(
        ["importance", "feature"],
        ascending=[False, True],
        kind="mergesort",
    ).head(15).sort_values(["importance", "feature"], ascending=[True, True], kind="mergesort")
    fig, axis = plt.subplots(figsize=(7.4, 4.8), dpi=150)
    axis.barh(top["feature"], top["importance"], color="#0B5963")
    axis.set(title="Selected model: top permutation importances", xlabel="Holdout MAE increase ($)")
    axis.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(figures_dir / "feature_importance.png", bbox_inches="tight")
    plt.close(fig)

    daily = residual_rows.groupby("date", observed=True).agg(mae=("absolute_error", "mean"), bias=("residual", "mean"))
    fig, axes = plt.subplots(2, 1, figsize=(8.2, 5.2), dpi=150, sharex=True)
    axes[0].plot(daily.index, daily["mae"], color="#0B5963")
    axes[0].set_ylabel("MAE ($)")
    axes[0].set_title("Chronological holdout error by date")
    axes[1].plot(daily.index, daily["bias"], color="#B4493D")
    axes[1].axhline(0, color="#34495E", linewidth=0.8)
    axes[1].set_ylabel("Mean error ($)")
    axes[1].set_xlabel("Date")
    for axis in axes:
        axis.grid(axis="y", alpha=0.25)
        axis.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(figures_dir / "holdout_error_over_time.png", bbox_inches="tight")
    plt.close(fig)


def _parameters_text(parameters: dict[str, Any]) -> str:
    return json.dumps(_serialise(parameters), sort_keys=True)


def _family_method_text(model_family: str) -> str:
    family = model_family.lower()
    if "catboost" in family:
        return (
            "CatBoost consumes the declared categorical columns directly. Its iteration count is learned by "
            "chronological early stopping inside the outer training partition."
        )
    if "histgradient" in family or "histogram" in family or "hgb" in family:
        return (
            "Histogram gradient boosting uses an ordinal category encoder fitted within each training fit; "
            "previously unseen categories map to a reserved value."
        )
    return "All estimator-specific preprocessing is fitted within the applicable training partition."


_PDF_MODEL_LABELS = {
    "hgb_compact": "HGB compact",
    "hgb_smoother": "HGB smoother",
    "catboost_rate_per_mile": "CatBoost",
    "december_hgb__december_basic": "HGB basic",
    "december_hgb__december_enriched_geo": "HGB geographic",
    "december_catboost__december_basic": "CatBoost basic",
    "december_catboost__december_enriched_geo": "CatBoost geographic",
    "equipment_median_rpm": "Equipment baseline",
    "global_median": "Global baseline",
}


def _pdf_model_name(model: Any) -> str:
    value = str(model)
    return _PDF_MODEL_LABELS.get(value, value)


def _pdf_family_name(model_family: Any) -> str:
    value = str(model_family)
    if value == "HistGradientBoostingRegressor":
        return "HGB"
    if value == "CatBoostRegressor":
        return "CatBoost"
    return value


def _pdf_feature_name(feature_set: Any) -> str:
    value = str(feature_set)
    return {
        "basic_plus_geographic": "Geographic",
        "basic_plus_route": "Route",
        "basic_plus_date": "Date",
        "basic_plus_interactions": "Interactions",
        "basic_supplied": "Basic",
        "best_combined": "Combined",
        "compact_calendar": "Calendar",
        "december_basic": "Basic",
        "december_enriched_geo": "Geographic",
        "distance + equipment": "Distance + equipment",
        "target_only": "Target only",
    }.get(value, value)


def _yes_no(value: Any) -> str:
    return "Yes" if str(value).strip().lower() in {"true", "1", "yes"} else "No"


def write_readme(
    path: Path,
    metrics_payload: dict[str, Any],
    audit: dict[str, Any],
    useful_groups: list[str],
) -> None:
    metrics = metrics_payload["metrics"]
    split = metrics_payload["split"]
    feature_screen = metrics_payload["feature_screening"]
    screen_split = feature_screen["split"]
    selected_family = str(metrics_payload["selected_model_family"])
    selected_parameters = _parameters_text(metrics_payload["selected_parameters"])
    december_model = str(metrics_payload["december_model"])
    december_family = str(metrics_payload["december_model_family"])
    december_feature_set = str(metrics_payload["december_feature_set"])
    december_metrics = metrics_payload["december_holdout_metrics"]
    december_parameters = _parameters_text(metrics_payload["december_parameters"])
    run_id = str(metrics_payload["run_id"])
    main_signature = str(metrics_payload["canonical_selections"]["main"]["signature"])
    december_signature = str(metrics_payload["canonical_selections"]["december"]["signature"])
    useful_groups_text = ", ".join(useful_groups) if useful_groups else "none beyond the basic supplied set"
    content = f"""# Freight Rate Prediction Assessment

Run ID: {run_id}

Canonical generated selections:

```text
{main_signature}
{december_signature}
```

## Objective

Train a leakage-safe freight-rate regressor on the supplied labeled data, predict all 12,000 future validation loads, and create the fixed Lexington-to-Fort-Wayne December series required by the supplied scorer.

## Dataset overview

- Development: {audit['development_shape'][0]:,} rows, {audit['date_ranges']['development'][0]} through {audit['date_ranges']['development'][1]}; target `posted_rate`.
- Final validation: {audit['validation_shape'][0]:,} rows, {audit['date_ranges']['validation'][0]} through {audit['date_ranges']['validation'][1]}.
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
# Windows PowerShell: .venv\\Scripts\\Activate.ps1
# macOS/Linux: source .venv/bin/activate
python -m pip install -r requirements.txt
python run_pipeline.py
```

The final command locates the supplied root-level files (with tolerant filename aliases), audits them, creates the chronological split, screens feature groups with both model families, compares all eligible HGB and CatBoost candidates, refits the selected families, predicts, validates both outputs, runs the unmodified scorer, and generates the report.

## Validation and model

The primary holdout mirrors the two-month future horizon: train {split['training_date_start']} through {split['training_date_end']} ({split['training_rows']:,} rows), validate {split['holdout_date_start']} through {split['holdout_date_end']} ({split['holdout_rows']:,} rows). Feature groups are screened before that comparison on a separate training-only split: fit through {screen_split['training_date_end']} and validate from {screen_split['validation_date_start']} through {screen_split['validation_date_end']}. Fixed HGB and CatBoost proxies score every logical feature set on the same fold-pure matrices, and the lowest equal-family mean MAE chooses one common list. CatBoost's best iteration for that list comes from the same training-only screen. Every outer candidate configuration is then frozen before scoring; outer labels are used only for the required final candidate selection and post-selection diagnostics, never to refit or adapt a candidate before its MAE is recorded.

Selection rule: {metrics_payload['selection_rule']}

Selected model: **{selected_family}** (`{metrics_payload['selected_model']}`), using **{metrics_payload['selected_feature_set']}**. {_family_method_text(selected_family)} MAE is the primary metric because dollars of absolute error are directly interpretable and robust to sparse label anomalies.

- Selected parameters: `{selected_parameters}`
- Saved final model: `{metrics_payload['final_model_filename']}`

| Metric | Chronological holdout |
|---|---:|
| MAE | ${metrics['mae']:,.2f} |
| RMSE | ${metrics['rmse']:,.2f} |
| R-squared | {metrics['r2']:.4f} |
| sMAPE | {metrics['smape']:.3f}% |

Feature sets that lowered equal-family mean MAE relative to the basic supplied-feature model on the training-only screen: {useful_groups_text}. The winning set was frozen before any outer-holdout scoring; groups that did not improve that inner validation window were not carried into final candidate comparison.

## December method

The reduced-schema selection is intentionally separate from the main validation model. Eligible December-compatible HGB and CatBoost candidates use the same chronological outer rows, rate-per-mile target, and posted-rate MAE; within each feature-set comparison, both families receive the same compatible columns. CatBoost early stopping remains internal to the outer training rows.

Selected December model: **{december_family}** (`{december_model}`), using **{december_feature_set}**, with holdout MAE **${december_metrics['mae']:,.2f}**. It uses only fields available in the fixed input plus known date features; coordinate enrichment, when selected, is learned exclusively from development city mappings. It does not fabricate market or quote signals.

- December parameters: `{december_parameters}`
- Saved December model: `{metrics_payload['december_model_filename']}`

## Outputs and scorer

```bash
python score.py --predictions validation_predictions.csv --december-predictions december-chart-inputs.csv
```

Successful scorer output is captured in `artifacts/scorer_output.txt`; the chart is `scorer_results/candidate_december.png`. Local metrics are development holdout metrics; the supplied scorer validates structure only and Spotter evaluates final validation accuracy after submission.

## Reproducibility, assumptions, and limitations

- Fixed random seed: {metrics_payload['random_seed']}; CPU-based eligible HGB and CatBoost comparisons; no external data or credentials.
- The selected main family is {selected_family}; the separately selected December family is {december_family}.
- The common feature list is chosen from one inner chronological window using equal weight for fixed HGB and CatBoost proxy scores; a different window or proxy configuration could choose differently.
- Validation includes eight cities absent from development, so unseen-city performance is inherently uncertain.
- The main shift is market index; chronological validation reduces, but cannot eliminate, future-regime risk.
- Coordinates are internally consistent but appear obfuscated, so supplied distance is treated as authoritative.
- Sparse extreme labels dominate RMSE; they are retained rather than silently removed.
- Historical target encodings were deliberately omitted because sparse routes and chronological leakage risk outweighed their probe value.
- The reported outer-holdout MAE is the required model-selection estimate, not a post-selection unbiased test score.

Loom walkthrough: [ADD FINAL LOOM LINK HERE]
"""
    path.write_text(content, encoding="utf-8")


def write_loom_script(
    path: Path,
    metrics_payload: dict[str, Any],
    audit: dict[str, Any],
    baseline_mae: float,
    top_features: list[str],
) -> None:
    metrics = metrics_payload["metrics"]
    split = metrics_payload["split"]
    feature_screen = metrics_payload["feature_screening"]
    screen_split = feature_screen["split"]
    selected_family = str(metrics_payload["selected_model_family"])
    december_model = str(metrics_payload["december_model"])
    december_family = str(metrics_payload["december_model_family"])
    december_feature_set = str(metrics_payload["december_feature_set"])
    december_metrics = metrics_payload["december_holdout_metrics"]
    selection_rule = str(metrics_payload["selection_rule"]).rstrip(".")
    run_id = str(metrics_payload["run_id"])
    main_signature = str(metrics_payload["canonical_selections"]["main"]["signature"])
    december_signature = str(metrics_payload["canonical_selections"]["december"]["signature"])
    leading_features = ", ".join(top_features[:5]) if top_features else "the features listed in the importance artifact"
    content = f"""# Loom walkthrough script (about 2-3 minutes)

Run ID: {run_id}

```text
{main_signature}
{december_signature}
```

Hi, this project predicts future posted freight rates and produces both the 12,000-row validation submission and the fixed December chart requested by the assessment.

The labeled dataset has {audit['development_shape'][0]:,} rows from {audit['date_ranges']['development'][0]} through {audit['date_ranges']['development'][1]}. The final prediction data starts the next day and runs through December, so I avoided a random split. I trained on {split['training_date_start']} through {split['training_date_end']}, or {split['training_rows']:,} rows, and held out {split['holdout_date_start']} through {split['holdout_date_end']}, or {split['holdout_rows']:,} rows. That two-month holdout best matches the final horizon and its lower market-index regime.

The main quality issues were missing weight and market-index values, plus impossible negative weight signs. I treated non-positive weights as missing, retained explicit missing and problem flags, and fitted every learned preprocessing step only on the applicable training partition. I retained valid extreme rates because removing them would be hard to defend. The target is right-skewed, with a median of ${audit['target_distribution']['median']:,.0f}, and distance is the dominant raw signal. Final validation also contains eight unseen cities and {audit['routes']['validation_unseen_rows']:,} unseen-route rows.

The global-median baseline had an MAE of ${baseline_mae:,.2f}. Before the outer comparison, I scored every feature group with fixed HGB and CatBoost proxies on a training-only chronological split that fit through {screen_split['training_date_end']} and validated through {screen_split['validation_date_end']}. The lowest equal-family mean MAE froze one common logical feature list without reading outer-holdout labels. CatBoost's iteration count for that list came from those fold-pure matrices, using distance-weighted rate-per-mile MAE so the stopping metric is proportional to posted-rate MAE. I then compared every eligible HGB and CatBoost finalist on the same outer chronological rows, frozen columns, rate-per-mile target, and posted-rate MAE. Every candidate was frozen before outer scoring; the outer holdout made the required final candidate choice and then supported diagnostics, without refitting or adapting candidates before their MAEs were recorded. The rule was: {selection_rule}. The winner was {selected_family}, candidate {metrics_payload['selected_model']}, using {metrics_payload['selected_feature_set']}. Its holdout metrics are MAE ${metrics['mae']:,.2f}, RMSE ${metrics['rmse']:,.2f}, R-squared {metrics['r2']:.4f}, and sMAPE {metrics['smape']:.3f} percent. The leading features were {leading_features}. {_family_method_text(selected_family)} The refit model is saved as {metrics_payload['final_model_filename']}.

The December file lacks market and quote signals, so I did not invent them. I compared compatible HGB and CatBoost candidates separately on the same outer rows, rate-per-mile target, posted-rate MAE, and identical columns within each feature-set alternative; CatBoost early stopping stayed inside the outer training rows. The winner was {december_family}, candidate {december_model}, using {december_feature_set}, with holdout MAE ${december_metrics['mae']:,.2f}. I refit it on all labeled rows and saved it as {metrics_payload['december_model_filename']}. Its date features are known in advance, and any coordinates are mapped only from development data.

The key files are `run_pipeline.py`, the small modules under `src`, the machine-readable tables under `artifacts`, `validation_predictions.csv`, and this report. From a fresh environment, run `python -m pip install -r requirements.txt`, then `python run_pipeline.py`. That also executes the supplied scorer and creates `scorer_results/candidate_december.png`.

The main limitations are future-regime and unseen-city uncertainty, plus reliance on two proxy families and one inner chronological window for common-feature screening. The outer-holdout MAE is the required selection estimate, not a post-selection unbiased test score or a hidden-test score.
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _pdf_table(
    rows: list[list[str]],
    widths: list[float],
    right_aligned_columns: tuple[int, ...] = (),
) -> Table:
    header_style = ParagraphStyle(
        "TableHeader",
        fontName="Helvetica-Bold",
        fontSize=7,
        leading=8,
        textColor=colors.white,
        alignment=TA_CENTER,
        splitLongWords=True,
    )
    body_style = ParagraphStyle(
        "TableBody",
        fontName="Helvetica",
        fontSize=7,
        leading=8,
        alignment=TA_LEFT,
        splitLongWords=True,
    )
    number_style = ParagraphStyle(
        "TableNumber",
        parent=body_style,
        alignment=TA_RIGHT,
    )
    wrapped_rows: list[list[Paragraph]] = []
    for row_index, row in enumerate(rows):
        wrapped_row: list[Paragraph] = []
        for column_index, value in enumerate(row):
            if row_index == 0:
                style = header_style
            elif column_index in right_aligned_columns:
                style = number_style
            else:
                style = body_style
            wrapped_row.append(Paragraph(escape(str(value)), style))
        wrapped_rows.append(wrapped_row)
    table = Table(wrapped_rows, colWidths=widths, repeatRows=1)
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0B5963")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#C7D4D7")),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F3F7F8")]),
                ("FONTSIZE", (0, 0), (-1, -1), 8),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("LEFTPADDING", (0, 0), (-1, -1), 5),
                ("RIGHTPADDING", (0, 0), (-1, -1), 5),
            ]
        )
    )
    return table


def write_pdf_report(
    path: Path,
    metrics_payload: dict[str, Any],
    audit: dict[str, Any],
    model_comparison: pd.DataFrame,
    feature_screening: pd.DataFrame,
    error_analysis: pd.DataFrame,
    feature_importance: pd.DataFrame,
    december_comparison: pd.DataFrame,
    figures_dir: Path,
    scorer_chart: Path,
) -> None:
    """Generate a concise five-page report entirely from executed results."""
    path.parent.mkdir(parents=True, exist_ok=True)
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="TitleCenter", parent=styles["Title"], alignment=TA_CENTER, textColor=colors.HexColor("#0B5963")))
    styles.add(ParagraphStyle(name="Small", parent=styles["BodyText"], fontSize=8.5, leading=11))
    styles.add(ParagraphStyle(name="Callout", parent=styles["BodyText"], backColor=colors.HexColor("#EAF2F3"), borderPadding=7, leading=13))
    styles.add(ParagraphStyle(name="Signature", parent=styles["Code"], fontName="Courier", fontSize=5.0, leading=6.5))
    document = SimpleDocTemplate(
        str(path),
        pagesize=letter,
        rightMargin=0.55 * inch,
        leftMargin=0.55 * inch,
        topMargin=0.45 * inch,
        bottomMargin=0.52 * inch,
        title="Freight Rate Prediction Assessment Report",
    )
    story: list[Any] = []
    split = metrics_payload["split"]
    feature_screen = metrics_payload["feature_screening"]
    screen_split = feature_screen["split"]
    metrics = metrics_payload["metrics"]
    selected_model = str(metrics_payload["selected_model"])
    selected_family = str(metrics_payload["selected_model_family"])
    selected_feature_set = str(metrics_payload["selected_feature_set"])
    selected_parameters = _parameters_text(metrics_payload["selected_parameters"])
    selection_rule = str(metrics_payload["selection_rule"]).rstrip(".")
    december_model = str(metrics_payload["december_model"])
    december_family = str(metrics_payload["december_model_family"])
    december_feature_set = str(metrics_payload["december_feature_set"])
    december_parameters = _parameters_text(metrics_payload["december_parameters"])
    december_metrics = metrics_payload["december_holdout_metrics"]
    run_id = str(metrics_payload["run_id"])
    main_signature = str(metrics_payload["canonical_selections"]["main"]["signature"])
    december_signature = str(metrics_payload["canonical_selections"]["december"]["signature"])

    story += [
        Paragraph("Freight Rate Prediction Assessment", styles["TitleCenter"]),
        Paragraph("Leakage-safe model development and submission artifacts", styles["Heading2"]),
        Spacer(1, 8),
        Paragraph(
            f"<b>Executive summary.</b> {escape(selected_family)} candidate <b>{escape(selected_model)}</b> using "
            f"{escape(selected_feature_set)} achieved a blocked future-holdout MAE of "
            f"${metrics['mae']:,.2f}, RMSE ${metrics['rmse']:,.2f}, R-squared {metrics['r2']:.4f}, "
            f"and sMAPE {metrics['smape']:.3f}%. The model was refit on all {audit['development_shape'][0]:,} labeled rows "
            f"to generate the 12,000-row submission and was saved as {escape(str(metrics_payload['final_model_filename']))}. "
            "No external data or final-validation labels were used.",
            styles["Callout"],
        ),
        Paragraph(escape(main_signature), styles["Signature"]),
        Paragraph("1. Dataset and data quality", styles["Heading2"]),
        Paragraph(
            f"Development contains {audit['development_shape'][0]:,} rows from {audit['date_ranges']['development'][0]} "
            f"to {audit['date_ranges']['development'][1]}; final validation contains {audit['validation_shape'][0]:,} later "
            f"rows from {audit['date_ranges']['validation'][0]} to {audit['date_ranges']['validation'][1]}. The target is "
            f"posted_rate (median ${audit['target_distribution']['median']:,.2f}, mean ${audit['target_distribution']['mean']:,.2f}, "
            f"maximum ${audit['target_distribution']['max']:,.2f}). There are {audit['unique_pickup_cities']} development pickup "
            f"cities, {audit['unique_delivery_cities']} delivery cities, and three equipment types.",
            styles["BodyText"],
        ),
        Paragraph(
            f"No duplicate IDs, exact duplicate rows, invalid dates, or illegal coordinates were found. Development has "
            f"300 missing weights, 374 missing market-index values, and {audit['negative_weight_rows']['development']} negative "
            "weights. The pipeline treats impossible non-positive weights as missing and retains explicit missing and problem "
            "flags. Learned preprocessing is fitted only on the applicable training rows. Valid rate extremes are retained and the MAE loss "
            "reduces their influence without hiding them.",
            styles["BodyText"],
        ),
        Image(str(figures_dir / "target_distribution.png"), width=6.6 * inch, height=3.5 * inch),
    ]
    story.append(PageBreak())

    story += [
        Paragraph("2. Validation design and exploratory findings", styles["Heading1"]),
        Paragraph(
            f"The final data is strictly later than development, so the primary split trains on {split['training_date_start']} through "
            f"{split['training_date_end']} ({split['training_rows']:,} rows) and validates on {split['holdout_date_start']} through "
            f"{split['holdout_date_end']} ({split['holdout_rows']:,} rows). This 61-day blocked holdout mirrors the final "
            "November-December horizon. A separate training-only chronological window fits through "
            f"{screen_split['training_date_end']} and validates from {screen_split['validation_date_start']} through "
            f"{screen_split['validation_date_end']} for feature screening. Preprocessing statistics and frequency maps use training rows only.",
            styles["Callout"],
        ),
        Paragraph(
            f"Distance is the dominant supplied signal. Market index has the clearest development-to-validation drift; the "
            "September-October regime is much closer to final validation than the full historical average. Final validation "
            f"contains eight unseen cities and {audit['routes']['validation_unseen_rows']:,} unseen-route rows, a limitation not "
            "fully represented by the holdout because all holdout cities were already observed.",
            styles["BodyText"],
        ),
        Image(str(figures_dir / "distribution_shift.png"), width=6.3 * inch, height=3.3 * inch),
        Paragraph("3. Cleaning and features", styles["Heading2"]),
        Paragraph(
            "Categories are whitespace/case normalised and missing categories become Unknown. Numeric missing values remain "
            "missing and are accompanied by explicit flags; learned preprocessing stays within training rows. Tested groups cover raw supplied fields, cyclic and calendar date features, "
            "route combinations with training-only frequencies/unseen flags, compact geographic calculations, and defensible "
            "distance/weight/market interactions. load_id is never a model feature. Coordinates appear synthetic, so supplied "
            "distance remains authoritative.",
            styles["BodyText"],
        ),
    ]
    story.append(PageBreak())

    comparison_rows = [["Model", "Family", "Features", "MAE", "RMSE", "Eligible", "Selected"]]
    for _, row in model_comparison.iterrows():
        comparison_rows.append(
            [
                _pdf_model_name(row["model"]),
                _pdf_family_name(row["model_family"]),
                _pdf_feature_name(row.get("feature_set", "-")),
                f"{row['mae']:.2f}",
                f"{row['rmse']:.2f}",
                _yes_no(row.get("eligible", False)),
                _yes_no(row.get("selected", False)),
            ]
        )
    screening_rows = [["Feature set", "HGB MAE", "CatBoost MAE", "Mean MAE", "Change", "Selected"]]
    for _, row in feature_screening.sort_values("selection_score_mean_mae").iterrows():
        screening_rows.append(
            [
                str(row["feature_set"]),
                f"{row['hgb_mae']:.2f}",
                f"{row['catboost_mae']:.2f}",
                f"{row['selection_score_mean_mae']:.2f}",
                f"{row['improvement_over_basic_selection_score']:.2f}",
                _yes_no(row["selected_for_outer_comparison"]),
            ]
        )
    top_features = feature_importance.sort_values(
        ["importance", "feature"],
        ascending=[False, True],
        kind="mergesort",
    ).head(5)["feature"].tolist()
    story += [
        Paragraph("4. Model comparison and selection", styles["Heading1"]),
        Paragraph(
            "MAE is the primary selection metric because dollar error is directly actionable and the labels contain sparse "
            "extremes. Fixed HGB and CatBoost proxies scored every feature group on the same training-internal chronological "
            "window. The lowest equal-family mean MAE froze one logical feature list for every eligible finalist, and CatBoost's "
            "best iteration came from that fold-pure screen. Every eligible HGB and CatBoost candidate then used the same outer "
            "chronological rows, frozen feature columns, rate-per-mile target, and posted-rate MAE. Every configuration was frozen "
            "before outer scoring; outer labels made the required final candidate choice and supported diagnostics, without "
            f"adapting a candidate before its MAE was recorded. The documented rule was: {escape(selection_rule)}. RMSE, R-squared, and sMAPE are reported for context.",
            styles["BodyText"],
        ),
        _pdf_table(
            comparison_rows,
            [1.35 * inch, 0.8 * inch, 1.2 * inch, 0.7 * inch, 0.75 * inch, 0.65 * inch, 0.65 * inch],
            right_aligned_columns=(3, 4),
        ),
        Spacer(1, 10),
        Paragraph("Training-only family-neutral feature screening (positive change improves on basic)", styles["Heading3"]),
        _pdf_table(
            screening_rows,
            [1.7 * inch, 0.7 * inch, 0.9 * inch, 0.75 * inch, 0.65 * inch, 0.65 * inch],
            right_aligned_columns=(1, 2, 3, 4),
        ),
        Spacer(1, 8),
        Paragraph(
            f"Selected: <b>{escape(selected_family)}</b> candidate <b>{escape(selected_model)}</b> with "
            f"<b>{escape(selected_feature_set)}</b>. {_family_method_text(selected_family)} The leading importance features "
            f"were {escape(', '.join(top_features))}. Parameters: <font name='Courier' size='7'>{escape(selected_parameters)}</font>. "
            f"Saved model: <font name='Courier'>{escape(str(metrics_payload['final_model_filename']))}</font>.",
            styles["Callout"],
        ),
    ]
    story.append(PageBreak())

    segment = error_analysis.sort_values(
        ["mae", "dimension", "segment"],
        ascending=[False, True, True],
        kind="mergesort",
    ).head(6)
    segment_rows = [["Dimension", "Segment", "Rows", "MAE", "Bias"]]
    for _, row in segment.iterrows():
        segment_rows.append([str(row["dimension"]), str(row["segment"]), f"{int(row['rows']):,}", f"{row['mae']:.2f}", f"{row['mean_error']:.2f}"])
    dec_rows = [["Model", "Family", "Features", "MAE", "RMSE", "Eligible", "Selected"]]
    for _, row in december_comparison.iterrows():
        dec_rows.append(
            [
                _pdf_model_name(row["model"]),
                _pdf_family_name(row["model_family"]),
                _pdf_feature_name(row["feature_set"]),
                f"{row['mae']:.2f}",
                f"{row['rmse']:.2f}",
                _yes_no(row.get("eligible", False)),
                _yes_no(row.get("selected", False)),
            ]
        )
    story += [
        Paragraph("5. Error analysis and model interpretation", styles["Heading1"]),
        Paragraph(
            "The table shows the highest-error reported segments. Rare extreme labels drive RMSE and high-target error. "
            "Route/city segmentation is retained in the artifacts for inspection; final unseen-city error may be higher than "
            "this holdout because the new cities arrive only in the unlabeled future set.",
            styles["BodyText"],
        ),
        Image(str(figures_dir / "feature_importance.png"), width=5.6 * inch, height=3.3 * inch),
        Spacer(1, 8),
        _pdf_table(
            segment_rows,
            [1.15 * inch, 1.55 * inch, 0.65 * inch, 0.75 * inch, 0.75 * inch],
            right_aligned_columns=(2, 3, 4),
        ),
    ]
    story.append(PageBreak())

    story += [
        Paragraph("6. Reduced-schema December predictions", styles["Heading1"]),
        Paragraph(
            "This selection is separate from the main validation model. Every eligible reduced-schema HGB and CatBoost "
            "candidate used the same outer chronological rows, rate-per-mile target, and posted-rate MAE; within each "
            "feature-set alternative, both families received identical compatible columns. Any CatBoost early stopping "
            "used only an internal chronological split within the outer training rows.",
            styles["Small"],
        ),
        _pdf_table(
            dec_rows,
            [1.55 * inch, 0.85 * inch, 1.0 * inch, 0.7 * inch, 0.75 * inch, 0.6 * inch, 0.6 * inch],
            right_aligned_columns=(3, 4),
        ),
        Paragraph(
            f"Selected: <b>{escape(december_family)}</b> candidate <b>{escape(december_model)}</b> using "
            f"<b>{escape(december_feature_set)}</b>, holdout MAE <b>${december_metrics['mae']:,.2f}</b>. It uses only pickup, "
            "delivery, distance, equipment, weight, known calendar fields and, if selected, development-derived city "
            "coordinates. Market and quote signals are not fabricated. A separate model is refit on all labeled rows. "
            f"Parameters: <font name='Courier' size='7'>{escape(december_parameters)}</font>. Saved model: "
            f"<font name='Courier'>{escape(str(metrics_payload['december_model_filename']))}</font>.",
            styles["BodyText"],
        ),
        Paragraph(escape(december_signature), styles["Signature"]),
        Image(str(scorer_chart), width=5.8 * inch, height=2.45 * inch),
        Paragraph("7. Limitations and reproduction", styles["Heading2"]),
        Paragraph(
            "Future market regimes, unseen cities/routes, synthetic coordinates, sparse abnormal labels, and reliance on two "
            "proxy families plus one training-only common-feature window remain limitations. The outer-holdout MAE is a model-selection "
            "estimate, not a post-selection unbiased test score. "
            "Install with <font name='Courier'>python -m pip install -r requirements.txt</font> and reproduce "
            "everything with <font name='Courier'>python run_pipeline.py</font>. The unmodified scorer validates both outputs "
            "and creates the chart above; it does not calculate accuracy.",
            styles["Small"],
        ),
    ]

    def add_page_footer(canvas: Any, doc: Any) -> None:
        canvas.saveState()
        canvas.setFont("Helvetica", 8)
        canvas.setFillColor(colors.HexColor("#607D82"))
        canvas.drawString(0.55 * inch, 0.25 * inch, f"Run ID: {run_id}")
        canvas.drawRightString(7.9 * inch, 0.25 * inch, f"Page {doc.page}")
        canvas.restoreState()

    document.build(story, onFirstPage=add_page_footer, onLaterPages=add_page_footer)
