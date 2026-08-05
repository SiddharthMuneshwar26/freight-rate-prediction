"""Reproduce the freight-rate assessment from raw inputs through scoring."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import platform
import subprocess
import sys
import warnings
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path
from typing import Any

# HGB uses OpenMP internally. Constraining every numerical backend before
# importing NumPy/sklearn prevents thread-reduction order from changing splits.
for thread_variable in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "BLIS_NUM_THREADS",
    "LOKY_MAX_CPU_COUNT",
):
    os.environ[thread_variable] = "1"
os.environ["PYTHONHASHSEED"] = "42"
warnings.filterwarnings("ignore", message=r"Could not find the number of physical cores.*")

import numpy as np
import pandas as pd
from joblib import load, dump

from src.evaluate import (
    build_error_analysis,
    permutation_importance,
    regression_metrics,
    require_nonempty_file,
    validate_generated_metrics_consistency,
    validate_december_output,
    validate_model_selection,
    validate_prediction_output,
)
from src.features import COMMON_FEATURE_SETS, DECEMBER_FEATURE_SETS, FeatureBuilder
from src.reporting import (
    create_eda_artifacts,
    create_model_figures,
    write_json,
)
from src.train import (
    HGB_SCREENING_PARAMETERS,
    HGB_FINAL_CANDIDATES,
    MODEL_FAMILY_CATBOOST,
    MODEL_FAMILY_HGB,
    RANDOM_SEED,
    ModelCandidate,
    equipment_median_predictions,
    fit_catboost_candidate,
    fit_hgb_candidate,
    fit_rate_model,
    global_median_predictions,
    learn_catboost_iterations,
    model_family_for_model,
    save_rate_model,
)


ROOT = Path(__file__).resolve().parent
ARTIFACTS = ROOT / "artifacts"
FIGURES = ROOT / "reports" / "figures"
MODELS = ROOT / "models"
REPORTS = ROOT / "reports"
SCORER_RESULTS = ROOT / "scorer_results"
LOGGER = logging.getLogger("freight_pipeline")
SUPPLIED_SCORER_SHA256 = "f73af4f34acf0c43cdd6e2cd6b2c0ec8b9da4420fd6fdcd69bd86e14d4868634"
SUPPLIED_DECEMBER_SHA256 = "d4a0cba969e315eae3c967268914960547dc6589568e07bf6182551bdc267e6e"
CATBOOST_CANDIDATE_PARAMETERS: dict[str, Any] = {
    "iterations": 600,
    "depth": 6,
    "learning_rate": 0.06,
    "l2_leaf_reg": 6.0,
}
EXPECTED_MAIN_CANDIDATES = {
    "hgb_compact": MODEL_FAMILY_HGB,
    "hgb_smoother": MODEL_FAMILY_HGB,
    "catboost_rate_per_mile": MODEL_FAMILY_CATBOOST,
}
EXPECTED_DECEMBER_CANDIDATES = {
    "december_hgb__december_basic": MODEL_FAMILY_HGB,
    "december_hgb__december_enriched_geo": MODEL_FAMILY_HGB,
    "december_catboost__december_basic": MODEL_FAMILY_CATBOOST,
    "december_catboost__december_enriched_geo": MODEL_FAMILY_CATBOOST,
}
EXPECTED_BASELINES = {"equipment_median_rpm", "global_median"}


@dataclass(frozen=True)
class InputPaths:
    development: Path
    validation: Path
    template: Path
    december: Path
    december_template: Path | None
    scorer: Path
    assessment_pdf: Path | None
    supplied_readme: Path | None


def _setup_logging() -> None:
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    handler_file = logging.FileHandler(ARTIFACTS / "pipeline.log", mode="w", encoding="utf-8")
    handler_console = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", datefmt="%H:%M:%S")
    handler_file.setFormatter(formatter)
    handler_console.setFormatter(formatter)
    LOGGER.handlers.clear()
    LOGGER.addHandler(handler_file)
    LOGGER.addHandler(handler_console)
    LOGGER.setLevel(logging.INFO)


def _find_file(aliases: list[str], required: bool = True) -> Path | None:
    for alias in aliases:
        candidate = ROOT / alias
        if candidate.is_file():
            return candidate
    fallback_files = sorted(
        (path for path in ROOT.rglob("*") if path.is_file()),
        key=lambda path: str(path.relative_to(ROOT)).casefold(),
    )
    by_name = {path.name.lower(): path for path in fallback_files}
    for alias in aliases:
        candidate = by_name.get(Path(alias).name.lower())
        if candidate is not None:
            return candidate
    if required:
        raise FileNotFoundError(f"Could not locate required input; tried: {aliases}")
    return None


def locate_inputs() -> InputPaths:
    paths = InputPaths(
        development=_find_file(["train-test.csv", "train_test.csv", "data/train_test.csv"]),  # type: ignore[arg-type]
        validation=_find_file(["validation.csv", "data/validation.csv"]),  # type: ignore[arg-type]
        template=_find_file(
            [
                "validation-predictions-template.csv",
                "validation_predictions_template.csv",
                "data/validation_predictions_template.csv",
            ]
        ),  # type: ignore[arg-type]
        december=_find_file(
            ["december-chart-inputs.csv", "december_chart_inputs.csv", "data/december_chart_inputs.csv"]
        ),  # type: ignore[arg-type]
        december_template=_find_file(
            ["december-chart-inputs-template.csv"], required=False
        ),
        scorer=_find_file(["score.py"]),  # type: ignore[arg-type]
        assessment_pdf=_find_file(
            ["freight-rate-ml-assessment.pdf", "Freight_Rate_ML_Assessment.pdf"], required=False
        ),
        supplied_readme=_find_file(["readme spotter.md"], required=False),
    )
    LOGGER.info("Located development data: %s", paths.development.relative_to(ROOT))
    LOGGER.info("Located validation data: %s", paths.validation.relative_to(ROOT))
    LOGGER.info("Located template: %s", paths.template.relative_to(ROOT))
    LOGGER.info("Located December input: %s", paths.december.relative_to(ROOT))
    if paths.december_template is not None:
        LOGGER.info("Located pristine December template: %s", paths.december_template.relative_to(ROOT))
    return paths


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_run_identity(inputs: InputPaths) -> tuple[str, str]:
    """Create a stable content-addressed identity for code and supplied inputs."""
    identity_paths = [
        ROOT / "run_pipeline.py",
        ROOT / "requirements.txt",
        ROOT / ".gitignore",
        inputs.development,
        inputs.validation,
        inputs.template,
        inputs.december_template or inputs.december,
        inputs.scorer,
    ]
    identity_paths.extend(sorted((ROOT / "src").glob("*.py"), key=lambda path: path.name))
    if inputs.assessment_pdf is not None:
        identity_paths.append(inputs.assessment_pdf)
    if inputs.supplied_readme is not None:
        identity_paths.append(inputs.supplied_readme)

    digest = hashlib.sha256()
    unique_paths = sorted(set(identity_paths), key=lambda path: str(path.relative_to(ROOT)).casefold())
    for path in unique_paths:
        relative = str(path.relative_to(ROOT)).replace("\\", "/")
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        digest.update(b"\0")
    fingerprint = digest.hexdigest()
    return f"submission-{fingerprint[:16]}", fingerprint


def read_and_validate_inputs(paths: InputPaths) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, str]:
    development = pd.read_csv(
        paths.development,
        dtype={"load_id": "string"},
    )
    validation = pd.read_csv(
        paths.validation,
        dtype={"load_id": "string"},
    )
    template = pd.read_csv(
        paths.template,
        dtype={"load_id": "string"},
    )
    december = pd.read_csv(paths.december_template or paths.december)
    target_candidates = [column for column in development.columns if column not in validation.columns]
    if len(target_candidates) != 1:
        raise ValueError(f"Expected exactly one development-only target column, found {target_candidates}")
    target = target_candidates[0]

    required_features = {
        "load_id",
        "pickup",
        "delivery",
        "pickup_lat",
        "pickup_lon",
        "delivery_lat",
        "delivery_lon",
        "distance",
        "equipment",
        "weight",
        "date",
        "market_index",
        "quote_signal",
    }
    missing_development = required_features - set(development.columns)
    missing_validation = required_features - set(validation.columns)
    if missing_development or missing_validation:
        raise ValueError(
            f"Required feature columns missing (development={sorted(missing_development)}, "
            f"validation={sorted(missing_validation)})"
        )
    if set(development.columns) - {target} != set(validation.columns):
        raise ValueError("Development and validation feature schemas do not match")
    if list(template.columns) != ["load_id", "predicted_rate"]:
        raise ValueError("Prediction template must contain load_id,predicted_rate in that order")
    if len(validation) != 12_000 or len(template) != 12_000:
        raise ValueError("Validation data and prediction template must each contain 12,000 rows")
    if set(template["load_id"].astype(str)) != set(validation["load_id"].astype(str)):
        raise ValueError("Template and validation IDs do not match")
    for label, frame in [("development", development), ("validation", validation)]:
        if frame["load_id"].isna().any() or frame["load_id"].duplicated().any():
            raise ValueError(f"{label} has missing or duplicate load IDs")
        parsed_dates = pd.to_datetime(frame["date"], errors="coerce")
        if parsed_dates.isna().any():
            raise ValueError(f"{label} contains invalid dates")
    target_values = pd.to_numeric(development[target], errors="coerce")
    if target_values.isna().any() or not np.isfinite(target_values).all() or target_values.le(0).any():
        raise ValueError(f"Inferred target {target!r} must be complete, finite, numeric, and positive")
    development[target] = target_values.astype(float)
    development = development.sort_values(
        ["date", "load_id"], kind="mergesort"
    ).reset_index(drop=True)
    validate_december_structure(december)
    LOGGER.info("Inferred regression target: %s", target)
    return development, validation, template, december, target


def validate_december_structure(frame: pd.DataFrame) -> None:
    expected_columns = [
        "pickup",
        "delivery",
        "distance",
        "equipment",
        "weight",
        "date",
        "predicted_rate",
    ]
    if list(frame.columns) != expected_columns:
        raise ValueError("December input columns or order are invalid")
    fixed = frame.copy()
    fixed["predicted_rate"] = 1.0
    validate_december_output(fixed)


def chronological_split(
    development: pd.DataFrame,
    validation: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    dev_dates = pd.to_datetime(development["date"])
    validation_dates = pd.to_datetime(validation["date"])
    if validation_dates.min() <= dev_dates.max():
        raise ValueError("Final validation is not strictly later than development; reassess split strategy")
    horizon_days = int((validation_dates.max() - validation_dates.min()).days) + 1
    cutoff = dev_dates.max() - pd.Timedelta(days=horizon_days - 1)
    training = development.loc[dev_dates < cutoff].copy().sort_values(
        ["date", "load_id"], kind="mergesort"
    ).reset_index(drop=True)
    holdout = development.loc[dev_dates >= cutoff].copy().sort_values(
        ["date", "load_id"], kind="mergesort"
    ).reset_index(drop=True)
    if len(training) < 1_000 or len(holdout) < 1_000:
        raise ValueError("Chronological split is too small for reliable modeling")

    train_routes = set(training["pickup"].astype(str) + " > " + training["delivery"].astype(str))
    holdout_routes = holdout["pickup"].astype(str) + " > " + holdout["delivery"].astype(str)
    train_cities = set(training["pickup"].astype(str)) | set(training["delivery"].astype(str))
    holdout_unseen_city = (~holdout["pickup"].astype(str).isin(train_cities)) | (
        ~holdout["delivery"].astype(str).isin(train_cities)
    )
    split = {
        "strategy": "blocked chronological holdout matching the 61-day final horizon",
        "training_date_start": str(pd.to_datetime(training["date"]).min().date()),
        "training_date_end": str(pd.to_datetime(training["date"]).max().date()),
        "holdout_date_start": str(pd.to_datetime(holdout["date"]).min().date()),
        "holdout_date_end": str(pd.to_datetime(holdout["date"]).max().date()),
        "training_rows": int(len(training)),
        "holdout_rows": int(len(holdout)),
        "holdout_unseen_route_rows": int((~holdout_routes.isin(train_routes)).sum()),
        "holdout_unseen_city_rows": int(holdout_unseen_city.sum()),
        "limitation": "The holdout has no new cities, while final validation contains new cities.",
    }
    return training, holdout, split


def chronological_inner_split(
    outer_training: pd.DataFrame,
    horizon_days: int = 61,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Create a training-only tail split for feature and iteration decisions."""
    dates = pd.to_datetime(outer_training["date"], errors="coerce")
    if dates.isna().any() or horizon_days < 1:
        raise ValueError("Training-only chronological split requires valid dates and a positive horizon")
    cutoff = dates.max() - pd.Timedelta(days=horizon_days - 1)
    inner_training = outer_training.loc[dates < cutoff].copy().sort_values(
        ["date", "load_id"], kind="mergesort"
    ).reset_index(drop=True)
    inner_validation = outer_training.loc[dates >= cutoff].copy().sort_values(
        ["date", "load_id"], kind="mergesort"
    ).reset_index(drop=True)
    if len(inner_training) < 1_000 or len(inner_validation) < 1_000:
        raise ValueError("Training-only chronological split is too small for model decisions")
    split = {
        "strategy": "training-only blocked chronological split",
        "training_date_start": str(pd.to_datetime(inner_training["date"]).min().date()),
        "training_date_end": str(pd.to_datetime(inner_training["date"]).max().date()),
        "validation_date_start": str(pd.to_datetime(inner_validation["date"]).min().date()),
        "validation_date_end": str(pd.to_datetime(inner_validation["date"]).max().date()),
        "training_rows": int(len(inner_training)),
        "validation_rows": int(len(inner_validation)),
        "horizon_days": horizon_days,
    }
    return inner_training, inner_validation, split


def _metric_row(model: str, feature_set: str, metrics: dict[str, float], **extra: Any) -> dict[str, Any]:
    return {"model": model, "feature_set": feature_set, **metrics, **extra}


def _candidate_metric_row(candidate: ModelCandidate, *, eligible: bool = True) -> dict[str, Any]:
    """Flatten the common candidate contract for the comparison artifact."""
    return _metric_row(
        candidate.name,
        candidate.feature_set,
        candidate.metrics,
        model_family=candidate.model_family,
        parameters=json.dumps(candidate.parameters, sort_keys=True),
        best_iteration=candidate.best_iteration,
        eligible=eligible,
        selected=False,
    )


def _selection_identity(candidate: ModelCandidate, label: str) -> dict[str, Any]:
    """Canonical values shared by predictions and machine-generated artifacts."""
    mae = float(candidate.metrics["mae"])
    identity = {
        "model": candidate.name,
        "family": candidate.model_family,
        "feature_set": candidate.feature_set,
        "mae": mae,
        "mae_text": repr(mae),
    }
    identity["signature"] = (
        f"{label.upper()}|model={identity['model']}|family={identity['family']}|"
        f"feature_set={identity['feature_set']}|mae={identity['mae_text']}"
    )
    return identity


def _select_lowest_mae(candidates: list[ModelCandidate]) -> ModelCandidate:
    """Select the exact lowest finite holdout MAE; break exact ties by name."""
    if not candidates:
        raise ValueError("At least one eligible model candidate is required")
    for candidate in candidates:
        mae = float(candidate.metrics.get("mae", np.nan))
        if not np.isfinite(mae):
            raise ValueError(f"Candidate {candidate.name!r} has an invalid holdout MAE")
    return min(candidates, key=lambda candidate: (float(candidate.metrics["mae"]), candidate.name))


def _validate_candidate_pool(
    candidates: list[ModelCandidate],
    expected: dict[str, str],
    label: str,
) -> None:
    """Require the documented candidate names and families exactly once."""
    names = [candidate.name for candidate in candidates]
    if len(names) != len(set(names)):
        raise RuntimeError(f"{label} candidate pool contains duplicate names")
    actual = {candidate.name: candidate.model_family for candidate in candidates}
    if actual != expected:
        raise RuntimeError(
            f"{label} candidate pool differs from the documented set: "
            f"actual={sorted(actual)}, expected={sorted(expected)}"
        )


def run_feature_screening(
    builder: FeatureBuilder,
    screening_training: pd.DataFrame,
    screening_validation: pd.DataFrame,
    target: str,
) -> tuple[pd.DataFrame, str, int]:
    """Choose one common feature list using equal-weight inner scores from both families."""
    rows: list[dict[str, Any]] = []
    for name in sorted(COMMON_FEATURE_SETS):
        columns = COMMON_FEATURE_SETS[name]
        LOGGER.info("Family-neutral feature screen: %s (%d features)", name, len(columns))
        training_features = builder.transform(screening_training, columns)
        validation_features = builder.transform(screening_validation, columns)
        hgb_result = fit_hgb_candidate(
            name=name,
            train_features=training_features,
            train_target=screening_training[target],
            holdout_features=validation_features,
            holdout_target=screening_validation[target],
            parameters=HGB_SCREENING_PARAMETERS,
            feature_set=name,
        )
        catboost_result = learn_catboost_iterations(
            training_features,
            screening_training[target],
            validation_features,
            screening_validation[target],
            CATBOOST_CANDIDATE_PARAMETERS,
        )
        selection_score = float(
            np.mean([hgb_result.metrics["mae"], catboost_result.metrics["mae"]])
        )
        rows.append(
            {
                "feature_set": name,
                "evaluation_scope": "training_only_inner_chronological",
                "feature_count": len(columns),
                "hgb_model_family": hgb_result.model_family,
                **{f"hgb_{metric}": value for metric, value in hgb_result.metrics.items()},
                "catboost_model_family": MODEL_FAMILY_CATBOOST,
                **{f"catboost_{metric}": value for metric, value in catboost_result.metrics.items()},
                "catboost_best_iteration": catboost_result.iterations,
                "selection_metric": "unweighted_mean_family_posted_rate_mae",
                "selection_score_mean_mae": selection_score,
            }
        )
        LOGGER.info(
            "  HGB MAE %.3f | CatBoost MAE %.3f | mean %.3f",
            hgb_result.metrics["mae"],
            catboost_result.metrics["mae"],
            selection_score,
        )
    results = pd.DataFrame(rows)
    if len(results) != len(COMMON_FEATURE_SETS) or results["feature_set"].duplicated().any():
        raise RuntimeError("Feature screening must emit exactly one row per common feature set")
    if set(results["hgb_model_family"]) != {MODEL_FAMILY_HGB} or set(
        results["catboost_model_family"]
    ) != {MODEL_FAMILY_CATBOOST}:
        raise RuntimeError("Both canonical model families must participate in feature screening")
    numeric_columns = [
        "hgb_mae",
        "catboost_mae",
        "selection_score_mean_mae",
        "catboost_best_iteration",
    ]
    if not np.isfinite(results[numeric_columns].to_numpy(dtype=float)).all():
        raise RuntimeError("Feature screening produced invalid metrics")
    expected_scores = results[["hgb_mae", "catboost_mae"]].mean(axis=1)
    if not np.allclose(
        results["selection_score_mean_mae"], expected_scores, rtol=0.0, atol=1e-12
    ):
        raise RuntimeError("Feature screening aggregation is inconsistent")
    basic_score = float(
        results.loc[
            results["feature_set"].eq("basic_supplied"), "selection_score_mean_mae"
        ].iloc[0]
    )
    results["improvement_over_basic_selection_score"] = (
        basic_score - results["selection_score_mean_mae"]
    )
    results = results.sort_values(
        ["selection_score_mean_mae", "feature_set"], kind="mergesort"
    ).reset_index(drop=True)
    selected_feature_set = str(results.iloc[0]["feature_set"])
    results["selected_for_outer_comparison"] = results["feature_set"].eq(
        selected_feature_set
    )
    if int(results["selected_for_outer_comparison"].sum()) != 1:
        raise RuntimeError("Feature screening must select exactly one common feature set")
    results.to_csv(ARTIFACTS / "feature_ablation.csv", index=False)
    selected_iterations = int(results.iloc[0]["catboost_best_iteration"])
    return results, selected_feature_set, selected_iterations


def run_temporal_stability(
    development: pd.DataFrame,
    target: str,
    selected_candidate: ModelCandidate,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    dates = pd.to_datetime(development["date"])
    for month_start in [pd.Timestamp("2025-08-01"), pd.Timestamp("2025-09-01"), pd.Timestamp("2025-10-01")]:
        month_end = month_start + pd.offsets.MonthEnd(0)
        train_fold = development.loc[dates < month_start].copy().sort_values(
            ["date", "load_id"], kind="mergesort"
        ).reset_index(drop=True)
        holdout_fold = development.loc[dates.between(month_start, month_end)].copy().sort_values(
            ["date", "load_id"], kind="mergesort"
        ).reset_index(drop=True)
        builder = FeatureBuilder().fit(train_fold)
        columns = selected_candidate.feature_columns
        train_features = builder.transform(train_fold, columns)
        holdout_features = builder.transform(holdout_fold, columns)
        model = fit_rate_model(
            selected_candidate.model_family,
            selected_candidate.parameters,
            train_features,
            train_fold[target],
        )
        predictions = model.predict(holdout_features)
        rows.append(
            {
                "holdout_month": month_start.strftime("%Y-%m"),
                "evaluation_role": "post_selection_temporal_diagnostic",
                "model": selected_candidate.name,
                "model_family": selected_candidate.model_family,
                "feature_set": selected_candidate.feature_set,
                "training_rows": len(train_fold),
                "holdout_rows": len(holdout_fold),
                **regression_metrics(holdout_fold[target], predictions),
            }
        )
    stability = pd.DataFrame(rows)
    stability.to_csv(ARTIFACTS / "temporal_stability.csv", index=False)
    return stability


def run_scorer(paths: InputPaths, predictions_path: Path) -> str:
    command = [
        sys.executable,
        str(paths.scorer.relative_to(ROOT)),
        "--predictions",
        str(predictions_path.relative_to(ROOT)),
        "--december-predictions",
        str(paths.december.relative_to(ROOT)),
        "--output-dir",
        str(SCORER_RESULTS.relative_to(ROOT)),
    ]
    completed = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, check=False)
    combined = completed.stdout + completed.stderr
    (ARTIFACTS / "scorer_output.txt").write_text(combined, encoding="utf-8")
    if completed.returncode != 0:
        raise RuntimeError(f"Supplied scorer failed with exit code {completed.returncode}:\n{combined}")
    LOGGER.info("Supplied scorer passed")
    for line in completed.stdout.strip().splitlines():
        LOGGER.info("scorer | %s", line)
    return combined


def main() -> None:
    _setup_logging()
    np.random.seed(RANDOM_SEED)
    for directory in [ARTIFACTS, FIGURES, MODELS, REPORTS, SCORER_RESULTS]:
        directory.mkdir(parents=True, exist_ok=True)
    static_readme = ROOT / "README.md"
    static_report = REPORTS / "assessment_report.pdf"
    require_nonempty_file(static_readme)
    require_nonempty_file(static_report)
    readme_hash_before = _sha256(static_readme)
    LOGGER.info("Static README SHA-256 before execution: %s", readme_hash_before)
    LOGGER.info("Starting end-to-end freight-rate pipeline")
    inputs = locate_inputs()
    december_source = inputs.december_template or inputs.december
    december_source_hash = _sha256(december_source)
    if december_source_hash != SUPPLIED_DECEMBER_SHA256:
        raise RuntimeError("The pristine December input does not match the supplied assessment file")
    run_id, source_fingerprint = build_run_identity(inputs)
    LOGGER.info("Run ID: %s", run_id)
    scorer_hash_before = _sha256(inputs.scorer)
    if scorer_hash_before != SUPPLIED_SCORER_SHA256:
        raise RuntimeError("score.py does not match the supplied assessment scorer")
    development, validation, template, december, target = read_and_validate_inputs(inputs)
    create_eda_artifacts(development, validation, december, ARTIFACTS, FIGURES)

    manifest = {
        "run_id": run_id,
        "source_fingerprint_sha256": source_fingerprint,
        "files": {
            "development": str(inputs.development.relative_to(ROOT)),
            "validation": str(inputs.validation.relative_to(ROOT)),
            "template": str(inputs.template.relative_to(ROOT)),
            "december": str(inputs.december.relative_to(ROOT)),
            "december_template": str(december_source.relative_to(ROOT)),
            "scorer": str(inputs.scorer.relative_to(ROOT)),
        },
        "sha256": {
            "development": _sha256(inputs.development),
            "validation": _sha256(inputs.validation),
            "template": _sha256(inputs.template),
            "december_template": december_source_hash,
            "scorer": scorer_hash_before,
        },
    }
    write_json(ARTIFACTS / "input_manifest.json", manifest)

    training, holdout, split = chronological_split(development, validation)
    LOGGER.info(
        "Chronological split: %s..%s (%d) -> %s..%s (%d)",
        split["training_date_start"],
        split["training_date_end"],
        split["training_rows"],
        split["holdout_date_start"],
        split["holdout_date_end"],
        split["holdout_rows"],
    )

    model_rows: list[dict[str, Any]] = []
    global_predictions = global_median_predictions(training[target], len(holdout))
    global_metrics = regression_metrics(holdout[target], global_predictions)
    model_rows.append(
        _metric_row(
            "global_median",
            "target_only",
            global_metrics,
            model_family="Baseline",
            parameters="{}",
            best_iteration=None,
            eligible=False,
            selected=False,
        )
    )
    equipment_predictions = equipment_median_predictions(training, training[target], holdout)
    equipment_metrics = regression_metrics(holdout[target], equipment_predictions)
    model_rows.append(
        _metric_row(
            "equipment_median_rpm",
            "distance + equipment",
            equipment_metrics,
            model_family="Baseline",
            parameters="{}",
            best_iteration=None,
            eligible=False,
            selected=False,
        )
    )

    screening_training, screening_validation, feature_screen_split = chronological_inner_split(training)
    if pd.to_datetime(screening_validation["date"]).max() >= pd.to_datetime(holdout["date"]).min():
        raise RuntimeError("Training-only feature screening overlaps the outer holdout")
    LOGGER.info(
        "Training-only feature screen: %s..%s (%d) -> %s..%s (%d)",
        feature_screen_split["training_date_start"],
        feature_screen_split["training_date_end"],
        feature_screen_split["training_rows"],
        feature_screen_split["validation_date_start"],
        feature_screen_split["validation_date_end"],
        feature_screen_split["validation_rows"],
    )
    screening_builder = FeatureBuilder().fit(screening_training)
    feature_screening, selected_feature_set, catboost_iterations = run_feature_screening(
        screening_builder,
        screening_training,
        screening_validation,
        target,
    )
    chronological_builder = FeatureBuilder().fit(training)
    selected_columns = COMMON_FEATURE_SETS[selected_feature_set]
    train_features = chronological_builder.transform(training, selected_columns)
    holdout_features = chronological_builder.transform(holdout, selected_columns)

    eligible_candidates: list[ModelCandidate] = []
    for name in sorted(HGB_FINAL_CANDIDATES):
        parameters = HGB_FINAL_CANDIDATES[name]
        LOGGER.info("Evaluating predeclared final candidate: %s", name)
        candidate = fit_hgb_candidate(
            name,
            train_features,
            training[target],
            holdout_features,
            holdout[target],
            parameters,
            selected_feature_set,
        )
        eligible_candidates.append(candidate)
        model_rows.append(_candidate_metric_row(candidate))

    LOGGER.info("Evaluating CatBoost on the same rows and selected features")
    catboost_parameters = dict(CATBOOST_CANDIDATE_PARAMETERS)
    catboost_parameters["iterations"] = catboost_iterations
    catboost = fit_catboost_candidate(
        "catboost_rate_per_mile",
        train_features,
        training[target],
        holdout_features,
        holdout[target],
        catboost_parameters,
        selected_feature_set,
    )
    eligible_candidates.append(catboost)
    model_rows.append(_candidate_metric_row(catboost))

    _validate_candidate_pool(eligible_candidates, EXPECTED_MAIN_CANDIDATES, "Main")

    candidate_families = {candidate.model_family for candidate in eligible_candidates}
    if not {MODEL_FAMILY_HGB, MODEL_FAMILY_CATBOOST}.issubset(candidate_families):
        raise RuntimeError("Both HGB and CatBoost must be eligible for final model selection")
    for candidate in eligible_candidates:
        if candidate.feature_columns != selected_columns:
            raise RuntimeError(f"Candidate {candidate.name!r} did not use the common selected feature set")
        forbidden_features = {"load_id", target, "predicted_rate"} & set(candidate.feature_columns)
        if forbidden_features:
            raise RuntimeError(f"Candidate {candidate.name!r} contains forbidden features: {forbidden_features}")
        if len(candidate.predictions) != len(holdout):
            raise RuntimeError(f"Candidate {candidate.name!r} did not predict every holdout row")
        if model_family_for_model(candidate.model) != candidate.model_family:
            raise RuntimeError(f"Candidate {candidate.name!r} family metadata does not match its fitted model")

    selected_candidate = _select_lowest_mae(eligible_candidates)
    selected_name = selected_candidate.name
    selected_feature_set = selected_candidate.feature_set
    selected_columns = selected_candidate.feature_columns
    selected_parameters = selected_candidate.parameters
    model_comparison = pd.DataFrame(model_rows).sort_values(
        ["mae", "model"], kind="mergesort"
    ).reset_index(drop=True)
    model_comparison["selected"] = model_comparison["model"].eq(selected_name)
    baseline_rows = model_comparison.loc[model_comparison["model"].isin(EXPECTED_BASELINES)]
    if (
        len(baseline_rows) != len(EXPECTED_BASELINES)
        or set(baseline_rows["model"]) != EXPECTED_BASELINES
        or baseline_rows["eligible"].any()
        or baseline_rows["selected"].any()
    ):
        raise RuntimeError("Both documented baselines must appear exactly once and remain ineligible")
    model_comparison.to_csv(ARTIFACTS / "model_comparison.csv", index=False)
    validate_model_selection(
        pd.read_csv(ARTIFACTS / "model_comparison.csv"),
        selected_name,
        selected_candidate.model_family,
        selected_candidate.metrics["mae"],
        EXPECTED_MAIN_CANDIDATES,
    )
    LOGGER.info(
        "Selected %s (%s) / %s: MAE %.3f, RMSE %.3f, R2 %.5f, sMAPE %.3f%%",
        selected_name,
        selected_candidate.model_family,
        selected_feature_set,
        selected_candidate.metrics["mae"],
        selected_candidate.metrics["rmse"],
        selected_candidate.metrics["r2"],
        selected_candidate.metrics["smape"],
    )

    feature_importance = permutation_importance(
        selected_candidate.model,
        holdout_features,
        holdout[target],
        sample_size=4_000,
        random_seed=RANDOM_SEED,
    )
    feature_importance.to_csv(ARTIFACTS / "feature_importance.csv", index=False)
    error_analysis, residual_rows = build_error_analysis(
        holdout,
        holdout[target],
        selected_candidate.predictions,
        training,
    )
    error_analysis.to_csv(ARTIFACTS / "error_analysis.csv", index=False)
    residual_rows[
        ["load_id", "date", "actual", "predicted", "residual", "absolute_error", "distance_band", "route_status", "city_status"]
    ].to_csv(ARTIFACTS / "holdout_predictions.csv", index=False)
    create_model_figures(feature_importance, residual_rows, FIGURES)
    stability = run_temporal_stability(development, target, selected_candidate)

    december_rows: list[dict[str, Any]] = []
    december_candidates: list[ModelCandidate] = []
    for name in sorted(DECEMBER_FEATURE_SETS):
        columns = DECEMBER_FEATURE_SETS[name]
        LOGGER.info("December reduced-schema validation: %s", name)
        december_train_features = chronological_builder.transform(training, columns)
        december_holdout_features = chronological_builder.transform(holdout, columns)
        hgb_candidate = fit_hgb_candidate(
            f"december_hgb__{name}",
            december_train_features,
            training[target],
            december_holdout_features,
            holdout[target],
            HGB_FINAL_CANDIDATES["hgb_compact"],
            name,
        )
        december_catboost_screen = learn_catboost_iterations(
            screening_builder.transform(screening_training, columns),
            screening_training[target],
            screening_builder.transform(screening_validation, columns),
            screening_validation[target],
            CATBOOST_CANDIDATE_PARAMETERS,
        )
        december_catboost_parameters = dict(CATBOOST_CANDIDATE_PARAMETERS)
        december_catboost_parameters["iterations"] = december_catboost_screen.iterations
        catboost_candidate = fit_catboost_candidate(
            f"december_catboost__{name}",
            december_train_features,
            training[target],
            december_holdout_features,
            holdout[target],
            december_catboost_parameters,
            name,
        )
        for candidate in [hgb_candidate, catboost_candidate]:
            december_candidates.append(candidate)
            december_rows.append(
                {
                    **_candidate_metric_row(candidate),
                    "feature_count": len(candidate.feature_columns),
                }
            )
    _validate_candidate_pool(december_candidates, EXPECTED_DECEMBER_CANDIDATES, "December")
    december_families = {candidate.model_family for candidate in december_candidates}
    if not {MODEL_FAMILY_HGB, MODEL_FAMILY_CATBOOST}.issubset(december_families):
        raise RuntimeError("Both HGB and CatBoost must be eligible for December model selection")
    for candidate in december_candidates:
        if len(candidate.predictions) != len(holdout):
            raise RuntimeError(f"December candidate {candidate.name!r} did not predict every holdout row")
        forbidden_features = {"load_id", target, "predicted_rate"} & set(candidate.feature_columns)
        if forbidden_features:
            raise RuntimeError(f"December candidate {candidate.name!r} contains forbidden features: {forbidden_features}")
        if model_family_for_model(candidate.model) != candidate.model_family:
            raise RuntimeError(f"December candidate {candidate.name!r} family metadata is inconsistent")
    selected_december_candidate = _select_lowest_mae(december_candidates)
    december_comparison = pd.DataFrame(december_rows).sort_values(
        ["mae", "model"], kind="mergesort"
    ).reset_index(drop=True)
    december_comparison["selected"] = december_comparison["model"].eq(selected_december_candidate.name)
    december_comparison.to_csv(ARTIFACTS / "december_model_comparison.csv", index=False)
    validate_model_selection(
        pd.read_csv(ARTIFACTS / "december_model_comparison.csv"),
        selected_december_candidate.name,
        selected_december_candidate.model_family,
        selected_december_candidate.metrics["mae"],
        EXPECTED_DECEMBER_CANDIDATES,
    )
    december_method = selected_december_candidate.feature_set
    LOGGER.info(
        "Selected December model %s (%s) / %s: MAE %.3f",
        selected_december_candidate.name,
        selected_december_candidate.model_family,
        december_method,
        selected_december_candidate.metrics["mae"],
    )

    LOGGER.info(
        "Refitting selected validation model %s (%s) on all labeled development rows",
        selected_name,
        selected_candidate.model_family,
    )
    full_builder = FeatureBuilder().fit(development)
    full_train_features = full_builder.transform(development, selected_columns)
    final_model = fit_rate_model(
        selected_candidate.model_family,
        selected_parameters,
        full_train_features,
        development[target],
    )
    if model_family_for_model(final_model) != selected_candidate.model_family:
        raise RuntimeError("Final model family does not match the selected candidate")
    final_model_path = MODELS / "final_rate_model.joblib"
    save_rate_model(final_model, final_model_path)
    if model_family_for_model(load(final_model_path)) != selected_candidate.model_family:
        raise RuntimeError("Saved final model family does not match the selected candidate")

    # Create an inference bundle containing preprocessor, columns and the fitted model
    final_bundle_path = MODELS / "final_inference_bundle.joblib"
    dump(
        {
            "feature_builder": full_builder,
            "feature_columns": selected_columns,
            "model": final_model,
            "target": target,
            "model_name": selected_name,
            "model_family": selected_candidate.model_family,
        },
        final_bundle_path,
    )

    validation_features = full_builder.transform(validation, selected_columns)
    validation_predictions = final_model.predict(validation_features)
    by_id = pd.Series(validation_predictions, index=validation["load_id"].astype(str), name="predicted_rate")
    submission = template[["load_id"]].copy()
    submission["load_id"] = submission["load_id"].astype(str)
    submission["predicted_rate"] = submission["load_id"].map(by_id)
    validate_prediction_output(submission, template, validation)
    predictions_path = ROOT / "validation_predictions.csv"
    submission.to_csv(predictions_path, index=False)

    LOGGER.info(
        "Refitting selected December-compatible model %s (%s) on all labeled rows",
        selected_december_candidate.name,
        selected_december_candidate.model_family,
    )
    december_columns = selected_december_candidate.feature_columns
    december_train_features = full_builder.transform(development, december_columns)
    december_model = fit_rate_model(
        selected_december_candidate.model_family,
        selected_december_candidate.parameters,
        december_train_features,
        development[target],
    )
    if model_family_for_model(december_model) != selected_december_candidate.model_family:
        raise RuntimeError("December model family does not match the selected December candidate")
    december_model_path = MODELS / "december_rate_model.joblib"
    save_rate_model(december_model, december_model_path)
    if model_family_for_model(load(december_model_path)) != selected_december_candidate.model_family:
        raise RuntimeError("Saved December model family does not match the selected candidate")

    # Optional: emit a December-compatible inference bundle as well
    december_bundle_path = MODELS / "december_inference_bundle.joblib"
    dump(
        {
            "feature_builder": full_builder,
            "feature_columns": december_columns,
            "model": december_model,
            "target": target,
            "model_name": selected_december_candidate.name,
            "model_family": selected_december_candidate.model_family,
        },
        december_bundle_path,
    )

    december_features = full_builder.transform(december, december_columns)
    completed_december = december.copy()
    completed_december["predicted_rate"] = december_model.predict(december_features)
    validate_december_output(completed_december)
    completed_december.to_csv(inputs.december, index=False)

    # Re-read both files so validation covers exactly what the reviewer receives.
    validate_prediction_output(pd.read_csv(predictions_path), template, validation)
    validate_december_output(pd.read_csv(inputs.december))
    scorer_output = run_scorer(inputs, predictions_path)
    scorer_chart = SCORER_RESULTS / "candidate_december.png"
    require_nonempty_file(scorer_chart)
    if _sha256(inputs.scorer) != scorer_hash_before:
        raise RuntimeError("score.py changed during the pipeline; supplied scorer must remain unmodified")

    selected_screen_score = float(
        feature_screening.loc[
            feature_screening["selected_for_outer_comparison"],
            "selection_score_mean_mae",
        ].iloc[0]
    )
    main_selection_identity = _selection_identity(selected_candidate, "main")
    december_selection_identity = _selection_identity(selected_december_candidate, "december")
    metrics_payload: dict[str, Any] = {
        "run_id": run_id,
        "source_fingerprint_sha256": source_fingerprint,
        "canonical_selections": {
            "main": main_selection_identity,
            "december": december_selection_identity,
        },
        "primary_metric": "mae",
        "selection_rule": (
            "Lowest finite MAE on the common chronological holdout among all eligible candidates; "
            "exact-equality ties are broken deterministically by candidate name."
        ),
        "candidate_configuration_policy": (
            "The common feature list is chosen by fixed two-family proxy configurations on training-only rows. "
            "Every eligible outer family, feature list, parameter set, and CatBoost iteration count is then frozen "
            "before outer-holdout scoring; no candidate is adapted and rescored on that holdout."
        ),
        "selected_model": selected_name,
        "selected_model_family": selected_candidate.model_family,
        "selected_feature_set": selected_feature_set,
        "selected_parameters": selected_parameters,
        "selected_best_iteration": selected_candidate.best_iteration,
        "selected_feature_columns": selected_candidate.feature_columns,
        "selected_categorical_columns": selected_candidate.categorical_columns,
        "metrics": selected_candidate.metrics,
        "minimum_eligible_mae": float(
            model_comparison.loc[model_comparison["eligible"], "mae"].min()
        ),
        "eligible_candidates": [candidate.name for candidate in eligible_candidates],
        "candidate_holdout_metrics": {
            candidate.name: candidate.metrics for candidate in eligible_candidates
        },
        "final_model_filename": str(final_model_path.relative_to(ROOT)),
        "baseline_metrics": {
            "global_median": global_metrics,
            "equipment_median_rpm": equipment_metrics,
        },
        "feature_screening": {
            "method": "family-neutral common-feature screening",
            "model_families": [MODEL_FAMILY_HGB, MODEL_FAMILY_CATBOOST],
            "selection_metric": "unweighted arithmetic mean of per-family posted-rate MAE",
            "selection_rule": (
                "Lowest finite mean family MAE on the training-only chronological validation window; "
                "exact ties are broken by feature-set name."
            ),
            "selected_feature_set": selected_feature_set,
            "selected_score": selected_screen_score,
            "outer_holdout_used": False,
            "preprocessing_fit_scope": "training-only inner-training rows",
            "artifact": str((ARTIFACTS / "feature_ablation.csv").relative_to(ROOT)),
            "screening_parameters": {
                MODEL_FAMILY_HGB: HGB_SCREENING_PARAMETERS,
                MODEL_FAMILY_CATBOOST: CATBOOST_CANDIDATE_PARAMETERS,
            },
            "split": feature_screen_split,
        },
        "catboost_iteration_learning": {
            "metric": "distance-weighted rate-per-mile MAE (proportional to posted-rate MAE)",
            "selected_main_iterations": catboost_iterations,
            "outer_holdout_used": False,
            "preprocessing_fit_scope": "training-only inner-training rows",
            "split": feature_screen_split,
        },
        "split": split,
        "temporal_stability": stability.to_dict(orient="records"),
        "december_model": selected_december_candidate.name,
        "december_model_family": selected_december_candidate.model_family,
        "december_feature_set": december_method,
        "december_parameters": selected_december_candidate.parameters,
        "december_best_iteration": selected_december_candidate.best_iteration,
        "december_method": december_method,
        "december_holdout_metrics": selected_december_candidate.metrics,
        "december_model_filename": str(december_model_path.relative_to(ROOT)),
        "validation_prediction_summary": {
            "rows": len(submission),
            "min": float(submission["predicted_rate"].min()),
            "mean": float(submission["predicted_rate"].mean()),
            "max": float(submission["predicted_rate"].max()),
        },
        "december_prediction_summary": {
            "rows": len(completed_december),
            "min": float(completed_december["predicted_rate"].min()),
            "mean": float(completed_december["predicted_rate"].mean()),
            "max": float(completed_december["predicted_rate"].max()),
        },
        "scorer_passed": True,
        "scorer_output": scorer_output.strip().splitlines(),
        "random_seed": RANDOM_SEED,
        "deterministic_execution": {
            "candidate_iteration_order": "sorted by candidate or feature-set name",
            "tie_break_rule": "exact MAE ties are resolved by candidate name",
            "row_order": "stable date then load_id mergesort",
            "thread_limits": {
                name: os.environ[name]
                for name in (
                    "OMP_NUM_THREADS",
                    "MKL_NUM_THREADS",
                    "OPENBLAS_NUM_THREADS",
                    "NUMEXPR_NUM_THREADS",
                    "VECLIB_MAXIMUM_THREADS",
                    "BLIS_NUM_THREADS",
                    "LOKY_MAX_CPU_COUNT",
                )
            },
            "versions": {
                "python": platform.python_version(),
                "numpy": version("numpy"),
                "pandas": version("pandas"),
                "scikit_learn": version("scikit-learn"),
                "matplotlib": version("matplotlib"),
                "catboost": version("catboost"),
                "joblib": version("joblib"),
            },
        },
    }
    write_json(ARTIFACTS / "validation_metrics.json", metrics_payload)

    generated_figures = [
        FIGURES / "target_distribution.png",
        FIGURES / "monthly_target.png",
        FIGURES / "distribution_shift.png",
        FIGURES / "feature_importance.png",
        FIGURES / "holdout_error_over_time.png",
    ]
    generated_outputs = [
        predictions_path,
        inputs.december,
        ARTIFACTS / "validation_metrics.json",
        ARTIFACTS / "model_comparison.csv",
        ARTIFACTS / "december_model_comparison.csv",
        ARTIFACTS / "feature_ablation.csv",
        ARTIFACTS / "feature_importance.csv",
        ARTIFACTS / "scorer_output.txt",
        final_model_path,
        final_bundle_path,
        december_model_path,
        december_bundle_path,
        scorer_chart,
        *generated_figures,
    ]
    for output in generated_outputs:
        require_nonempty_file(output)
    validate_generated_metrics_consistency(
        ARTIFACTS / "model_comparison.csv",
        ARTIFACTS / "december_model_comparison.csv",
        ARTIFACTS / "validation_metrics.json",
        EXPECTED_MAIN_CANDIDATES,
        EXPECTED_DECEMBER_CANDIDATES,
        EXPECTED_BASELINES,
    )
    require_nonempty_file(static_readme)
    require_nonempty_file(static_report)
    readme_hash_after = _sha256(static_readme)
    if readme_hash_after != readme_hash_before:
        raise RuntimeError("README.md changed during pipeline execution")
    LOGGER.info("Static README SHA-256 unchanged: %s", readme_hash_after)
    LOGGER.info("Machine-generated metrics consistency checks passed for run ID %s", run_id)
    LOGGER.info("Single-run programmatic checks passed; compare two clean runs before submission")


if __name__ == "__main__":
    main()
