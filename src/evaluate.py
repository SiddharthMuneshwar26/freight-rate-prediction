"""Metrics, segmented error analysis, and submission validation."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def regression_metrics(actual: pd.Series | np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    """Calculate business-readable regression metrics with safe sMAPE."""
    y_true = np.asarray(actual, dtype=float)
    y_pred = np.asarray(predicted, dtype=float)
    denominator = np.abs(y_true) + np.abs(y_pred)
    smape_terms = np.divide(
        2.0 * np.abs(y_pred - y_true),
        denominator,
        out=np.zeros_like(y_true, dtype=float),
        where=denominator > 0,
    )
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
        "smape": float(np.mean(smape_terms) * 100.0),
    }


def validate_model_selection(
    comparison: pd.DataFrame,
    selected_name: str,
    selected_family: str,
    selected_mae: float,
    expected_eligible: dict[str, str] | None = None,
) -> None:
    """Verify that a comparison table truthfully identifies its best eligible model."""
    required_columns = {"model", "model_family", "mae", "eligible", "selected"}
    missing_columns = required_columns - set(comparison.columns)
    if missing_columns:
        raise ValueError(f"Model comparison is missing required columns: {sorted(missing_columns)}")

    def coerce_boolean(value: object, column: str) -> bool:
        if isinstance(value, (bool, np.bool_)):
            return bool(value)
        if isinstance(value, (int, np.integer)) and value in (0, 1):
            return bool(value)
        if isinstance(value, (float, np.floating)) and np.isfinite(value) and value in (0.0, 1.0):
            return bool(value)
        if isinstance(value, str):
            normalised = value.strip().lower()
            if normalised in {"true", "1", "yes", "y"}:
                return True
            if normalised in {"false", "0", "no", "n"}:
                return False
        raise ValueError(f"Model comparison column {column!r} contains an invalid boolean value: {value!r}")

    eligible = comparison["eligible"].map(lambda value: coerce_boolean(value, "eligible"))
    marked_selected = comparison["selected"].map(lambda value: coerce_boolean(value, "selected"))
    maes = pd.to_numeric(comparison["mae"], errors="coerce")

    if not eligible.any():
        raise ValueError("Model comparison has no eligible candidates")
    eligible_maes = maes.loc[eligible].to_numpy(dtype=float)
    if not np.isfinite(eligible_maes).all():
        raise ValueError("Eligible model comparison rows must have finite MAE values")
    if expected_eligible is not None:
        eligible_rows = comparison.loc[eligible]
        eligible_names = eligible_rows["model"].astype(str)
        if eligible_names.duplicated().any():
            raise ValueError("Eligible model names must be unique")
        if set(eligible_names) != set(expected_eligible):
            raise ValueError("Emitted eligible models do not match the evaluated candidate pool")
        emitted_families = dict(zip(eligible_names, eligible_rows["model_family"].astype(str), strict=True))
        if emitted_families != {str(name): str(family) for name, family in expected_eligible.items()}:
            raise ValueError("Emitted eligible model families do not match the evaluated candidate pool")

    matching_name = comparison["model"].astype("string").eq(str(selected_name))
    if int(matching_name.sum()) != 1:
        raise ValueError(f"Selected candidate {selected_name!r} must appear exactly once in model comparison")
    if int(marked_selected.sum()) != 1:
        raise ValueError("Model comparison must mark exactly one row as selected")

    selected_position = int(np.flatnonzero(marked_selected.to_numpy(dtype=bool))[0])
    selected_row = comparison.iloc[selected_position]
    if str(selected_row["model"]) != str(selected_name):
        raise ValueError("Marked selected model does not match the selected candidate name")
    if str(selected_row["model_family"]) != str(selected_family):
        raise ValueError("Marked selected model family does not match the selected family")
    if not bool(eligible.iloc[selected_position]):
        raise ValueError("Marked selected model is not eligible")

    try:
        selected_mae_value = float(selected_mae)
    except (TypeError, ValueError) as exc:
        raise ValueError("Selected MAE must be finite and numeric") from exc
    if not np.isfinite(selected_mae_value):
        raise ValueError("Selected MAE must be finite and numeric")

    tolerance = {"rtol": 1e-12, "atol": 1e-12}
    row_mae = float(maes.iloc[selected_position])
    if not np.isclose(row_mae, selected_mae_value, **tolerance):
        raise ValueError("Marked selected row MAE does not match the selected MAE")
    minimum_eligible_mae = float(np.min(eligible_maes))
    if not np.isclose(selected_mae_value, minimum_eligible_mae, **tolerance):
        raise ValueError("Selected MAE is not the minimum eligible model MAE")


def _segment_summary(frame: pd.DataFrame, dimension: str, segment_column: str) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for segment, group in frame.groupby(segment_column, observed=True, dropna=False):
        metrics = regression_metrics(group["actual"], group["predicted"])
        rows.append(
            {
                "dimension": dimension,
                "segment": str(segment),
                "rows": int(len(group)),
                **metrics,
                "mean_error": float(group["predicted"].sub(group["actual"]).mean()),
            }
        )
    return pd.DataFrame(rows)


def build_error_analysis(
    holdout_raw: pd.DataFrame,
    actual: pd.Series,
    predicted: np.ndarray,
    training_raw: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Summarise residuals across decision-relevant holdout segments."""
    analysis = holdout_raw.copy().reset_index(drop=True)
    analysis["actual"] = np.asarray(actual, dtype=float)
    analysis["predicted"] = np.asarray(predicted, dtype=float)
    analysis["residual"] = analysis["predicted"] - analysis["actual"]
    analysis["absolute_error"] = analysis["residual"].abs()
    analysis["date"] = pd.to_datetime(analysis["date"])
    analysis["month"] = analysis["date"].dt.to_period("M").astype(str)
    analysis["distance_band"] = pd.cut(
        analysis["distance"],
        bins=[-np.inf, 250, 800, 1600, np.inf],
        labels=["Short", "Medium", "Long", "Very Long"],
    )
    analysis["target_band"] = pd.qcut(
        analysis["actual"],
        q=[0, 0.25, 0.75, 1],
        labels=["Low", "Medium", "High"],
        duplicates="drop",
    )

    train_routes = training_raw["pickup"].astype(str) + " > " + training_raw["delivery"].astype(str)
    holdout_routes = analysis["pickup"].astype(str) + " > " + analysis["delivery"].astype(str)
    route_counts = train_routes.value_counts()
    analysis["route_status"] = np.select(
        [
            holdout_routes.map(route_counts).fillna(0).eq(0),
            holdout_routes.map(route_counts).fillna(0).lt(10),
        ],
        ["Unseen", "Rare (<10)"],
        default="Common (>=10)",
    )
    train_pickups = set(training_raw["pickup"].astype(str))
    train_deliveries = set(training_raw["delivery"].astype(str))
    city_unseen = (~analysis["pickup"].astype(str).isin(train_pickups)) | (
        ~analysis["delivery"].astype(str).isin(train_deliveries)
    )
    analysis["city_status"] = np.where(city_unseen, "Unseen city", "Seen cities")

    tables = [
        _segment_summary(analysis, "month", "month"),
        _segment_summary(analysis, "distance_band", "distance_band"),
        _segment_summary(analysis, "equipment", "equipment"),
        _segment_summary(analysis, "target_band", "target_band"),
        _segment_summary(analysis, "route_status", "route_status"),
        _segment_summary(analysis, "city_status", "city_status"),
    ]
    return pd.concat(tables, ignore_index=True), analysis


def permutation_importance(
    model: object,
    features: pd.DataFrame,
    actual: pd.Series,
    sample_size: int = 4_000,
    random_seed: int = 42,
) -> pd.DataFrame:
    """Measure holdout MAE increase from independently shuffling each feature."""
    rng = np.random.default_rng(random_seed)
    if len(features) > sample_size:
        positions = np.sort(rng.choice(len(features), size=sample_size, replace=False))
        sample_features = features.iloc[positions].reset_index(drop=True)
        sample_actual = actual.iloc[positions].reset_index(drop=True)
    else:
        sample_features = features.reset_index(drop=True)
        sample_actual = actual.reset_index(drop=True)
    base_predictions = model.predict(sample_features)  # type: ignore[attr-defined]
    base_mae = mean_absolute_error(sample_actual, base_predictions)
    rows: list[dict[str, float | str]] = []
    for column in sample_features.columns:
        permuted = sample_features.copy()
        permuted[column] = rng.permutation(permuted[column].to_numpy())
        changed = model.predict(permuted)  # type: ignore[attr-defined]
        changed_mae = mean_absolute_error(sample_actual, changed)
        rows.append(
            {
                "feature": column,
                "importance": float(changed_mae - base_mae),
                "permuted_mae": float(changed_mae),
                "baseline_sample_mae": float(base_mae),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["importance", "feature"],
        ascending=[False, True],
        kind="mergesort",
    ).reset_index(drop=True)


def validate_prediction_output(
    output: pd.DataFrame,
    template: pd.DataFrame,
    validation: pd.DataFrame,
) -> None:
    """Enforce both assessment and scorer constraints before writing success."""
    if list(output.columns) != ["load_id", "predicted_rate"]:
        raise ValueError("Validation output columns or order are invalid")
    if len(output) != 12_000:
        raise ValueError(f"Expected 12,000 validation predictions, found {len(output):,}")
    if output["load_id"].isna().any() or output["load_id"].duplicated().any():
        raise ValueError("Validation output has missing or duplicate load IDs")
    if output["load_id"].tolist() != template["load_id"].astype(str).tolist():
        raise ValueError("Validation output does not preserve template ID order")
    if set(output["load_id"]) != set(validation["load_id"].astype(str)):
        raise ValueError("Validation output IDs do not exactly match validation.csv")
    values = pd.to_numeric(output["predicted_rate"], errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(values).all() or (values <= 0).any():
        raise ValueError("Validation predicted rates must be finite and positive")


def validate_december_output(frame: pd.DataFrame) -> None:
    """Enforce the fixed December chart-input contract."""
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
        raise ValueError("December columns or order changed")
    dates = pd.to_datetime(frame["date"], errors="coerce")
    expected_dates = pd.date_range("2025-12-01", "2025-12-31", freq="D")
    if len(frame) != 31 or dates.isna().any() or list(dates) != list(expected_dates):
        raise ValueError("December file must contain ordered dates 2025-12-01 through 2025-12-31")
    fixed_checks = {
        "pickup": frame["pickup"].eq("Lexington").all(),
        "delivery": frame["delivery"].eq("Fort Wayne").all(),
        "distance": np.isclose(pd.to_numeric(frame["distance"]), 360.0).all(),
        "equipment": frame["equipment"].eq("Dry Van").all(),
        "weight": np.isclose(pd.to_numeric(frame["weight"]), 32_000.0).all(),
    }
    failures = [name for name, passed in fixed_checks.items() if not passed]
    if failures:
        raise ValueError(f"December fixed inputs changed: {failures}")
    predicted = pd.to_numeric(frame["predicted_rate"], errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(predicted).all() or (predicted <= 0).any():
        raise ValueError("December predicted rates must be finite and positive")


def require_nonempty_file(path: Path) -> None:
    if not path.is_file() or path.stat().st_size == 0:
        raise FileNotFoundError(f"Required file is missing or empty: {path}")


def validate_generated_metrics_consistency(
    model_comparison_path: Path,
    december_comparison_path: Path,
    metrics_path: Path,
    expected_main: dict[str, str],
    expected_december: dict[str, str],
    expected_baselines: set[str],
) -> None:
    """Validate canonical selections across machine-generated CSV and JSON outputs."""
    metrics: dict[str, Any] = json.loads(metrics_path.read_text(encoding="utf-8"))
    run_id = str(metrics.get("run_id", ""))
    if not run_id:
        raise ValueError("validation_metrics.json is missing the run ID")
    selections = metrics.get("canonical_selections")
    if not isinstance(selections, dict) or set(selections) != {"main", "december"}:
        raise ValueError("validation_metrics.json is missing canonical selection identities")

    required_columns = {
        "model",
        "model_family",
        "feature_set",
        "mae",
        "eligible",
        "selected",
    }

    def read_rows(path: Path) -> list[dict[str, str]]:
        with path.open("r", encoding="utf-8", newline="") as stream:
            reader = csv.DictReader(stream)
            fieldnames = set(reader.fieldnames or [])
            missing_columns = required_columns - fieldnames
            if missing_columns:
                raise ValueError(
                    f"{path.name} is missing required columns: {sorted(missing_columns)}"
                )
            rows = list(reader)
        if not rows:
            raise ValueError(f"{path.name} must contain at least one comparison row")
        return rows

    def parse_boolean(value: str, column: str, label: str) -> bool:
        normalised = str(value).strip().lower()
        if normalised in {"true", "1", "yes"}:
            return True
        if normalised in {"false", "0", "no"}:
            return False
        raise ValueError(
            f"{label} comparison column {column!r} contains an invalid boolean value: {value!r}"
        )

    def validate_table(
        rows: list[dict[str, str]],
        selection: dict[str, Any],
        expected: dict[str, str],
        label: str,
        expected_ineligible: set[str] | None = None,
    ) -> float:
        ineligible_names = expected_ineligible or set()
        names = [str(row["model"]) for row in rows]
        expected_names = set(expected) | ineligible_names
        if len(names) != len(set(names)):
            raise ValueError(f"{label} comparison model names must be unique")
        if set(names) != expected_names:
            raise ValueError(
                f"{label} comparison model set changed: "
                f"actual={sorted(set(names))}, expected={sorted(expected_names)}"
            )

        eligible_rows = [
            row
            for row in rows
            if parse_boolean(row["eligible"], "eligible", label)
        ]
        emitted = {str(row["model"]): str(row["model_family"]) for row in eligible_rows}
        if emitted != expected:
            raise ValueError(f"{label} eligible candidate set or families changed")

        selected_rows = [
            row
            for row in rows
            if parse_boolean(row["selected"], "selected", label)
        ]
        if len(selected_rows) != 1:
            raise ValueError(f"{label} comparison must select exactly one row")
        selected = selected_rows[0]
        if not parse_boolean(selected["eligible"], "eligible", label):
            raise ValueError(f"{label} selected row must be eligible")

        exact_fields = {
            "model": str(selection["model"]),
            "model_family": str(selection["family"]),
            "feature_set": str(selection["feature_set"]),
            "mae": str(selection["mae_text"]),
        }
        for column, expected_value in exact_fields.items():
            if str(selected[column]) != expected_value:
                raise ValueError(
                    f"{label} comparison {column} disagrees with canonical selection: "
                    f"{selected[column]!r} != {expected_value!r}"
                )

        try:
            eligible_maes = [float(row["mae"]) for row in eligible_rows]
            selected_mae = float(selected["mae"])
            canonical_mae = float(selection["mae"])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{label} comparison MAEs must be numeric") from exc
        if not np.isfinite(np.asarray(eligible_maes, dtype=float)).all():
            raise ValueError(f"{label} eligible MAEs must be finite")
        minimum_eligible_mae = min(eligible_maes)
        if selected_mae != minimum_eligible_mae:
            raise ValueError(f"{label} selected MAE is not the exact minimum eligible MAE")
        if selected_mae != canonical_mae or repr(canonical_mae) != str(selection["mae_text"]):
            raise ValueError(f"{label} canonical MAE fields disagree")

        for row in rows:
            name = str(row["model"])
            if name not in ineligible_names:
                continue
            if (
                parse_boolean(row["eligible"], "eligible", label)
                or parse_boolean(row["selected"], "selected", label)
                or str(row["model_family"]) != "Baseline"
            ):
                raise ValueError(
                    f"{label} baseline {name!r} must use family 'Baseline' and remain ineligible"
                )
        return minimum_eligible_mae

    main_rows = read_rows(model_comparison_path)
    december_rows = read_rows(december_comparison_path)
    minimum_main_mae = validate_table(
        main_rows,
        selections["main"],
        expected_main,
        "Main",
        expected_baselines,
    )
    validate_table(
        december_rows,
        selections["december"],
        expected_december,
        "December",
    )

    main_identity = selections["main"]
    december_identity = selections["december"]
    json_checks = {
        "selected_model": main_identity["model"],
        "selected_model_family": main_identity["family"],
        "selected_feature_set": main_identity["feature_set"],
        "december_model": december_identity["model"],
        "december_model_family": december_identity["family"],
        "december_feature_set": december_identity["feature_set"],
    }
    for key, expected_value in json_checks.items():
        if metrics.get(key) != expected_value:
            raise ValueError(
                f"validation_metrics.json field {key!r} disagrees with canonical selection"
            )

    main_metrics = metrics.get("metrics")
    december_metrics = metrics.get("december_holdout_metrics")
    if not isinstance(main_metrics, dict) or not isinstance(december_metrics, dict):
        raise ValueError("validation_metrics.json is missing selected-model metric objects")
    if repr(float(main_metrics["mae"])) != str(main_identity["mae_text"]):
        raise ValueError("Main JSON MAE disagrees with canonical selection")
    if repr(float(december_metrics["mae"])) != str(december_identity["mae_text"]):
        raise ValueError("December JSON MAE disagrees with canonical selection")
    if float(metrics.get("minimum_eligible_mae", np.nan)) != minimum_main_mae:
        raise ValueError("validation_metrics.json minimum eligible MAE disagrees with comparison CSV")

    eligible_candidates = metrics.get("eligible_candidates")
    if (
        not isinstance(eligible_candidates, list)
        or len(eligible_candidates) != len(set(map(str, eligible_candidates)))
        or set(map(str, eligible_candidates)) != set(expected_main)
    ):
        raise ValueError("validation_metrics.json eligible candidate set changed")
    baseline_metrics = metrics.get("baseline_metrics")
    if not isinstance(baseline_metrics, dict) or set(baseline_metrics) != expected_baselines:
        raise ValueError("validation_metrics.json baseline set changed")
