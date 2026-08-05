"""Create evidence-backed EDA, figures, and machine-readable artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


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
