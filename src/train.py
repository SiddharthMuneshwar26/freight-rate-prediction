"""Focused model training helpers for the freight assessment."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool
from joblib import dump
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.preprocessing import OrdinalEncoder
from threadpoolctl import threadpool_limits

from .evaluate import regression_metrics
from .features import categorical_columns


RANDOM_SEED = 42
MODEL_FAMILY_HGB = "HistGradientBoostingRegressor"
MODEL_FAMILY_CATBOOST = "CatBoostRegressor"

HGB_SCREENING_PARAMETERS: dict[str, Any] = {
    "max_iter": 250,
    "learning_rate": 0.06,
    "max_leaf_nodes": 15,
    "min_samples_leaf": 30,
    "l2_regularization": 3.0,
}

HGB_FINAL_CANDIDATES: dict[str, dict[str, Any]] = {
    "hgb_compact": HGB_SCREENING_PARAMETERS,
    "hgb_smoother": {
        "max_iter": 300,
        "learning_rate": 0.05,
        "max_leaf_nodes": 15,
        "min_samples_leaf": 45,
        "l2_regularization": 5.0,
    },
}


@dataclass
class ModelCandidate:
    """One holdout-evaluated model eligible for common model selection."""

    name: str
    model_family: str
    model: HGBRateModel | CatBoostRateModel
    parameters: dict[str, Any]
    feature_columns: list[str]
    categorical_columns: list[str]
    predictions: np.ndarray
    metrics: dict[str, float]
    feature_set: str
    best_iteration: int | None = None


@dataclass(frozen=True)
class CatBoostTuningResult:
    """Training-only CatBoost stopping result and its posted-rate metrics."""

    iterations: int
    metrics: dict[str, float]


@dataclass
class HGBRateModel:
    """HistGradientBoosting model for rate-per-mile with safe ordinal categories."""

    parameters: dict[str, Any]
    feature_names: list[str] = field(default_factory=list)
    category_names: list[str] = field(default_factory=list)
    encoder: OrdinalEncoder | None = None
    regressor: HistGradientBoostingRegressor | None = None

    def fit(self, features: pd.DataFrame, target: pd.Series) -> "HGBRateModel":
        self.feature_names = list(features.columns)
        self.category_names = categorical_columns(self.feature_names)
        matrix = self._fit_matrix(features)
        distance = pd.to_numeric(features["distance"], errors="coerce").to_numpy(dtype=float)
        rate_per_mile = target.to_numpy(dtype=float) / distance
        categorical_mask = [name in self.category_names for name in self.feature_names]
        self.regressor = HistGradientBoostingRegressor(
            **self.parameters,
            loss="absolute_error",
            categorical_features=categorical_mask,
            early_stopping=False,
            random_state=RANDOM_SEED,
        )
        with threadpool_limits(limits=1):
            self.regressor.fit(matrix, rate_per_mile)
        return self

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        if self.regressor is None or self.encoder is None:
            raise RuntimeError("HGBRateModel must be fitted before prediction")
        matrix = self._transform_matrix(features)
        with threadpool_limits(limits=1):
            rpm = self.regressor.predict(matrix)
        distance = pd.to_numeric(features["distance"], errors="coerce").to_numpy(dtype=float)
        return np.maximum(rpm * distance, 1.0)

    def _fit_matrix(self, features: pd.DataFrame) -> np.ndarray:
        matrix = features.loc[:, self.feature_names].copy()
        if self.category_names:
            self.encoder = OrdinalEncoder(
                handle_unknown="use_encoded_value",
                unknown_value=-1,
                dtype=np.float64,
            )
            encoded_categories = self.encoder.fit_transform(
                matrix[self.category_names].fillna("Unknown").astype(str)
            )
            for index, name in enumerate(self.category_names):
                matrix[name] = encoded_categories[:, index]
        else:
            self.encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
            self.encoder.fit(np.full((len(matrix), 1), "unused"))
        return matrix.astype(float).to_numpy()

    def _transform_matrix(self, features: pd.DataFrame) -> np.ndarray:
        if list(features.columns) != self.feature_names:
            features = features.loc[:, self.feature_names]
        matrix = features.copy()
        if self.category_names:
            encoded_categories = self.encoder.transform(
                matrix[self.category_names].fillna("Unknown").astype(str)
            )
            for index, name in enumerate(self.category_names):
                matrix[name] = encoded_categories[:, index]
        return matrix.astype(float).to_numpy()


@dataclass
class CatBoostRateModel:
    """CatBoost model trained on rate per mile and predicting positive dollar rates."""

    parameters: dict[str, Any]
    feature_names: list[str] = field(default_factory=list)
    category_names: list[str] = field(default_factory=list)
    regressor: CatBoostRegressor | None = None

    def fit(self, features: pd.DataFrame, target: pd.Series) -> "CatBoostRateModel":
        self.feature_names = list(features.columns)
        self.category_names = categorical_columns(self.feature_names)
        matrix = features.loc[:, self.feature_names]
        distance = pd.to_numeric(matrix["distance"], errors="raise").to_numpy(dtype=float)
        rate_per_mile = target.to_numpy(dtype=float) / distance
        self.regressor = _new_catboost_regressor(self.parameters)
        self.regressor.fit(
            matrix,
            rate_per_mile,
            cat_features=self.category_names,
            verbose=False,
        )
        return self

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        if self.regressor is None:
            raise RuntimeError("CatBoostRateModel must be fitted before prediction")
        matrix = features.loc[:, self.feature_names]
        rpm = self.regressor.predict(matrix).astype(float)
        distance = pd.to_numeric(matrix["distance"], errors="raise").to_numpy(dtype=float)
        return np.maximum(rpm * distance, 1.0)


def _new_catboost_regressor(parameters: dict[str, Any]) -> CatBoostRegressor:
    configured = dict(parameters)
    configured.update(
        {
            "loss_function": "MAE",
            "eval_metric": "MAE",
            "random_seed": RANDOM_SEED,
            "task_type": "CPU",
            "allow_writing_files": False,
            "thread_count": 1,
            "verbose": False,
        }
    )
    return CatBoostRegressor(**configured)


def global_median_predictions(training_target: pd.Series, rows: int) -> np.ndarray:
    return np.full(rows, float(training_target.median()))


def equipment_median_predictions(
    training: pd.DataFrame,
    training_target: pd.Series,
    holdout: pd.DataFrame,
) -> np.ndarray:
    equipment = training["equipment"].astype(str).str.strip().str.title()
    training_distance = pd.to_numeric(training["distance"], errors="raise").to_numpy(dtype=float)
    rate_per_mile = training_target.to_numpy(dtype=float) / training_distance
    medians = pd.DataFrame({"equipment": equipment, "rpm": rate_per_mile}).groupby(
        "equipment", observed=True
    )["rpm"].median()
    fallback = float(np.median(rate_per_mile))
    cleaned_holdout = holdout["equipment"].astype(str).str.strip().str.title()
    holdout_rpm = cleaned_holdout.map(medians).fillna(fallback).to_numpy(dtype=float)
    holdout_distance = pd.to_numeric(holdout["distance"], errors="raise").to_numpy(dtype=float)
    return holdout_rpm * holdout_distance


def fit_catboost_candidate(
    name: str,
    train_features: pd.DataFrame,
    train_target: pd.Series,
    holdout_features: pd.DataFrame,
    holdout_target: pd.Series,
    parameters: dict[str, Any],
    feature_set: str,
) -> ModelCandidate:
    """Fit a CatBoost finalist with parameters frozen before holdout scoring."""
    feature_columns = list(train_features.columns)
    category_names = categorical_columns(feature_columns)
    effective_parameters = dict(parameters)
    best_iteration = int(effective_parameters["iterations"])
    if best_iteration < 1:
        raise ValueError("CatBoost finalist requires a positive frozen iteration count")
    model = CatBoostRateModel(parameters=effective_parameters).fit(train_features, train_target)
    predictions = model.predict(holdout_features)
    return ModelCandidate(
        name=name,
        model_family=MODEL_FAMILY_CATBOOST,
        model=model,
        parameters=effective_parameters,
        feature_columns=feature_columns,
        categorical_columns=category_names,
        predictions=predictions,
        metrics=regression_metrics(holdout_target, predictions),
        feature_set=feature_set,
        best_iteration=best_iteration,
    )


def learn_catboost_iterations(
    tuning_train_features: pd.DataFrame,
    tuning_train_target: pd.Series,
    tuning_validation_features: pd.DataFrame,
    tuning_validation_target: pd.Series,
    parameters: dict[str, Any],
) -> CatBoostTuningResult:
    """Learn iterations and score them on fold-pure, training-only matrices."""
    feature_columns = list(tuning_train_features.columns)
    if list(tuning_validation_features.columns) != feature_columns:
        raise ValueError("CatBoost iteration-learning feature schemas do not match")
    category_names = categorical_columns(feature_columns)
    train_distance = pd.to_numeric(
        tuning_train_features["distance"], errors="raise"
    ).to_numpy(dtype=float)
    validation_distance = pd.to_numeric(
        tuning_validation_features["distance"], errors="raise"
    ).to_numpy(dtype=float)
    train_rpm = tuning_train_target.to_numpy(dtype=float) / train_distance
    validation_rpm = tuning_validation_target.to_numpy(dtype=float) / validation_distance

    # Distance-weighted RPM MAE is proportional to posted-rate MAE, so early
    # stopping uses the same business metric as final candidate comparison.
    validation_pool = Pool(
        tuning_validation_features,
        label=validation_rpm,
        cat_features=category_names,
        weight=validation_distance,
    )
    provisional_model = _new_catboost_regressor(parameters)
    provisional_model.fit(
        tuning_train_features,
        train_rpm,
        cat_features=category_names,
        eval_set=validation_pool,
        early_stopping_rounds=60,
        verbose=False,
    )
    best_iteration = int(provisional_model.get_best_iteration()) + 1
    if best_iteration <= 0:
        best_iteration = int(parameters.get("iterations", 1_000))
    validation_rpm_predictions = provisional_model.predict(
        tuning_validation_features,
        ntree_end=best_iteration,
    ).astype(float)
    validation_rate_predictions = np.maximum(validation_rpm_predictions * validation_distance, 1.0)
    return CatBoostTuningResult(
        iterations=best_iteration,
        metrics=regression_metrics(tuning_validation_target, validation_rate_predictions),
    )


def fit_hgb_candidate(
    name: str,
    train_features: pd.DataFrame,
    train_target: pd.Series,
    holdout_features: pd.DataFrame,
    holdout_target: pd.Series,
    parameters: dict[str, Any],
    feature_set: str,
) -> ModelCandidate:
    """Fit and evaluate a robust nonlinear rate-per-mile model."""
    model = HGBRateModel(parameters=dict(parameters)).fit(train_features, train_target)
    predictions = model.predict(holdout_features)
    feature_columns = list(train_features.columns)
    return ModelCandidate(
        name=name,
        model_family=MODEL_FAMILY_HGB,
        model=model,
        parameters=dict(parameters),
        feature_columns=feature_columns,
        categorical_columns=categorical_columns(feature_columns),
        predictions=predictions,
        metrics=regression_metrics(holdout_target, predictions),
        feature_set=feature_set,
    )


def fit_rate_model(
    model_family: str,
    parameters: dict[str, Any],
    features: pd.DataFrame,
    target: pd.Series,
) -> HGBRateModel | CatBoostRateModel:
    """Refit one supported rate-per-mile model with fixed effective parameters."""
    if model_family == MODEL_FAMILY_HGB:
        return HGBRateModel(parameters=dict(parameters)).fit(features, target)
    if model_family == MODEL_FAMILY_CATBOOST:
        return CatBoostRateModel(parameters=dict(parameters)).fit(features, target)
    raise ValueError(f"Unsupported model family: {model_family!r}")


def model_family_for_model(model: object) -> str:
    """Return the stable family identifier for a supported fitted wrapper."""
    if isinstance(model, HGBRateModel):
        if not isinstance(model.regressor, HistGradientBoostingRegressor):
            raise TypeError("HGBRateModel does not contain a fitted HistGradientBoostingRegressor")
        return MODEL_FAMILY_HGB
    if isinstance(model, CatBoostRateModel):
        if not isinstance(model.regressor, CatBoostRegressor):
            raise TypeError("CatBoostRateModel does not contain a fitted CatBoostRegressor")
        return MODEL_FAMILY_CATBOOST
    raise TypeError(f"Unsupported rate model type: {type(model).__name__}")


def save_rate_model(model: HGBRateModel | CatBoostRateModel, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    dump(model, path)
