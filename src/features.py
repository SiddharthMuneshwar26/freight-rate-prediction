"""Leakage-safe cleaning and focused freight feature engineering."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import numpy as np
import pandas as pd


RAW_NUMERIC = [
    "pickup_lat",
    "pickup_lon",
    "delivery_lat",
    "delivery_lon",
    "distance",
    "weight",
    "market_index",
    "quote_signal",
]
RAW_CATEGORICAL = ["pickup", "delivery", "equipment"]

BASIC_FEATURES = [
    "distance",
    "weight",
    "weight_missing",
    "weight_sign_corrected",
    "market_index",
    "market_index_missing",
    "quote_signal",
    "equipment",
]

DATE_FEATURES = [
    "year",
    "month",
    "quarter",
    "week_of_year",
    "day_of_month",
    "day_of_year",
    "day_of_week",
    "weekend",
    "month_start",
    "month_end",
    "days_since_start",
    "month_sin",
    "month_cos",
    "weekday_sin",
    "weekday_cos",
    "annual_sin",
    "annual_cos",
    "federal_holiday",
    "near_thanksgiving",
    "near_christmas",
]

COMPACT_DATE_FEATURES = [
    "days_since_start",
    "month",
    "day_of_week",
    "day_of_month",
    "week_of_year",
    "weekend",
    "annual_sin",
    "annual_cos",
    "weekday_sin",
    "weekday_cos",
]

ROUTE_FEATURES = [
    "pickup",
    "delivery",
    "route",
    "pickup_equipment",
    "delivery_equipment",
    "route_equipment",
    "pickup_frequency",
    "delivery_frequency",
    "route_frequency",
    "equipment_frequency",
    "pickup_equipment_frequency",
    "delivery_equipment_frequency",
    "route_equipment_frequency",
    "rare_pickup",
    "rare_delivery",
    "rare_route",
    "unseen_pickup",
    "unseen_delivery",
    "unseen_route",
]

GEOGRAPHIC_FEATURES = [
    "pickup_lat",
    "pickup_lon",
    "delivery_lat",
    "delivery_lon",
    "haversine_distance",
    "distance_haversine_difference",
    "distance_haversine_ratio",
    "latitude_difference",
    "longitude_difference",
    "absolute_latitude_difference",
    "absolute_longitude_difference",
    "midpoint_latitude",
    "midpoint_longitude",
    "bearing_sin",
    "bearing_cos",
    "distance_band",
]

INTERACTION_FEATURES = [
    "log_distance",
    "log_weight",
    "distance_squared_scaled",
    "weight_to_distance",
    "distance_market_index",
    "distance_quote_signal",
    "market_quote_interaction",
    "equipment_distance_band",
]

FEATURE_SETS: dict[str, list[str]] = {
    "basic_supplied": BASIC_FEATURES,
    "basic_plus_date": BASIC_FEATURES + DATE_FEATURES,
    "basic_plus_route": BASIC_FEATURES + ROUTE_FEATURES,
    "basic_plus_geographic": BASIC_FEATURES + GEOGRAPHIC_FEATURES,
    "basic_plus_interactions": BASIC_FEATURES + INTERACTION_FEATURES,
    "best_combined": BASIC_FEATURES
    + DATE_FEATURES
    + ROUTE_FEATURES
    + GEOGRAPHIC_FEATURES
    + INTERACTION_FEATURES,
}

# HistGradientBoosting supports at most 255 values per categorical feature. These
# sets retain route information through endpoints/frequencies instead of feeding
# thousands of route strings as ordinal categories.
HGB_ROUTE_FEATURES = [
    "pickup",
    "delivery",
    "pickup_frequency",
    "delivery_frequency",
    "route_frequency",
    "equipment_frequency",
    "pickup_equipment_frequency",
    "delivery_equipment_frequency",
    "route_equipment_frequency",
    "rare_pickup",
    "rare_delivery",
    "rare_route",
    "unseen_pickup",
    "unseen_delivery",
    "unseen_route",
]

COMMON_FEATURE_SETS: dict[str, list[str]] = {
    "basic_supplied": BASIC_FEATURES,
    "basic_plus_date": BASIC_FEATURES + DATE_FEATURES,
    "basic_plus_route": BASIC_FEATURES + HGB_ROUTE_FEATURES,
    "basic_plus_geographic": BASIC_FEATURES + GEOGRAPHIC_FEATURES,
    "basic_plus_interactions": BASIC_FEATURES + INTERACTION_FEATURES,
    "best_combined": BASIC_FEATURES
    + DATE_FEATURES
    + HGB_ROUTE_FEATURES
    + GEOGRAPHIC_FEATURES
    + INTERACTION_FEATURES,
    "compact_calendar": [
        "distance",
        "weight",
        "equipment",
        "pickup",
        "delivery",
        "log_distance",
    ]
    + COMPACT_DATE_FEATURES,
}

DECEMBER_FEATURE_SETS: dict[str, list[str]] = {
    "december_basic": [
        "distance",
        "weight",
        "equipment",
        "pickup",
        "delivery",
    ]
    + COMPACT_DATE_FEATURES
    + ["log_distance"],
    "december_enriched_geo": [
        "distance",
        "weight",
        "equipment",
        "pickup",
        "delivery",
    ]
    + COMPACT_DATE_FEATURES
    + GEOGRAPHIC_FEATURES
    + ["log_distance"],
}

CATEGORICAL_FEATURES = {
    "pickup",
    "delivery",
    "equipment",
    "route",
    "pickup_equipment",
    "delivery_equipment",
    "route_equipment",
    "distance_band",
    "equipment_distance_band",
}


def _normalise_category(values: pd.Series) -> pd.Series:
    """Canonicalise whitespace and capitalisation, preserving missing as Unknown."""
    result = (
        values.astype("string")
        .fillna("Unknown")
        .str.strip()
        .str.replace(r"\s+", " ", regex=True)
        .str.title()
    )
    return result.mask(result.eq(""), "Unknown").astype(str)


def _federal_holidays_2025() -> set[pd.Timestamp]:
    return {
        pd.Timestamp(value)
        for value in [
            "2025-01-01",
            "2025-01-20",
            "2025-02-17",
            "2025-05-26",
            "2025-06-19",
            "2025-07-04",
            "2025-09-01",
            "2025-10-13",
            "2025-11-11",
            "2025-11-27",
            "2025-12-25",
        ]
    }


def _near_date(dates: pd.Series, anchors: Iterable[pd.Timestamp], days: int = 2) -> np.ndarray:
    result = np.zeros(len(dates), dtype=np.int8)
    for anchor in anchors:
        result |= (dates.sub(anchor).abs().dt.days <= days).to_numpy(dtype=np.int8)
    return result


@dataclass
class FeatureBuilder:
    """Fit cleaning statistics on training rows and reuse them at inference."""

    rare_threshold: int = 10
    impute_numeric: bool = False
    reference_date: pd.Timestamp | None = None
    numeric_medians: dict[str, float] = field(default_factory=dict)
    frequency_maps: dict[str, dict[str, float]] = field(default_factory=dict)
    count_maps: dict[str, dict[str, int]] = field(default_factory=dict)
    coordinate_map: dict[str, tuple[float, float]] = field(default_factory=dict)
    fitted_: bool = False

    def fit(self, frame: pd.DataFrame) -> "FeatureBuilder":
        base = self._base_clean(frame, fit=True)
        self.reference_date = base["date"].min()
        self.numeric_medians = {
            column: float(base[column].median())
            for column in RAW_NUMERIC
            if column in base and base[column].notna().any()
        }
        self._fit_coordinate_map(base)
        base = self._fill_numeric(base)
        base = self._add_categorical_composites(base)
        frequency_columns = [
            "pickup",
            "delivery",
            "route",
            "equipment",
            "pickup_equipment",
            "delivery_equipment",
            "route_equipment",
        ]
        denominator = max(len(base), 1)
        for column in frequency_columns:
            counts = base[column].value_counts(dropna=False)
            self.count_maps[column] = counts.astype(int).to_dict()
            self.frequency_maps[column] = (counts / denominator).astype(float).to_dict()
        self.fitted_ = True
        return self

    def transform(self, frame: pd.DataFrame, features: list[str]) -> pd.DataFrame:
        if not self.fitted_ or self.reference_date is None:
            raise RuntimeError("FeatureBuilder must be fitted before transform")
        base = self._base_clean(frame, fit=False)
        base = self._enrich_coordinates(base)
        base = self._fill_numeric(base)
        base = self._add_categorical_composites(base)
        base = self._add_date_features(base)
        base = self._add_geographic_features(base)
        base = self._add_interactions(base)
        base = self._add_frequency_features(base)
        missing = [column for column in features if column not in base]
        if missing:
            raise ValueError(f"Could not create required features: {missing}")
        output = base.loc[:, features].copy()
        for column in features:
            if column not in CATEGORICAL_FEATURES:
                continue
            output[column] = output[column].fillna("Unknown").astype(str)
        return output

    def fit_transform(self, frame: pd.DataFrame, features: list[str]) -> pd.DataFrame:
        return self.fit(frame).transform(frame, features)

    def _base_clean(self, frame: pd.DataFrame, fit: bool) -> pd.DataFrame:
        result = frame.copy()
        if "date" not in result:
            raise ValueError("Required column missing: date")
        result["date"] = pd.to_datetime(result["date"], errors="coerce")
        if result["date"].isna().any():
            label = "training" if fit else "inference"
            raise ValueError(f"Invalid dates found in {label} data")

        for column in RAW_CATEGORICAL:
            if column not in result:
                result[column] = "Unknown"
            result[column] = _normalise_category(result[column])

        for column in RAW_NUMERIC:
            if column not in result:
                result[column] = np.nan
            result[column] = pd.to_numeric(result[column], errors="coerce")

        result["weight_missing"] = result["weight"].isna().astype(np.int8)
        result["weight_sign_corrected"] = result["weight"].lt(0).fillna(False).astype(np.int8)
        result["market_index_missing"] = result["market_index"].isna().astype(np.int8)

        for column in ["distance", "weight", "market_index", "quote_signal"]:
            result.loc[result[column] <= 0, column] = np.nan
        for latitude in ["pickup_lat", "delivery_lat"]:
            result.loc[~result[latitude].between(-90, 90), latitude] = np.nan
        for longitude in ["pickup_lon", "delivery_lon"]:
            result.loc[~result[longitude].between(-180, 180), longitude] = np.nan
        return result

    def _fill_numeric(self, frame: pd.DataFrame) -> pd.DataFrame:
        result = frame.copy()
        if not self.impute_numeric:
            return result
        for column in RAW_NUMERIC:
            if column not in result:
                continue
            median = self.numeric_medians.get(column)
            if median is not None:
                result[column] = result[column].fillna(median)
        return result

    def _fit_coordinate_map(self, frame: pd.DataFrame) -> None:
        pickup = frame[["pickup", "pickup_lat", "pickup_lon"]].rename(
            columns={"pickup": "city", "pickup_lat": "lat", "pickup_lon": "lon"}
        )
        delivery = frame[["delivery", "delivery_lat", "delivery_lon"]].rename(
            columns={"delivery": "city", "delivery_lat": "lat", "delivery_lon": "lon"}
        )
        coordinates = pd.concat([pickup, delivery], ignore_index=True).dropna()
        medians = coordinates.groupby("city", observed=True)[["lat", "lon"]].median()
        self.coordinate_map = {
            str(city): (float(row["lat"]), float(row["lon"]))
            for city, row in medians.iterrows()
        }

    def _enrich_coordinates(self, frame: pd.DataFrame) -> pd.DataFrame:
        result = frame.copy()
        for role in ["pickup", "delivery"]:
            mapped_lat = result[role].map(lambda city: self.coordinate_map.get(city, (np.nan, np.nan))[0])
            mapped_lon = result[role].map(lambda city: self.coordinate_map.get(city, (np.nan, np.nan))[1])
            result[f"{role}_lat"] = result[f"{role}_lat"].fillna(mapped_lat)
            result[f"{role}_lon"] = result[f"{role}_lon"].fillna(mapped_lon)
        return result

    @staticmethod
    def _add_categorical_composites(frame: pd.DataFrame) -> pd.DataFrame:
        result = frame.copy()
        result["route"] = result["pickup"] + " > " + result["delivery"]
        result["pickup_equipment"] = result["pickup"] + " | " + result["equipment"]
        result["delivery_equipment"] = result["delivery"] + " | " + result["equipment"]
        result["route_equipment"] = result["route"] + " | " + result["equipment"]
        return result

    def _add_date_features(self, frame: pd.DataFrame) -> pd.DataFrame:
        result = frame.copy()
        dates = result["date"]
        result["year"] = dates.dt.year
        result["month"] = dates.dt.month
        result["quarter"] = dates.dt.quarter
        result["week_of_year"] = dates.dt.isocalendar().week.astype(int)
        result["day_of_month"] = dates.dt.day
        result["day_of_year"] = dates.dt.dayofyear
        result["day_of_week"] = dates.dt.dayofweek
        result["weekend"] = dates.dt.dayofweek.ge(5).astype(np.int8)
        result["month_start"] = dates.dt.is_month_start.astype(np.int8)
        result["month_end"] = dates.dt.is_month_end.astype(np.int8)
        result["days_since_start"] = dates.sub(self.reference_date).dt.days
        result["month_sin"] = np.sin(2 * np.pi * result["month"] / 12)
        result["month_cos"] = np.cos(2 * np.pi * result["month"] / 12)
        result["weekday_sin"] = np.sin(2 * np.pi * result["day_of_week"] / 7)
        result["weekday_cos"] = np.cos(2 * np.pi * result["day_of_week"] / 7)
        result["annual_sin"] = np.sin(2 * np.pi * result["day_of_year"] / 365.25)
        result["annual_cos"] = np.cos(2 * np.pi * result["day_of_year"] / 365.25)
        holidays = _federal_holidays_2025()
        result["federal_holiday"] = dates.dt.normalize().isin(holidays).astype(np.int8)
        result["near_thanksgiving"] = _near_date(dates, [pd.Timestamp("2025-11-27")])
        result["near_christmas"] = _near_date(dates, [pd.Timestamp("2025-12-25")])
        return result

    @staticmethod
    def _add_geographic_features(frame: pd.DataFrame) -> pd.DataFrame:
        result = frame.copy()
        lat1 = np.radians(result["pickup_lat"])
        lat2 = np.radians(result["delivery_lat"])
        delta_lat = lat2 - lat1
        delta_lon = np.radians(result["delivery_lon"] - result["pickup_lon"])
        haversine_a = (
            np.sin(delta_lat / 2) ** 2
            + np.cos(lat1) * np.cos(lat2) * np.sin(delta_lon / 2) ** 2
        ).clip(0, 1)
        angular = 2 * np.arctan2(np.sqrt(haversine_a), np.sqrt(1 - haversine_a))
        result["haversine_distance"] = 3958.7613 * angular
        result["distance_haversine_difference"] = result["distance"] - result["haversine_distance"]
        result["distance_haversine_ratio"] = result["distance"] / result["haversine_distance"].clip(lower=1.0)
        result["latitude_difference"] = result["delivery_lat"] - result["pickup_lat"]
        result["longitude_difference"] = result["delivery_lon"] - result["pickup_lon"]
        result["absolute_latitude_difference"] = result["latitude_difference"].abs()
        result["absolute_longitude_difference"] = result["longitude_difference"].abs()
        result["midpoint_latitude"] = (result["pickup_lat"] + result["delivery_lat"]) / 2
        result["midpoint_longitude"] = (result["pickup_lon"] + result["delivery_lon"]) / 2
        y_component = np.sin(delta_lon) * np.cos(lat2)
        x_component = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(delta_lon)
        bearing = np.arctan2(y_component, x_component)
        result["bearing_sin"] = np.sin(bearing)
        result["bearing_cos"] = np.cos(bearing)
        result["distance_band"] = pd.cut(
            result["distance"],
            bins=[-np.inf, 250, 800, 1600, np.inf],
            labels=["Short", "Medium", "Long", "Very Long"],
        ).astype(str)
        return result

    @staticmethod
    def _add_interactions(frame: pd.DataFrame) -> pd.DataFrame:
        result = frame.copy()
        result["log_distance"] = np.log1p(result["distance"])
        result["log_weight"] = np.log1p(result["weight"])
        result["distance_squared_scaled"] = result["distance"].pow(2) / 1_000.0
        result["weight_to_distance"] = result["weight"] / result["distance"].clip(lower=1.0)
        result["distance_market_index"] = result["distance"] * result["market_index"]
        result["distance_quote_signal"] = result["distance"] * result["quote_signal"]
        result["market_quote_interaction"] = result["market_index"] * result["quote_signal"]
        if "distance_band" not in result:
            result["distance_band"] = pd.cut(
                result["distance"],
                bins=[-np.inf, 250, 800, 1600, np.inf],
                labels=["Short", "Medium", "Long", "Very Long"],
            ).astype(str)
        result["equipment_distance_band"] = result["equipment"] + " | " + result["distance_band"]
        return result

    def _add_frequency_features(self, frame: pd.DataFrame) -> pd.DataFrame:
        result = frame.copy()
        for column, mapping in self.frequency_maps.items():
            result[f"{column}_frequency"] = result[column].map(mapping).fillna(0.0)
            counts = result[column].map(self.count_maps[column]).fillna(0).astype(int)
            if column in {"pickup", "delivery", "route"}:
                result[f"rare_{column}"] = counts.between(1, self.rare_threshold - 1).astype(np.int8)
                result[f"unseen_{column}"] = counts.eq(0).astype(np.int8)
        return result


def categorical_columns(features: list[str]) -> list[str]:
    """Return CatBoost categorical columns in stable feature order."""
    return [column for column in features if column in CATEGORICAL_FEATURES]
