"""Small validation and data-interpretation helpers."""

from __future__ import annotations

import re
from typing import Iterable

import numpy as np
import pandas as pd


def finite_series(series: pd.Series) -> pd.Series:
    """Return a numeric series with non-finite values converted to NaN."""

    numeric = pd.to_numeric(series, errors="coerce")
    return numeric.where(np.isfinite(numeric), np.nan)


def is_pressure_property(name: str | None) -> bool:
    if not name:
        return False
    normalized = re.sub(r"[^A-Z0-9]+", "", name.upper())
    pressure_names = {
        "PRES",
        "PRESS",
        "PRESSURE",
        "AVG_PRESSURE",
        "AVERAGE_PRESSURE",
        "RESERVOIR_PRESSURE",
        "PRESSURE_AT_MAP_DATE",
        "EXTRAPOLATED_PRESSURE",
        "PSI",
        "BHP",
        "SBHP",
        "WHP",
    }
    return "PRESSURE" in normalized or normalized in {re.sub(r"[^A-Z0-9]+", "", value) for value in pressure_names}


def is_phi_property(name: str | None) -> bool:
    if not name:
        return False
    normalized = re.sub(r"[^A-Z0-9]+", "", name.upper())
    return normalized in {"PHI", "POR", "POROSITY"} or "POROSITY" in normalized


def skewness_is_high(values: Iterable[float], threshold: float = 2.0) -> bool:
    series = pd.Series(values, dtype="float64").replace([np.inf, -np.inf], np.nan).dropna()
    if len(series) < 8 or series.nunique() < 3:
        return False
    skewness = float(series.skew())
    return skewness > threshold


def non_collinear_points(x: Iterable[float], y: Iterable[float]) -> bool:
    points = np.column_stack([np.asarray(x, dtype=float), np.asarray(y, dtype=float)])
    points = points[np.isfinite(points).all(axis=1)]
    if len(points) < 3:
        return False
    centered = points - points.mean(axis=0)
    return np.linalg.matrix_rank(centered) >= 2
