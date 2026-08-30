"""Compartment-aware interpolation helpers."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from shapely.geometry import Point

from core.geometry.masking import polygon_keep_mask
from core.geometry.models import GeometryLayer
from core.interpolation import InterpolationError, interpolate_surface_result


@dataclass
class CompartmentInterpolationResult:
    surface: np.ndarray
    variance: np.ndarray | None = None
    panel_grid: np.ndarray | None = None
    warnings: list[str] = field(default_factory=list)


def observations_in_feature(prepared_df: pd.DataFrame, geometry) -> pd.DataFrame:
    if prepared_df.empty:
        return prepared_df.copy()
    mask = []
    for _, row in prepared_df.iterrows():
        x = float(row["X"])
        y = float(row["Y"])
        mask.append(bool(np.isfinite(x) and np.isfinite(y) and geometry.covers(Point(x, y))))
    return prepared_df.loc[mask].copy()


def compartment_interpolate(
    prepared_df: pd.DataFrame,
    panel_layer: GeometryLayer,
    grid_x: np.ndarray,
    grid_y: np.ndarray,
    method: str,
    method_parameters: dict | None = None,
    min_observations: int = 3,
) -> CompartmentInterpolationResult:
    surface = np.full_like(grid_x, np.nan, dtype=float)
    variance = np.full_like(grid_x, np.nan, dtype=float)
    has_variance = False
    panel_grid = np.full(grid_x.shape, None, dtype=object)
    warnings: list[str] = []

    for feature in panel_layer.polygon_features:
        panel_points = observations_in_feature(prepared_df, feature.geometry)
        if len(panel_points) < min_observations:
            warnings.append(
                f"{feature.name} has insufficient observations for {method} interpolation "
                f"({len(panel_points)} available, {min_observations} required)."
            )
            continue
        try:
            panel_result = interpolate_surface_result(
                panel_points["X"],
                panel_points["Y"],
                panel_points["Z"],
                grid_x,
                grid_y,
                method,
                method_parameters or {},
            )
        except InterpolationError as exc:
            warnings.append(f"{feature.name}: {exc}")
            continue
        keep = polygon_keep_mask([feature], grid_x, grid_y)
        surface[keep] = panel_result.estimate[keep]
        if panel_result.variance is not None:
            variance[keep] = panel_result.variance[keep]
            has_variance = True
        panel_grid[keep] = feature.name

    return CompartmentInterpolationResult(
        surface=surface,
        variance=variance if has_variance else None,
        panel_grid=panel_grid,
        warnings=warnings,
    )
