"""Compartment-aware interpolation helpers."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from shapely.geometry import Point
from shapely.ops import unary_union

from core.active_data import PANEL_MODE_COMBINED, PANEL_MODE_INDEPENDENT, panel_mode_from_legacy
from core.geometry.validation import detect_polygon_overlaps
from core.geometry.masking import polygon_keep_mask
from core.geometry.models import GeometryFeature, GeometryLayer
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


def panel_feature_names(panel_layer: GeometryLayer | None) -> list[str]:
    if panel_layer is None:
        return []
    names: list[str] = []
    for feature in panel_layer.polygon_features:
        if feature.name not in names:
            names.append(feature.name)
    return names


def selected_panel_features(
    panel_layer: GeometryLayer | None,
    selected_panels: list[object] | tuple[object, ...] | None = None,
) -> list[GeometryFeature]:
    if panel_layer is None:
        return []
    features = panel_layer.polygon_features
    selected = {str(value) for value in (selected_panels or []) if value not in (None, "")}
    if not selected:
        return features
    return [feature for feature in features if str(feature.name) in selected]


def selected_panel_union(
    panel_layer: GeometryLayer | None,
    selected_panels: list[object] | tuple[object, ...] | None = None,
):
    features = selected_panel_features(panel_layer, selected_panels)
    if not features:
        return None
    return unary_union([feature.geometry for feature in features])


def selected_panel_bounds(
    panel_layer: GeometryLayer | None,
    selected_panels: list[object] | tuple[object, ...] | None = None,
) -> tuple[float, float, float, float] | None:
    geometry = selected_panel_union(panel_layer, selected_panels)
    if geometry is None or geometry.is_empty:
        return None
    return tuple(float(value) for value in geometry.bounds)


def panel_domain_keep_mask(
    panel_layer: GeometryLayer | None,
    selected_panels: list[object] | tuple[object, ...] | None,
    grid_x: np.ndarray,
    grid_y: np.ndarray,
) -> np.ndarray:
    features = selected_panel_features(panel_layer, selected_panels)
    if not features:
        return np.ones_like(grid_x, dtype=bool)
    return polygon_keep_mask(features, grid_x, grid_y)


def filter_dataframe_to_selected_panels(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    panel_layer: GeometryLayer | None,
    selected_panels: list[object] | tuple[object, ...] | None = None,
) -> pd.DataFrame:
    features = selected_panel_features(panel_layer, selected_panels)
    if not features or not selected_panels:
        return df.copy()
    keep: list[bool] = []
    for _, row in df.iterrows():
        x = pd.to_numeric(pd.Series([row.get(x_col)]), errors="coerce").iloc[0]
        y = pd.to_numeric(pd.Series([row.get(y_col)]), errors="coerce").iloc[0]
        if pd.isna(x) or pd.isna(y):
            keep.append(False)
            continue
        point = Point(float(x), float(y))
        keep.append(any(feature.geometry.covers(point) for feature in features))
    return df.loc[keep].copy()


def assert_independent_panels_do_not_overlap(
    panel_layer: GeometryLayer | None,
    selected_panels: list[object] | tuple[object, ...] | None = None,
) -> None:
    features = selected_panel_features(panel_layer, selected_panels)
    if not features:
        return
    layer = GeometryLayer(
        name=panel_layer.name if panel_layer is not None else "Selected Panels",
        layer_type="Panel / Compartment",
        features=features,
        source_name=panel_layer.source_name if panel_layer is not None else "",
        name_attribute=panel_layer.name_attribute if panel_layer is not None else None,
    )
    overlaps = detect_polygon_overlaps(layer)
    if overlaps:
        details = "; ".join(f"{item.first} / {item.second}" for item in overlaps[:3])
        raise ValueError(
            "Selected panel polygons overlap. Independent compartment interpolation "
            f"requires non-overlapping compartment geometry. Overlap: {details}."
        )


def _explicit_panel_label(value, valid_panels: set[str]) -> str | None:
    if value in (None, "") or pd.isna(value):
        return None
    labels = [item.strip() for item in str(value).split("|") if item.strip()]
    if len(labels) != 1:
        return None
    return labels[0] if labels[0] in valid_panels else None


def assign_prepared_observations_to_panels(
    prepared_df: pd.DataFrame,
    panel_layer: GeometryLayer,
    selected_panels: list[object] | tuple[object, ...] | None = None,
    dataset_panel_col: str | None = None,
) -> tuple[pd.Series, list[str]]:
    features = selected_panel_features(panel_layer, selected_panels)
    valid_panels = {feature.name for feature in features}
    labels: list[str | None] = []
    warnings: list[str] = []

    for _, row in prepared_df.iterrows():
        explicit = (
            _explicit_panel_label(row.get(dataset_panel_col), valid_panels)
            if dataset_panel_col and dataset_panel_col in prepared_df.columns
            else None
        )
        if explicit is not None:
            labels.append(explicit)
            continue

        x = float(row["X"])
        y = float(row["Y"])
        point = Point(x, y)
        matches = [feature.name for feature in features if feature.geometry.covers(point)]
        if len(matches) == 1:
            labels.append(matches[0])
        elif len(matches) > 1:
            labels.append(None)
            warnings.append(
                f"Observation at X={x:.6g}, Y={y:.6g} lies on multiple panel polygons and was excluded from independent interpolation."
            )
        else:
            labels.append(None)

    return pd.Series(labels, index=prepared_df.index, dtype=object), warnings


def compartment_interpolate(
    prepared_df: pd.DataFrame,
    panel_layer: GeometryLayer,
    grid_x: np.ndarray,
    grid_y: np.ndarray,
    method: str,
    method_parameters: dict | None = None,
    min_observations: int = 3,
    selected_panels: list[object] | tuple[object, ...] | None = None,
    dataset_panel_col: str | None = None,
) -> CompartmentInterpolationResult:
    assert_independent_panels_do_not_overlap(panel_layer, selected_panels)
    surface = np.full_like(grid_x, np.nan, dtype=float)
    variance = np.full_like(grid_x, np.nan, dtype=float)
    has_variance = False
    panel_grid = np.full(grid_x.shape, None, dtype=object)
    warnings: list[str] = []
    panel_labels, assignment_warnings = assign_prepared_observations_to_panels(
        prepared_df,
        panel_layer,
        selected_panels=selected_panels,
        dataset_panel_col=dataset_panel_col,
    )
    warnings.extend(assignment_warnings)

    for feature in selected_panel_features(panel_layer, selected_panels):
        panel_points = prepared_df.loc[panel_labels == feature.name].copy()
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


def is_independent_panel_mode(panel_interpolation_mode: str | None) -> bool:
    return panel_mode_from_legacy(panel_interpolation_mode) == PANEL_MODE_INDEPENDENT


def is_combined_panel_mode(panel_interpolation_mode: str | None) -> bool:
    return panel_mode_from_legacy(panel_interpolation_mode) == PANEL_MODE_COMBINED
