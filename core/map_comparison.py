"""Saved-map comparison, grid alignment, and delta calculations."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import numpy as np
from scipy.interpolate import griddata

from core.crs import crs_are_compatible
from core.pressure_dates import format_map_date


PRESSURE_CHANGE_DISABLED_MESSAGE = (
    "Pressure Change requires two pressure scenarios with different Pressure Map Reference Dates. "
    "Use Map B - Map A to compare methods or scenarios at the same date."
)


@dataclass(frozen=True)
class CompatibilityReport:
    side_by_side_allowed: bool
    delta_allowed: bool
    issues: list[str]
    warnings: list[str]


@dataclass(frozen=True)
class DeltaResult:
    grid_x: np.ndarray
    grid_y: np.ndarray
    grid_z: np.ndarray
    metadata: dict[str, object]


def grids_are_identical(first: dict[str, object], second: dict[str, object], tolerance: float = 1e-9) -> bool:
    return (
        np.asarray(first["grid_x"]).shape == np.asarray(second["grid_x"]).shape
        and np.asarray(first["grid_y"]).shape == np.asarray(second["grid_y"]).shape
        and np.allclose(np.asarray(first["grid_x"], dtype=float), np.asarray(second["grid_x"], dtype=float), atol=tolerance, rtol=0)
        and np.allclose(np.asarray(first["grid_y"], dtype=float), np.asarray(second["grid_y"], dtype=float), atol=tolerance, rtol=0)
    )


def grid_spacing(scenario: dict[str, object]) -> tuple[float | None, float | None]:
    grid_x = np.asarray(scenario["grid_x"], dtype=float)
    grid_y = np.asarray(scenario["grid_y"], dtype=float)
    dx = float(np.nanmedian(np.diff(grid_x[0, :]))) if grid_x.shape[1] > 1 else None
    dy = float(np.nanmedian(np.diff(grid_y[:, 0]))) if grid_y.shape[0] > 1 else None
    return dx, dy


def grid_extent(scenario: dict[str, object]) -> tuple[float, float, float, float]:
    grid_x = np.asarray(scenario["grid_x"], dtype=float)
    grid_y = np.asarray(scenario["grid_y"], dtype=float)
    return float(np.nanmin(grid_x)), float(np.nanmax(grid_x)), float(np.nanmin(grid_y)), float(np.nanmax(grid_y))


def grid_definition(scenario: dict[str, object]) -> dict[str, object]:
    dx, dy = grid_spacing(scenario)
    x_min, x_max, y_min, y_max = grid_extent(scenario)
    grid_z = np.asarray(scenario["grid_z"], dtype=float)
    return {
        "shape": list(grid_z.shape),
        "extent": {"x_min": x_min, "x_max": x_max, "y_min": y_min, "y_max": y_max},
        "dx": dx,
        "dy": dy,
        "valid_cells": int(np.isfinite(grid_z).sum()),
    }


def compatibility_report(first: dict[str, object], second: dict[str, object]) -> CompatibilityReport:
    issues: list[str] = []
    warnings: list[str] = []
    if first.get("coordinate_unit") != second.get("coordinate_unit"):
        issues.append("Coordinate units differ.")
    if not crs_are_compatible(first.get("crs"), second.get("crs")):
        issues.append("Coordinate reference systems differ.")
    if first.get("property") != second.get("property"):
        issues.append("Properties differ.")
    if (first.get("property_unit") or "") != (second.get("property_unit") or ""):
        issues.append("Property units differ.")
    if not grids_are_identical(first, second):
        warnings.append("Grid alignment required before arithmetic.")
    if first.get("geometry_context") != second.get("geometry_context"):
        warnings.append("Geometry context differs; review before interpreting the comparison.")
    return CompatibilityReport(
        side_by_side_allowed=True,
        delta_allowed=not issues,
        issues=issues,
        warnings=warnings,
    )


def resample_to_grid(source: dict[str, object], target_grid_x: np.ndarray, target_grid_y: np.ndarray) -> np.ndarray:
    source_x = np.asarray(source["grid_x"], dtype=float)
    source_y = np.asarray(source["grid_y"], dtype=float)
    source_z = np.asarray(source["grid_z"], dtype=float)
    finite = np.isfinite(source_x) & np.isfinite(source_y) & np.isfinite(source_z)
    if finite.sum() < 3:
        raise ValueError("Source map has too few finite grid cells for alignment.")
    return np.asarray(
        griddata(
            np.column_stack([source_x[finite], source_y[finite]]),
            source_z[finite],
            (target_grid_x, target_grid_y),
            method="linear",
        ),
        dtype=float,
    )


def resample_support_mask(source: dict[str, object], target_grid_x: np.ndarray, target_grid_y: np.ndarray) -> np.ndarray:
    source_x = np.asarray(source["grid_x"], dtype=float)
    source_y = np.asarray(source["grid_y"], dtype=float)
    source_z = np.asarray(source["grid_z"], dtype=float)
    finite_xy = np.isfinite(source_x) & np.isfinite(source_y)
    if finite_xy.sum() < 3:
        return np.zeros_like(target_grid_x, dtype=bool)
    support = griddata(
        np.column_stack([source_x[finite_xy], source_y[finite_xy]]),
        np.isfinite(source_z[finite_xy]).astype(float),
        (target_grid_x, target_grid_y),
        method="linear",
    )
    return np.isfinite(support) & (support >= 0.999)


def delta_support_mask(
    map_a: dict[str, object],
    map_b: dict[str, object],
    aligned_b_z: np.ndarray,
) -> np.ndarray:
    a_z = np.asarray(map_a["grid_z"], dtype=float)
    a_support = np.isfinite(a_z)
    if grids_are_identical(map_a, map_b):
        b_support = np.isfinite(np.asarray(map_b["grid_z"], dtype=float))
    else:
        b_support = resample_support_mask(map_b, np.asarray(map_a["grid_x"], dtype=float), np.asarray(map_a["grid_y"], dtype=float))
    return a_support & b_support & np.isfinite(aligned_b_z)


def calculate_delta(
    map_a: dict[str, object],
    map_b: dict[str, object],
    operation: str = "Map B - Map A",
) -> DeltaResult:
    report = compatibility_report(map_a, map_b)
    if not report.delta_allowed:
        raise ValueError("Delta map cannot be calculated: " + "; ".join(report.issues))
    grid_x = np.asarray(map_a["grid_x"], dtype=float)
    grid_y = np.asarray(map_a["grid_y"], dtype=float)
    a_z = np.asarray(map_a["grid_z"], dtype=float)
    if grids_are_identical(map_a, map_b):
        b_z = np.asarray(map_b["grid_z"], dtype=float)
        alignment = "Direct subtraction on identical grids"
        resampling = "none"
    else:
        b_z = resample_to_grid(map_b, grid_x, grid_y)
        alignment = "Map B resampled to Map A grid"
        resampling = "linear"
    valid_delta = delta_support_mask(map_a, map_b, b_z)
    if not np.any(valid_delta):
        raise ValueError("Delta map has no overlapping valid support after grid alignment and mask intersection.")
    delta = np.full_like(a_z, np.nan, dtype=float)
    delta[valid_delta] = b_z[valid_delta] - a_z[valid_delta]
    metadata = {
        "Source_Map_A": map_a.get("name"),
        "Source_Map_B": map_b.get("name"),
        "Operation": operation,
        "Property": map_a.get("property"),
        "Property_Unit": map_a.get("property_unit") or "",
        "Date_A": map_a.get("pressure_reference_date") or "",
        "Date_B": map_b.get("pressure_reference_date") or "",
        "Grid_Alignment": alignment,
        "Grid_A_Definition": grid_definition(map_a),
        "Grid_B_Definition": grid_definition(map_b),
        "Target_Grid_Definition": grid_definition(map_a),
        "Alignment_Resampling_Method": resampling,
        "Mask_Intersection_Rule": "Delta is finite only where aligned Map A and Map B are both finite.",
        "Valid_Delta_Cells": int(np.isfinite(delta).sum()),
        "Panel_Selection_A": map_a.get("selected_panels", []),
        "Panel_Selection_B": map_b.get("selected_panels", []),
        "Panel_Interpolation_Mode_A": map_a.get("panel_interpolation_mode", ""),
        "Panel_Interpolation_Mode_B": map_b.get("panel_interpolation_mode", ""),
        "Selected_Layers_A": map_a.get("selected_layers", []),
        "Selected_Layers_B": map_b.get("selected_layers", []),
    }
    return DeltaResult(grid_x=grid_x, grid_y=grid_y, grid_z=delta, metadata=metadata)


def _parse_date(value) -> date | None:
    if value in (None, ""):
        return None
    try:
        return date.fromisoformat(str(value)[:10])
    except ValueError:
        return None


def _is_pressure_scenario(scenario: dict[str, object]) -> bool:
    property_type = str(scenario.get("property_type") or "").strip().lower()
    property_name = str(scenario.get("property") or scenario.get("property_col") or "").strip().lower()
    return bool(scenario.get("is_pressure_map")) or property_type == "pressure" or property_name == "pressure"


def pressure_change_allowed(first: dict[str, object], second: dict[str, object]) -> tuple[bool, str]:
    if not (_is_pressure_scenario(first) and _is_pressure_scenario(second)):
        return False, PRESSURE_CHANGE_DISABLED_MESSAGE
    first_date = _parse_date(first.get("pressure_reference_date") or first.get("map_reference_date"))
    second_date = _parse_date(second.get("pressure_reference_date") or second.get("map_reference_date"))
    if first_date is None or second_date is None:
        return False, PRESSURE_CHANGE_DISABLED_MESSAGE
    if first_date == second_date:
        return False, PRESSURE_CHANGE_DISABLED_MESSAGE
    return True, ""


def pressure_date_order(first: dict[str, object], second: dict[str, object]) -> tuple[dict[str, object], dict[str, object]] | None:
    first_date = _parse_date(first.get("pressure_reference_date") or first.get("map_reference_date"))
    second_date = _parse_date(second.get("pressure_reference_date") or second.get("map_reference_date"))
    if first_date is None or second_date is None:
        return None
    return (first, second) if first_date <= second_date else (second, first)


def calculate_pressure_change(first: dict[str, object], second: dict[str, object]) -> DeltaResult:
    allowed, message = pressure_change_allowed(first, second)
    if not allowed:
        raise ValueError(message)
    ordered = pressure_date_order(first, second)
    if ordered is None:
        raise ValueError(PRESSURE_CHANGE_DISABLED_MESSAGE)
    earlier, later = ordered
    earlier_date = earlier.get("pressure_reference_date") or earlier.get("map_reference_date") or ""
    later_date = later.get("pressure_reference_date") or later.get("map_reference_date") or ""
    label = f"Pressure Change {format_map_date(later_date)} minus {format_map_date(earlier_date)}"
    result = calculate_delta(earlier, later, operation=label)
    result.metadata["Source_Map_Earlier"] = earlier.get("name")
    result.metadata["Source_Map_Later"] = later.get("name")
    result.metadata["Date_Earlier"] = earlier_date
    result.metadata["Date_Later"] = later_date
    result.metadata["Pressure_Change_Label"] = label
    result.metadata["Delta_P_Convention"] = "Later pressure map minus earlier pressure map."
    return result


def symmetric_delta_range(values) -> tuple[float, float] | None:
    array = np.asarray(values, dtype=float)
    valid = array[np.isfinite(array)]
    if valid.size == 0:
        return None
    limit = float(np.nanmax(np.abs(valid)))
    if limit == 0:
        limit = 1.0
    return -limit, limit
