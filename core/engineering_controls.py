"""Engineering control points and soft control regions."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Any

import numpy as np
import pandas as pd
from shapely.geometry import Point, mapping, shape
from shapely.geometry.base import BaseGeometry

from core.active_data import PANEL_MODE_INDEPENDENT, parse_reference_date, panel_mode_from_legacy
from core.geometry.compartment import selected_panel_features, selected_panel_union
from core.geometry.masking import polygon_union
from core.geometry.models import GeometryLayer
from utils.constants import INCLUDE_COLUMN


CONTROL_TYPE = "Engineering Control"
MANUAL_CONTROL_SOURCE = "Manual Control Point"
REGION_CONTROL_SOURCE = "Soft Control Region"
CONTROL_REGION_TYPE = "Soft Control Region"
REGION_CONTROL_WARNING_THRESHOLD = 1000
REGION_CONTROL_HARD_LIMIT = 5000

CONTROL_POINT_COLUMNS = [
    "Active",
    "Control_ID",
    "X",
    "Y",
    "Property",
    "Value",
    "Property_Unit",
    "Reservoir_Layer",
    "Panel",
    "Pressure_Map_Reference_Date",
    "Comment",
    "Source_Type",
]

CONTROL_REGION_COLUMNS = [
    "Active",
    "Region_ID",
    "Region_Name",
    "Property",
    "Target_Value",
    "Property_Unit",
    "Reservoir_Layer",
    "Panel",
    "Pressure_Map_Reference_Date",
    "Control_Point_Spacing",
    "Generated_Control_Count",
    "Comment",
]


@dataclass(frozen=True)
class ControlSelection:
    dataframe: pd.DataFrame
    manual_points: list[dict[str, Any]]
    region_points: list[dict[str, Any]]
    regions: list[dict[str, Any]]
    warnings: tuple[str, ...]
    signature_state: dict[str, object]

    @property
    def control_count(self) -> int:
        return len(self.dataframe)


def iso_date_or_blank(value: Any) -> str:
    parsed = parse_reference_date(value)
    return parsed.isoformat() if parsed else ""


def _next_identifier(existing: list[dict[str, Any]], key: str, prefix: str) -> str:
    used = {str(item.get(key, "")) for item in existing}
    index = 1
    while True:
        candidate = f"{prefix}-{index:03d}"
        if candidate not in used:
            return candidate
        index += 1


def next_control_id(existing: list[dict[str, Any]] | None = None) -> str:
    return _next_identifier(existing or [], "Control_ID", "CP")


def next_region_id(existing: list[dict[str, Any]] | None = None) -> str:
    return _next_identifier(existing or [], "Region_ID", "CR")


def normalize_control_point(control: dict[str, Any]) -> dict[str, Any]:
    normalized = {column: control.get(column) for column in CONTROL_POINT_COLUMNS}
    normalized["Control_ID"] = str(normalized.get("Control_ID") or "")
    normalized["Active"] = bool(normalized.get("Active", True))
    normalized["Source_Type"] = str(normalized.get("Source_Type") or MANUAL_CONTROL_SOURCE)
    normalized["Property"] = str(normalized.get("Property") or "")
    normalized["Property_Unit"] = str(normalized.get("Property_Unit") or "")
    normalized["Reservoir_Layer"] = str(normalized.get("Reservoir_Layer") or "")
    normalized["Panel"] = str(normalized.get("Panel") or "")
    normalized["Pressure_Map_Reference_Date"] = iso_date_or_blank(normalized.get("Pressure_Map_Reference_Date"))
    normalized["Comment"] = str(normalized.get("Comment") or "")
    for key in ("X", "Y", "Value"):
        normalized[key] = pd.to_numeric(pd.Series([normalized.get(key)]), errors="coerce").iloc[0]
    if control.get("Region_ID") not in (None, ""):
        normalized["Region_ID"] = str(control.get("Region_ID"))
    if control.get("Region_Name") not in (None, ""):
        normalized["Region_Name"] = str(control.get("Region_Name"))
    normalized["Control_Type"] = CONTROL_TYPE
    return normalized


def create_control_point(
    *,
    x: float,
    y: float,
    property_name: str,
    value: float,
    property_unit: str | None = "",
    reservoir_layer: str | None = None,
    panel: str | None = None,
    pressure_reference_date: Any = None,
    active: bool = True,
    comment: str = "",
    source_type: str = MANUAL_CONTROL_SOURCE,
    control_id: str | None = None,
    existing_controls: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    return normalize_control_point(
        {
            "Control_ID": control_id or next_control_id(existing_controls),
            "X": x,
            "Y": y,
            "Property": property_name,
            "Value": value,
            "Property_Unit": property_unit or "",
            "Reservoir_Layer": reservoir_layer or "",
            "Panel": panel or "",
            "Pressure_Map_Reference_Date": iso_date_or_blank(pressure_reference_date),
            "Active": active,
            "Comment": comment,
            "Source_Type": source_type,
        }
    )


def _geometry_from_region(region: dict[str, Any]) -> BaseGeometry | None:
    geometry = region.get("Geometry")
    if isinstance(geometry, BaseGeometry):
        return geometry
    if isinstance(geometry, dict):
        try:
            return shape(geometry)
        except Exception:
            return None
    return None


def normalize_control_region(region: dict[str, Any]) -> dict[str, Any]:
    geometry = _geometry_from_region(region)
    normalized = {
        "Active": bool(region.get("Active", True)),
        "Region_ID": str(region.get("Region_ID") or ""),
        "Region_Name": str(region.get("Region_Name") or ""),
        "Geometry": geometry,
        "Property": str(region.get("Property") or ""),
        "Target_Value": pd.to_numeric(pd.Series([region.get("Target_Value")]), errors="coerce").iloc[0],
        "Property_Unit": str(region.get("Property_Unit") or ""),
        "Reservoir_Layer": str(region.get("Reservoir_Layer") or ""),
        "Panel": str(region.get("Panel") or ""),
        "Pressure_Map_Reference_Date": iso_date_or_blank(region.get("Pressure_Map_Reference_Date")),
        "Control_Point_Spacing": pd.to_numeric(
            pd.Series([region.get("Control_Point_Spacing")]),
            errors="coerce",
        ).iloc[0],
        "Generated_Control_Count": int(region.get("Generated_Control_Count") or 0),
        "Comment": str(region.get("Comment") or ""),
    }
    return normalized


def create_control_region(
    *,
    region_name: str,
    geometry: BaseGeometry,
    property_name: str,
    target_value: float,
    property_unit: str | None = "",
    reservoir_layer: str | None = None,
    panel: str | None = None,
    pressure_reference_date: Any = None,
    control_point_spacing: float = 250.0,
    active: bool = True,
    comment: str = "",
    region_id: str | None = None,
    existing_regions: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    return normalize_control_region(
        {
            "Region_ID": region_id or next_region_id(existing_regions),
            "Region_Name": region_name,
            "Geometry": geometry,
            "Property": property_name,
            "Target_Value": target_value,
            "Property_Unit": property_unit or "",
            "Reservoir_Layer": reservoir_layer or "",
            "Panel": panel or "",
            "Pressure_Map_Reference_Date": iso_date_or_blank(pressure_reference_date),
            "Control_Point_Spacing": control_point_spacing,
            "Active": active,
            "Comment": comment,
        }
    )


def serialize_control_point(control: dict[str, Any]) -> dict[str, Any]:
    point = normalize_control_point(control)
    return {key: _jsonable(value) for key, value in point.items() if key in point}


def serialize_control_points(controls: list[dict[str, Any]] | tuple[dict[str, Any], ...] | None) -> list[dict[str, Any]]:
    return [serialize_control_point(control) for control in controls or []]


def deserialize_control_points(controls: list[dict[str, Any]] | tuple[dict[str, Any], ...] | None) -> list[dict[str, Any]]:
    return [normalize_control_point(control) for control in controls or []]


def serialize_control_region(region: dict[str, Any]) -> dict[str, Any]:
    normalized = normalize_control_region(region)
    output = {key: _jsonable(value) for key, value in normalized.items() if key != "Geometry"}
    geometry = normalized.get("Geometry")
    output["Geometry"] = mapping(geometry) if isinstance(geometry, BaseGeometry) else None
    return output


def serialize_control_regions(regions: list[dict[str, Any]] | tuple[dict[str, Any], ...] | None) -> list[dict[str, Any]]:
    return [serialize_control_region(region) for region in regions or []]


def deserialize_control_regions(regions: list[dict[str, Any]] | tuple[dict[str, Any], ...] | None) -> list[dict[str, Any]]:
    return [normalize_control_region(region) for region in regions or []]


def controls_dataframe(controls: list[dict[str, Any]] | tuple[dict[str, Any], ...] | None) -> pd.DataFrame:
    rows = [normalize_control_point(control) for control in controls or []]
    return pd.DataFrame(rows, columns=CONTROL_POINT_COLUMNS)


def regions_dataframe(regions: list[dict[str, Any]] | tuple[dict[str, Any], ...] | None) -> pd.DataFrame:
    rows = []
    for region in regions or []:
        normalized = normalize_control_region(region)
        rows.append({column: normalized.get(column) for column in CONTROL_REGION_COLUMNS})
    return pd.DataFrame(rows, columns=CONTROL_REGION_COLUMNS)


def _jsonable(value: Any) -> Any:
    if isinstance(value, (date, pd.Timestamp)):
        return iso_date_or_blank(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if not isinstance(value, (list, tuple, dict, BaseGeometry)):
        try:
            if pd.isna(value):
                return None
        except (TypeError, ValueError):
            pass
    return value


def _text(value: Any) -> str:
    if value in (None, ""):
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    return str(value)


def _finite_float(value: Any) -> float | None:
    number = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(number) or not np.isfinite(float(number)):
        return None
    return float(number)


def assign_panel_from_point(
    x: float,
    y: float,
    panel_layer: GeometryLayer | None,
    selected_panels: list[object] | tuple[object, ...] | None = None,
) -> tuple[str | None, str | None]:
    if panel_layer is None:
        return None, None
    point = Point(float(x), float(y))
    matches = [feature.name for feature in selected_panel_features(panel_layer, selected_panels) if feature.geometry.covers(point)]
    if len(matches) == 1:
        return str(matches[0]), None
    if len(matches) > 1:
        return None, "Control point lies on an ambiguous panel boundary and was not assigned to multiple panels."
    return None, "Control point is outside the selected panel polygons."


def create_region_from_wells(
    wells: pd.DataFrame,
    x_col: str,
    y_col: str,
    buffer_distance: float = 0.0,
) -> BaseGeometry:
    points = [
        Point(float(row[x_col]), float(row[y_col]))
        for _, row in wells.dropna(subset=[x_col, y_col]).iterrows()
        if np.isfinite(float(row[x_col])) and np.isfinite(float(row[y_col]))
    ]
    if len(points) < 3:
        raise ValueError("At least three finite well locations are required to create a control region.")
    geometry = pd.Series(points).tolist()
    from shapely.geometry import MultiPoint

    hull = MultiPoint(geometry).convex_hull
    if buffer_distance:
        hull = hull.buffer(float(buffer_distance))
    if hull.is_empty or hull.geom_type not in {"Polygon", "MultiPolygon"}:
        raise ValueError("Selected wells do not form a polygonal control region. Add a buffer or choose different wells.")
    return hull


def _regular_points_in_geometry(geometry: BaseGeometry, spacing: float) -> list[Point]:
    if spacing <= 0:
        raise ValueError("Control Point Spacing must be greater than zero.")
    min_x, min_y, max_x, max_y = geometry.bounds
    width = max_x - min_x
    height = max_y - min_y
    estimate = max(int(np.ceil(width / spacing)), 1) * max(int(np.ceil(height / spacing)), 1)
    if estimate > REGION_CONTROL_WARNING_THRESHOLD:
        # The caller records a warning; the hard limit prevents accidental lockups.
        if estimate > REGION_CONTROL_HARD_LIMIT:
            raise ValueError(
                f"Control region would generate about {estimate:,} controls. Increase spacing before generating."
            )
    x_values = np.arange(min_x + spacing / 2.0, max_x + spacing / 2.0, spacing)
    y_values = np.arange(min_y + spacing / 2.0, max_y + spacing / 2.0, spacing)
    points: list[Point] = []
    for y in y_values:
        for x in x_values:
            point = Point(float(x), float(y))
            if geometry.covers(point):
                points.append(point)
    if not points:
        point = geometry.representative_point()
        if geometry.covers(point):
            points.append(point)
    return points


def _matching_scope_warnings(
    *,
    identifier: str,
    property_name: str,
    active_property: str,
    value: Any,
    control_layer: str,
    active_layer: str | None,
    control_panel: str,
    selected_panels: tuple[str, ...],
    control_date: str,
    active_reference_date,
    property_type: str,
    property_unit: str,
    active_property_unit: str,
) -> tuple[bool, list[str]]:
    warnings: list[str] = []
    if property_name != active_property:
        return False, warnings
    if property_unit and active_property_unit and property_unit != active_property_unit:
        return False, [f"{identifier} uses unit {property_unit}, not active unit {active_property_unit}."]
    if _finite_float(value) is None:
        return False, [f"{identifier} has a non-finite control value."]
    if active_layer not in (None, "") and control_layer != str(active_layer):
        if control_layer:
            return False, [f"{identifier} is scoped to layer {control_layer}, not active layer {active_layer}."]
        return False, [f"{identifier} is missing a Reservoir Layer for active layer {active_layer}."]
    if selected_panels and control_panel and control_panel not in selected_panels:
        return False, [f"{identifier} is scoped to panel {control_panel}, outside selected panels."]
    if property_type == "Pressure":
        active_date = iso_date_or_blank(active_reference_date)
        if not control_date:
            return False, [f"{identifier} is a pressure control without a Pressure Map Reference Date."]
        if control_date != active_date:
            return False, [f"{identifier} is scoped to pressure date {control_date}, not active date {active_date}."]
    return True, warnings


def _region_points(
    region: dict[str, Any],
    *,
    selected_panels: tuple[str, ...],
    panel_interpolation_mode: str,
    panel_layer: GeometryLayer | None,
    property_col: str,
) -> tuple[list[dict[str, Any]], list[str]]:
    normalized = normalize_control_region(region)
    geometry = normalized.get("Geometry")
    if not isinstance(geometry, BaseGeometry) or geometry.is_empty:
        return [], [f"{normalized['Region_ID']} has no valid polygon geometry."]
    spacing = _finite_float(normalized.get("Control_Point_Spacing"))
    if spacing is None or spacing <= 0:
        return [], [f"{normalized['Region_ID']} has invalid Control Point Spacing."]

    warnings: list[str] = []
    region_panel = str(normalized.get("Panel") or "")
    effective_selected_panels = (region_panel,) if region_panel and panel_layer is not None else selected_panels
    if effective_selected_panels and panel_layer is not None:
        if panel_mode_from_legacy(panel_interpolation_mode) == PANEL_MODE_INDEPENDENT:
            pieces = []
            for feature in selected_panel_features(panel_layer, effective_selected_panels):
                intersection = geometry.intersection(feature.geometry)
                if not intersection.is_empty:
                    pieces.append((intersection, str(feature.name)))
        else:
            selected_domain = selected_panel_union(panel_layer, effective_selected_panels)
            intersection = geometry.intersection(selected_domain) if selected_domain is not None else geometry
            pieces = [(intersection, region_panel)]
    else:
        pieces = [(geometry, normalized.get("Panel", ""))]

    points: list[dict[str, Any]] = []
    sequence = 1
    for piece, panel_name in pieces:
        if piece.is_empty:
            continue
        try:
            generated_points = _regular_points_in_geometry(piece, float(spacing))
        except ValueError as exc:
            return [], [str(exc)]
        if len(generated_points) > REGION_CONTROL_WARNING_THRESHOLD:
            warnings.append(
                f"{normalized['Region_ID']} generated {len(generated_points):,} controls; increase spacing if this is unintended."
            )
        for point in generated_points:
            assigned_panel = panel_name
            if not assigned_panel and panel_layer is not None:
                assigned_panel, assignment_warning = assign_panel_from_point(point.x, point.y, panel_layer, effective_selected_panels)
                if assignment_warning and effective_selected_panels:
                    warnings.append(f"{normalized['Region_ID']}: {assignment_warning}")
            points.append(
                create_control_point(
                    x=point.x,
                    y=point.y,
                    property_name=property_col,
                    value=float(normalized["Target_Value"]),
                    property_unit=normalized.get("Property_Unit", ""),
                    reservoir_layer=normalized.get("Reservoir_Layer", ""),
                    panel=assigned_panel or normalized.get("Panel", ""),
                    pressure_reference_date=normalized.get("Pressure_Map_Reference_Date", ""),
                    active=True,
                    comment=normalized.get("Comment", ""),
                    source_type=REGION_CONTROL_SOURCE,
                    control_id=f"{normalized['Region_ID']}-GC-{sequence:03d}",
                )
                | {"Region_ID": normalized["Region_ID"], "Region_Name": normalized["Region_Name"]}
            )
            sequence += 1

    normalized["Generated_Control_Count"] = len(points)
    return points, warnings


def _outside_reservoir_warnings(
    controls: list[dict[str, Any]],
    reservoir_boundary_layer: GeometryLayer | None,
) -> list[str]:
    if reservoir_boundary_layer is None:
        return []
    reservoir_geometry = polygon_union(reservoir_boundary_layer.polygon_features)
    if reservoir_geometry is None or reservoir_geometry.is_empty:
        return []
    warnings: list[str] = []
    for control in controls:
        x = _finite_float(control.get("X"))
        y = _finite_float(control.get("Y"))
        if x is None or y is None:
            continue
        if not reservoir_geometry.covers(Point(x, y)):
            warnings.append(f"{control.get('Control_ID')} is outside the active reservoir boundary.")
    return warnings


def _exclude_outside_selected_panels(
    controls: list[dict[str, Any]],
    *,
    panel_layer: GeometryLayer | None,
    selected_panels: tuple[str, ...],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[str]]:
    if panel_layer is None or not selected_panels:
        return controls, [], []
    panel_geometry = selected_panel_union(panel_layer, selected_panels)
    if panel_geometry is None or panel_geometry.is_empty:
        return controls, [], []
    used: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    warnings: list[str] = []
    for control in controls:
        x = _finite_float(control.get("X"))
        y = _finite_float(control.get("Y"))
        if x is None or y is None:
            used.append(control)
            continue
        if panel_geometry.covers(Point(x, y)):
            used.append(control)
            continue
        control = dict(control)
        control["Used_In_Interpolation"] = False
        control["Exclusion_Reason"] = "Outside selected panels"
        excluded.append(control)
        warnings.append(f"Engineering control {control.get('Control_ID')} is outside selected panel polygons.")
    return used, excluded, warnings


def _control_point_candidates(
    control_points: list[dict[str, Any]] | tuple[dict[str, Any], ...] | None,
    *,
    property_col: str,
    property_type: str,
    property_unit: str,
    pressure_reference_date,
    reservoir_layer: str | None,
    selected_panels: tuple[str, ...],
) -> tuple[list[dict[str, Any]], list[str]]:
    controls: list[dict[str, Any]] = []
    warnings: list[str] = []
    for raw_control in control_points or []:
        control = normalize_control_point(raw_control)
        if not control.get("Active", True):
            continue
        include, control_warnings = _matching_scope_warnings(
            identifier=control["Control_ID"],
            property_name=control.get("Property", ""),
            active_property=property_col,
            value=control.get("Value"),
            control_layer=control.get("Reservoir_Layer", ""),
            active_layer=reservoir_layer,
            control_panel=control.get("Panel", ""),
            selected_panels=selected_panels,
            control_date=control.get("Pressure_Map_Reference_Date", ""),
            active_reference_date=pressure_reference_date,
            property_type=property_type,
            property_unit=control.get("Property_Unit", ""),
            active_property_unit=property_unit,
        )
        warnings.extend(control_warnings)
        if include:
            controls.append(control)
    return controls, warnings


def _control_region_candidates(
    control_regions: list[dict[str, Any]] | tuple[dict[str, Any], ...] | None,
    *,
    property_col: str,
    property_type: str,
    property_unit: str,
    pressure_reference_date,
    reservoir_layer: str | None,
    selected_panels: tuple[str, ...],
    panel_interpolation_mode: str,
    panel_layer: GeometryLayer | None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[str]]:
    active_regions: list[dict[str, Any]] = []
    generated_controls: list[dict[str, Any]] = []
    warnings: list[str] = []
    for raw_region in control_regions or []:
        region = normalize_control_region(raw_region)
        if not region.get("Active", True):
            continue
        include, region_warnings = _matching_scope_warnings(
            identifier=region["Region_ID"],
            property_name=region.get("Property", ""),
            active_property=property_col,
            value=region.get("Target_Value"),
            control_layer=region.get("Reservoir_Layer", ""),
            active_layer=reservoir_layer,
            control_panel=region.get("Panel", ""),
            selected_panels=selected_panels,
            control_date=region.get("Pressure_Map_Reference_Date", ""),
            active_reference_date=pressure_reference_date,
            property_type=property_type,
            property_unit=region.get("Property_Unit", ""),
            active_property_unit=property_unit,
        )
        warnings.extend(region_warnings)
        if not include:
            continue
        points, point_warnings = _region_points(
            region,
            selected_panels=selected_panels,
            panel_interpolation_mode=panel_interpolation_mode,
            panel_layer=panel_layer,
            property_col=property_col,
        )
        warnings.extend(point_warnings)
        if selected_panels and panel_layer is not None and not points:
            warnings.append(f"{region['Region_ID']} does not intersect selected panels.")
        region["Generated_Control_Count"] = len(points)
        active_regions.append(region)
        generated_controls.extend(points)
    return active_regions, generated_controls, warnings


def _coordinate_key(x: Any, y: Any) -> tuple[float, float] | None:
    x_float = _finite_float(x)
    y_float = _finite_float(y)
    if x_float is None or y_float is None:
        return None
    return (round(x_float, 8), round(y_float, 8))


def _measured_coordinate_values(measured_dataframe: pd.DataFrame, x_col: str, y_col: str, property_col: str) -> dict[tuple[float, float], float]:
    values: dict[tuple[float, float], float] = {}
    if measured_dataframe.empty or property_col not in measured_dataframe.columns:
        return values
    working = measured_dataframe
    if INCLUDE_COLUMN in working.columns:
        working = working[working[INCLUDE_COLUMN].astype(bool)]
    for _, row in working.iterrows():
        key = _coordinate_key(row.get(x_col), row.get(y_col))
        value = _finite_float(row.get(property_col))
        if key is not None and value is not None:
            values.setdefault(key, value)
    return values


def _duplicate_control_warnings(controls: list[dict[str, Any]]) -> list[str]:
    seen: dict[tuple[float, float], list[str]] = {}
    for control in controls:
        key = _coordinate_key(control.get("X"), control.get("Y"))
        if key is not None:
            seen.setdefault(key, []).append(str(control.get("Control_ID")))
    warnings = []
    for ids in seen.values():
        if len(ids) > 1:
            warnings.append(f"Duplicate engineering control coordinates: {', '.join(ids)}.")
    return warnings


def _overlapping_region_conflict_warnings(regions: list[dict[str, Any]]) -> list[str]:
    warnings: list[str] = []
    normalized = [normalize_control_region(region) for region in regions]
    for index, first in enumerate(normalized):
        first_geometry = first.get("Geometry")
        first_value = _finite_float(first.get("Target_Value"))
        if not isinstance(first_geometry, BaseGeometry) or first_geometry.is_empty or first_value is None:
            continue
        for second in normalized[index + 1 :]:
            second_geometry = second.get("Geometry")
            second_value = _finite_float(second.get("Target_Value"))
            if not isinstance(second_geometry, BaseGeometry) or second_geometry.is_empty or second_value is None:
                continue
            if np.isclose(first_value, second_value):
                continue
            try:
                overlap_area = float(first_geometry.intersection(second_geometry).area)
            except Exception:
                overlap_area = 0.0
            if overlap_area > 0.0:
                warnings.append(
                    "Overlapping control regions have conflicting target values: "
                    f"{first.get('Region_ID')} ({first_value:.6g}) and {second.get('Region_ID')} ({second_value:.6g})."
                )
    return warnings


def _exclude_measured_conflicts(
    controls: list[dict[str, Any]],
    *,
    measured_dataframe: pd.DataFrame,
    x_col: str,
    y_col: str,
    property_col: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[str]]:
    measured_values = _measured_coordinate_values(measured_dataframe, x_col, y_col, property_col)
    used: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    warnings: list[str] = []
    for control in controls:
        key = _coordinate_key(control.get("X"), control.get("Y"))
        value = _finite_float(control.get("Value"))
        if key is not None and value is not None and key in measured_values and not np.isclose(measured_values[key], value):
            control = dict(control)
            control["Used_In_Interpolation"] = False
            control["Exclusion_Reason"] = "Measured observation conflict"
            warnings.append(
                f"Engineering control {control.get('Control_ID')} conflicts with a measured observation at the same location; measured observation kept."
            )
            excluded.append(control)
            continue
        control = dict(control)
        control["Used_In_Interpolation"] = True
        control["Exclusion_Reason"] = ""
        used.append(control)
    return used, excluded, warnings


def _controls_to_interpolation_frame(
    controls: list[dict[str, Any]],
    *,
    x_col: str,
    y_col: str,
    property_col: str,
    layer_col: str | None,
    panel_col: str | None,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for control in controls:
        rows.append(
            {
                x_col: float(control["X"]),
                y_col: float(control["Y"]),
                property_col: float(control["Value"]),
                INCLUDE_COLUMN: True,
                "Control_ID": control.get("Control_ID"),
                "Control_Type": CONTROL_TYPE,
                "Source_Type": control.get("Source_Type", MANUAL_CONTROL_SOURCE),
                "Region_ID": control.get("Region_ID", ""),
                "Region_Name": control.get("Region_Name", ""),
                "Comment": control.get("Comment", ""),
                panel_col or "Panel": control.get("Panel", ""),
                layer_col or "Reservoir_Layer": control.get("Reservoir_Layer", ""),
                "Pressure_Map_Reference_Date": control.get("Pressure_Map_Reference_Date", ""),
            }
        )
    return pd.DataFrame(rows)


def _signature_control(control: dict[str, Any]) -> dict[str, Any]:
    normalized = normalize_control_point(control)
    return {
        "id": normalized.get("Control_ID", ""),
        "x": _finite_float(normalized.get("X")),
        "y": _finite_float(normalized.get("Y")),
        "property": normalized.get("Property", ""),
        "value": _finite_float(normalized.get("Value")),
        "unit": normalized.get("Property_Unit", ""),
        "layer": normalized.get("Reservoir_Layer", ""),
        "panel": normalized.get("Panel", ""),
        "reference_date": normalized.get("Pressure_Map_Reference_Date", ""),
        "active": bool(normalized.get("Active", True)),
        "source_type": normalized.get("Source_Type", ""),
        "region_id": normalized.get("Region_ID", ""),
        "used": bool(control.get("Used_In_Interpolation", True)),
        "exclusion_reason": control.get("Exclusion_Reason", ""),
    }


def _signature_region(region: dict[str, Any]) -> dict[str, Any]:
    normalized = normalize_control_region(region)
    geometry = normalized.get("Geometry")
    return {
        "id": normalized.get("Region_ID", ""),
        "name": normalized.get("Region_Name", ""),
        "property": normalized.get("Property", ""),
        "target_value": _finite_float(normalized.get("Target_Value")),
        "unit": normalized.get("Property_Unit", ""),
        "layer": normalized.get("Reservoir_Layer", ""),
        "panel": normalized.get("Panel", ""),
        "reference_date": normalized.get("Pressure_Map_Reference_Date", ""),
        "spacing": _finite_float(normalized.get("Control_Point_Spacing")),
        "active": bool(normalized.get("Active", True)),
        "generated_count": int(normalized.get("Generated_Control_Count") or 0),
        "geometry_wkt": geometry.wkt if isinstance(geometry, BaseGeometry) else "",
    }


def engineering_controls_for_context(
    *,
    control_points: list[dict[str, Any]] | tuple[dict[str, Any], ...] | None,
    control_regions: list[dict[str, Any]] | tuple[dict[str, Any], ...] | None,
    measured_dataframe: pd.DataFrame,
    mappings: dict[str, str | None],
    property_col: str,
    property_type: str,
    property_unit: str,
    pressure_reference_date,
    reservoir_layer: str | None,
    selected_panels: list[object] | tuple[object, ...] | None,
    panel_interpolation_mode: str,
    x_col: str,
    y_col: str,
    panel_layer: GeometryLayer | None = None,
    reservoir_boundary_layer: GeometryLayer | None = None,
) -> ControlSelection:
    selected_panel_tuple = tuple(str(value) for value in (selected_panels or ()) if value not in (None, ""))
    manual_points, warnings = _control_point_candidates(
        control_points,
        property_col=property_col,
        property_type=property_type,
        property_unit=property_unit,
        pressure_reference_date=pressure_reference_date,
        reservoir_layer=reservoir_layer,
        selected_panels=selected_panel_tuple,
    )
    active_regions, region_points, region_warnings = _control_region_candidates(
        control_regions,
        property_col=property_col,
        property_type=property_type,
        property_unit=property_unit,
        pressure_reference_date=pressure_reference_date,
        reservoir_layer=reservoir_layer,
        selected_panels=selected_panel_tuple,
        panel_interpolation_mode=panel_interpolation_mode,
        panel_layer=panel_layer,
    )
    warnings.extend(region_warnings)
    warnings.extend(_overlapping_region_conflict_warnings(active_regions))
    all_candidates = manual_points + region_points
    warnings.extend(_outside_reservoir_warnings(all_candidates, reservoir_boundary_layer))
    warnings.extend(_duplicate_control_warnings(all_candidates))
    panel_scoped_controls, panel_excluded_controls, panel_warnings = _exclude_outside_selected_panels(
        all_candidates,
        panel_layer=panel_layer,
        selected_panels=selected_panel_tuple,
    )
    warnings.extend(panel_warnings)
    used_controls, excluded_controls, conflict_warnings = _exclude_measured_conflicts(
        panel_scoped_controls,
        measured_dataframe=measured_dataframe,
        x_col=x_col,
        y_col=y_col,
        property_col=property_col,
    )
    excluded_controls = panel_excluded_controls + excluded_controls
    warnings.extend(conflict_warnings)
    controls_frame = _controls_to_interpolation_frame(
        used_controls,
        x_col=x_col,
        y_col=y_col,
        property_col=property_col,
        layer_col=mappings.get("layer"),
        panel_col=mappings.get("panel"),
    )
    if control_points or control_regions:
        signature_state = {
            "manual_control_points": sorted((_signature_control(item) for item in manual_points), key=lambda item: item["id"]),
            "region_control_points": sorted((_signature_control(item) for item in region_points), key=lambda item: item["id"]),
            "excluded_control_points": sorted((_signature_control(item) for item in excluded_controls), key=lambda item: item["id"]),
            "control_regions": sorted((_signature_region(item) for item in active_regions), key=lambda item: item["id"]),
        }
    else:
        signature_state = {}
    return ControlSelection(
        dataframe=controls_frame,
        manual_points=manual_points,
        region_points=region_points,
        regions=active_regions,
        warnings=tuple(dict.fromkeys(warnings)),
        signature_state=signature_state,
    )


def controls_for_plot(selection: ControlSelection | None) -> pd.DataFrame:
    if selection is None:
        return pd.DataFrame()
    return pd.DataFrame(
        [normalize_control_point(item) for item in selection.manual_points + selection.region_points],
        columns=CONTROL_POINT_COLUMNS + ["Region_ID", "Region_Name", "Control_Type"],
    )
