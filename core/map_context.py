"""Map title and metadata helpers."""

from __future__ import annotations

from datetime import date

import numpy as np

from core.active_data import panel_mode_from_legacy
from core.crs import normalize_crs_config
from core.pressure_dates import format_map_date
from utils.constants import APP_NAME
from utils.units import coordinate_unit_symbol


def build_default_map_title(
    property_name: str,
    filter_values: dict[str, list[object]] | None = None,
    filter_columns: dict[str, str | None] | None = None,
    is_pressure_map: bool = False,
    map_reference_date: date | None = None,
) -> str:
    parts = [f"{property_name} Map"]
    filter_values = filter_values or {}
    filter_columns = filter_columns or {}
    for key in ("reservoir", "panel", "layer", "field"):
        column = filter_columns.get(key)
        selected = filter_values.get(column or "", [])
        if selected and len(selected) == 1:
            parts.append(str(selected[0]))
    if is_pressure_map and map_reference_date is not None:
        parts.append(f"Reference Date: {format_map_date(map_reference_date)}")
    return " - ".join(parts)


def build_map_metadata(
    property_name: str,
    property_unit: str | None,
    x_column: str,
    y_column: str,
    coordinate_unit: str,
    interpolation_method: str,
    grid_parameters: dict,
    method_parameters: dict,
    mask_parameters: dict,
    duplicate_method: str,
    is_pressure_map: bool = False,
    map_reference_date: date | None = None,
    geometry_context: dict[str, object] | None = None,
    property_type: str | None = None,
    crs: dict[str, object] | None = None,
    selected_panels: list[object] | tuple[object, ...] | None = None,
    panel_interpolation_mode: str | None = None,
    selected_layers: list[object] | tuple[object, ...] | None = None,
    interpolation_domain: str | None = None,
    domain_bounds: tuple[float, float, float, float] | list[float] | None = None,
    grid_x=None,
    grid_y=None,
    validation_metrics: dict[str, object] | None = None,
    model_signature_hash: str | None = None,
    reservoir_layer: str | None = None,
    layer_mapping_scope: str | None = None,
    measured_observation_count: int | None = None,
    engineering_control_count: int | None = None,
    control_region_count: int | None = None,
    control_point_ids: list[object] | tuple[object, ...] | None = None,
    control_region_ids: list[object] | tuple[object, ...] | None = None,
) -> dict[str, object]:
    unit_symbol = coordinate_unit_symbol(coordinate_unit)
    crs_config = normalize_crs_config(crs)
    metadata: dict[str, object] = {
        "Application": APP_NAME,
        "Property": property_name,
        "Property_Type": property_type or ("Pressure" if is_pressure_map else "Generic"),
        "Property_Unit": property_unit or "",
        "X_Column": x_column,
        "Y_Column": y_column,
        "X_Field": x_column,
        "Y_Field": y_column,
        "Coordinate_Unit": unit_symbol,
        "CRS_Mode": crs_config.get("mode", ""),
        "CRS_EPSG": crs_config.get("epsg", ""),
        "EPSG": crs_config.get("epsg", ""),
        "CRS_Name": crs_config.get("name", ""),
        "CRS_Authority": crs_config.get("authority", ""),
        "Interpolation_Method": interpolation_method,
        "Grid_NX": grid_parameters.get("nx"),
        "Grid_NY": grid_parameters.get("ny"),
        "Grid_Buffer_Percent": round(float(grid_parameters.get("buffer_fraction", 0.0)) * 100.0, 6),
        "Duplicate_Coordinate_Handling": duplicate_method,
        "Mask": mask_parameters.get("mode"),
        "Interpolation_Domain": interpolation_domain or grid_parameters.get("interpolation_domain", ""),
        "Interpolation_Domain_Type": interpolation_domain or grid_parameters.get("interpolation_domain", ""),
        "Panel_Interpolation_Mode": panel_mode_from_legacy(panel_interpolation_mode, False),
        "Selected_Panels": ", ".join(str(value) for value in (selected_panels or [])),
        "Selected_Layers": ", ".join(str(value) for value in (selected_layers or [])),
        "Reservoir_Layer": reservoir_layer or "",
        "Layer_Mapping_Scope": layer_mapping_scope or "",
        "Measured_Observation_Count": int(measured_observation_count or 0),
        "Engineering_Control_Count": int(engineering_control_count or 0),
        "Control_Region_Count": int(control_region_count or 0),
        "Control_Point_IDs": ", ".join(str(value) for value in (control_point_ids or [])),
        "Control_Region_IDs": ", ".join(str(value) for value in (control_region_ids or [])),
        "Variogram_Range_Convention": method_parameters.get("variogram_range_convention", "Practical Range")
        if interpolation_method in {"Ordinary Kriging", "Universal Kriging"}
        else "",
    }
    if model_signature_hash:
        metadata["Model_Signature_Hash"] = model_signature_hash
    if is_pressure_map and map_reference_date:
        metadata["Pressure_Map_Reference_Date"] = map_reference_date.isoformat()
    if domain_bounds is not None:
        metadata["Domain_Min_X"] = float(domain_bounds[0])
        metadata["Domain_Min_Y"] = float(domain_bounds[1])
        metadata["Domain_Max_X"] = float(domain_bounds[2])
        metadata["Domain_Max_Y"] = float(domain_bounds[3])
        metadata["Domain_Bounds"] = {
            "min_x": float(domain_bounds[0]),
            "min_y": float(domain_bounds[1]),
            "max_x": float(domain_bounds[2]),
            "max_y": float(domain_bounds[3]),
        }
    domain_type = str(metadata.get("Interpolation_Domain_Type") or "")
    if domain_type == "Reservoir Boundary Extent":
        metadata["Domain_Geometry_Source"] = "Reservoir Boundary"
    elif domain_type in {"Selected Panel Union Extent", "Selected Panel Extent"}:
        metadata["Domain_Geometry_Source"] = "Selected Panel Union"
    elif domain_type:
        metadata["Domain_Geometry_Source"] = "Well Data"
    if grid_x is not None and grid_y is not None:
        x = np.asarray(grid_x, dtype=float)
        y = np.asarray(grid_y, dtype=float)
        if x.ndim == 2 and y.ndim == 2 and x.shape[1] > 1 and y.shape[0] > 1:
            dx = float(np.nanmedian(np.diff(x[0, :])))
            dy = float(np.nanmedian(np.diff(y[:, 0])))
            abs_dx = abs(dx)
            abs_dy = abs(dy)
            metadata["Grid_NX"] = int(x.shape[1])
            metadata["Grid_NY"] = int(x.shape[0])
            metadata["NX"] = int(x.shape[1])
            metadata["NY"] = int(x.shape[0])
            metadata["Grid_DX_Value"] = dx
            metadata["Grid_DX_Unit"] = unit_symbol
            metadata["Grid_DY_Value"] = dy
            metadata["Grid_DY_Unit"] = unit_symbol
            metadata["X_Spacing"] = abs_dx
            metadata["Y_Spacing"] = abs_dy
            metadata["X_Spacing_Unit"] = unit_symbol
            metadata["Y_Spacing_Unit"] = unit_symbol
            metadata["Grid_X_Min"] = float(np.nanmin(x))
            metadata["Grid_X_Max"] = float(np.nanmax(x))
            metadata["Grid_Y_Min"] = float(np.nanmin(y))
            metadata["Grid_Y_Max"] = float(np.nanmax(y))
            metadata["Bounds"] = {
                "x_min_center": float(np.nanmin(x)),
                "x_max_center": float(np.nanmax(x)),
                "y_min_center": float(np.nanmin(y)),
                "y_max_center": float(np.nanmax(y)),
                "west_edge": float(np.nanmin(x)) - abs_dx / 2.0,
                "east_edge": float(np.nanmax(x)) + abs_dx / 2.0,
                "south_edge": float(np.nanmin(y)) - abs_dy / 2.0,
                "north_edge": float(np.nanmax(y)) + abs_dy / 2.0,
            }
            metadata["Grid_Node_Convention"] = (
                "Grid X/Y arrays are cell centers; raster edge bounds are one half spacing outside the center limits."
            )

    geometry_context = geometry_context or {}
    if geometry_context:
        metadata["Reservoir_Boundary_Used"] = bool(geometry_context.get("reservoir_boundary_used", False))
        metadata["Panel_Constraint_Used"] = bool(geometry_context.get("panel_constraint_used", False))
        metadata["Active_Panels"] = geometry_context.get("active_panels", "")
        metadata["Fault_Layer_Loaded"] = bool(geometry_context.get("fault_layer_loaded", False))
        metadata["Custom_Layer_Count"] = int(geometry_context.get("custom_layer_count", 0) or 0)
        for key in (
            "reservoir_boundary_name",
            "reservoir_boundary_source",
            "reservoir_boundary_bounds",
            "panel_layer_name",
            "panel_layer_source",
            "panel_bounds",
            "selected_panel_names",
            "selected_panel_bounds",
            "selected_panel_feature_count",
            "panel_interpolation_mode",
        ):
            if key in geometry_context:
                metadata[key.title().replace("_", "_")] = geometry_context[key]

    for key, value in method_parameters.items():
        if key == "search_radius":
            metadata["Search_Radius_Value"] = value
            metadata["Search_Radius_Unit"] = unit_symbol
        elif key == "range":
            metadata["Range_Value"] = value
            metadata["Range_Unit"] = unit_symbol
        elif key == "variance":
            metadata["Variance_or_Sill"] = value
        elif key == "nugget":
            metadata["Nugget"] = value
        elif key == "variogram_model":
            metadata["Variogram_Model"] = value
        elif key == "variogram_mode":
            metadata["Variogram_Mode"] = value
        elif key == "anisotropy_enabled":
            metadata["Anisotropy_Enabled"] = value
        elif key == "anisotropy_angle":
            metadata["Anisotropy_Direction"] = value
        elif key == "anisotropy_ratio":
            metadata["Anisotropy_Ratio"] = value
        elif key == "max_neighbors":
            metadata["Maximum_Neighbors"] = value
        elif key == "epsilon":
            metadata["RBF_Epsilon_Value"] = value
            metadata["RBF_Epsilon_Unit"] = unit_symbol
        elif value is not None:
            metadata[f"Method_{key}"] = value

    max_distance = mask_parameters.get("max_distance")
    if max_distance is not None:
        metadata["Maximum_Distance_Value"] = max_distance
        metadata["Maximum_Distance_Unit"] = unit_symbol
    if validation_metrics:
        for key, value in validation_metrics.items():
            metadata[f"Validation_{key}"] = value
    return metadata
