"""Map title and metadata helpers."""

from __future__ import annotations

from datetime import date

from core.pressure_dates import format_map_date
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
) -> dict[str, object]:
    unit_symbol = coordinate_unit_symbol(coordinate_unit)
    metadata: dict[str, object] = {
        "Property": property_name,
        "Property_Unit": property_unit or "",
        "X_Column": x_column,
        "Y_Column": y_column,
        "Coordinate_Unit": unit_symbol,
        "Interpolation_Method": interpolation_method,
        "Grid_NX": grid_parameters.get("nx"),
        "Grid_NY": grid_parameters.get("ny"),
        "Grid_Buffer_Percent": round(float(grid_parameters.get("buffer_fraction", 0.0)) * 100.0, 6),
        "Duplicate_Coordinate_Handling": duplicate_method,
        "Mask": mask_parameters.get("mode"),
    }
    if is_pressure_map and map_reference_date:
        metadata["Pressure_Map_Reference_Date"] = map_reference_date.isoformat()

    geometry_context = geometry_context or {}
    if geometry_context:
        metadata["Reservoir_Boundary_Used"] = bool(geometry_context.get("reservoir_boundary_used", False))
        metadata["Panel_Constraint_Used"] = bool(geometry_context.get("panel_constraint_used", False))
        metadata["Active_Panels"] = geometry_context.get("active_panels", "")
        metadata["Fault_Layer_Loaded"] = bool(geometry_context.get("fault_layer_loaded", False))
        metadata["Custom_Layer_Count"] = int(geometry_context.get("custom_layer_count", 0) or 0)

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
    return metadata
