"""Saved map scenario helpers."""

from __future__ import annotations

from copy import deepcopy
from datetime import date, datetime, timezone
from uuid import uuid4

import numpy as np
import pandas as pd


ARRAY_KEYS = {"grid_x", "grid_y", "grid_z", "grid_variance", "panel_grid"}
DATAFRAME_KEYS = {"included_observations", "excluded_observations"}


def _now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def json_safe(value):
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, (datetime, date, pd.Timestamp)):
        return value.isoformat()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, pd.DataFrame):
        return value.to_dict(orient="records")
    return value


def create_map_scenario(
    name: str,
    generated_map: dict[str, object],
    project_context: dict[str, object] | None = None,
    validation: dict[str, object] | None = None,
    scenario_id: str | None = None,
) -> dict[str, object]:
    if not generated_map:
        raise ValueError("Generate a map before saving a scenario.")
    required = {"grid_x", "grid_y", "grid_z", "property_col", "method"}
    missing = [key for key in required if key not in generated_map]
    if missing:
        raise ValueError(f"Generated map is missing required field(s): {', '.join(missing)}.")

    scenario = {
        "scenario_schema_version": "1.0",
        "id": scenario_id or uuid4().hex,
        "name": name.strip() or str(generated_map.get("title") or "Saved Map"),
        "created_time": _now_iso(),
        "title": generated_map.get("title") or name.strip() or "Saved Map",
        "property": generated_map.get("property_col"),
        "property_unit": generated_map.get("unit") or "",
        "coordinate_unit": generated_map.get("coordinate_unit"),
        "is_pressure_map": bool(generated_map.get("is_pressure_map", False)),
        "pressure_reference_date": json_safe(generated_map.get("map_reference_date")),
        "measurement_date_col": generated_map.get("measurement_date_col"),
        "map_reference_date_col": generated_map.get("map_reference_date_col"),
        "x_col": generated_map.get("x_col"),
        "y_col": generated_map.get("y_col"),
        "well_col": generated_map.get("well_col"),
        "interpolation_method": generated_map.get("method"),
        "interpolation_parameters": json_safe(generated_map.get("method_parameters", {})),
        "grid_parameters": json_safe(generated_map.get("grid_parameters", {})),
        "mask_parameters": json_safe(generated_map.get("mask_parameters", {})),
        "mask_info": json_safe(generated_map.get("mask_info", {})),
        "duplicate_method": generated_map.get("duplicate_method"),
        "respect_compartments": bool(generated_map.get("respect_compartments", False)),
        "geometry_context": json_safe(generated_map.get("geometry_context", {})),
        "style_settings": json_safe((project_context or {}).get("style_settings", {})),
        "layer_settings": json_safe((project_context or {}).get("layer_settings", {})),
        "filter_values": json_safe((project_context or {}).get("filter_values", {})),
        "include_state": json_safe((project_context or {}).get("include_state", {})),
        "crs": json_safe((project_context or {}).get("crs", {})),
        "export_metadata": json_safe(generated_map.get("export_metadata", {})),
        "hover_columns": json_safe(generated_map.get("hover_columns", [])),
        "validation_metrics": json_safe((validation or {}).get("metrics", validation or {})),
        "grid_x": np.asarray(generated_map["grid_x"], dtype=float),
        "grid_y": np.asarray(generated_map["grid_y"], dtype=float),
        "grid_z": np.asarray(generated_map["grid_z"], dtype=float),
        "grid_variance": None
        if generated_map.get("grid_variance") is None
        else np.asarray(generated_map["grid_variance"], dtype=float),
        "panel_grid": None if generated_map.get("panel_grid") is None else np.asarray(generated_map["panel_grid"], dtype=object),
        "included_observations": generated_map.get("included_observations", pd.DataFrame()).copy(),
        "excluded_observations": generated_map.get("excluded_observations", pd.DataFrame()).copy(),
    }
    return scenario


def scenario_to_generated_map(scenario: dict[str, object]) -> dict[str, object]:
    return {
        "grid_x": np.asarray(scenario["grid_x"], dtype=float),
        "grid_y": np.asarray(scenario["grid_y"], dtype=float),
        "grid_z": np.asarray(scenario["grid_z"], dtype=float),
        "grid_variance": None
        if scenario.get("grid_variance") is None
        else np.asarray(scenario["grid_variance"], dtype=float),
        "panel_grid": scenario.get("panel_grid"),
        "included_observations": scenario.get("included_observations", pd.DataFrame()).copy(),
        "excluded_observations": scenario.get("excluded_observations", pd.DataFrame()).copy(),
        "property_col": scenario.get("property"),
        "unit": scenario.get("property_unit") or "",
        "x_col": scenario.get("x_col"),
        "y_col": scenario.get("y_col"),
        "well_col": scenario.get("well_col"),
        "coordinate_unit": scenario.get("coordinate_unit"),
        "is_pressure_map": bool(scenario.get("is_pressure_map", False)),
        "map_reference_date": scenario.get("pressure_reference_date"),
        "measurement_date_col": scenario.get("measurement_date_col"),
        "map_reference_date_col": scenario.get("map_reference_date_col"),
        "method": scenario.get("interpolation_method"),
        "method_parameters": deepcopy(scenario.get("interpolation_parameters", {})),
        "grid_parameters": deepcopy(scenario.get("grid_parameters", {})),
        "mask_parameters": deepcopy(scenario.get("mask_parameters", {})),
        "mask_info": deepcopy(scenario.get("mask_info", {})),
        "respect_compartments": bool(scenario.get("respect_compartments", False)),
        "geometry_context": deepcopy(scenario.get("geometry_context", {})),
        "duplicate_method": scenario.get("duplicate_method"),
        "hover_columns": deepcopy(scenario.get("hover_columns", [])),
        "title": scenario.get("title") or scenario.get("name"),
        "export_metadata": deepcopy(scenario.get("export_metadata", {})),
    }


def duplicate_scenario(scenario: dict[str, object], new_name: str | None = None) -> dict[str, object]:
    copied = deepcopy(scenario)
    copied["id"] = uuid4().hex
    copied["name"] = (new_name or f"{scenario.get('name', 'Saved Map')} Copy").strip()
    copied["created_time"] = _now_iso()
    return copied


def rename_scenario(scenario: dict[str, object], new_name: str) -> dict[str, object]:
    updated = deepcopy(scenario)
    updated["name"] = new_name.strip() or str(scenario.get("name") or "Saved Map")
    return updated


def scenario_summary_table(scenarios: list[dict[str, object]]) -> pd.DataFrame:
    rows = []
    for scenario in scenarios:
        rows.append(
            {
                "Name": scenario.get("name"),
                "Property": scenario.get("property"),
                "Unit": scenario.get("property_unit"),
                "Method": scenario.get("interpolation_method"),
                "Pressure Reference Date": scenario.get("pressure_reference_date") or "",
                "Grid": _grid_label(scenario),
                "Created": scenario.get("created_time"),
            }
        )
    return pd.DataFrame(rows)


def _grid_label(scenario: dict[str, object]) -> str:
    grid_z = scenario.get("grid_z")
    if grid_z is None:
        return ""
    shape = np.asarray(grid_z).shape
    return f"{shape[1]} x {shape[0]}" if len(shape) == 2 else str(shape)


def scenario_metadata(scenario: dict[str, object]) -> dict[str, object]:
    return {
        key: json_safe(value)
        for key, value in scenario.items()
        if key not in ARRAY_KEYS and key not in DATAFRAME_KEYS
    }
