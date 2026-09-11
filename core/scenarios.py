"""Saved map scenario helpers."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, is_dataclass
from datetime import date, datetime, timezone
from uuid import uuid4

import numpy as np
import pandas as pd
from shapely.geometry import mapping
from shapely.geometry.base import BaseGeometry

from core.active_data import panel_mode_from_legacy
from core.crs import normalize_crs_config, map_crs_snapshot
from core.engineering_controls import (
    deserialize_control_points,
    deserialize_control_regions,
    serialize_control_points,
    serialize_control_regions,
)
from utils.constants import INTERNAL_ROW_ID


ARRAY_KEYS = {"grid_x", "grid_y", "grid_z", "grid_variance", "grid_stddev", "panel_grid"}
DATAFRAME_KEYS = {"included_observations", "excluded_observations"}


def _now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def json_safe(value):
    if is_dataclass(value):
        return json_safe(asdict(value))
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
    if isinstance(value, BaseGeometry):
        return mapping(value)
    return value


def _observation_ids(frame: pd.DataFrame) -> list[int]:
    if not isinstance(frame, pd.DataFrame) or INTERNAL_ROW_ID not in frame.columns:
        return []
    ids: list[int] = []
    for value in frame[INTERNAL_ROW_ID].dropna().tolist():
        try:
            ids.append(int(value))
        except (TypeError, ValueError):
            continue
    return ids


def _current_validation(generated_map: dict[str, object], validation: dict[str, object] | None) -> dict[str, object] | None:
    if not validation:
        return None
    generated_signature = generated_map.get("model_signature") or {}
    validation_signature = validation.get("signature") or {}
    if generated_signature and validation_signature and generated_signature.get("hash") == validation_signature.get("hash"):
        return validation
    if generated_signature or validation_signature:
        return None
    return validation


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

    included_observations = generated_map.get("included_observations", pd.DataFrame()).copy()
    excluded_observations = generated_map.get("excluded_observations", pd.DataFrame()).copy()
    engineering_controls = generated_map.get("engineering_controls", pd.DataFrame())
    if isinstance(engineering_controls, pd.DataFrame):
        engineering_controls = engineering_controls.copy()
    else:
        engineering_controls = pd.DataFrame(engineering_controls)
    grid_variance = generated_map.get("grid_variance")
    validation_current = _current_validation(generated_map, validation)
    panel_mode = panel_mode_from_legacy(
        generated_map.get("panel_interpolation_mode"),
        bool(generated_map.get("respect_compartments", False)),
    )
    crs_snapshot = map_crs_snapshot(generated_map)
    scenario = {
        "CRS_Mode": crs_snapshot["mode"],
        "EPSG": crs_snapshot["epsg"],
        "CRS_Name": crs_snapshot["name"],
        "Coordinate_Unit": generated_map.get("coordinate_unit", ""),
        "scenario_schema_version": "1.0",
        "id": scenario_id or uuid4().hex,
        "name": name.strip() or str(generated_map.get("title") or "Saved Map"),
        "created_time": _now_iso(),
        "title": generated_map.get("title") or name.strip() or "Saved Map",
        "property": generated_map.get("property_col"),
        "property_type": generated_map.get("property_type") or ("Pressure" if generated_map.get("is_pressure_map") else "Generic"),
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
        "interpolation_family": generated_map.get("interpolation_family", ""),
        "interpolation_parameters": json_safe(generated_map.get("method_parameters", {})),
        "interpolation_metadata": json_safe(generated_map.get("interpolation_metadata", {})),
        "grid_parameters": json_safe(generated_map.get("grid_parameters", {})),
        "interpolation_domain": generated_map.get("interpolation_domain"),
        "reservoir_layer": generated_map.get("reservoir_layer"),
        "layer_mapping_scope": generated_map.get("layer_mapping_scope"),
        "domain_bounds": json_safe(generated_map.get("domain_bounds")),
        "mask_parameters": json_safe(generated_map.get("mask_parameters", {})),
        "mask_info": json_safe(generated_map.get("mask_info", {})),
        "duplicate_method": generated_map.get("duplicate_method"),
        "respect_compartments": panel_mode == "Independent by Panel / Compartment",
        "panel_interpolation_mode": panel_mode,
        "selected_panels": json_safe(generated_map.get("selected_panels", [])),
        "selected_layers": json_safe(generated_map.get("selected_layers", [])),
        "geometry_context": json_safe(generated_map.get("geometry_context", {})),
        "geometry_references": json_safe(generated_map.get("geometry_references", {})),
        "style_settings": json_safe((project_context or {}).get("style_settings", {})),
        "layer_settings": json_safe((project_context or {}).get("layer_settings", {})),
        "filter_values": json_safe((project_context or {}).get("filter_values", {})),
        "include_state": json_safe((project_context or {}).get("include_state", {})),
        "included_observation_ids": _observation_ids(included_observations),
        "excluded_observation_ids": _observation_ids(excluded_observations),
        "crs": json_safe(crs_snapshot),
        "export_metadata": json_safe(generated_map.get("export_metadata", {})),
        "measured_observation_count": int(generated_map.get("measured_observation_count", len(included_observations))),
        "conditioning_observation_count": int(
            generated_map.get("conditioning_observation_count", len(included_observations) + len(engineering_controls))
        ),
        "engineering_control_count": int(generated_map.get("engineering_control_count", len(engineering_controls))),
        "control_region_count": int(generated_map.get("control_region_count", 0)),
        "control_warnings": json_safe(generated_map.get("control_warnings", [])),
        "engineering_controls": engineering_controls,
        "engineering_control_points": serialize_control_points(generated_map.get("engineering_control_points", [])),
        "engineering_control_regions": serialize_control_regions(generated_map.get("engineering_control_regions", [])),
        "hover_columns": json_safe(generated_map.get("hover_columns", [])),
        "validation_metrics": json_safe((validation_current or {}).get("metrics", {})),
        "validation_signature": json_safe((validation_current or {}).get("signature", {})),
        "model_signature": json_safe(generated_map.get("model_signature", {})),
        "model_signature_hash": (generated_map.get("model_signature") or {}).get("hash", ""),
        "grid_x": np.asarray(generated_map["grid_x"], dtype=float).copy(),
        "grid_y": np.asarray(generated_map["grid_y"], dtype=float).copy(),
        "grid_z": np.asarray(generated_map["grid_z"], dtype=float).copy(),
        "grid_variance": None
        if grid_variance is None
        else np.asarray(grid_variance, dtype=float).copy(),
        "grid_stddev": None
        if grid_variance is None
        else np.sqrt(np.maximum(np.asarray(grid_variance, dtype=float), 0.0)).copy(),
        "panel_grid": None if generated_map.get("panel_grid") is None else np.asarray(generated_map["panel_grid"], dtype=object).copy(),
        "included_observations": included_observations,
        "excluded_observations": excluded_observations,
    }
    return scenario


def scenario_to_generated_map(scenario: dict[str, object]) -> dict[str, object]:
    engineering_controls = scenario.get("engineering_controls", pd.DataFrame())
    if not isinstance(engineering_controls, pd.DataFrame):
        engineering_controls = pd.DataFrame(engineering_controls)
    return {
        "grid_x": np.asarray(scenario["grid_x"], dtype=float).copy(),
        "grid_y": np.asarray(scenario["grid_y"], dtype=float).copy(),
        "grid_z": np.asarray(scenario["grid_z"], dtype=float).copy(),
        "grid_variance": None
        if scenario.get("grid_variance") is None
        else np.asarray(scenario["grid_variance"], dtype=float).copy(),
        "panel_grid": None
        if scenario.get("panel_grid") is None
        else np.asarray(scenario["panel_grid"], dtype=object).copy(),
        "included_observations": scenario.get("included_observations", pd.DataFrame()).copy(),
        "excluded_observations": scenario.get("excluded_observations", pd.DataFrame()).copy(),
        "property_col": scenario.get("property"),
        "property_type": scenario.get("property_type") or ("Pressure" if scenario.get("is_pressure_map") else "Generic"),
        "unit": scenario.get("property_unit") or "",
        "x_col": scenario.get("x_col"),
        "y_col": scenario.get("y_col"),
        "well_col": scenario.get("well_col"),
        "coordinate_unit": scenario.get("coordinate_unit"),
        "crs": deepcopy(scenario.get("crs", {})),
        "is_pressure_map": bool(scenario.get("is_pressure_map", False)),
        "map_reference_date": scenario.get("pressure_reference_date"),
        "measurement_date_col": scenario.get("measurement_date_col"),
        "map_reference_date_col": scenario.get("map_reference_date_col"),
        "method": scenario.get("interpolation_method"),
        "interpolation_family": scenario.get("interpolation_family", ""),
        "method_parameters": deepcopy(scenario.get("interpolation_parameters", {})),
        "interpolation_metadata": deepcopy(scenario.get("interpolation_metadata", {})),
        "grid_parameters": deepcopy(scenario.get("grid_parameters", {})),
        "interpolation_domain": scenario.get("interpolation_domain"),
        "reservoir_layer": scenario.get("reservoir_layer"),
        "layer_mapping_scope": scenario.get("layer_mapping_scope"),
        "domain_bounds": deepcopy(scenario.get("domain_bounds")),
        "mask_parameters": deepcopy(scenario.get("mask_parameters", {})),
        "mask_info": deepcopy(scenario.get("mask_info", {})),
        "respect_compartments": bool(scenario.get("respect_compartments", False)),
        "panel_interpolation_mode": panel_mode_from_legacy(
            scenario.get("panel_interpolation_mode"),
            bool(scenario.get("respect_compartments", False)),
        ),
        "selected_panels": deepcopy(scenario.get("selected_panels", [])),
        "selected_layers": deepcopy(scenario.get("selected_layers", [])),
        "geometry_context": deepcopy(scenario.get("geometry_context", {})),
        "geometry_references": deepcopy(scenario.get("geometry_references", {})),
        "display_style_settings": deepcopy(scenario.get("style_settings", {})),
        "display_layer_settings": deepcopy(scenario.get("layer_settings", {})),
        "duplicate_method": scenario.get("duplicate_method"),
        "hover_columns": deepcopy(scenario.get("hover_columns", [])),
        "title": scenario.get("title") or scenario.get("name"),
        "export_metadata": deepcopy(scenario.get("export_metadata", {})),
        "measured_observation_count": int(
            scenario.get("measured_observation_count", len(scenario.get("included_observations", pd.DataFrame())))
        ),
        "conditioning_observation_count": int(scenario.get("conditioning_observation_count", 0)),
        "engineering_controls": engineering_controls.copy(),
        "engineering_control_points": deserialize_control_points(scenario.get("engineering_control_points", [])),
        "engineering_control_regions": deserialize_control_regions(scenario.get("engineering_control_regions", [])),
        "engineering_control_count": int(scenario.get("engineering_control_count", len(engineering_controls))),
        "control_region_count": int(scenario.get("control_region_count", 0)),
        "control_warnings": deepcopy(scenario.get("control_warnings", [])),
        "model_signature": deepcopy(scenario.get("model_signature", {})),
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
                "Reservoir Layer": scenario.get("reservoir_layer") or ", ".join(scenario.get("selected_layers", []) or []),
                "Grid": _grid_label(scenario),
                "Controls": scenario.get("engineering_control_count", 0),
                "Regions": scenario.get("control_region_count", 0),
                "Created": scenario.get("created_time"),
            }
        )
    return pd.DataFrame(rows)


def scenario_display_label(scenario: dict[str, object]) -> str:
    """Return a human-readable scenario label while keeping ID separate."""

    property_name = str(scenario.get("property") or scenario.get("name") or "Map")
    parts = [property_name]
    if bool(scenario.get("is_pressure_map")) and scenario.get("pressure_reference_date"):
        formatted_date = scenario.get("pressure_reference_date")
        try:
            formatted_date = pd.to_datetime(formatted_date).strftime("%d-%b-%Y")
        except Exception:
            pass
        parts.append(str(formatted_date))
    layer = scenario.get("reservoir_layer") or ", ".join(scenario.get("selected_layers", []) or [])
    if layer:
        parts.append(str(layer))
    method = scenario.get("interpolation_method")
    if method:
        parts.append(str(method))
    return " | ".join(parts)


def scenario_by_id(scenarios: list[dict[str, object]], scenario_id: str | None) -> dict[str, object] | None:
    if not scenario_id:
        return None
    for scenario in scenarios or []:
        if str(scenario.get("id") or "") == str(scenario_id):
            return scenario
    return None


def validate_scenario_snapshot(scenario: dict[str, object]) -> list[str]:
    errors: list[str] = []
    for key in ("id", "grid_x", "grid_y", "grid_z", "property", "interpolation_method"):
        if key not in scenario or scenario.get(key) is None:
            errors.append(f"missing {key}")
    try:
        grid_x = np.asarray(scenario.get("grid_x"), dtype=float)
        grid_y = np.asarray(scenario.get("grid_y"), dtype=float)
        grid_z = np.asarray(scenario.get("grid_z"), dtype=float)
        if grid_x.shape != grid_y.shape or grid_x.shape != grid_z.shape or grid_z.ndim != 2:
            errors.append("grid arrays must be 2-D with matching shapes")
    except Exception:
        errors.append("grid arrays are not numeric")
    if scenario.get("grid_variance") is not None:
        try:
            if np.asarray(scenario.get("grid_variance"), dtype=float).shape != np.asarray(scenario.get("grid_z"), dtype=float).shape:
                errors.append("uncertainty grid shape does not match property grid")
        except Exception:
            errors.append("uncertainty grid is not numeric")
    return errors


def safe_scenario_to_generated_map(scenario: dict[str, object]) -> tuple[dict[str, object] | None, list[str]]:
    errors = validate_scenario_snapshot(scenario)
    if errors:
        return None, errors
    try:
        generated = scenario_to_generated_map(scenario)
    except Exception as exc:
        return None, [str(exc)]
    return generated, []


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
