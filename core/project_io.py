"""Portable Reservoir Mapping Studio project archives."""

from __future__ import annotations

import json
import zipfile
from datetime import date
from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd
from shapely.geometry import mapping

from core.active_data import panel_mode_from_legacy
from core.column_mapper import normalize_column_mappings
from core.crs import normalize_crs_config
from core.data_loader import add_internal_row_id
from core.engineering_controls import (
    deserialize_control_points,
    deserialize_control_regions,
    serialize_control_points,
    serialize_control_regions,
)
from core.geometry.loader import load_geojson_bytes
from core.geometry.models import GeometryLayer
from core.layer_mapping import DOMAIN_WELL_DATA_EXTENT, normalize_interpolation_domain
from core.geostatistics.variogram import ExperimentalVariogram, VariogramFit
from core.scenarios import json_safe, scenario_metadata
from utils.constants import INTERNAL_ROW_ID


PROJECT_SCHEMA_VERSION = "1.3"
SUPPORTED_PROJECT_SCHEMA_VERSIONS = {"1.0", "1.1", "1.2", PROJECT_SCHEMA_VERSION}


class ProjectArchiveError(ValueError):
    """Raised when a project archive is missing required content."""


def _json_bytes(payload: dict[str, object]) -> bytes:
    return json.dumps(payload, indent=2, default=str).encode("utf-8")


def _read_json(archive: zipfile.ZipFile, name: str) -> dict[str, object]:
    try:
        return json.loads(archive.read(name).decode("utf-8"))
    except KeyError as exc:
        raise ProjectArchiveError(f"Project archive is missing {name}.") from exc
    except json.JSONDecodeError as exc:
        raise ProjectArchiveError(f"{name} is not valid JSON.") from exc


def _geometry_layer_to_geojson(layer: GeometryLayer) -> dict[str, object]:
    features = []
    for feature in layer.features:
        properties = dict(feature.attributes)
        properties["__rms_name"] = feature.name
        properties["__rms_layer_type"] = feature.layer_type
        features.append(
            {
                "type": "Feature",
                "properties": properties,
                "geometry": mapping(feature.geometry),
            }
        )
    return {"type": "FeatureCollection", "features": features}


def _iter_geometry_layers(geometry_layers: dict[str, object]):
    for key in ("reservoir_boundary", "panels", "faults"):
        layer = geometry_layers.get(key)
        if layer is not None:
            yield key, layer
    for index, layer in enumerate(geometry_layers.get("custom", []) or []):
        yield f"custom_{index}", layer


def _write_scenario(archive: zipfile.ZipFile, scenario: dict[str, object]) -> dict[str, object]:
    scenario_id = str(scenario["id"])
    base = f"maps/{scenario_id}"
    archive.writestr(f"{base}.json", _json_bytes(scenario_metadata(scenario)))
    arrays = {
        "grid_x": np.asarray(scenario["grid_x"], dtype=float),
        "grid_y": np.asarray(scenario["grid_y"], dtype=float),
        "grid_z": np.asarray(scenario["grid_z"], dtype=float),
    }
    if scenario.get("grid_variance") is not None:
        arrays["grid_variance"] = np.asarray(scenario["grid_variance"], dtype=float)
    if scenario.get("grid_stddev") is not None:
        arrays["grid_stddev"] = np.asarray(scenario["grid_stddev"], dtype=float)
    if scenario.get("panel_grid") is not None:
        panel = np.asarray(scenario["panel_grid"], dtype=object)
        arrays["panel_grid"] = np.where(pd.isna(panel), "", panel.astype(str))
    array_buffer = BytesIO()
    np.savez_compressed(array_buffer, **arrays)
    archive.writestr(f"{base}.npz", array_buffer.getvalue())
    for key, filename in [
        ("included_observations", f"{base}_included.csv"),
        ("excluded_observations", f"{base}_excluded.csv"),
    ]:
        frame = scenario.get(key)
        if isinstance(frame, pd.DataFrame) and not frame.empty:
            archive.writestr(filename, frame.to_csv(index=False).encode("utf-8"))
    return {"id": scenario_id, "metadata_path": f"{base}.json", "arrays_path": f"{base}.npz"}


def _map_result_metadata(map_result: dict[str, object]) -> dict[str, object]:
    metadata = dict(map_result)
    metadata["engineering_control_points"] = serialize_control_points(
        metadata.get("engineering_control_points", [])
    )
    metadata["engineering_control_regions"] = serialize_control_regions(
        metadata.get("engineering_control_regions", [])
    )
    return scenario_metadata(metadata)


def _write_map_result(
    archive: zipfile.ZipFile,
    map_result: dict[str, object],
    base: str,
) -> dict[str, object]:
    archive.writestr(f"{base}.json", _json_bytes(_map_result_metadata(map_result)))
    arrays = {
        "grid_x": np.asarray(map_result["grid_x"], dtype=float),
        "grid_y": np.asarray(map_result["grid_y"], dtype=float),
        "grid_z": np.asarray(map_result["grid_z"], dtype=float),
    }
    if map_result.get("grid_variance") is not None:
        arrays["grid_variance"] = np.asarray(map_result["grid_variance"], dtype=float)
    if map_result.get("panel_grid") is not None:
        panel = np.asarray(map_result["panel_grid"], dtype=object)
        arrays["panel_grid"] = np.where(pd.isna(panel), "", panel.astype(str))
    array_buffer = BytesIO()
    np.savez_compressed(array_buffer, **arrays)
    archive.writestr(f"{base}.npz", array_buffer.getvalue())
    for key, filename in [
        ("included_observations", f"{base}_included.csv"),
        ("excluded_observations", f"{base}_excluded.csv"),
    ]:
        frame = map_result.get(key)
        if isinstance(frame, pd.DataFrame) and not frame.empty:
            archive.writestr(filename, frame.to_csv(index=False).encode("utf-8"))
    return {"metadata_path": f"{base}.json", "arrays_path": f"{base}.npz"}


def _serialize_geostatistics(geostatistics: dict[str, object] | None) -> dict[str, object]:
    geostatistics = geostatistics or {}
    serialized = {
        "variogram_fit": json_safe(geostatistics.get("variogram_fit")),
        "variogram_fits": json_safe(geostatistics.get("variogram_fits", [])),
        "experimental_variogram": json_safe(geostatistics.get("experimental_variogram")),
        "variogram_settings": json_safe(geostatistics.get("variogram_settings", {})),
        "anisotropy_enabled": bool(geostatistics.get("anisotropy_enabled", False)),
        "anisotropy_angle": geostatistics.get("anisotropy_angle", 0.0),
        "anisotropy_ratio": geostatistics.get("anisotropy_ratio", 1.0),
    }
    cross_validation = geostatistics.get("cross_validation") or {}
    cross_validation_signature = geostatistics.get("cross_validation_signature", {})
    if isinstance(cross_validation, dict):
        cross_validation_signature = cross_validation.get("signature", {}) or cross_validation_signature
    serialized["cross_validation_signature"] = json_safe(cross_validation_signature)
    method_comparison = geostatistics.get("method_comparison")
    if isinstance(method_comparison, pd.DataFrame):
        serialized["method_comparison"] = method_comparison.to_dict(orient="records")
    elif isinstance(method_comparison, dict):
        serialized["method_comparison_signature"] = json_safe(method_comparison.get("signature", {}))
    return serialized


def _restore_variogram_fit(payload) -> VariogramFit | None:
    if not payload:
        return None
    return VariogramFit(
        model=str(payload["model"]),
        range_value=float(payload["range_value"]),
        variance=float(payload["variance"]),
        nugget=float(payload["nugget"]),
        fit_error=float(payload.get("fit_error", 0.0)),
    )


def _restore_experimental_variogram(payload) -> ExperimentalVariogram | None:
    if not payload:
        return None
    return ExperimentalVariogram(
        lag_distance=np.asarray(payload["lag_distance"], dtype=float),
        semivariance=np.asarray(payload["semivariance"], dtype=float),
        pair_count=np.asarray(payload["pair_count"], dtype=int),
        max_lag=float(payload["max_lag"]),
        n_lags=int(payload["n_lags"]),
    )


def _restore_geostatistics(payload: dict[str, object] | None) -> dict[str, object]:
    payload = payload or {}
    return {
        "variogram_fit": _restore_variogram_fit(payload.get("variogram_fit")),
        "variogram_fits": [
            fit for fit in (_restore_variogram_fit(item) for item in payload.get("variogram_fits", []) or []) if fit
        ],
        "experimental_variogram": _restore_experimental_variogram(payload.get("experimental_variogram")),
        "variogram_settings": payload.get("variogram_settings", {}),
        "anisotropy_enabled": bool(payload.get("anisotropy_enabled", False)),
        "anisotropy_angle": float(payload.get("anisotropy_angle", 0.0)),
        "anisotropy_ratio": float(payload.get("anisotropy_ratio", 1.0)),
        "cross_validation": None,
        "cross_validation_signature": payload.get("cross_validation_signature", {}),
        "method_comparison": None,
        "method_comparison_signature": payload.get("method_comparison_signature", {}),
    }


def save_project_archive(state: dict[str, object]) -> bytes:
    buffer = BytesIO()
    manifest: dict[str, object] = {
        "project_schema_version": PROJECT_SCHEMA_VERSION,
        "project_metadata": state.get("project_metadata", {}),
        "source_name": state.get("source_name"),
        "column_mappings": normalize_column_mappings(state.get("column_mappings", {})),
        "additional_filter_columns": state.get("additional_filter_columns", []),
        "filter_values": state.get("filter_values", {}),
        "coordinate_unit": state.get("coordinate_unit"),
        "property_unit": state.get("property_unit", ""),
        "pressure_reference_date": state.get("pressure_reference_date"),
        "selected_panels": state.get("selected_panels", []),
        "panel_interpolation_mode": panel_mode_from_legacy(state.get("panel_interpolation_mode"), False),
        "mapping_interpolation_domain": normalize_interpolation_domain(
            state.get("mapping_interpolation_domain", DOMAIN_WELL_DATA_EXTENT)
        ),
        "layer_mapping_scope": state.get("layer_mapping_scope", "Selected Layer"),
        "selected_reservoir_layer": state.get("selected_reservoir_layer"),
        "active_generated_layer": state.get("active_generated_layer"),
        "generated_layer_statuses": json_safe(state.get("generated_layer_statuses", [])),
        "generated_layer_batch_signature": json_safe(state.get("generated_layer_batch_signature", {})),
        "engineering_control_points": serialize_control_points(state.get("engineering_control_points", [])),
        "engineering_control_regions": serialize_control_regions(state.get("engineering_control_regions", [])),
        "crs": normalize_crs_config(state.get("crs", {})),
        "include_state": state.get("include_state", {}),
        "layer_settings": state.get("layer_settings", {}),
        "style_settings": state.get("style_settings", {}),
        "current_property": state.get("current_property"),
        "geostatistics": _serialize_geostatistics(state.get("geostatistics", {})),
        "geometry_layers": [],
        "map_scenarios": [],
        "generated_layer_maps": [],
        "current_scenario_id": state.get("current_scenario_id"),
    }
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        working_df = state.get("working_df")
        if isinstance(working_df, pd.DataFrame):
            portable = working_df.drop(columns=[INTERNAL_ROW_ID], errors="ignore")
            archive.writestr("data/working_data.csv", portable.to_csv(index=False).encode("utf-8"))
            manifest["working_data_path"] = "data/working_data.csv"

        geometry_layers = state.get("geometry_layers", {}) or {}
        for key, layer in _iter_geometry_layers(geometry_layers):
            path = f"geometry/{key}.geojson"
            archive.writestr(path, _json_bytes(_geometry_layer_to_geojson(layer)))
            manifest["geometry_layers"].append(
                {
                    "key": key,
                    "path": path,
                    "name": layer.name,
                    "layer_type": layer.layer_type,
                    "source_name": layer.source_name,
                    "name_attribute": "__rms_name",
                }
            )

        for scenario in state.get("map_scenarios", []) or []:
            manifest["map_scenarios"].append(_write_scenario(archive, scenario))

        for index, (layer_name, map_result) in enumerate((state.get("generated_layer_maps", {}) or {}).items()):
            base = f"generated_layers/layer_{index}"
            info = _write_map_result(archive, map_result, base)
            manifest["generated_layer_maps"].append({"layer": layer_name, **info})

        archive.writestr("project.json", _json_bytes(manifest))
    return buffer.getvalue()


def _read_optional_csv(archive: zipfile.ZipFile, path: str) -> pd.DataFrame:
    try:
        return pd.read_csv(BytesIO(archive.read(path)))
    except KeyError:
        return pd.DataFrame()


def _read_map_result(
    archive: zipfile.ZipFile,
    metadata_path: str,
    arrays_path: str,
    base: str,
) -> dict[str, object]:
    metadata = _read_json(archive, metadata_path)
    arrays = np.load(BytesIO(archive.read(arrays_path)), allow_pickle=False)
    result = dict(metadata)
    for key in ("grid_x", "grid_y", "grid_z", "grid_variance"):
        if key in arrays:
            result[key] = arrays[key]
    if "panel_grid" in arrays:
        panel = arrays["panel_grid"].astype(object)
        result["panel_grid"] = np.where(panel == "", None, panel)
    else:
        result["panel_grid"] = None
    result["included_observations"] = _read_optional_csv(archive, f"{base}_included.csv")
    result["excluded_observations"] = _read_optional_csv(archive, f"{base}_excluded.csv")
    controls = result.get("engineering_controls", [])
    result["engineering_controls"] = controls if isinstance(controls, pd.DataFrame) else pd.DataFrame(controls)
    result["engineering_control_points"] = deserialize_control_points(result.get("engineering_control_points", []))
    result["engineering_control_regions"] = deserialize_control_regions(result.get("engineering_control_regions", []))
    return result


def _parse_iso_date(value) -> date | None:
    if value in (None, ""):
        return None
    try:
        return date.fromisoformat(str(value)[:10])
    except ValueError:
        return None


def load_project_archive(data: bytes) -> dict[str, object]:
    try:
        archive = zipfile.ZipFile(BytesIO(data))
    except zipfile.BadZipFile as exc:
        raise ProjectArchiveError("Project file is not a valid .rmsproj ZIP archive.") from exc

    with archive:
        manifest = _read_json(archive, "project.json")
        if str(manifest.get("project_schema_version")) not in SUPPORTED_PROJECT_SCHEMA_VERSIONS:
            raise ProjectArchiveError(
                f"Unsupported project schema version: {manifest.get('project_schema_version')}."
            )
        state: dict[str, object] = {
            "project_metadata": manifest.get("project_metadata", {}),
            "source_name": manifest.get("source_name"),
            "source_key": manifest.get("source_name"),
            "column_mappings": normalize_column_mappings(manifest.get("column_mappings", {})),
            "additional_filter_columns": manifest.get("additional_filter_columns", []),
            "filter_values": manifest.get("filter_values", {}),
            "coordinate_unit": manifest.get("coordinate_unit"),
            "property_unit": manifest.get("property_unit", ""),
            "pressure_reference_date": _parse_iso_date(manifest.get("pressure_reference_date")),
            "selected_panels": manifest.get("selected_panels", []),
            "panel_interpolation_mode": panel_mode_from_legacy(manifest.get("panel_interpolation_mode"), False),
            "mapping_interpolation_domain": normalize_interpolation_domain(
                manifest.get("mapping_interpolation_domain", DOMAIN_WELL_DATA_EXTENT)
            ),
            "layer_mapping_scope": manifest.get("layer_mapping_scope", "Selected Layer"),
            "selected_reservoir_layer": manifest.get("selected_reservoir_layer"),
            "active_generated_layer": manifest.get("active_generated_layer"),
            "generated_layer_maps": {},
            "generated_layer_statuses": manifest.get("generated_layer_statuses", []),
            "generated_layer_batch_signature": manifest.get("generated_layer_batch_signature", {}),
            "engineering_control_points": deserialize_control_points(manifest.get("engineering_control_points", [])),
            "engineering_control_regions": deserialize_control_regions(manifest.get("engineering_control_regions", [])),
            "crs": normalize_crs_config(manifest.get("crs", {})),
            "include_state": {int(key): bool(value) for key, value in dict(manifest.get("include_state", {})).items()},
            "layer_settings": manifest.get("layer_settings", {}),
            "style_settings": manifest.get("style_settings", {}),
            "geostatistics": _restore_geostatistics(manifest.get("geostatistics", {})),
            "current_property": manifest.get("current_property"),
            "geometry_layers": {"reservoir_boundary": None, "panels": None, "faults": None, "custom": []},
            "map_scenarios": [],
            "current_scenario_id": manifest.get("current_scenario_id"),
        }
        data_path = manifest.get("working_data_path")
        if data_path:
            raw_df = pd.read_csv(BytesIO(archive.read(str(data_path))))
            state["original_df"] = raw_df.copy()
            state["working_df"] = add_internal_row_id(raw_df)

        for layer_info in manifest.get("geometry_layers", []) or []:
            layer = load_geojson_bytes(
                archive.read(str(layer_info["path"])),
                str(layer_info["layer_type"]),
                str(layer_info["name"]),
                "__rms_name",
                str(layer_info.get("source_name") or Path(str(layer_info["path"])).name),
            )
            key = str(layer_info["key"])
            if key.startswith("custom_"):
                state["geometry_layers"]["custom"].append(layer)
            else:
                state["geometry_layers"][key] = layer

        for scenario_info in manifest.get("map_scenarios", []) or []:
            metadata = _read_json(archive, str(scenario_info["metadata_path"]))
            arrays = np.load(BytesIO(archive.read(str(scenario_info["arrays_path"]))), allow_pickle=False)
            scenario = dict(metadata)
            for key in ("grid_x", "grid_y", "grid_z", "grid_variance", "grid_stddev"):
                if key in arrays:
                    scenario[key] = arrays[key]
            scenario["panel_interpolation_mode"] = panel_mode_from_legacy(
                scenario.get("panel_interpolation_mode"),
                bool(scenario.get("respect_compartments", False)),
            )
            if "panel_grid" in arrays:
                panel = arrays["panel_grid"].astype(object)
                scenario["panel_grid"] = np.where(panel == "", None, panel)
            else:
                scenario["panel_grid"] = None
            base = f"maps/{scenario['id']}"
            scenario["included_observations"] = _read_optional_csv(archive, f"{base}_included.csv")
            scenario["excluded_observations"] = _read_optional_csv(archive, f"{base}_excluded.csv")
            state["map_scenarios"].append(scenario)
        for index, map_info in enumerate(manifest.get("generated_layer_maps", []) or []):
            layer_name = str(map_info.get("layer") or f"Layer {index + 1}")
            base = str(map_info.get("metadata_path", f"generated_layers/layer_{index}.json")).rsplit(".", 1)[0]
            state["generated_layer_maps"][layer_name] = _read_map_result(
                archive,
                str(map_info["metadata_path"]),
                str(map_info["arrays_path"]),
                base,
            )
        return state
