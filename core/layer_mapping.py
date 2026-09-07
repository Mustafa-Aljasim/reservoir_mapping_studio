"""Reservoir-layer map generation and stale-state helpers."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Callable

import numpy as np
import pandas as pd

from core.active_data import (
    PANEL_MODE_INDEPENDENT,
    build_model_signature,
    panel_mode_from_legacy,
    prepare_active_property_data,
    respect_compartments_from_mode,
    signatures_match,
)
from core.crs import normalize_crs_config
from core.data_qc import prepare_interpolation_dataframe
from core.engineering_controls import engineering_controls_for_context
from core.geometry.compartment import (
    compartment_interpolate,
    filter_dataframe_to_selected_panels,
    panel_domain_keep_mask,
    selected_panel_bounds,
    selected_panel_features,
)
from core.geometry.masking import layer_keep_mask, polygon_union
from core.geometry.models import GeometryLayer
from core.geostatistics.variogram import (
    VARIOGRAM_RANGE_CONVENTION,
    compute_experimental_variogram,
    fit_candidate_models,
)
from core.grid import generate_grid
from core.interpolation import InterpolationError, interpolate_surface_result
from core.map_context import build_default_map_title, build_map_metadata
from core.masking import (
    apply_keep_mask,
    auto_maximum_distance,
    combine_masks,
    convex_hull_keep_mask,
    maximum_distance_keep_mask,
)
from utils.constants import INCLUDE_COLUMN, INTERNAL_ROW_ID


LAYER_SCOPE_SELECTED = "Selected Layer"
LAYER_SCOPE_ALL = "All Layers"
LAYER_MAPPING_SCOPES = (LAYER_SCOPE_SELECTED, LAYER_SCOPE_ALL)
UNSPECIFIED_LAYER = "Unspecified"
MAP_STATUS_UP_TO_DATE = "Up to date"
MAP_STATUS_STALE = "Parameters changed - update required"
DOMAIN_WELL_DATA_EXTENT = "Well Data Extent"
DOMAIN_RESERVOIR_BOUNDARY_EXTENT = "Reservoir Boundary Extent"
DOMAIN_SELECTED_PANEL_UNION_EXTENT = "Selected Panel Union Extent"
LEGACY_DOMAIN_SELECTED_PANEL_EXTENT = "Selected Panel Extent"
INTERPOLATION_DOMAIN_OPTIONS = (
    DOMAIN_WELL_DATA_EXTENT,
    DOMAIN_RESERVOIR_BOUNDARY_EXTENT,
    DOMAIN_SELECTED_PANEL_UNION_EXTENT,
)


@dataclass(frozen=True)
class LayerGenerationStatus:
    layer: str
    status: str
    observations: int
    message: str = ""


@dataclass(frozen=True)
class LayerMapCollection:
    maps: dict[str, dict[str, object]]
    statuses: tuple[LayerGenerationStatus, ...]
    active_layer: str | None
    batch_signature: dict[str, object]


def normalize_layer_scope(layer_scope: str | None) -> str:
    return str(layer_scope) if layer_scope in LAYER_MAPPING_SCOPES else LAYER_SCOPE_SELECTED


def layer_column(mappings: dict[str, str | None]) -> str | None:
    column = mappings.get("layer")
    return column or None


def filter_values_excluding_semantics(
    mappings: dict[str, str | None],
    filter_values: dict[str, list[object]] | None,
    semantic_keys: tuple[str, ...] | list[str] | set[str],
) -> dict[str, list[object]]:
    excluded = {mappings.get(key) for key in semantic_keys if mappings.get(key)}
    return {
        column: list(values)
        for column, values in (filter_values or {}).items()
        if column not in excluded
    }


def reservoir_layer_values(df: pd.DataFrame, mappings: dict[str, str | None]) -> list[str]:
    column = layer_column(mappings)
    if not column or column not in df.columns:
        return []
    values = df[column].dropna().astype(str)
    values = values[values.str.strip() != ""]
    return sorted(values.drop_duplicates().tolist(), key=str)


def _attach_include_state(
    df: pd.DataFrame,
    include_state: dict[object, object] | None,
    include_col: str = INCLUDE_COLUMN,
) -> pd.DataFrame:
    output = df.copy()
    include_state = include_state or {}
    if INTERNAL_ROW_ID in output.columns:
        output[include_col] = output[INTERNAL_ROW_ID].map(
            lambda row_id: bool(include_state.get(int(row_id), True))
        )
    elif include_col not in output.columns:
        output[include_col] = True
    else:
        output[include_col] = output[include_col].astype(bool)
    return output


def _minimum_observations(method: str, parameters: dict[str, object]) -> int:
    method_key = method.strip().lower().replace(" ", "_")
    if method_key == "idw":
        return max(1, int(parameters.get("min_neighbors", 1) or 1))
    return 3


def resolve_layer_method_parameters(
    method: str,
    method_parameters: dict[str, object],
    prepared: pd.DataFrame,
) -> dict[str, object]:
    parameters = dict(method_parameters or {})
    if method != "Ordinary Kriging" or parameters.get("variogram_mode") != "Auto Fit":
        return parameters
    values = pd.to_numeric(prepared["Z"], errors="coerce").dropna()
    if len(prepared) < 5 or values.nunique() <= 1:
        return parameters
    experimental = compute_experimental_variogram(prepared["X"], prepared["Y"], prepared["Z"], n_lags=10)
    fits = fit_candidate_models(experimental)
    if not fits:
        return parameters
    best = fits[0]
    parameters.update(
        {
            "variogram_model": best.model,
            "variogram_range_convention": VARIOGRAM_RANGE_CONVENTION,
            "range": best.range_value,
            "variance": best.variance,
            "nugget": best.nugget,
            "fit_error": best.fit_error,
        }
    )
    return parameters


def batch_signature_dataframe(
    *,
    dataframe: pd.DataFrame,
    mappings: dict[str, str | None],
    property_column: str,
    property_type: str,
    pressure_reference_date,
    filter_values: dict[str, list[object]] | None,
    selected_panels: list[object] | tuple[object, ...] | None,
    requested_layers: list[str],
    panel_layer: GeometryLayer | None,
    include_state: dict[object, object] | None,
    x_col: str,
    y_col: str,
) -> pd.DataFrame:
    selected_layers = requested_layers if layer_column(mappings) else None
    active = prepare_active_property_data(
        dataframe,
        mappings,
        property_column,
        property_type,
        pressure_reference_date,
        filter_values_excluding_semantics(mappings, filter_values, ("layer",)),
        selected_panels if panel_layer is not None else None,
        selected_layers=selected_layers,
    )
    filtered = active.dataframe
    if selected_panels and panel_layer is not None and not mappings.get("panel"):
        filtered = filter_dataframe_to_selected_panels(filtered, x_col, y_col, panel_layer, selected_panels)
    return _attach_include_state(filtered, include_state)


def resolve_domain_bounds(
    interpolation_domain: str,
    reservoir_boundary_layer: GeometryLayer | None = None,
    panel_layer: GeometryLayer | None = None,
    selected_panels: list[object] | tuple[object, ...] | None = None,
) -> tuple[float, float, float, float] | None:
    if interpolation_domain == DOMAIN_RESERVOIR_BOUNDARY_EXTENT and reservoir_boundary_layer:
        reservoir_geometry = polygon_union(reservoir_boundary_layer.polygon_features)
        if reservoir_geometry is not None and not reservoir_geometry.is_empty:
            return tuple(float(value) for value in reservoir_geometry.bounds)
    if interpolation_domain in {DOMAIN_SELECTED_PANEL_UNION_EXTENT, LEGACY_DOMAIN_SELECTED_PANEL_EXTENT}:
        return selected_panel_bounds(panel_layer, selected_panels)
    return None


def normalize_interpolation_domain(interpolation_domain: str | None) -> str:
    if interpolation_domain == LEGACY_DOMAIN_SELECTED_PANEL_EXTENT:
        return DOMAIN_SELECTED_PANEL_UNION_EXTENT
    if interpolation_domain in {
        DOMAIN_WELL_DATA_EXTENT,
        DOMAIN_RESERVOIR_BOUNDARY_EXTENT,
        DOMAIN_SELECTED_PANEL_UNION_EXTENT,
    }:
        return str(interpolation_domain)
    return DOMAIN_WELL_DATA_EXTENT


def interpolation_domain_options(
    *,
    has_reservoir_boundary: bool = False,
    has_selected_panel_union: bool = False,
) -> list[str]:
    options = [DOMAIN_WELL_DATA_EXTENT]
    if has_reservoir_boundary:
        options.append(DOMAIN_RESERVOIR_BOUNDARY_EXTENT)
    if has_selected_panel_union:
        options.append(DOMAIN_SELECTED_PANEL_UNION_EXTENT)
    return options


def coerce_interpolation_domain_selection(
    interpolation_domain: str | None,
    options: list[str] | tuple[str, ...],
) -> str:
    valid_options = [option for option in options if option in INTERPOLATION_DOMAIN_OPTIONS]
    if not valid_options:
        return DOMAIN_WELL_DATA_EXTENT
    normalized = normalize_interpolation_domain(interpolation_domain)
    return normalized if normalized in valid_options else valid_options[0]


def domain_geometry_source(interpolation_domain: str | None) -> str:
    domain = normalize_interpolation_domain(interpolation_domain)
    if domain == DOMAIN_RESERVOIR_BOUNDARY_EXTENT:
        return "Reservoir Boundary"
    if domain == DOMAIN_SELECTED_PANEL_UNION_EXTENT:
        return "Selected Panel Union"
    return "Well Data"


def _apply_masks(
    grid_z: np.ndarray,
    grid_variance: np.ndarray | None,
    prepared: pd.DataFrame,
    grid_x: np.ndarray,
    grid_y: np.ndarray,
    mask_parameters: dict[str, object],
    reservoir_boundary_layer: GeometryLayer | None,
    panel_layer: GeometryLayer | None,
    selected_panels: list[object] | tuple[object, ...],
    apply_panel_domain: bool,
) -> tuple[np.ndarray, np.ndarray | None, dict[str, object]]:
    mode = str(mask_parameters.get("mode", "Convex Hull"))
    max_distance_value = mask_parameters.get("max_distance")
    masks: list[np.ndarray] = []
    if "Reservoir Boundary" in mode and reservoir_boundary_layer is not None:
        masks.append(layer_keep_mask(reservoir_boundary_layer, grid_x, grid_y))
    selected_panel_mask = "Selected Panel Union" in mode and panel_layer is not None and selected_panels
    if selected_panel_mask:
        masks.append(panel_domain_keep_mask(panel_layer, selected_panels, grid_x, grid_y))
    if "Convex Hull" in mode:
        masks.append(convex_hull_keep_mask(prepared["X"], prepared["Y"], grid_x, grid_y))
    if "Maximum Distance" in mode:
        if max_distance_value is None:
            max_distance_value = auto_maximum_distance(prepared["X"], prepared["Y"])
        masks.append(maximum_distance_keep_mask(prepared["X"], prepared["Y"], grid_x, grid_y, float(max_distance_value)))
    keep_mask = combine_masks(*masks) if masks else np.ones_like(grid_z, dtype=bool)
    masked_z = apply_keep_mask(grid_z, keep_mask)
    masked_variance = apply_keep_mask(grid_variance, keep_mask) if grid_variance is not None else None
    return masked_z, masked_variance, {
        "mask_mode": mode,
        "max_distance": max_distance_value,
        "masked_cells": int((~keep_mask).sum()),
        "valid_grid_cells": int(np.isfinite(masked_z).sum()),
        "finite_grid_cells_before_mask": int(np.isfinite(grid_z).sum()),
        "grid_cells_before_mask": int(grid_z.size),
        "panel_domain_masked_cells": int((~panel_domain_keep_mask(panel_layer, selected_panels, grid_x, grid_y)).sum())
        if selected_panel_mask
        else 0,
    }


def _interpolate_layer_surface(
    prepared: pd.DataFrame,
    method: str,
    method_parameters: dict[str, object],
    grid_parameters: dict[str, object],
    mask_parameters: dict[str, object],
    domain_bounds: tuple[float, float, float, float] | None,
    panel_interpolation_mode: str,
    panel_layer: GeometryLayer | None,
    reservoir_boundary_layer: GeometryLayer | None,
    selected_panels: list[object] | tuple[object, ...],
    dataset_panel_col: str | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None, dict[str, object]]:
    grid_x, grid_y = generate_grid(
        prepared["X"],
        prepared["Y"],
        nx=int(grid_parameters["nx"]),
        ny=int(grid_parameters["ny"]),
        buffer_fraction=float(grid_parameters.get("buffer_fraction", 0.0)),
        bounds=domain_bounds,
    )
    respect_compartments = respect_compartments_from_mode(panel_interpolation_mode)
    if respect_compartments and panel_layer is not None:
        result = compartment_interpolate(
            prepared,
            panel_layer,
            grid_x,
            grid_y,
            method,
            method_parameters,
            min_observations=max(3, int(method_parameters.get("min_neighbors", 3) or 3)),
            selected_panels=selected_panels,
            dataset_panel_col=dataset_panel_col,
        )
        grid_z, grid_variance, mask_info = _apply_masks(
            result.surface,
            result.variance,
            prepared,
            grid_x,
            grid_y,
            mask_parameters,
            reservoir_boundary_layer,
            panel_layer,
            selected_panels,
            apply_panel_domain=False,
        )
        mask_info["panel_constraint"] = True
        mask_info["compartment_warnings"] = result.warnings
        panel_grid = result.panel_grid
    else:
        surface = interpolate_surface_result(
            prepared["X"],
            prepared["Y"],
            prepared["Z"],
            grid_x,
            grid_y,
            method,
            method_parameters,
        )
        grid_z, grid_variance, mask_info = _apply_masks(
            surface.estimate,
            surface.variance,
            prepared,
            grid_x,
            grid_y,
            mask_parameters,
            reservoir_boundary_layer,
            panel_layer,
            selected_panels,
            apply_panel_domain=True,
        )
        mask_info["panel_constraint"] = False
        panel_grid = None
    if int(mask_info.get("valid_grid_cells", 0)) == 0:
        raise ValueError("Generated surface has no finite cells after masks were applied.")
    return grid_x, grid_y, grid_z, grid_variance, panel_grid, mask_info


def build_layer_model_signature(
    *,
    property_column: str,
    property_type: str,
    pressure_reference_date: date | str | None,
    selected_panels: list[str] | tuple[str, ...],
    reservoir_layer: str | None,
    layer_mapping_scope: str,
    panel_interpolation_mode: str,
    filter_values: dict[str, list[object]],
    active_dataframe: pd.DataFrame,
    duplicate_method: str,
    interpolation_method: str,
    interpolation_parameters: dict[str, object],
    grid_parameters: dict[str, object],
    interpolation_domain: str,
    domain_bounds,
    mask_parameters: dict[str, object],
    control_state: dict[str, object] | None = None,
    crs: dict[str, object] | None = None,
) -> dict[str, object]:
    selected_layers = [] if reservoir_layer is None else [str(reservoir_layer)]
    variogram = {
        "model": interpolation_parameters.get("variogram_model"),
        "range": interpolation_parameters.get("range"),
        "variance": interpolation_parameters.get("variance"),
        "nugget": interpolation_parameters.get("nugget"),
        "range_convention": interpolation_parameters.get("variogram_range_convention"),
    } if interpolation_method == "Ordinary Kriging" else {}
    return build_model_signature(
        property_column=property_column,
        property_type=property_type,
        pressure_reference_date=pressure_reference_date,
        selected_panels=selected_panels,
        selected_layers=selected_layers,
        panel_interpolation_mode=panel_interpolation_mode,
        filter_values=filter_values,
        active_dataframe=active_dataframe,
        duplicate_method=duplicate_method,
        interpolation_method=interpolation_method,
        interpolation_parameters={
            "method": interpolation_parameters,
            "grid": grid_parameters or {},
            "crs_epsg": normalize_crs_config(crs).get("epsg"),
        },
        layer_mapping_scope=normalize_layer_scope(layer_mapping_scope),
        interpolation_domain=interpolation_domain,
        domain_bounds=domain_bounds,
        mask_parameters=mask_parameters,
        control_state=control_state,
        variogram=variogram,
        anisotropy={
            "enabled": interpolation_parameters.get("anisotropy_enabled", False),
            "angle": interpolation_parameters.get("anisotropy_angle", 0.0),
            "ratio": interpolation_parameters.get("anisotropy_ratio", 1.0),
        },
    )


def prepare_layer_observations(
    *,
    dataframe: pd.DataFrame,
    mappings: dict[str, str | None],
    property_column: str,
    property_type: str,
    pressure_reference_date,
    filter_values: dict[str, list[object]] | None,
    selected_panels: list[object] | tuple[object, ...] | None,
    reservoir_layer: str | None,
    panel_layer: GeometryLayer | None,
    include_state: dict[object, object] | None,
    duplicate_method: str,
    x_col: str,
    y_col: str,
    well_col: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    layer_values = [reservoir_layer] if reservoir_layer is not None and layer_column(mappings) else None
    active = prepare_active_property_data(
        dataframe,
        mappings,
        property_column,
        property_type,
        pressure_reference_date,
        filter_values_excluding_semantics(mappings, filter_values, ("layer",)),
        selected_panels if panel_layer is not None else None,
        selected_layers=layer_values,
    )
    filtered = active.dataframe
    if selected_panels and panel_layer is not None and not mappings.get("panel"):
        filtered = filter_dataframe_to_selected_panels(filtered, x_col, y_col, panel_layer, selected_panels)
    filtered_with_include = _attach_include_state(filtered, include_state)
    prepared = prepare_interpolation_dataframe(
        filtered_with_include,
        x_col,
        y_col,
        property_column,
        include_col=INCLUDE_COLUMN,
        duplicate_method=duplicate_method,
        metadata_columns=[column for column in [well_col, mappings.get("panel"), layer_column(mappings)] if column],
        row_id_col=INTERNAL_ROW_ID,
    )
    return filtered_with_include, prepared


def generate_single_layer_map(
    *,
    dataframe: pd.DataFrame,
    mappings: dict[str, str | None],
    property_column: str,
    property_type: str,
    property_unit: str | None,
    pressure_reference_date,
    is_pressure_map: bool,
    filter_values: dict[str, list[object]] | None,
    selected_panels: list[object] | tuple[object, ...],
    reservoir_layer: str | None,
    layer_mapping_scope: str,
    panel_interpolation_mode: str,
    include_state: dict[object, object] | None,
    duplicate_method: str,
    interpolation_method: str,
    interpolation_parameters: dict[str, object],
    grid_parameters: dict[str, object],
    mask_parameters: dict[str, object],
    interpolation_domain: str,
    coordinate_unit: str,
    crs: dict[str, object] | None,
    panel_layer: GeometryLayer | None = None,
    reservoir_boundary_layer: GeometryLayer | None = None,
    control_points: list[dict[str, object]] | tuple[dict[str, object], ...] | None = None,
    control_regions: list[dict[str, object]] | tuple[dict[str, object], ...] | None = None,
    fault_layer_loaded: bool = False,
    custom_layer_count: int = 0,
    x_col: str = "X",
    y_col: str = "Y",
    well_col: str | None = None,
    hover_columns: list[tuple[str, str]] | None = None,
) -> dict[str, object]:
    interpolation_domain = normalize_interpolation_domain(interpolation_domain)
    domain_bounds = resolve_domain_bounds(interpolation_domain, reservoir_boundary_layer, panel_layer, selected_panels)
    filtered_with_include, prepared = prepare_layer_observations(
        dataframe=dataframe,
        mappings=mappings,
        property_column=property_column,
        property_type=property_type,
        pressure_reference_date=pressure_reference_date,
        filter_values=filter_values,
        selected_panels=selected_panels,
        reservoir_layer=reservoir_layer,
        panel_layer=panel_layer,
        include_state=include_state,
        duplicate_method=duplicate_method,
        x_col=x_col,
        y_col=y_col,
        well_col=well_col,
    )
    layer_name = str(reservoir_layer) if reservoir_layer is not None else UNSPECIFIED_LAYER
    minimum = _minimum_observations(interpolation_method, interpolation_parameters)
    control_selection = engineering_controls_for_context(
        control_points=control_points,
        control_regions=control_regions,
        measured_dataframe=filtered_with_include,
        mappings=mappings,
        property_col=property_column,
        property_type=property_type,
        property_unit=property_unit or "",
        pressure_reference_date=pressure_reference_date,
        reservoir_layer=reservoir_layer,
        selected_panels=selected_panels,
        panel_interpolation_mode=panel_interpolation_mode,
        x_col=x_col,
        y_col=y_col,
        panel_layer=panel_layer,
        reservoir_boundary_layer=reservoir_boundary_layer,
    )
    conditioning_input = filtered_with_include.copy()
    if not control_selection.dataframe.empty:
        conditioning_input = pd.concat([conditioning_input, control_selection.dataframe], ignore_index=True, sort=False)
    conditioning_prepared = prepare_interpolation_dataframe(
        conditioning_input,
        x_col,
        y_col,
        property_column,
        include_col=INCLUDE_COLUMN,
        duplicate_method=duplicate_method,
        metadata_columns=[
            column
            for column in [
                well_col,
                mappings.get("panel"),
                layer_column(mappings),
                "Control_ID",
                "Control_Type",
                "Source_Type",
                "Region_ID",
                "Region_Name",
            ]
            if column
        ],
        row_id_col=INTERNAL_ROW_ID,
    )
    if len(conditioning_prepared) < minimum:
        raise ValueError(
            f"{layer_name} has insufficient data ({len(conditioning_prepared)} available, {minimum} required)."
        )
    layer_parameters = resolve_layer_method_parameters(interpolation_method, interpolation_parameters, prepared)
    grid_x, grid_y, grid_z, grid_variance, panel_grid, mask_info = _interpolate_layer_surface(
        conditioning_prepared,
        interpolation_method,
        layer_parameters,
        grid_parameters,
        mask_parameters,
        domain_bounds,
        panel_interpolation_mode,
        panel_layer,
        reservoir_boundary_layer,
        selected_panels,
        mappings.get("panel"),
    )

    selected_panel_names = [str(value) for value in selected_panels or []]
    selected_layers = [] if reservoir_layer is None else [layer_name]
    filter_values_no_layer = filter_values_excluding_semantics(mappings, filter_values, ("layer",))
    signature = build_layer_model_signature(
        property_column=property_column,
        property_type=property_type,
        pressure_reference_date=pressure_reference_date,
        selected_panels=selected_panel_names,
        reservoir_layer=reservoir_layer,
        layer_mapping_scope=layer_mapping_scope,
        panel_interpolation_mode=panel_interpolation_mode,
        filter_values=filter_values_no_layer,
        active_dataframe=filtered_with_include,
        duplicate_method=duplicate_method,
        interpolation_method=interpolation_method,
        interpolation_parameters=layer_parameters,
        grid_parameters=grid_parameters,
        interpolation_domain=interpolation_domain,
        domain_bounds=domain_bounds,
        mask_parameters=mask_parameters,
        control_state=control_selection.signature_state,
        crs=crs,
    )

    reservoir_geometry = polygon_union(reservoir_boundary_layer.polygon_features) if reservoir_boundary_layer else None
    selected_features = selected_panel_features(panel_layer, selected_panel_names)
    selected_panel_domain_bounds = selected_panel_bounds(panel_layer, selected_panel_names)
    geometry_references = {
        "reservoir_boundary_name": reservoir_boundary_layer.name if reservoir_boundary_layer else "",
        "reservoir_boundary_source": reservoir_boundary_layer.source_name if reservoir_boundary_layer else "",
        "reservoir_boundary_bounds": tuple(float(value) for value in reservoir_geometry.bounds)
        if reservoir_geometry is not None
        else None,
        "panel_layer_name": panel_layer.name if panel_layer else "",
        "panel_layer_source": panel_layer.source_name if panel_layer else "",
        "selected_panel_names": selected_panel_names,
        "selected_panel_bounds": selected_panel_domain_bounds,
        "selected_panel_feature_count": len(selected_features),
        "panel_interpolation_mode": panel_interpolation_mode,
    }
    export_metadata = build_map_metadata(
        property_column,
        property_unit,
        x_col,
        y_col,
        coordinate_unit,
        interpolation_method,
        grid_parameters,
        layer_parameters,
        mask_parameters,
        duplicate_method,
        is_pressure_map=is_pressure_map,
        map_reference_date=pressure_reference_date,
        geometry_context={
            "reservoir_boundary_used": "Reservoir Boundary" in str(mask_parameters.get("mode", "")),
            "panel_constraint_used": panel_mode_from_legacy(panel_interpolation_mode) == PANEL_MODE_INDEPENDENT,
            "active_panels": ", ".join(selected_panel_names),
            "fault_layer_loaded": fault_layer_loaded,
            "custom_layer_count": custom_layer_count,
            **geometry_references,
        },
        property_type=property_type,
        crs=crs,
        selected_panels=selected_panel_names,
        selected_layers=selected_layers,
        panel_interpolation_mode=panel_interpolation_mode,
        interpolation_domain=interpolation_domain,
        domain_bounds=domain_bounds,
        grid_x=grid_x,
        grid_y=grid_y,
        model_signature_hash=signature["hash"],
        reservoir_layer=reservoir_layer,
        layer_mapping_scope=normalize_layer_scope(layer_mapping_scope),
        measured_observation_count=int((filtered_with_include[INCLUDE_COLUMN]).sum()),
        engineering_control_count=control_selection.control_count,
        control_region_count=len(control_selection.regions),
        control_point_ids=control_selection.dataframe.get("Control_ID", pd.Series(dtype=object)).dropna().tolist()
        if not control_selection.dataframe.empty
        else [],
        control_region_ids=[region.get("Region_ID") for region in control_selection.regions],
    )
    export_metadata["Interpolation_Domain_Type"] = interpolation_domain
    export_metadata["Domain_Geometry_Source"] = domain_geometry_source(interpolation_domain)
    if domain_bounds is not None:
        export_metadata["Domain_Bounds"] = {
            "min_x": float(domain_bounds[0]),
            "min_y": float(domain_bounds[1]),
            "max_x": float(domain_bounds[2]),
            "max_y": float(domain_bounds[3]),
        }
    title = build_default_map_title(
        property_column,
        filter_values_no_layer,
        mappings,
        is_pressure_map=is_pressure_map,
        map_reference_date=pressure_reference_date,
    )
    if reservoir_layer is not None:
        title = f"{title} - Reservoir Layer: {layer_name}"
    included = filtered_with_include[filtered_with_include[INCLUDE_COLUMN]].copy()
    excluded = filtered_with_include[~filtered_with_include[INCLUDE_COLUMN]].copy()
    well_extent = (
        float(prepared["X"].min()),
        float(prepared["Y"].min()),
        float(prepared["X"].max()),
        float(prepared["Y"].max()),
    )
    grid_extent = (float(grid_x.min()), float(grid_y.min()), float(grid_x.max()), float(grid_y.max()))
    mask_info.update(
        {
            "well_extent": well_extent,
            "reservoir_extent": domain_bounds,
            "generated_grid_extent": grid_extent,
            "grid_cells_inside_reservoir": int(layer_keep_mask(reservoir_boundary_layer, grid_x, grid_y).sum())
            if reservoir_boundary_layer is not None
            else int(grid_x.size),
        }
    )
    return {
        "layer_name": layer_name,
        "reservoir_layer": reservoir_layer,
        "layer_mapping_scope": normalize_layer_scope(layer_mapping_scope),
        "grid_x": grid_x,
        "grid_y": grid_y,
        "grid_z": grid_z,
        "grid_variance": grid_variance,
        "panel_grid": panel_grid,
        "included_observations": included,
        "excluded_observations": excluded,
        "engineering_controls": control_selection.dataframe.copy(),
        "engineering_control_points": control_selection.manual_points,
        "engineering_control_regions": control_selection.regions,
        "engineering_control_count": control_selection.control_count,
        "control_region_count": len(control_selection.regions),
        "control_warnings": list(control_selection.warnings),
        "measured_observation_count": int(len(included)),
        "conditioning_observation_count": int(len(conditioning_prepared)),
        "property_col": property_column,
        "unit": property_unit or "",
        "x_col": x_col,
        "y_col": y_col,
        "well_col": well_col,
        "coordinate_unit": coordinate_unit,
        "crs": normalize_crs_config(crs),
        "is_pressure_map": is_pressure_map,
        "property_type": property_type,
        "map_reference_date": pressure_reference_date,
        "measurement_date_col": mappings.get("measurement_date"),
        "map_reference_date_col": mappings.get("map_reference_date"),
        "method": interpolation_method,
        "method_parameters": layer_parameters,
        "grid_parameters": grid_parameters,
        "mask_parameters": mask_parameters,
        "interpolation_domain": interpolation_domain,
        "domain_bounds": domain_bounds,
        "mask_info": mask_info,
        "respect_compartments": panel_mode_from_legacy(panel_interpolation_mode) == PANEL_MODE_INDEPENDENT,
        "panel_interpolation_mode": panel_interpolation_mode,
        "selected_panels": selected_panel_names,
        "selected_layers": selected_layers,
        "geometry_context": {
            "reservoir_boundary_loaded": reservoir_boundary_layer is not None,
            "panel_layer_loaded": panel_layer is not None,
            "fault_layer_loaded": fault_layer_loaded,
            "custom_layer_count": custom_layer_count,
            **geometry_references,
        },
        "geometry_references": geometry_references,
        "duplicate_method": duplicate_method,
        "hover_columns": hover_columns or [],
        "title": title,
        "export_metadata": export_metadata,
        "model_signature": signature,
    }


def build_batch_signature(
    *,
    layer_scope: str,
    requested_layers: list[str],
    property_column: str,
    property_type: str,
    pressure_reference_date,
    selected_panels: list[str],
    panel_interpolation_mode: str,
    filter_values: dict[str, list[object]],
    duplicate_method: str,
    interpolation_method: str,
    interpolation_parameters: dict[str, object],
    grid_parameters: dict[str, object],
    mask_parameters: dict[str, object],
    interpolation_domain: str,
    domain_bounds,
    active_dataframe: pd.DataFrame | None = None,
    control_state: dict[str, object] | None = None,
) -> dict[str, object]:
    return build_model_signature(
        property_column=property_column,
        property_type=property_type,
        pressure_reference_date=pressure_reference_date,
        selected_panels=selected_panels,
        selected_layers=requested_layers,
        panel_interpolation_mode=panel_interpolation_mode,
        filter_values=filter_values,
        active_dataframe=active_dataframe,
        duplicate_method=duplicate_method,
        interpolation_method=interpolation_method,
        interpolation_parameters={
            "method": interpolation_parameters,
            "grid": grid_parameters,
        },
        layer_mapping_scope=normalize_layer_scope(layer_scope),
        interpolation_domain=interpolation_domain,
        domain_bounds=domain_bounds,
        mask_parameters=mask_parameters,
        control_state=control_state,
    )


def generate_layer_map_collection(
    *,
    dataframe: pd.DataFrame,
    mappings: dict[str, str | None],
    property_column: str,
    property_type: str,
    property_unit: str | None,
    pressure_reference_date,
    is_pressure_map: bool,
    filter_values: dict[str, list[object]] | None,
    selected_panels: list[object] | tuple[object, ...],
    layer_scope: str,
    selected_layer: str | None,
    panel_interpolation_mode: str,
    include_state: dict[object, object] | None,
    duplicate_method: str,
    interpolation_method: str,
    interpolation_parameters: dict[str, object],
    grid_parameters: dict[str, object],
    mask_parameters: dict[str, object],
    interpolation_domain: str,
    coordinate_unit: str,
    crs: dict[str, object] | None,
    panel_layer: GeometryLayer | None = None,
    reservoir_boundary_layer: GeometryLayer | None = None,
    control_points: list[dict[str, object]] | tuple[dict[str, object], ...] | None = None,
    control_regions: list[dict[str, object]] | tuple[dict[str, object], ...] | None = None,
    fault_layer_loaded: bool = False,
    custom_layer_count: int = 0,
    x_col: str = "X",
    y_col: str = "Y",
    well_col: str | None = None,
    hover_columns: list[tuple[str, str]] | None = None,
    progress_callback: Callable[[int, int, str], None] | None = None,
) -> LayerMapCollection:
    scope = normalize_layer_scope(layer_scope)
    filter_values_no_layer = filter_values_excluding_semantics(mappings, filter_values, ("layer",))
    if layer_column(mappings):
        panel_selection_for_layers = selected_panels if panel_layer is not None and mappings.get("panel") else None
        active_for_layers = prepare_active_property_data(
            dataframe,
            mappings,
            property_column,
            property_type,
            pressure_reference_date,
            filter_values_no_layer,
            panel_selection_for_layers,
        ).dataframe
        available_layers = reservoir_layer_values(active_for_layers, mappings)
    else:
        available_layers = []

    if scope == LAYER_SCOPE_ALL:
        if not layer_column(mappings):
            raise ValueError("All Layers requires a mapped Reservoir Layer column.")
        if not available_layers:
            raise ValueError("All Layers requested, but no valid Reservoir Layer values are available after filters.")
        requested_layers = available_layers
    elif layer_column(mappings):
        if selected_layer is None:
            selected_layer = available_layers[0] if available_layers else None
        if selected_layer is None or str(selected_layer) not in available_layers:
            raise ValueError("Selected Reservoir Layer has no active observations after filters.")
        requested_layers = [str(selected_layer)]
    else:
        requested_layers = [UNSPECIFIED_LAYER]

    maps: dict[str, dict[str, object]] = {}
    statuses: list[LayerGenerationStatus] = []
    total = len(requested_layers)
    for index, layer_name in enumerate(requested_layers, start=1):
        if progress_callback:
            progress_callback(index, total, layer_name)
        layer_value = None if not layer_column(mappings) else layer_name
        try:
            generated = generate_single_layer_map(
                dataframe=dataframe,
                mappings=mappings,
                property_column=property_column,
                property_type=property_type,
                property_unit=property_unit,
                pressure_reference_date=pressure_reference_date,
                is_pressure_map=is_pressure_map,
                filter_values=filter_values_no_layer,
                selected_panels=selected_panels,
                reservoir_layer=layer_value,
                layer_mapping_scope=scope,
                panel_interpolation_mode=panel_interpolation_mode,
                include_state=include_state,
                duplicate_method=duplicate_method,
                interpolation_method=interpolation_method,
                interpolation_parameters=interpolation_parameters,
                grid_parameters=grid_parameters,
                mask_parameters=mask_parameters,
                interpolation_domain=interpolation_domain,
                coordinate_unit=coordinate_unit,
                crs=crs,
                panel_layer=panel_layer,
                reservoir_boundary_layer=reservoir_boundary_layer,
                control_points=control_points,
                control_regions=control_regions,
                fault_layer_loaded=fault_layer_loaded,
                custom_layer_count=custom_layer_count,
                x_col=x_col,
                y_col=y_col,
                well_col=well_col,
                hover_columns=hover_columns,
            )
            maps[layer_name] = generated
            statuses.append(LayerGenerationStatus(layer_name, "Generated", len(generated["included_observations"])))
        except (InterpolationError, ValueError) as exc:
            statuses.append(LayerGenerationStatus(layer_name, "Insufficient data", 0, str(exc)))

    active_layer = None
    if maps:
        if selected_layer in maps:
            active_layer = str(selected_layer)
        else:
            active_layer = next(iter(maps))
    batch_active_dataframe = batch_signature_dataframe(
        dataframe=dataframe,
        mappings=mappings,
        property_column=property_column,
        property_type=property_type,
        pressure_reference_date=pressure_reference_date,
        filter_values=filter_values_no_layer,
        selected_panels=selected_panels,
        requested_layers=requested_layers,
        panel_layer=panel_layer,
        include_state=include_state,
        x_col=x_col,
        y_col=y_col,
    )
    batch_control_state = {
        layer_name: map_result.get("model_signature", {}).get("engineering_controls", {})
        for layer_name, map_result in maps.items()
    }
    batch_signature = build_batch_signature(
        layer_scope=scope,
        requested_layers=requested_layers,
        property_column=property_column,
        property_type=property_type,
        pressure_reference_date=pressure_reference_date,
        selected_panels=[str(value) for value in selected_panels or []],
        panel_interpolation_mode=panel_interpolation_mode,
        filter_values=filter_values_no_layer,
        duplicate_method=duplicate_method,
        interpolation_method=interpolation_method,
        interpolation_parameters=interpolation_parameters,
        grid_parameters=grid_parameters,
        mask_parameters=mask_parameters,
        interpolation_domain=interpolation_domain,
        domain_bounds=resolve_domain_bounds(interpolation_domain, reservoir_boundary_layer, panel_layer, selected_panels),
        active_dataframe=batch_active_dataframe,
        control_state=batch_control_state,
    )
    return LayerMapCollection(maps=maps, statuses=tuple(statuses), active_layer=active_layer, batch_signature=batch_signature)


def select_generated_layer_map(
    generated_layer_maps: dict[str, dict[str, object]],
    layer_name: str | None,
) -> dict[str, object] | None:
    if not generated_layer_maps:
        return None
    if layer_name in generated_layer_maps:
        return generated_layer_maps[layer_name]
    return generated_layer_maps[next(iter(generated_layer_maps))]


def map_status(current_signature: dict[str, object] | None, generated_map: dict[str, object] | None) -> str:
    if not generated_map:
        return MAP_STATUS_STALE
    return MAP_STATUS_UP_TO_DATE if signatures_match(generated_map.get("model_signature"), current_signature) else MAP_STATUS_STALE
