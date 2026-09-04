"""Mapping Studio page."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from shapely.geometry.base import BaseGeometry

from components.control_region_drawer import control_region_drawer
from core.active_data import prepare_active_property_data
from core.column_mapper import normalize_column_mappings, numeric_property_candidates
from core.data_qc import build_qc_summary, descriptive_statistics, flag_outliers
from core.engineering_controls import (
    CONTROL_POINT_COLUMNS,
    CONTROL_REGION_COLUMNS,
    DRAWN_CONTROL_REGION_SOURCE,
    EXISTING_GEOMETRY_REGION_SOURCE,
    REGION_CONTROL_SOURCE,
    SELECTED_WELLS_REGION_SOURCE,
    assign_panel_from_point,
    clip_control_region_to_active_domain,
    controls_dataframe,
    controls_for_plot,
    create_control_point,
    create_control_region,
    create_region_from_wells,
    engineering_controls_for_context,
    next_region_id,
    normalize_control_point,
    normalize_control_region,
    normalize_polygon_vertices,
    polygon_from_vertices,
    regions_dataframe,
    validate_control_region_parameters,
)
from core.filtering import build_filter_column_list
from core.geometry.assignment import assign_points_to_polygons, outside_panel_count
from core.geometry.compartment import filter_dataframe_to_selected_panels, selected_panel_bounds
from core.geometry.masking import polygon_union
from core.geostatistics.variogram import (
    VARIOGRAM_RANGE_CONVENTION,
    compute_experimental_variogram,
    fit_candidate_models,
)
from core.crs import crs_coordinate_unit_warning, crs_display_name
from core.interpolation.rbf import estimate_epsilon
from core.layer_mapping import (
    DOMAIN_RESERVOIR_BOUNDARY_EXTENT,
    DOMAIN_SELECTED_PANEL_UNION_EXTENT,
    DOMAIN_WELL_DATA_EXTENT,
    LAYER_SCOPE_ALL,
    LAYER_SCOPE_SELECTED,
    MAP_STATUS_UP_TO_DATE,
    UNSPECIFIED_LAYER,
    build_layer_model_signature,
    coerce_interpolation_domain_selection,
    filter_values_excluding_semantics,
    generate_layer_map_collection,
    interpolation_domain_options,
    layer_column,
    map_status,
    prepare_layer_observations,
    reservoir_layer_values,
    resolve_domain_bounds,
    resolve_layer_method_parameters,
    select_generated_layer_map,
)
from core.masking import auto_maximum_distance
from core.pressure_dates import format_map_date, summarize_measurement_dates
from core.plotting.geometry_layers import add_fault_layer, add_polygon_layer
from core.plotting.engineering_controls import add_control_region_overlays, add_engineering_control_traces
from core.plotting.map_builder import build_context_map_figure, build_map_figure, move_observation_traces_to_top
from core.plotting.styling import format_numeric
from core.scenarios import (
    create_map_scenario,
    duplicate_scenario,
    rename_scenario,
    scenario_summary_table,
    scenario_to_generated_map,
)
from core.unit_conversion import available_display_units, convert_values
from pages.shared import (
    attach_include_column,
    color_scale_options,
    coordinate_unit_input,
    ensure_session_state,
    layer_mapping_scope_control,
    mark_project_dirty,
    panel_interpolation_mode_control,
    panel_selection_control,
    pressure_reference_date_control,
    render_filter_controls,
    reservoir_layer_control,
    unit_input,
    update_include_state_from_editor,
)
from utils.constants import (
    DUPLICATE_METHODS,
    GRID_PRESETS,
    INCLUDE_COLUMN,
    INTERNAL_ROW_ID,
    INTERPOLATION_METHODS,
    MASK_OPTIONS,
    OUTLIER_COLUMN,
    RBF_KERNELS,
)
from utils.export import (
    GEOTIFF_NODATA,
    ZMAP_NULL_VALUE,
    batch_map_package_zip_bytes,
    build_map_export_metadata,
    control_regions_export_dataframe,
    dataframe_to_csv_bytes,
    dataframe_to_excel_bytes,
    drop_internal_columns,
    engineering_controls_export_dataframe,
    figure_to_image_bytes,
    grid_to_dataframe,
    grid_to_excel_bytes,
    grid_to_xyz_ascii_bytes,
    map_package_zip_bytes,
    map_base_filename,
    map_geotiff_export_files,
    map_zmap_export_files,
    metadata_to_json_bytes,
)
from utils.units import coordinate_unit_symbol, format_distance
from utils.validators import is_phi_property, is_pressure_property, skewness_is_high


ensure_session_state()


def auto_idw_search_radius(prepared: pd.DataFrame, buffer_fraction: float = 0.03) -> float:
    if prepared.empty:
        return 0.0
    x = pd.to_numeric(prepared["X"], errors="coerce").dropna()
    y = pd.to_numeric(prepared["Y"], errors="coerce").dropna()
    if x.empty or y.empty:
        return 0.0
    x_span = float(x.max() - x.min())
    y_span = float(y.max() - y.min())
    diagonal = float(np.hypot(x_span, y_span))
    if diagonal == 0:
        return 1.0
    return diagonal * (1.0 + 2.0 * max(buffer_fraction, 0.0))


def generated_hover_columns(mappings: dict[str, str | None]) -> list[tuple[str, str]]:
    columns = list(build_filter_column_list(mappings, st.session_state.get("additional_filter_columns", [])))
    used = {column for _, column in columns}
    for label, column in [("Reservoir Layer", mappings.get("layer")), ("Panel", mappings.get("panel"))]:
        if column and column not in used:
            columns.append((label, column))
            used.add(column)
    return columns


def convert_grid_for_display(grid: np.ndarray, source_unit: str | None, display_unit: str | None) -> np.ndarray:
    if (source_unit or "") == (display_unit or ""):
        return grid
    return convert_values(grid, source_unit, display_unit)


def convert_observations_for_display(
    observations: pd.DataFrame,
    property_col: str,
    source_unit: str | None,
    display_unit: str | None,
) -> pd.DataFrame:
    if observations.empty or (source_unit or "") == (display_unit or ""):
        return observations
    converted = observations.copy()
    converted[property_col] = convert_values(converted[property_col], source_unit, display_unit)
    return converted


def render_static_image_export_controls(figure, base_name: str) -> None:
    export_cols = st.columns(4)
    with export_cols[0]:
        width = st.number_input("Image Width", min_value=600, max_value=6000, value=1800, step=100)
    with export_cols[1]:
        height = st.number_input("Image Height", min_value=400, max_value=5000, value=1100, step=100)
    with export_cols[2]:
        scale = st.number_input("Image Scale", min_value=1.0, max_value=5.0, value=2.0, step=0.5)
    with export_cols[3]:
        show_legend = st.checkbox("Legend", value=True)
    export_title = st.text_input("Export Title", value=str(figure.layout.title.text or base_name))
    export_figure = go.Figure(figure.to_plotly_json())
    export_figure.update_layout(title=export_title, showlegend=show_legend)
    if st.button("PREPARE MAP IMAGE EXPORTS"):
        try:
            st.session_state.map_image_exports = {
                "png": figure_to_image_bytes(export_figure, "png", int(width), int(height), float(scale)),
                "svg": figure_to_image_bytes(export_figure, "svg", int(width), int(height), float(scale)),
                "pdf": figure_to_image_bytes(export_figure, "pdf", int(width), int(height), float(scale)),
                "base_name": base_name,
            }
            st.success("Image exports prepared.")
        except RuntimeError as exc:
            st.error(str(exc))
    exports = st.session_state.get("map_image_exports", {})
    if exports:
        button_cols = st.columns(3)
        button_cols[0].download_button("Download PNG", exports["png"], f"{exports['base_name']}.png", "image/png")
        button_cols[1].download_button("Download SVG", exports["svg"], f"{exports['base_name']}.svg", "image/svg+xml")
        button_cols[2].download_button("Download PDF", exports["pdf"], f"{exports['base_name']}.pdf", "application/pdf")


def render_static_image_exports(figure, base_name: str) -> None:
    with st.expander("Image Export", expanded=False):
        render_static_image_export_controls(figure, base_name)


def method_parameter_controls(
    method: str,
    prepared: pd.DataFrame,
    coordinate_unit: str,
    grid_buffer_fraction: float,
) -> dict[str, object]:
    unit_symbol = coordinate_unit_symbol(coordinate_unit)
    if method == "IDW":
        cols = st.columns(2)
        with cols[0]:
            power = st.number_input("Power", min_value=0.1, max_value=10.0, value=2.0, step=0.1)
            neighbors = st.number_input("Neighbors", min_value=1, max_value=200, value=12, step=1)
        with cols[1]:
            min_neighbors = st.number_input("Minimum Neighbors", min_value=1, max_value=50, value=3, step=1)
            radius_mode = st.radio("Search Radius Mode", ["Auto", "Manual"], horizontal=True)
        search_radius = auto_idw_search_radius(prepared, grid_buffer_fraction)
        if radius_mode == "Manual":
            search_radius = st.number_input(
                f"Manual Search Radius ({unit_symbol})",
                min_value=0.0001,
                value=float(search_radius or 1000.0),
                step=100.0,
            )
        else:
            st.caption(f"Auto-calculated search radius: {format_distance(search_radius, coordinate_unit)}")
        return {
            "power": power,
            "neighbors": int(neighbors),
            "min_neighbors": int(min_neighbors),
            "search_radius": search_radius,
            "search_radius_mode": radius_mode,
        }

    if method == "Linear":
        return {}

    if method == "Cubic":
        st.info("Cubic interpolation may overshoot observed property ranges and create smooth but potentially unrealistic reservoir values.")
        return {}

    if method == "RBF":
        cols = st.columns(2)
        with cols[0]:
            kernel_label = st.selectbox("Kernel", list(RBF_KERNELS), index=0)
            smoothing = st.number_input("Smoothing", min_value=0.0, value=0.0, step=0.1)
        with cols[1]:
            neighbor_mode = st.radio("Neighbors", ["All", "Limited"], horizontal=True)
            neighbors = None
            if neighbor_mode == "Limited":
                neighbors = st.number_input("Neighbor Count", min_value=1, max_value=500, value=30, step=1)
        epsilon = None
        kernel = RBF_KERNELS[kernel_label]
        if kernel in {"multiquadric", "gaussian"}:
            epsilon_mode = st.radio("Shape Parameter", ["Auto", "Manual"], horizontal=True)
            if epsilon_mode == "Manual":
                epsilon = st.number_input(
                    f"RBF Epsilon ({unit_symbol})",
                    min_value=0.0001,
                    value=1000.0,
                    step=100.0,
                )
            elif not prepared.empty:
                epsilon = estimate_epsilon(prepared["X"], prepared["Y"])
                st.caption(f"Auto RBF epsilon estimate: {format_distance(epsilon, coordinate_unit)}")
        return {"kernel": kernel, "smoothing": smoothing, "neighbors": neighbors, "epsilon": epsilon}

    if method == "Ordinary Kriging":
        values = pd.to_numeric(prepared["Z"], errors="coerce").dropna() if not prepared.empty else pd.Series(dtype=float)
        x_values = pd.to_numeric(prepared["X"], errors="coerce").dropna() if not prepared.empty else pd.Series(dtype=float)
        y_values = pd.to_numeric(prepared["Y"], errors="coerce").dropna() if not prepared.empty else pd.Series(dtype=float)
        x_span = float(x_values.max() - x_values.min()) if len(x_values) else 1.0
        y_span = float(y_values.max() - y_values.min()) if len(y_values) else 1.0
        default_range = max(float(np.hypot(x_span, y_span)) / 3.0, 1.0)
        default_variance = max(float(values.var(ddof=1)) if len(values) > 1 else 1.0, 1e-6)

        variogram_mode = st.radio("Variogram Parameters", ["Auto Fit", "Manual"], horizontal=True)
        best_fit = None
        fit_error = None
        if variogram_mode == "Auto Fit" and len(prepared) >= 5 and values.nunique() > 1:
            try:
                experimental = compute_experimental_variogram(prepared["X"], prepared["Y"], prepared["Z"], n_lags=10)
                fits = fit_candidate_models(experimental)
                if fits:
                    best_fit = fits[0]
                    fit_error = best_fit.fit_error
                    st.caption(
                        f"Best numerical variogram fit: {best_fit.model}; range "
                        f"{format_distance(best_fit.range_value, coordinate_unit)}, "
                        f"variance {best_fit.variance:.4g}, nugget {best_fit.nugget:.4g}."
                    )
            except ValueError as exc:
                st.warning(str(exc))

        cols = st.columns(3)
        with cols[0]:
            model = st.selectbox(
                "Model",
                ["Spherical", "Exponential", "Gaussian"],
                index=["Spherical", "Exponential", "Gaussian"].index(best_fit.model) if best_fit else 0,
            )
        with cols[1]:
            range_value = st.number_input(
                f"Range ({unit_symbol})",
                min_value=0.0001,
                value=float(best_fit.range_value if best_fit else default_range),
                step=max(default_range / 20.0, 1.0),
            )
        with cols[2]:
            variance = st.number_input(
                "Variance / Partial Sill",
                min_value=0.000001,
                value=float(best_fit.variance if best_fit else default_variance),
            )
        nugget = st.number_input("Nugget", min_value=0.0, value=float(best_fit.nugget if best_fit else 0.0))

        anis_cols = st.columns(3)
        with anis_cols[0]:
            anisotropy_enabled = st.checkbox("Enable Anisotropy", value=False)
        with anis_cols[1]:
            anisotropy_angle = st.number_input(
                "Major Continuity Direction (degrees)",
                min_value=0.0,
                max_value=180.0,
                value=0.0,
                help="0 degrees = +X direction, 90 degrees = +Y direction, counterclockwise positive.",
            )
        with anis_cols[2]:
            anisotropy_ratio = st.number_input(
                "Anisotropy Ratio",
                min_value=0.01,
                max_value=1.0,
                value=1.0,
                step=0.05,
                help="Minor range divided by major range.",
            )

        neigh_cols = st.columns(3)
        with neigh_cols[0]:
            neighborhood = st.radio("Kriging Neighborhood", ["All Observations", "Local"], horizontal=True)
        max_neighbors = None
        search_radius = None
        if neighborhood == "Local":
            with neigh_cols[1]:
                max_neighbors = st.number_input("Maximum Neighbors", min_value=3, max_value=500, value=30, step=1)
            with neigh_cols[2]:
                search_radius = st.number_input(
                    f"Search Radius ({unit_symbol})",
                    min_value=0.0001,
                    value=float(auto_idw_search_radius(prepared, grid_buffer_fraction)),
                    step=100.0,
                )
        return {
            "variogram_model": model,
            "variogram_mode": variogram_mode,
            "variogram_range_convention": VARIOGRAM_RANGE_CONVENTION,
            "range": range_value,
            "variance": variance,
            "nugget": nugget,
            "fit_error": fit_error,
            "anisotropy_enabled": anisotropy_enabled,
            "anisotropy_angle": anisotropy_angle,
            "anisotropy_ratio": anisotropy_ratio,
            "max_neighbors": int(max_neighbors) if max_neighbors is not None else None,
            "search_radius": search_radius,
        }

    return {}


def grid_controls() -> dict[str, object]:
    preset = st.selectbox("Grid Resolution", list(GRID_PRESETS), index=1)
    if preset == "Custom":
        cols = st.columns(2)
        with cols[0]:
            nx = st.number_input("NX", min_value=10, max_value=500, value=150, step=10)
        with cols[1]:
            ny = st.number_input("NY", min_value=10, max_value=500, value=150, step=10)
    else:
        nx, ny = GRID_PRESETS[preset]
        metric_cols = st.columns(2)
        metric_cols[0].metric("NX", nx)
        metric_cols[1].metric("NY", ny)
    buffer_percent = st.number_input("Buffer (%)", min_value=0.0, max_value=50.0, value=3.0, step=0.5)
    return {"preset": preset, "nx": int(nx), "ny": int(ny), "buffer_fraction": float(buffer_percent) / 100.0}


def mask_controls(
    prepared: pd.DataFrame,
    coordinate_unit: str,
    has_reservoir_boundary: bool = False,
    has_selected_panel_union: bool = False,
) -> dict[str, object]:
    unit_symbol = coordinate_unit_symbol(coordinate_unit)
    options = list(MASK_OPTIONS)
    if not has_reservoir_boundary:
        options = [option for option in options if "Reservoir Boundary" not in option]
    if not has_selected_panel_union:
        options = [option for option in options if "Selected Panel Union" not in option]
    if not options:
        options = ["No Mask"]
    mask_key = "mapping_mask_mode"
    if mask_key not in st.session_state:
        st.session_state[mask_key] = options[0]
    elif st.session_state[mask_key] not in options:
        st.session_state[mask_key] = options[0]
    mode = st.radio("Spatial Mask", options=options, key=mask_key)
    max_distance = None
    distance_mode = None
    if "Maximum Distance" in mode:
        cols = st.columns(2)
        with cols[0]:
            distance_mode = st.radio("Maximum Distance Mode", ["Auto", "Manual"], horizontal=True)
        if distance_mode == "Manual":
            with cols[1]:
                max_distance = st.number_input(
                    f"Maximum Distance ({unit_symbol})",
                    min_value=0.0001,
                    value=1000.0,
                    step=100.0,
                )
        else:
            max_distance = 0.0 if prepared.empty else float(auto_maximum_distance(prepared["X"], prepared["Y"]))
            cols[1].metric(f"Auto Distance ({unit_symbol})", f"{max_distance:,.4g}")
    return {"mode": mode, "max_distance": max_distance, "distance_mode": distance_mode}


def render_layer_manager() -> dict[str, object]:
    settings = dict(st.session_state.get("layer_settings", {}))
    geometry_layers = st.session_state.geometry_layers
    custom_layers = geometry_layers.get("custom", [])
    generated_available = bool(st.session_state.get("generated_map") or st.session_state.get("generated_layer_maps"))

    raw_default = bool(settings.get("show_raw_points", settings.get("show_wells", True)))
    settings["show_raw_points"] = st.checkbox("Raw Measured Points", value=raw_default)
    settings["show_wells"] = settings["show_raw_points"]
    settings["show_excluded"] = st.checkbox("Excluded Observations", value=bool(settings.get("show_excluded", True)))
    settings["show_surface"] = st.checkbox(
        "Property Surface",
        value=bool(settings.get("show_surface", True)),
        disabled=not generated_available,
    )
    settings["show_contours"] = st.checkbox(
        "Contours",
        value=bool(settings.get("show_contours", True)),
        disabled=not generated_available,
    )
    settings["show_engineering_controls"] = st.checkbox(
        "Engineering Controls",
        value=bool(settings.get("show_engineering_controls", True)),
    )
    settings["show_control_regions"] = st.checkbox(
        "Control Regions",
        value=bool(settings.get("show_control_regions", True)),
    )
    settings["show_region_control_points"] = st.checkbox(
        "Generated Region Points",
        value=bool(settings.get("show_region_control_points", False)),
    )
    if geometry_layers.get("reservoir_boundary"):
        settings["show_reservoir_boundary"] = st.checkbox(
            "Reservoir Boundary",
            value=bool(settings.get("show_reservoir_boundary", True)),
        )
        settings["reservoir_boundary_width"] = st.slider(
            "Boundary Width",
            min_value=0.5,
            max_value=6.0,
            value=float(settings.get("reservoir_boundary_width", 2.5)),
            step=0.5,
        )
    if geometry_layers.get("panels"):
        settings["show_panels"] = st.checkbox("Panel Boundaries", value=bool(settings.get("show_panels", True)))
        settings["show_panel_labels"] = st.checkbox("Panel Labels", value=bool(settings.get("show_panel_labels", False)))
        settings["panel_boundary_width"] = st.slider(
            "Panel Width",
            min_value=0.5,
            max_value=6.0,
            value=float(settings.get("panel_boundary_width", 1.5)),
            step=0.5,
        )
    if geometry_layers.get("faults"):
        settings["show_faults"] = st.checkbox("Faults", value=bool(settings.get("show_faults", True)))
        settings["show_fault_labels"] = st.checkbox("Fault Labels", value=bool(settings.get("show_fault_labels", False)))
        settings["fault_line_width"] = st.slider(
            "Fault Width",
            min_value=0.5,
            max_value=6.0,
            value=float(settings.get("fault_line_width", 2.0)),
            step=0.5,
        )
    if custom_layers:
        settings["show_custom_layers"] = st.checkbox("Custom Layers", value=bool(settings.get("show_custom_layers", True)))
        settings["show_custom_labels"] = st.checkbox("Custom Labels", value=bool(settings.get("show_custom_labels", False)))
        settings["custom_line_width"] = st.slider(
            "Custom Width",
            min_value=0.5,
            max_value=6.0,
            value=float(settings.get("custom_line_width", 1.5)),
            step=0.5,
        )
    settings["geometry_fill_opacity"] = st.slider(
        "Polygon Fill Opacity",
        min_value=0.0,
        max_value=0.6,
        value=float(settings.get("geometry_fill_opacity", 0.0)),
        step=0.05,
    )
    st.session_state.layer_settings = settings
    return settings


def finite_default(series, fallback: float = 0.0) -> float:
    values = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    return float(values.mean()) if not values.empty else fallback


def data_panel_options(frame: pd.DataFrame, panel_col: str | None) -> list[str]:
    if not panel_col or panel_col not in frame.columns:
        return []
    values = frame[panel_col].dropna().astype(str)
    values = values[values.str.strip() != ""]
    return sorted(values.drop_duplicates().tolist(), key=str)


def selected_region_geometry_choices(
    *,
    reservoir_boundary_layer,
    panel_layer,
    custom_layers,
    selected_panels: list[object] | tuple[object, ...],
) -> dict[str, dict[str, object]]:
    choices: dict[str, dict[str, object]] = {}
    selected_panel_names = {str(value) for value in selected_panels or [] if value not in (None, "")}

    def add_layer(prefix: str, layer, panel_from_feature: bool = False) -> None:
        if layer is None:
            return
        for index, feature in enumerate(layer.polygon_features):
            geometry = feature.geometry
            if geometry.is_empty or not geometry.is_valid:
                continue
            if panel_from_feature and selected_panel_names and str(feature.name) not in selected_panel_names:
                continue
            key = f"{prefix}:{index}"
            choices[key] = {
                "label": f"{layer.name} - {feature.name}",
                "geometry": geometry,
                "panel": str(feature.name) if panel_from_feature else "",
            }

    add_layer("reservoir", reservoir_boundary_layer)
    add_layer("panel", panel_layer, panel_from_feature=True)
    for custom_index, custom_layer in enumerate(custom_layers or []):
        add_layer(f"custom-{custom_index}", custom_layer)
    return choices


def well_seed_options(frame: pd.DataFrame, well_col: str | None, x_col: str, y_col: str) -> dict[str, str]:
    options: dict[str, str] = {}
    if frame.empty:
        return options
    for index, row in frame.iterrows():
        row_id = row.get(INTERNAL_ROW_ID, index)
        key = str(row_id)
        well_name = row.get(well_col) if well_col and well_col in frame.columns else f"Row {key}"
        x_value = pd.to_numeric(pd.Series([row.get(x_col)]), errors="coerce").iloc[0]
        y_value = pd.to_numeric(pd.Series([row.get(y_col)]), errors="coerce").iloc[0]
        x_label = "" if pd.isna(x_value) else f"{float(x_value):.6g}"
        y_label = "" if pd.isna(y_value) else f"{float(y_value):.6g}"
        options[key] = f"{well_name} ({x_label}, {y_label})"
    return options


def controls_display_frame(selection, property_col: str, source_unit: str | None, display_unit: str | None) -> pd.DataFrame:
    frame = controls_for_plot(selection)
    if frame.empty:
        return frame
    if property_col not in frame.columns:
        frame[property_col] = pd.to_numeric(frame["Value"], errors="coerce")
    frame = convert_observations_for_display(frame, property_col, source_unit, display_unit)
    frame["Value"] = frame[property_col]
    return frame


def plot_style_with_overlay_settings(
    style: dict[str, object],
    layer_settings: dict[str, object],
    *,
    generated_surface: bool,
) -> dict[str, object]:
    combined = {**style, **layer_settings}
    raw_points_visible = bool(style.get("show_wells", True)) and bool(
        layer_settings.get("show_raw_points", layer_settings.get("show_wells", True))
    )
    combined["show_raw_points"] = raw_points_visible
    combined["show_wells"] = raw_points_visible
    combined["show_excluded"] = bool(style.get("show_excluded", True)) and bool(layer_settings.get("show_excluded", True))
    if not generated_surface:
        combined["show_surface"] = False
        combined["show_contour_lines"] = False
        combined["show_contour_labels"] = False
    elif not bool(layer_settings.get("show_contours", True)):
        combined["show_contour_lines"] = False
        combined["show_contour_labels"] = False
    return combined


def split_observations_for_display(
    frame: pd.DataFrame,
    property_col: str,
    source_unit: str | None,
    display_unit: str | None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if frame.empty:
        return frame.copy(), frame.copy()
    included = frame[frame[INCLUDE_COLUMN]].copy() if INCLUDE_COLUMN in frame else frame.copy()
    excluded = frame[~frame[INCLUDE_COLUMN]].copy() if INCLUDE_COLUMN in frame else frame.iloc[0:0].copy()
    return (
        convert_observations_for_display(included, property_col, source_unit, display_unit),
        convert_observations_for_display(excluded, property_col, source_unit, display_unit),
    )


def _polygon_rings_payload(geometry: BaseGeometry | None) -> list[list[list[float]]]:
    if not isinstance(geometry, BaseGeometry) or geometry.is_empty:
        return []
    polygons = [geometry] if geometry.geom_type == "Polygon" else list(geometry.geoms) if geometry.geom_type == "MultiPolygon" else []
    rings: list[list[list[float]]] = []
    for polygon in polygons:
        x_values, y_values = polygon.exterior.xy
        rings.append([[float(x), float(y)] for x, y in zip(x_values, y_values)])
    return rings


def _line_points_payload(geometry: BaseGeometry | None) -> list[list[list[float]]]:
    if not isinstance(geometry, BaseGeometry) or geometry.is_empty:
        return []
    lines = [geometry] if geometry.geom_type == "LineString" else list(geometry.geoms) if geometry.geom_type == "MultiLineString" else []
    return [
        [[float(x), float(y)] for x, y in zip(*line.xy)]
        for line in lines
    ]


def geometry_polygons_payload(layer, *, stroke: str, fill: str, width: float) -> list[dict[str, object]]:
    if layer is None:
        return []
    payload: list[dict[str, object]] = []
    for feature in layer.polygon_features:
        rings = _polygon_rings_payload(feature.geometry)
        if rings:
            payload.append({"name": feature.name, "rings": rings, "stroke": stroke, "fill": fill, "width": width})
    return payload


def geometry_lines_payload(layer, *, stroke: str, width: float) -> list[dict[str, object]]:
    if layer is None:
        return []
    payload: list[dict[str, object]] = []
    for feature in layer.line_features:
        for points in _line_points_payload(feature.geometry):
            payload.append({"name": feature.name, "points": points, "stroke": stroke, "width": width})
    return payload


def control_regions_payload(regions: list[dict[str, object]] | tuple[dict[str, object], ...]) -> list[dict[str, object]]:
    payload: list[dict[str, object]] = []
    for region in regions or []:
        normalized = normalize_control_region(region)
        rings = _polygon_rings_payload(normalized.get("Geometry"))
        if rings:
            payload.append({"name": normalized.get("Region_Name") or normalized.get("Region_ID"), "rings": rings})
    return payload


def observation_points_payload(
    frame: pd.DataFrame,
    *,
    x_col: str,
    y_col: str,
    property_col: str,
    well_col: str | None,
    layer_col: str | None,
    panel_col: str | None,
    unit: str,
    is_pressure_map: bool,
    pressure_reference_date,
) -> list[dict[str, object]]:
    payload: list[dict[str, object]] = []
    if frame.empty:
        return payload
    for index, row in frame.iterrows():
        x_value = pd.to_numeric(pd.Series([row.get(x_col)]), errors="coerce").iloc[0]
        y_value = pd.to_numeric(pd.Series([row.get(y_col)]), errors="coerce").iloc[0]
        if pd.isna(x_value) or pd.isna(y_value) or not np.isfinite(float(x_value)) or not np.isfinite(float(y_value)):
            continue
        label = str(row.get(well_col)) if well_col and pd.notna(row.get(well_col)) else f"Row {index}"
        value = pd.to_numeric(pd.Series([row.get(property_col)]), errors="coerce").iloc[0]
        hover_lines = [
            f"Well: {label}",
            f"{property_col}: {format_numeric(value)} {unit}".strip(),
        ]
        if layer_col and layer_col in frame.columns and pd.notna(row.get(layer_col)):
            hover_lines.append(f"Reservoir Layer: {row.get(layer_col)}")
        if panel_col and panel_col in frame.columns and pd.notna(row.get(panel_col)):
            hover_lines.append(f"Panel: {row.get(panel_col)}")
        if is_pressure_map:
            hover_lines.append(f"Map Reference Date: {format_map_date(pressure_reference_date)}")
        hover_lines.append(f"{x_col}: {format_numeric(x_value)}")
        hover_lines.append(f"{y_col}: {format_numeric(y_value)}")
        payload.append(
            {
                "x": float(x_value),
                "y": float(y_value),
                "label": label,
                "value": None if pd.isna(value) else float(value),
                "hover": " | ".join(hover_lines),
            }
        )
    return payload


def controls_payload(controls: pd.DataFrame, *, x_col: str, y_col: str) -> list[dict[str, object]]:
    payload: list[dict[str, object]] = []
    if controls is None or controls.empty:
        return payload
    working = controls.copy()
    if x_col not in working.columns and "X" in working.columns:
        working[x_col] = working["X"]
    if y_col not in working.columns and "Y" in working.columns:
        working[y_col] = working["Y"]
    for _, row in working.iterrows():
        x_value = pd.to_numeric(pd.Series([row.get(x_col)]), errors="coerce").iloc[0]
        y_value = pd.to_numeric(pd.Series([row.get(y_col)]), errors="coerce").iloc[0]
        if pd.isna(x_value) or pd.isna(y_value) or not np.isfinite(float(x_value)) or not np.isfinite(float(y_value)):
            continue
        payload.append(
            {
                "x": float(x_value),
                "y": float(y_value),
                "label": str(row.get("Control_ID") or ""),
                "region_id": str(row.get("Region_ID") or ""),
                "hover": f"Control: {row.get('Control_ID') or ''} | X {format_numeric(x_value)} | Y {format_numeric(y_value)}",
            }
        )
    return payload


def controls_for_overlay(controls: pd.DataFrame, layer_settings: dict[str, object]) -> pd.DataFrame:
    if controls is None or controls.empty:
        return pd.DataFrame()
    source = controls.get("Source_Type", pd.Series("", index=controls.index)).astype(str)
    manual_keep = (source != REGION_CONTROL_SOURCE) & bool(layer_settings.get("show_engineering_controls", True))
    region_keep = (source == REGION_CONTROL_SOURCE) & bool(layer_settings.get("show_region_control_points", False))
    return controls.loc[manual_keep | region_keep].copy()


def drawing_vertices_state() -> list[dict[str, float]]:
    try:
        return normalize_polygon_vertices(st.session_state.get("control_region_draw_vertices", []))
    except ValueError:
        vertices = []
        for vertex in st.session_state.get("control_region_draw_vertices", []) or []:
            if isinstance(vertex, dict) and "x" in vertex and "y" in vertex:
                vertices.append({"X": vertex["x"], "Y": vertex["y"]})
            elif isinstance(vertex, dict):
                vertices.append({"X": vertex.get("X"), "Y": vertex.get("Y")})
        return vertices


def draw_payload_bounds(*payload_groups: list[dict[str, object]]) -> dict[str, float]:
    xs: list[float] = []
    ys: list[float] = []

    def collect_point(x_value, y_value) -> None:
        x = pd.to_numeric(pd.Series([x_value]), errors="coerce").iloc[0]
        y = pd.to_numeric(pd.Series([y_value]), errors="coerce").iloc[0]
        if pd.notna(x) and pd.notna(y) and np.isfinite(float(x)) and np.isfinite(float(y)):
            xs.append(float(x))
            ys.append(float(y))

    for group in payload_groups:
        for item in group:
            if "x" in item and "y" in item:
                collect_point(item.get("x"), item.get("y"))
            for ring in item.get("rings", []) or []:
                for point in ring:
                    if isinstance(point, (list, tuple)) and len(point) >= 2:
                        collect_point(point[0], point[1])
            for point in item.get("points", []) or []:
                if isinstance(point, (list, tuple)) and len(point) >= 2:
                    collect_point(point[0], point[1])
    if not xs or not ys:
        return {}
    return {"x_min": min(xs), "x_max": max(xs), "y_min": min(ys), "y_max": max(ys)}


def render_vertices_editor(vertices_key: str, editor_key: str) -> None:
    vertices = st.session_state.get(vertices_key, []) or []
    if not vertices:
        return
    rows = []
    for index, vertex in enumerate(vertices, start=1):
        x_value, y_value = (vertex.get("x"), vertex.get("y")) if isinstance(vertex, dict) else (None, None)
        if isinstance(vertex, dict):
            x_value = vertex.get("X", x_value)
            y_value = vertex.get("Y", y_value)
        rows.append({"Vertex": index, "X": x_value, "Y": y_value})
    table = pd.DataFrame(rows)
    edited = st.data_editor(
        table,
        width="stretch",
        hide_index=True,
        disabled=["Vertex"],
        column_config={
            "X": st.column_config.NumberColumn("X"),
            "Y": st.column_config.NumberColumn("Y"),
        },
        key=editor_key,
    )
    if not edited[["X", "Y"]].equals(table[["X", "Y"]]):
        st.session_state[vertices_key] = [
            {"X": row.get("X"), "Y": row.get("Y")}
            for _, row in edited.iterrows()
        ]
        st.rerun()


def statuses_frame(statuses: list[object] | tuple[object, ...]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for status in statuses or []:
        if isinstance(status, dict):
            rows.append(
                {
                    "Layer": status.get("layer", ""),
                    "Status": status.get("status", ""),
                    "Observations": status.get("observations", 0),
                    "Message": status.get("message", ""),
                }
            )
        else:
            rows.append(
                {
                    "Layer": getattr(status, "layer", ""),
                    "Status": getattr(status, "status", ""),
                    "Observations": getattr(status, "observations", 0),
                    "Message": getattr(status, "message", ""),
                }
            )
    return pd.DataFrame(rows)


def layer_map_key(generated_map: dict[str, object]) -> str:
    layer_name = generated_map.get("reservoir_layer")
    if layer_name not in (None, ""):
        return str(layer_name)
    selected_layers = generated_map.get("selected_layers") or []
    if selected_layers:
        return str(selected_layers[0])
    return UNSPECIFIED_LAYER


def current_open_scenario() -> dict[str, object] | None:
    current_id = st.session_state.get("current_scenario_id")
    if not current_id:
        return None
    for scenario in st.session_state.get("map_scenarios", []) or []:
        if scenario.get("id") == current_id:
            return scenario
    return None


def render_status(status: str) -> None:
    if status == MAP_STATUS_UP_TO_DATE:
        st.success(f"Map Status: {status}")
    else:
        st.warning(f"Map Status: {status}")


def render_style_controls(filtered_with_include: pd.DataFrame, property_col: str, well_col: str | None) -> dict[str, object]:
    style = dict(st.session_state.get("style_settings", {}))
    scale_options = color_scale_options()
    current_scale = style.get("color_scale", "Turbo")
    if current_scale not in scale_options:
        current_scale = "Turbo"
    style["color_scale"] = st.selectbox("Color Scale", scale_options, index=scale_options.index(current_scale))
    style["reverse_colors"] = st.checkbox("Reverse Colors", value=bool(style.get("reverse_colors", False)))
    style["height"] = st.slider("Map Height", min_value=450, max_value=1000, value=int(style.get("height", 720)), step=25)
    style["z_range_mode"] = st.radio(
        "Property Range",
        ["Auto", "Manual"],
        index=0 if style.get("z_range_mode") == "Auto" else 1,
        horizontal=True,
    )
    if style["z_range_mode"] == "Manual":
        current_values = pd.to_numeric(filtered_with_include[property_col], errors="coerce").dropna()
        default_min = float(current_values.min()) if not current_values.empty else 0.0
        default_max = float(current_values.max()) if not current_values.empty else 1.0
        cols = st.columns(2)
        with cols[0]:
            style["zmin"] = st.number_input("Minimum", value=float(style.get("zmin") or default_min))
        with cols[1]:
            style["zmax"] = st.number_input("Maximum", value=float(style.get("zmax") or default_max))

    style["contour_mode"] = st.radio(
        "Contours",
        ["Auto interval", "Manual interval"],
        index=0 if style.get("contour_mode") == "Auto interval" else 1,
        horizontal=True,
    )
    if style["contour_mode"] == "Manual interval":
        style["contour_interval"] = st.number_input(
            "Interval",
            min_value=0.000001,
            value=float(style.get("contour_interval") or 10.0),
        )
    cols = st.columns(2)
    with cols[0]:
        style["show_contour_lines"] = st.checkbox("Show Contour Lines", value=bool(style.get("show_contour_lines", True)))
        style["show_wells"] = st.checkbox("Show Wells", value=bool(style.get("show_wells", True)))
        style["marker_outline"] = st.checkbox("Marker Outline", value=bool(style.get("marker_outline", True)))
    with cols[1]:
        style["show_contour_labels"] = st.checkbox("Show Contour Labels", value=bool(style.get("show_contour_labels", False)))
        style["show_excluded"] = st.checkbox("Show Excluded Observations", value=bool(style.get("show_excluded", True)))

    style["contour_line_width"] = st.slider(
        "Contour Width",
        min_value=0.0,
        max_value=3.0,
        value=float(style.get("contour_line_width", 0.75)),
        step=0.25,
    )
    style["marker_size"] = st.slider("Marker Size", min_value=4, max_value=20, value=int(style.get("marker_size", 9)))
    style["marker_opacity"] = st.slider(
        "Marker Opacity",
        min_value=0.1,
        max_value=1.0,
        value=float(style.get("marker_opacity", 0.9)),
    )
    label_options = ["None", "Property Value"]
    if well_col:
        label_options = ["None", "Well Name", "Property Value", "Well Name + Property Value"]
    current_label = style.get("well_label_mode", "None")
    if current_label not in label_options:
        current_label = "None"
    style["well_label_mode"] = st.selectbox("Well Labels", label_options, index=label_options.index(current_label))
    style["label_text_size"] = st.slider("Label Text Size", min_value=8, max_value=18, value=int(style.get("label_text_size", 11)))
    style["title_override"] = st.text_input("Map Title Override", value=style.get("title_override", ""))
    st.session_state.style_settings = style
    return style


def add_geometry_overlays(
    figure,
    *,
    reservoir_boundary_layer,
    panel_layer,
    fault_layer,
    custom_layers,
    layer_settings: dict[str, object],
    coordinate_unit: str,
    show_debug_boundary: bool,
) -> None:
    add_polygon_layer(
        figure,
        reservoir_boundary_layer,
        coordinate_unit,
        visible=bool(layer_settings.get("show_reservoir_boundary", True)) and show_debug_boundary,
        line_color="#0F172A",
        line_width=float(layer_settings.get("reservoir_boundary_width", 2.5)),
        fill_opacity=float(layer_settings.get("geometry_fill_opacity", 0.0)),
        name="Reservoir Boundary",
    )
    add_polygon_layer(
        figure,
        panel_layer,
        coordinate_unit,
        visible=bool(layer_settings.get("show_panels", True)),
        show_labels=bool(layer_settings.get("show_panel_labels", False)),
        line_color="#92400E",
        line_width=float(layer_settings.get("panel_boundary_width", 1.5)),
        fill_opacity=0.0,
        name="Panel Boundaries",
    )
    add_fault_layer(
        figure,
        fault_layer,
        visible=bool(layer_settings.get("show_faults", True)),
        show_labels=bool(layer_settings.get("show_fault_labels", False)),
        line_width=float(layer_settings.get("fault_line_width", 2.0)),
    )
    if bool(layer_settings.get("show_custom_layers", True)):
        for custom_layer in custom_layers:
            add_polygon_layer(
                figure,
                custom_layer,
                coordinate_unit,
                visible=True,
                show_labels=bool(layer_settings.get("show_custom_labels", False)),
                line_color="#2563EB",
                line_width=float(layer_settings.get("custom_line_width", 1.5)),
                fill_opacity=float(layer_settings.get("geometry_fill_opacity", 0.0)),
                name=custom_layer.name,
            )
            add_fault_layer(
                figure,
                custom_layer,
                visible=True,
                show_labels=bool(layer_settings.get("show_custom_labels", False)),
                line_color="#2563EB",
                line_width=float(layer_settings.get("custom_line_width", 1.5)),
                line_dash="solid",
                label_prefix="Layer",
                name=custom_layer.name,
            )


def extent_text(extent) -> str:
    return "Unavailable" if extent is None else f"X {extent[0]:,.6g} to {extent[2]:,.6g}; Y {extent[1]:,.6g} to {extent[3]:,.6g}"


st.title("Reservoir Mapping Studio")
st.subheader("Mapping Studio")

df = st.session_state.get("working_df")
if df is None:
    st.info("Load data in the Data Manager before opening the Mapping Studio.")
    st.stop()

mappings = normalize_column_mappings(st.session_state.get("column_mappings", {}))
st.session_state.column_mappings = mappings
x_col = mappings.get("x")
y_col = mappings.get("y")
well_col = mappings.get("well")
geometry_layers = st.session_state.geometry_layers
reservoir_boundary_layer = geometry_layers.get("reservoir_boundary")
panel_layer = geometry_layers.get("panels")
fault_layer = geometry_layers.get("faults")
custom_layers = geometry_layers.get("custom", [])
if not x_col or not y_col:
    st.warning("Map X and Y coordinate columns in the Data Manager before generating a reservoir map.")
    st.stop()

property_options = numeric_property_candidates(df, mappings)
if not property_options:
    st.warning("No numeric property columns are available for mapping.")
    st.stop()

controls_col, map_col = st.columns([0.36, 0.64], gap="large")

with controls_col:
    with st.expander("Property", expanded=True):
        coordinate_unit = coordinate_unit_input("mapping_studio_coordinate_unit")
        st.caption(
            f"Spatial distances use {coordinate_unit_symbol(coordinate_unit)}. "
            "Uploaded X/Y coordinates are not converted."
        )
        unit_warning = crs_coordinate_unit_warning(st.session_state.get("crs", {}), coordinate_unit)
        if unit_warning:
            st.warning(unit_warning)
        current_property = st.session_state.get("current_property")
        if current_property not in property_options:
            current_property = property_options[0]
        property_col = st.selectbox(
            "Property",
            property_options,
            index=property_options.index(current_property),
            key="mapping_property",
        )
        st.session_state.current_property = property_col
        unit = unit_input("mapping_unit")
        display_unit_options = available_display_units(unit)
        current_display_unit = st.session_state.get("display_property_unit", unit)
        if current_display_unit not in display_unit_options:
            current_display_unit = unit if unit in display_unit_options else display_unit_options[0]
        display_unit = st.selectbox(
            "Convert Display To",
            display_unit_options,
            index=display_unit_options.index(current_display_unit),
            help="Display-only conversion. The source data and saved computational grid remain in the original property unit.",
        )
        st.session_state.display_property_unit = display_unit
        inferred_type = "Pressure" if is_pressure_property(property_col) else "Generic"
        property_type = st.radio(
            "Property Type",
            ["Pressure", "Generic"],
            index=0 if inferred_type == "Pressure" else 1,
            horizontal=True,
            key=f"property_type_{property_col}",
            help="Auto-detection can be overridden when a pressure-like column name is ambiguous.",
        )

is_pressure_map = property_type == "Pressure"
pressure_reference_date = None

with controls_col:
    with st.expander("Filters, Panels, Layers", expanded=True):
        filtered_base = render_filter_controls(
            df,
            mappings,
            "mapping_studio",
            exclude_semantic_keys=("layer",),
        )
        filter_values_no_layer = filter_values_excluding_semantics(
            mappings,
            st.session_state.get("filter_values", {}),
            ("layer",),
        )

        if panel_layer is not None and panel_layer.polygon_features:
            selected_panels = panel_selection_control(panel_layer, "mapping_selected_panels")
            panel_interpolation_mode = panel_interpolation_mode_control(True, "mapping_panel_interpolation_mode")
        else:
            selected_panels = []
            panel_interpolation_mode = panel_interpolation_mode_control(False, "mapping_panel_interpolation_mode")

        if is_pressure_map:
            st.info(
                "Pressure values should already be prepared/extrapolated to the map reference date before import. "
                "Reservoir Mapping Studio performs spatial interpolation only."
            )
            pressure_reference_date, _ = pressure_reference_date_control(
                filtered_base,
                mappings,
                "mapping_pressure_reference_date_input",
            )

        active_for_layers = pd.DataFrame()
        try:
            panel_selection_for_layers = selected_panels if panel_layer is not None and mappings.get("panel") else None
            active_for_layers = prepare_active_property_data(
                df,
                mappings,
                property_col,
                property_type,
                pressure_reference_date,
                filter_values_no_layer,
                panel_selection_for_layers,
            ).dataframe
        except ValueError as exc:
            st.warning(str(exc))

        layer_col = layer_column(mappings)
        has_layer_column = bool(layer_col and layer_col in df.columns)
        layer_options = reservoir_layer_values(active_for_layers, mappings) if has_layer_column else []
        layer_scope = layer_mapping_scope_control(has_layer_column, "mapping_layer_mapping_scope")
        selected_layer = None
        if has_layer_column:
            if layer_options:
                if layer_scope == LAYER_SCOPE_SELECTED:
                    selected_layer = reservoir_layer_control(layer_options, "mapping_selected_reservoir_layer")
                else:
                    if st.session_state.get("selected_reservoir_layer") not in layer_options:
                        st.session_state.selected_reservoir_layer = layer_options[0]
                    selected_layer = st.session_state.get("selected_reservoir_layer")
                    st.caption(f"{len(layer_options):,} active reservoir layer(s) will be generated independently.")
            else:
                st.warning("No Reservoir Layer values are available after the active filters.")
        else:
            st.caption("Map a Layer column in the Data Manager to enable reservoir-layer selection.")

generated_layer_maps = dict(st.session_state.get("generated_layer_maps", {}) or {})
if generated_layer_maps:
    active_generated_layer = st.session_state.get("active_generated_layer")
    if active_generated_layer not in generated_layer_maps:
        active_generated_layer = next(iter(generated_layer_maps))
        st.session_state.active_generated_layer = active_generated_layer
    st.session_state.generated_map = generated_layer_maps[active_generated_layer]


def observation_layer_for_controls() -> str | None:
    if not has_layer_column:
        return None
    if layer_scope == LAYER_SCOPE_SELECTED:
        return selected_layer
    active_layer = st.session_state.get("active_generated_layer")
    if active_layer in layer_options:
        return str(active_layer)
    return layer_options[0] if layer_options else None


def prepare_observations_for_layer(reservoir_layer: str | None) -> tuple[pd.DataFrame, pd.DataFrame]:
    if has_layer_column and reservoir_layer is None:
        empty = df.iloc[0:0].copy()
        return attach_include_column(empty), pd.DataFrame()
    return prepare_layer_observations(
        dataframe=df,
        mappings=mappings,
        property_column=property_col,
        property_type=property_type,
        pressure_reference_date=pressure_reference_date,
        filter_values=filter_values_no_layer,
        selected_panels=selected_panels,
        reservoir_layer=reservoir_layer,
        panel_layer=panel_layer,
        include_state=st.session_state.get("include_state", {}),
        duplicate_method=st.session_state.get("mapping_duplicate_method", DUPLICATE_METHODS[0]),
        x_col=x_col,
        y_col=y_col,
        well_col=well_col,
    )


def control_selection_for_layer(reservoir_layer: str | None, measured_dataframe: pd.DataFrame):
    return engineering_controls_for_context(
        control_points=st.session_state.get("engineering_control_points", []),
        control_regions=st.session_state.get("engineering_control_regions", []),
        measured_dataframe=measured_dataframe,
        mappings=mappings,
        property_col=property_col,
        property_type=property_type,
        property_unit=unit,
        pressure_reference_date=pressure_reference_date,
        reservoir_layer=None if reservoir_layer in (None, UNSPECIFIED_LAYER) else str(reservoir_layer),
        selected_panels=selected_panels,
        panel_interpolation_mode=panel_interpolation_mode,
        x_col=x_col,
        y_col=y_col,
        panel_layer=panel_layer,
        reservoir_boundary_layer=reservoir_boundary_layer,
    )


preview_layer = observation_layer_for_controls()
filtered_with_include, prepared = prepare_observations_for_layer(preview_layer)
filtered_with_include = flag_outliers(filtered_with_include, property_col)

property_values = (
    pd.to_numeric(filtered_with_include[property_col], errors="coerce")
    if property_col in filtered_with_include
    else pd.Series(dtype=float)
)
with controls_col:
    with st.expander("Active Observations", expanded=False):
        if has_layer_column:
            st.caption(f"Preview layer: {preview_layer or 'None'}")
        if is_phi_property(property_col):
            valid_phi = property_values.replace([np.inf, -np.inf], np.nan).dropna()
            if len(valid_phi):
                fraction_share = ((valid_phi >= 0) & (valid_phi <= 1)).mean()
                percent_share = ((valid_phi > 1) & (valid_phi <= 100)).mean()
                if fraction_share > 0.7:
                    st.caption("Porosity-like values mostly fall between 0 and 1 and may represent fractions.")
                elif percent_share > 0.7:
                    st.caption("Porosity-like values mostly fall between 1 and 100 and may represent percentages.")
        if skewness_is_high(property_values):
            st.info("This property is strongly skewed. Deterministic interpolation may be sensitive to high-value observations.")

        display_columns = [INCLUDE_COLUMN]
        for column in [well_col, x_col, y_col, property_col, layer_col]:
            if column and column in filtered_with_include.columns and column not in display_columns:
                display_columns.append(column)
        if is_pressure_map:
            for column in [mappings.get("measurement_date"), mappings.get("map_reference_date")]:
                if column and column in filtered_with_include.columns and column not in display_columns:
                    display_columns.append(column)
        for _, column in build_filter_column_list(mappings, st.session_state.get("additional_filter_columns", [])):
            if column not in display_columns and column in filtered_with_include.columns:
                display_columns.append(column)
        if OUTLIER_COLUMN in filtered_with_include.columns:
            display_columns.append(OUTLIER_COLUMN)
        display_columns_with_id = [INTERNAL_ROW_ID] + [column for column in display_columns if column != INTERNAL_ROW_ID]

        if filtered_with_include.empty:
            st.warning("No observations are active for the selected property, filters, panels, and layer.")
        else:
            edited = st.data_editor(
                filtered_with_include[display_columns_with_id],
                width="stretch",
                hide_index=True,
                disabled=[column for column in display_columns_with_id if column != INCLUDE_COLUMN],
                column_config={
                    INTERNAL_ROW_ID: st.column_config.NumberColumn("Row ID", disabled=True),
                    INCLUDE_COLUMN: st.column_config.CheckboxColumn("Include"),
                },
                key="observation_editor",
            )
            update_include_state_from_editor(edited)
            filtered_with_include, prepared = prepare_observations_for_layer(preview_layer)
            filtered_with_include = flag_outliers(filtered_with_include, property_col)

        included = (
            filtered_with_include[filtered_with_include[INCLUDE_COLUMN]]
            if INCLUDE_COLUMN in filtered_with_include
            else pd.DataFrame()
        )
        stats = descriptive_statistics(
            pd.to_numeric(included[property_col], errors="coerce") if property_col in included else pd.Series(dtype=float)
        )
        stat_cols = st.columns(2)
        stat_cols[0].metric("Included", f"{stats['observations']:,}")
        stat_cols[1].metric("Mean", "" if stats["mean"] is None else f"{stats['mean']:.4g}")
        stat_cols_2 = st.columns(3)
        stat_cols_2[0].metric("Minimum", "" if stats["minimum"] is None else f"{stats['minimum']:.4g}")
        stat_cols_2[1].metric("Median", "" if stats["median"] is None else f"{stats['median']:.4g}")
        stat_cols_2[2].metric("Maximum", "" if stats["maximum"] is None else f"{stats['maximum']:.4g}")

        qc = build_qc_summary(filtered_with_include, x_col, y_col, property_col, well_col)
        if qc.missing_property:
            st.warning(f"{qc.missing_property} record(s) have missing or non-numeric {property_col} values.")
        if qc.duplicate_xy_rows:
            st.warning(f"{qc.duplicate_xy_rows} record(s) share duplicate XY coordinates. Choose duplicate handling before interpolation.")
        if qc.outlier_count:
            st.warning(f"{qc.outlier_count} potential outlier(s) are flagged. They remain included unless you clear Include.")

        if panel_layer is not None and panel_layer.polygon_features and not filtered_with_include.empty:
            with st.expander("Panel Assignment QC", expanded=False):
                assignments = assign_points_to_polygons(
                    filtered_with_include,
                    x_col,
                    y_col,
                    panel_layer,
                    dataset_panel_col=mappings.get("panel"),
                )
                outside_count = outside_panel_count(assignments)
                if outside_count:
                    st.warning(f"{outside_count} observation(s) are outside all active panel polygons.")
                mismatch_count = int((assignments["Assignment_Status"] == "Mismatch").sum()) if not assignments.empty else 0
                if mismatch_count:
                    st.warning(f"{mismatch_count} dataset/spatial panel mismatch(es) detected.")
                st.dataframe(assignments, width="stretch", hide_index=True)

        if is_pressure_map:
            if pressure_reference_date:
                st.metric("Pressure Map Reference Date", format_map_date(pressure_reference_date))
            measurement_col = mappings.get("measurement_date")
            if measurement_col and measurement_col in filtered_with_include.columns:
                measurement_summary = summarize_measurement_dates(filtered_with_include[measurement_col])
                date_cols = st.columns(3)
                date_cols[0].metric(
                    "Earliest Measurement",
                    format_map_date(measurement_summary.earliest) if measurement_summary.earliest else "",
                )
                date_cols[1].metric(
                    "Latest Measurement",
                    format_map_date(measurement_summary.latest) if measurement_summary.latest else "",
                )
                date_cols[2].metric(
                    "Date Span",
                    "" if measurement_summary.span_days is None else f"{measurement_summary.span_days:,} days",
                )
                if measurement_summary.failed_count:
                    st.warning(f"{measurement_summary.failed_count} original measurement date value(s) could not be parsed.")
            else:
                st.caption("No original pressure measurement date column is mapped.")

        valid_values = pd.to_numeric(included[property_col], errors="coerce").dropna() if property_col in included else pd.Series(dtype=float)
        if not valid_values.empty:
            histogram = px.histogram(valid_values, nbins=20, labels={"value": property_col})
            histogram.update_layout(showlegend=False, height=260, margin={"l": 20, "r": 20, "t": 20, "b": 20})
            st.plotly_chart(histogram, width="stretch")

with controls_col:
    with st.expander("Engineering Controls", expanded=False):
        controls_scope_layer = preview_layer if has_layer_column else None
        controls_reference_date = pressure_reference_date if is_pressure_map else None
        if has_layer_column:
            st.caption(f"Control layer scope: {controls_scope_layer or 'None'}")
        if is_pressure_map:
            if controls_reference_date:
                st.caption(f"Control pressure date: {format_map_date(controls_reference_date)}")
            else:
                st.warning("Pressure controls added now will have no reference date and will be excluded until a reference date is selected.")

        control_points = list(st.session_state.get("engineering_control_points", []))
        control_regions = list(st.session_state.get("engineering_control_regions", []))
        default_x = finite_default(filtered_with_include[x_col], 0.0) if x_col in filtered_with_include else 0.0
        default_y = finite_default(filtered_with_include[y_col], 0.0) if y_col in filtered_with_include else 0.0
        default_value = finite_default(filtered_with_include[property_col], 0.0) if property_col in filtered_with_include else 0.0
        panel_options = data_panel_options(active_for_layers if not active_for_layers.empty else df, mappings.get("panel"))

        point_tab, region_tab = st.tabs(["Point", "Region"])
        with point_tab:
            point_cols = st.columns(3)
            control_x = point_cols[0].number_input("Control X", value=default_x, key="engineering_control_x")
            control_y = point_cols[1].number_input("Control Y", value=default_y, key="engineering_control_y")
            control_value = point_cols[2].number_input("Control Value", value=default_value, key="engineering_control_value")
            manual_panel = ""
            if panel_layer is not None and panel_layer.polygon_features:
                manual_panel, panel_warning = assign_panel_from_point(
                    float(control_x),
                    float(control_y),
                    panel_layer,
                    selected_panels,
                )
                if manual_panel:
                    st.caption(f"Assigned panel: {manual_panel}")
                elif panel_warning:
                    st.warning(panel_warning)
            elif panel_options:
                manual_panel = st.selectbox("Control Panel", [""] + panel_options, key="engineering_control_panel")
            control_active = st.checkbox("Active Control", value=True, key="engineering_control_active")
            control_comment = st.text_input("Control Comment", key="engineering_control_comment")
            if st.button("Add Control Point", type="primary", width="stretch"):
                new_control = create_control_point(
                    x=float(control_x),
                    y=float(control_y),
                    property_name=property_col,
                    value=float(control_value),
                    property_unit=unit,
                    reservoir_layer=controls_scope_layer,
                    panel=manual_panel or "",
                    pressure_reference_date=controls_reference_date,
                    active=control_active,
                    comment=control_comment,
                    existing_controls=control_points,
                )
                st.session_state.engineering_control_points = control_points + [new_control]
                mark_project_dirty()
                st.success("Engineering control point added. Update the map to apply it.")
                st.rerun()

        with region_tab:
            source_options = [
                DRAWN_CONTROL_REGION_SOURCE,
                SELECTED_WELLS_REGION_SOURCE,
                EXISTING_GEOMETRY_REGION_SOURCE,
            ]
            source_aliases = {
                "Selected Wells Convex Hull": SELECTED_WELLS_REGION_SOURCE,
                "Existing Polygon": EXISTING_GEOMETRY_REGION_SOURCE,
            }
            current_source = source_aliases.get(
                st.session_state.get("engineering_control_region_source"),
                st.session_state.get("engineering_control_region_source"),
            )
            region_source_key = "engineering_control_region_source"
            if region_source_key not in st.session_state:
                st.session_state[region_source_key] = DRAWN_CONTROL_REGION_SOURCE
            else:
                st.session_state[region_source_key] = (
                    current_source if current_source in source_options else DRAWN_CONTROL_REGION_SOURCE
                )
            region_source = st.radio(
                "Region Source",
                options=source_options,
                horizontal=True,
                key=region_source_key,
            )
            region_name = st.text_input("Region Name", key="engineering_control_region_name")
            region_scope_layer = controls_scope_layer
            if has_layer_column:
                if layer_options:
                    current_region_layer = st.session_state.get("engineering_control_region_layer")
                    default_layer = (
                        current_region_layer
                        if current_region_layer in layer_options
                        else controls_scope_layer
                        if controls_scope_layer in layer_options
                        else layer_options[0]
                    )
                    region_scope_layer = st.selectbox(
                        "Reservoir Layer",
                        layer_options,
                        index=layer_options.index(default_layer),
                        key="engineering_control_region_layer",
                    )
                else:
                    region_scope_layer = None
                    st.warning("Select filters that leave at least one Reservoir Layer before saving a control region.")
            else:
                st.caption("Reservoir Layer: Unspecified")
            st.caption(f"Region unit: {unit or 'unitless'}")

            region_cols = st.columns(3)
            region_target = region_cols[0].number_input(
                "Target Value",
                value=default_value,
                key="engineering_control_region_target",
            )
            region_spacing = region_cols[1].number_input(
                f"Point Spacing ({coordinate_unit_symbol(coordinate_unit)})",
                min_value=0.000001,
                value=250.0,
                key="engineering_control_region_spacing",
            )
            region_active = region_cols[2].checkbox("Active Region", value=True, key="engineering_control_region_active")
            region_panel = ""
            if panel_layer is not None and panel_layer.polygon_features and len(selected_panels) == 1:
                region_panel = str(selected_panels[0])
                st.caption(f"Panel context: {region_panel}")
            elif panel_layer is not None and panel_layer.polygon_features and selected_panels:
                st.caption(f"Panel context: selected panel union ({len(selected_panels):,} panels)")
            elif panel_layer is not None and panel_layer.polygon_features:
                st.caption("Panel context: all loaded panels")
            elif panel_layer is None and panel_options:
                region_panel = st.selectbox("Region Panel", [""] + panel_options, key="engineering_control_region_panel")
            region_comment = st.text_input("Region Comment", key="engineering_control_region_comment")

            def save_region_from_geometry(geometry, fallback_name: str, panel_scope: str, source_label: str) -> None:
                if has_layer_column and not region_scope_layer:
                    st.error("Select a Reservoir Layer before saving this control region.")
                    return
                try:
                    target_value, spacing_value, layer_value = validate_control_region_parameters(
                        target_value=region_target,
                        spacing=region_spacing,
                        reservoir_layer=region_scope_layer,
                        valid_layers=layer_options if has_layer_column else None,
                        property_type=property_type,
                        pressure_reference_date=controls_reference_date,
                    )
                    new_region = create_control_region(
                        region_name=region_name or fallback_name,
                        geometry=geometry,
                        property_name=property_col,
                        target_value=target_value,
                        property_unit=unit,
                        reservoir_layer=layer_value,
                        panel=panel_scope,
                        pressure_reference_date=controls_reference_date,
                        control_point_spacing=spacing_value,
                        active=region_active,
                        comment=region_comment,
                        region_source=source_label,
                        existing_regions=control_regions,
                    )
                    st.session_state.engineering_control_regions = control_regions + [new_region]
                    st.session_state.pending_control_region_vertices = []
                    st.session_state.control_region_draw_vertices = []
                    st.session_state.control_region_drawing_active = False
                    mark_project_dirty()
                    st.success("Soft control region added. Update the map to apply its generated controls.")
                    st.rerun()
                except ValueError as exc:
                    st.error(str(exc))

            if region_source == DRAWN_CONTROL_REGION_SOURCE:
                draw_cols = st.columns(3)
                if draw_cols[0].button("[DRAW CONTROL REGION]", type="primary", width="stretch"):
                    st.session_state.control_region_drawing_active = True
                    st.session_state.control_region_draw_vertices = []
                    st.session_state.pending_control_region_vertices = []
                    st.session_state.last_control_region_draw_event_id = ""
                    st.rerun()
                if draw_cols[1].button("Clear Drawing", width="stretch"):
                    st.session_state.control_region_draw_vertices = []
                    st.session_state.pending_control_region_vertices = []
                    st.session_state.control_region_drawing_active = False
                    st.rerun()
                if st.session_state.get("control_region_drawing_active"):
                    st.info("Drawing is active on the Mapping Studio canvas.")
                    with st.expander("Polygon Vertices", expanded=False):
                        render_vertices_editor("control_region_draw_vertices", "engineering_control_draw_vertices_editor")

                pending_vertices = st.session_state.get("pending_control_region_vertices", []) or []
                if pending_vertices:
                    with st.expander("Polygon Vertices", expanded=True):
                        render_vertices_editor("pending_control_region_vertices", "engineering_control_pending_vertices_editor")
                    try:
                        drawn = polygon_from_vertices(st.session_state.get("pending_control_region_vertices", []))
                        clip_result = clip_control_region_to_active_domain(
                            drawn.geometry,
                            reservoir_boundary_layer=reservoir_boundary_layer,
                            panel_layer=panel_layer,
                            selected_panels=selected_panels,
                        )
                        for warning in drawn.warnings:
                            st.warning(warning)
                        for warning in clip_result.warnings:
                            st.warning(warning)
                        if clip_result.clipped:
                            clip_choice = st.radio(
                                "Domain Adjustment",
                                ["Clip to Reservoir Domain", "Cancel and Redraw"],
                                index=0,
                                horizontal=True,
                                key="engineering_control_region_clip_choice",
                            )
                        else:
                            clip_choice = "Clip to Reservoir Domain"
                        st.caption(
                            f"Drawn region area after domain handling: "
                            f"{format_numeric(clip_result.geometry.area)} {coordinate_unit_symbol(coordinate_unit)}^2"
                        )
                        action_cols = st.columns(2)
                        if action_cols[0].button("Save Control Region", type="primary", width="stretch"):
                            if clip_choice == "Cancel and Redraw":
                                st.error("Choose Clip to Reservoir Domain before saving, or clear the drawing and redraw.")
                            else:
                                save_region_from_geometry(
                                    clip_result.geometry,
                                    "Drawn Control Region",
                                    region_panel,
                                    DRAWN_CONTROL_REGION_SOURCE,
                                )
                        if action_cols[1].button("Cancel and Redraw", width="stretch"):
                            st.session_state.pending_control_region_vertices = []
                            st.session_state.control_region_draw_vertices = []
                            st.session_state.control_region_drawing_active = True
                            st.session_state.last_control_region_draw_event_id = ""
                            st.rerun()
                    except ValueError as exc:
                        st.error(str(exc))

            elif region_source == SELECTED_WELLS_REGION_SOURCE:
                buffer_distance = st.number_input(
                    f"Hull Buffer ({coordinate_unit_symbol(coordinate_unit)})",
                    min_value=0.0,
                    value=0.0,
                    key="engineering_control_region_buffer",
                )
                seed_frame = included.copy()
                seed_options = well_seed_options(seed_frame, well_col, x_col, y_col)
                valid_seed_ids = set(seed_options)
                st.session_state.selected_control_region_well_ids = [
                    value for value in st.session_state.get("selected_control_region_well_ids", []) if value in valid_seed_ids
                ]
                selected_seed_ids = st.multiselect(
                    "Seed Wells",
                    options=list(seed_options),
                    format_func=lambda value: seed_options.get(value, value),
                    key="selected_control_region_well_ids",
                )
                if st.button("Create Region From Wells", type="primary", width="stretch"):
                    if len(selected_seed_ids) < 3:
                        st.error("Select at least three included wells.")
                    else:
                        seed_frame = seed_frame.copy()
                        seed_frame["_Seed_Key"] = [
                            str(row.get(INTERNAL_ROW_ID, index)) for index, row in seed_frame.iterrows()
                        ]
                        selected_wells = seed_frame[seed_frame["_Seed_Key"].isin(selected_seed_ids)]
                        try:
                            geometry = create_region_from_wells(selected_wells, x_col, y_col, float(buffer_distance))
                            save_region_from_geometry(
                                geometry,
                                "Well Hull Control Region",
                                region_panel,
                                SELECTED_WELLS_REGION_SOURCE,
                            )
                        except ValueError as exc:
                            st.error(str(exc))
            else:
                polygon_choices = selected_region_geometry_choices(
                    reservoir_boundary_layer=reservoir_boundary_layer,
                    panel_layer=panel_layer,
                    custom_layers=custom_layers,
                    selected_panels=selected_panels,
                )
                if not polygon_choices:
                    st.caption("No valid polygon geometry is loaded.")
                else:
                    selected_polygon_key = st.selectbox(
                        "Polygon",
                        options=list(polygon_choices),
                        format_func=lambda value: str(polygon_choices[value]["label"]),
                        key="engineering_control_polygon_choice",
                    )
                    chosen_polygon = polygon_choices[selected_polygon_key]
                    chosen_panel = str(chosen_polygon.get("panel") or region_panel or "")
                    if chosen_panel:
                        st.caption(f"Region panel scope: {chosen_panel}")
                    if st.button("Create Region From Polygon", type="primary", width="stretch"):
                        save_region_from_geometry(
                            chosen_polygon["geometry"],
                            str(chosen_polygon["label"]),
                            chosen_panel,
                            EXISTING_GEOMETRY_REGION_SOURCE,
                        )

        control_table = controls_dataframe(st.session_state.get("engineering_control_points", []))
        if not control_table.empty:
            edited_controls = st.data_editor(
                control_table[CONTROL_POINT_COLUMNS],
                width="stretch",
                hide_index=True,
                disabled=["Control_ID", "Property", "Property_Unit", "Source_Type"],
                column_config={
                    "Active": st.column_config.CheckboxColumn("Active"),
                    "X": st.column_config.NumberColumn("X"),
                    "Y": st.column_config.NumberColumn("Y"),
                    "Value": st.column_config.NumberColumn("Value"),
                },
                key="engineering_control_point_editor",
            )
            if not edited_controls.equals(control_table[CONTROL_POINT_COLUMNS]):
                st.session_state.engineering_control_points = [
                    normalize_control_point(row.to_dict()) for _, row in edited_controls.iterrows()
                ]
                mark_project_dirty()
            delete_control_ids = st.multiselect(
                "Delete Control Points",
                control_table["Control_ID"].dropna().astype(str).tolist(),
                key="engineering_control_delete_ids",
            )
            if st.button("Delete Selected Control Points", width="stretch") and delete_control_ids:
                delete_set = set(delete_control_ids)
                st.session_state.engineering_control_points = [
                    control
                    for control in st.session_state.get("engineering_control_points", [])
                    if str(control.get("Control_ID")) not in delete_set
                ]
                mark_project_dirty()
                st.rerun()
        else:
            st.caption("No manual engineering control points.")

        region_table = regions_dataframe(st.session_state.get("engineering_control_regions", []))
        if not region_table.empty:
            edited_regions = st.data_editor(
                region_table[CONTROL_REGION_COLUMNS],
                width="stretch",
                hide_index=True,
                disabled=["Region_ID", "Region_Source", "Property", "Property_Unit", "Generated_Control_Count"],
                column_config={
                    "Active": st.column_config.CheckboxColumn("Active"),
                    "Target_Value": st.column_config.NumberColumn("Target Value"),
                    "Control_Point_Spacing": st.column_config.NumberColumn("Spacing"),
                },
                key="engineering_control_region_editor",
            )
            if not edited_regions.equals(region_table[CONTROL_REGION_COLUMNS]):
                existing_regions = {
                    str(region.get("Region_ID")): normalize_control_region(region)
                    for region in st.session_state.get("engineering_control_regions", [])
                }
                updated_regions = []
                for _, row in edited_regions.iterrows():
                    region_id = str(row.get("Region_ID") or "")
                    merged = {**existing_regions.get(region_id, {}), **row.to_dict()}
                    updated_regions.append(normalize_control_region(merged))
                st.session_state.engineering_control_regions = updated_regions
                mark_project_dirty()
            duplicate_region_ids = st.multiselect(
                "Duplicate Control Regions",
                region_table["Region_ID"].dropna().astype(str).tolist(),
                key="engineering_control_region_duplicate_ids",
            )
            if st.button("Duplicate Selected Control Regions", width="stretch") and duplicate_region_ids:
                duplicate_set = set(duplicate_region_ids)
                existing_regions = [normalize_control_region(region) for region in st.session_state.get("engineering_control_regions", [])]
                copied_regions = []
                for region in existing_regions:
                    if str(region.get("Region_ID")) not in duplicate_set:
                        continue
                    copied = normalize_control_region(
                        {
                            **region,
                            "Region_ID": next_region_id(existing_regions + copied_regions),
                            "Region_Name": f"{region.get('Region_Name') or region.get('Region_ID')} Copy",
                            "Generated_Control_Count": 0,
                        }
                    )
                    copied_regions.append(copied)
                if copied_regions:
                    st.session_state.engineering_control_regions = existing_regions + copied_regions
                    mark_project_dirty()
                    st.rerun()
            delete_region_ids = st.multiselect(
                "Delete Control Regions",
                region_table["Region_ID"].dropna().astype(str).tolist(),
                key="engineering_control_region_delete_ids",
            )
            if st.button("Delete Selected Control Regions", width="stretch") and delete_region_ids:
                delete_set = set(delete_region_ids)
                st.session_state.engineering_control_regions = [
                    region
                    for region in st.session_state.get("engineering_control_regions", [])
                    if str(region.get("Region_ID")) not in delete_set
                ]
                mark_project_dirty()
                st.rerun()
        else:
            st.caption("No soft control regions.")

        preview_control_selection = control_selection_for_layer(controls_scope_layer, filtered_with_include)
        st.caption(
            f"{preview_control_selection.control_count:,} active matching engineering control point(s); "
            f"{len(preview_control_selection.regions):,} active matching control region(s)."
        )
        for warning in preview_control_selection.warnings:
            st.warning(warning)

with controls_col:
    with st.expander("Interpolation", expanded=True):
        duplicate_default = st.session_state.get("mapping_duplicate_method", DUPLICATE_METHODS[0])
        duplicate_index = DUPLICATE_METHODS.index(duplicate_default) if duplicate_default in DUPLICATE_METHODS else 0
        duplicate_method = st.selectbox(
            "Duplicate Coordinate Handling",
            DUPLICATE_METHODS,
            index=duplicate_index,
            key="mapping_duplicate_method",
        )
        filtered_with_include, prepared = prepare_observations_for_layer(preview_layer)
        filtered_with_include = flag_outliers(filtered_with_include, property_col)
        grid_parameters = grid_controls()
        method = st.radio("Interpolation Method", INTERPOLATION_METHODS, horizontal=True, index=0)
        if panel_layer is not None and panel_layer.polygon_features:
            if panel_interpolation_mode == "Independent by Panel / Compartment":
                st.caption("Independent mode: each selected panel is interpolated from its own assigned observations.")
            else:
                st.caption("Combined mode: selected panel observations are pooled into one interpolation model.")
        method_parameters = method_parameter_controls(
            method,
            prepared,
            coordinate_unit,
            float(grid_parameters["buffer_fraction"]),
        )
        selected_panel_domain_bounds = selected_panel_bounds(panel_layer, selected_panels)
        st.markdown("##### Spatial Domain")
        domain_options = interpolation_domain_options(
            has_reservoir_boundary=reservoir_boundary_layer is not None and bool(reservoir_boundary_layer.polygon_features),
            has_selected_panel_union=selected_panel_domain_bounds is not None,
        )
        domain_key = "mapping_interpolation_domain"
        if domain_key not in st.session_state:
            st.session_state[domain_key] = domain_options[0]
        else:
            st.session_state[domain_key] = coerce_interpolation_domain_selection(
                st.session_state[domain_key],
                domain_options,
            )
        interpolation_domain = st.radio(
            "Interpolation Domain",
            options=domain_options,
            horizontal=True,
            key=domain_key,
        )
        domain_bounds = resolve_domain_bounds(
            interpolation_domain,
            reservoir_boundary_layer,
            panel_layer,
            selected_panels,
        )
        if interpolation_domain == DOMAIN_RESERVOIR_BOUNDARY_EXTENT:
            st.caption("The full rectangular reservoir boundary extent will be interpolated before any polygon mask is applied.")
        if interpolation_domain == DOMAIN_SELECTED_PANEL_UNION_EXTENT:
            st.caption("The selected panel polygon union extent will be interpolated before any spatial mask is applied.")
        if method in {"Linear", "Cubic"} and interpolation_domain != DOMAIN_WELL_DATA_EXTENT:
            st.info("Linear and Cubic interpolation may remain NaN outside the convex hull of the observations.")

        st.markdown("##### Spatial Mask")
        mask_parameters = mask_controls(
            prepared,
            coordinate_unit,
            has_reservoir_boundary=reservoir_boundary_layer is not None and bool(reservoir_boundary_layer.polygon_features),
            has_selected_panel_union=selected_panel_domain_bounds is not None,
        )
        interpolation_control_selection = control_selection_for_layer(preview_layer, filtered_with_include)
        st.caption(
            f"{len(prepared):,} finite included measured observation(s) and "
            f"{interpolation_control_selection.control_count:,} active matching engineering control(s) "
            "will participate after duplicate handling."
        )
        if method == "Ordinary Kriging" and interpolation_control_selection.control_count:
            st.caption("Ordinary Kriging variogram fitting remains measured-only; controls condition the kriging estimate.")
        for warning in interpolation_control_selection.warnings:
            st.warning(warning)

        button_label = "GENERATE ALL LAYER MAPS" if layer_scope == LAYER_SCOPE_ALL else "GENERATE MAP"
        if st.session_state.get("generated_map") is not None:
            button_label = "UPDATE ALL LAYER MAPS" if layer_scope == LAYER_SCOPE_ALL else "UPDATE MAP"
        if st.button(button_label, type="primary", width="stretch"):
            if layer_scope == LAYER_SCOPE_SELECTED and has_layer_column and selected_layer is None:
                st.error("Select a Reservoir Layer before generating the map.")
            elif layer_scope == LAYER_SCOPE_ALL and not layer_options:
                st.error("No Reservoir Layer values are available to generate.")
            else:
                progress = st.progress(0.0)
                progress_text = st.empty()

                def progress_callback(index: int, total: int, layer_name: str) -> None:
                    progress.progress(index / max(total, 1))
                    progress_text.caption(f"Generating layer {index} of {total}: {layer_name}")

                try:
                    with st.spinner("Generating reservoir map surfaces..."):
                        collection = generate_layer_map_collection(
                            dataframe=df,
                            mappings=mappings,
                            property_column=property_col,
                            property_type=property_type,
                            property_unit=unit,
                            pressure_reference_date=pressure_reference_date,
                            is_pressure_map=is_pressure_map,
                            filter_values=filter_values_no_layer,
                            selected_panels=selected_panels,
                            layer_scope=layer_scope,
                            selected_layer=selected_layer,
                            panel_interpolation_mode=panel_interpolation_mode,
                            include_state=st.session_state.get("include_state", {}),
                            duplicate_method=duplicate_method,
                            interpolation_method=method,
                            interpolation_parameters=method_parameters,
                            grid_parameters=grid_parameters,
                            mask_parameters=mask_parameters,
                            interpolation_domain=interpolation_domain,
                            coordinate_unit=coordinate_unit,
                            crs=st.session_state.get("crs", {}),
                            panel_layer=panel_layer,
                            reservoir_boundary_layer=reservoir_boundary_layer,
                            control_points=st.session_state.get("engineering_control_points", []),
                            control_regions=st.session_state.get("engineering_control_regions", []),
                            fault_layer_loaded=fault_layer is not None,
                            custom_layer_count=len(custom_layers),
                            x_col=x_col,
                            y_col=y_col,
                            well_col=well_col,
                            hover_columns=generated_hover_columns(mappings),
                            progress_callback=progress_callback,
                        )
                    progress.empty()
                    progress_text.empty()
                    st.session_state.generated_layer_maps = dict(collection.maps)
                    st.session_state.generated_layer_statuses = [
                        {
                            "layer": item.layer,
                            "status": item.status,
                            "observations": item.observations,
                            "message": item.message,
                        }
                        for item in collection.statuses
                    ]
                    st.session_state.generated_layer_batch_signature = collection.batch_signature
                    st.session_state.active_generated_layer = collection.active_layer
                    st.session_state.generated_map = select_generated_layer_map(collection.maps, collection.active_layer)
                    mark_project_dirty()
                    if collection.maps:
                        st.success(f"Generated {len(collection.maps):,} layer map(s).")
                    else:
                        st.error("No layer maps could be generated.")
                    status_table = statuses_frame(st.session_state.generated_layer_statuses)
                    if not status_table.empty:
                        st.dataframe(status_table, width="stretch", hide_index=True)
                except ValueError as exc:
                    progress.empty()
                    progress_text.empty()
                    st.error(str(exc))

        if st.session_state.get("generated_layer_statuses"):
            with st.expander("Layer Generation Status", expanded=False):
                st.dataframe(statuses_frame(st.session_state.generated_layer_statuses), width="stretch", hide_index=True)

    with st.expander("Map Overlays", expanded=False):
        layer_settings = render_layer_manager()

    with st.expander("Style", expanded=False):
        style = render_style_controls(filtered_with_include, property_col, well_col)


def current_signature_for_display(generated_map: dict[str, object] | None) -> dict[str, object] | None:
    if not generated_map:
        return None
    if layer_scope == LAYER_SCOPE_SELECTED:
        signature_layer = selected_layer if has_layer_column else None
    elif layer_scope == LAYER_SCOPE_ALL and has_layer_column:
        signature_layer = generated_map.get("reservoir_layer")
    else:
        signature_layer = None
    if signature_layer == UNSPECIFIED_LAYER:
        signature_layer = None
    if has_layer_column and signature_layer is None:
        return None
    try:
        signature_filtered, signature_prepared = prepare_observations_for_layer(
            None if signature_layer is None else str(signature_layer)
        )
    except ValueError:
        return None
    signature_control_selection = control_selection_for_layer(signature_layer, signature_filtered)
    if signature_prepared.empty and signature_control_selection.dataframe.empty:
        return None
    signature_parameters = resolve_layer_method_parameters(method, method_parameters, signature_prepared)
    return build_layer_model_signature(
        property_column=property_col,
        property_type=property_type,
        pressure_reference_date=pressure_reference_date,
        selected_panels=[str(value) for value in selected_panels or []],
        reservoir_layer=None if signature_layer is None else str(signature_layer),
        layer_mapping_scope=layer_scope,
        panel_interpolation_mode=panel_interpolation_mode,
        filter_values=filter_values_no_layer,
        active_dataframe=signature_filtered,
        duplicate_method=duplicate_method,
        interpolation_method=method,
        interpolation_parameters=signature_parameters,
        grid_parameters=grid_parameters,
        interpolation_domain=interpolation_domain,
        domain_bounds=domain_bounds,
        mask_parameters=mask_parameters,
        control_state=signature_control_selection.signature_state,
    )


context_style = st.session_state.get("style_settings", {})
context_layer_settings = st.session_state.get("layer_settings", {})
context_plot_style = plot_style_with_overlay_settings(
    context_style,
    context_layer_settings,
    generated_surface=bool(st.session_state.get("generated_map") or st.session_state.get("generated_layer_maps")),
)
context_included_display, context_excluded_display = split_observations_for_display(
    filtered_with_include,
    property_col,
    unit,
    display_unit,
)
try:
    context_control_selection = control_selection_for_layer(preview_layer, filtered_with_include)
except ValueError:
    context_control_selection = None
context_display_controls = controls_display_frame(
    context_control_selection,
    property_col,
    unit,
    display_unit,
)
context_display_controls = controls_for_overlay(context_display_controls, context_layer_settings)
context_display_regions = list(getattr(context_control_selection, "regions", []) or [])
context_polygons = []
if bool(context_layer_settings.get("show_reservoir_boundary", True)):
    context_polygons.extend(
        geometry_polygons_payload(reservoir_boundary_layer, stroke="#0f172a", fill="rgba(15, 23, 42, 0.035)", width=2.0)
    )
if bool(context_layer_settings.get("show_panels", True)):
    context_polygons.extend(
        geometry_polygons_payload(panel_layer, stroke="#92400e", fill="rgba(146, 64, 14, 0.035)", width=1.5)
    )
if bool(context_layer_settings.get("show_custom_layers", True)):
    for custom_layer in custom_layers:
        context_polygons.extend(
            geometry_polygons_payload(custom_layer, stroke="#2563eb", fill="rgba(37, 99, 235, 0.035)", width=1.5)
        )
context_lines = []
if bool(context_layer_settings.get("show_faults", True)):
    context_lines.extend(geometry_lines_payload(fault_layer, stroke="#b91c1c", width=2.0))
if bool(context_layer_settings.get("show_custom_layers", True)):
    for custom_layer in custom_layers:
        context_lines.extend(geometry_lines_payload(custom_layer, stroke="#2563eb", width=1.5))
context_points_payload = observation_points_payload(
    context_included_display,
    x_col=x_col,
    y_col=y_col,
    property_col=property_col,
    well_col=well_col,
    layer_col=layer_col,
    panel_col=mappings.get("panel"),
    unit=display_unit,
    is_pressure_map=is_pressure_map,
    pressure_reference_date=pressure_reference_date,
)
context_excluded_points_payload = observation_points_payload(
    context_excluded_display,
    x_col=x_col,
    y_col=y_col,
    property_col=property_col,
    well_col=well_col,
    layer_col=layer_col,
    panel_col=mappings.get("panel"),
    unit=display_unit,
    is_pressure_map=is_pressure_map,
    pressure_reference_date=pressure_reference_date,
)
context_controls_payload = controls_payload(context_display_controls, x_col=x_col, y_col=y_col)
context_regions_payload = (
    control_regions_payload(context_display_regions)
    if bool(context_layer_settings.get("show_control_regions", True))
    else []
)
context_bounds = draw_payload_bounds(
    context_points_payload,
    context_excluded_points_payload,
    context_controls_payload,
    context_regions_payload,
    context_polygons,
    context_lines,
)

current_plot_figure = None

with map_col:
    generated_layer_maps = dict(st.session_state.get("generated_layer_maps", {}) or {})
    generated = st.session_state.get("generated_map")
    if generated_layer_maps:
        layer_keys = list(generated_layer_maps)
        current_layer_key = st.session_state.get("active_generated_layer")
        if current_layer_key not in layer_keys:
            current_layer_key = layer_keys[0]
        selected_generated_layer = st.selectbox(
            "Generated Layer Map",
            layer_keys,
            index=layer_keys.index(current_layer_key),
            key="mapping_active_generated_layer_selector",
        )
        st.session_state.active_generated_layer = selected_generated_layer
        generated = generated_layer_maps[selected_generated_layer]
        st.session_state.generated_map = generated

    if st.session_state.get("control_region_drawing_active"):
        st.subheader("Draw Control Region")
        draw_result = control_region_drawer(
            points=context_points_payload,
            excluded_points=context_excluded_points_payload,
            controls=context_controls_payload,
            regions=context_regions_payload,
            polygons=context_polygons,
            lines=context_lines,
            bounds=context_bounds,
            vertices=st.session_state.get("control_region_draw_vertices", []),
            coordinate_unit=coordinate_unit_symbol(coordinate_unit),
            x_label=x_col,
            y_label=y_col,
            height=int(context_style.get("height", 720)),
            key="control_region_drawer_canvas",
        )
        if draw_result:
            event_id = str(draw_result.get("event_id") or "")
            if event_id and event_id != st.session_state.get("last_control_region_draw_event_id"):
                st.session_state.last_control_region_draw_event_id = event_id
                action = str(draw_result.get("action") or "update")
                vertices = draw_result.get("vertices") or []
                if action == "cancel":
                    st.session_state.control_region_drawing_active = False
                    st.session_state.control_region_draw_vertices = []
                    st.session_state.pending_control_region_vertices = []
                elif action == "clear":
                    st.session_state.control_region_draw_vertices = []
                    st.session_state.pending_control_region_vertices = []
                elif action == "finish":
                    st.session_state.pending_control_region_vertices = vertices
                    st.session_state.control_region_draw_vertices = vertices
                    st.session_state.control_region_drawing_active = False
                else:
                    st.session_state.control_region_draw_vertices = vertices
                st.rerun()
    elif not generated:
        measured_count = len(context_included_display)
        control_count = len(context_display_controls)
        region_count = len(context_display_regions)
        status_cols = st.columns([1.2, 1, 1, 1])
        status_cols[0].warning("Map Status: no surface generated")
        status_cols[1].metric("Preview Layer", preview_layer or UNSPECIFIED_LAYER)
        status_cols[2].metric("Measured", f"{measured_count:,}")
        status_cols[3].metric("Controls", f"{control_count:,}")
        title = context_style.get("title_override") or f"{property_col} Base Map"
        figure = build_context_map_figure(
            context_included_display,
            context_excluded_display,
            x_col,
            y_col,
            property_col,
            well_col=well_col,
            hover_columns=generated_hover_columns(mappings),
            title=title,
            unit=display_unit,
            coordinate_unit=coordinate_unit,
            is_pressure_map=is_pressure_map,
            map_reference_date=pressure_reference_date,
            measurement_date_col=mappings.get("measurement_date"),
            map_reference_date_col=mappings.get("map_reference_date"),
            style=context_plot_style,
        )
        add_geometry_overlays(
            figure,
            reservoir_boundary_layer=reservoir_boundary_layer,
            panel_layer=panel_layer,
            fault_layer=fault_layer,
            custom_layers=custom_layers,
            layer_settings=context_layer_settings,
            coordinate_unit=coordinate_unit,
            show_debug_boundary=True,
        )
        add_control_region_overlays(
            figure,
            context_display_regions,
            coordinate_unit=coordinate_unit,
            visible=bool(context_layer_settings.get("show_control_regions", True)),
        )
        add_engineering_control_traces(
            figure,
            context_display_controls,
            x_col=x_col,
            y_col=y_col,
            property_col=property_col,
            unit=display_unit,
            show_manual=bool(context_layer_settings.get("show_engineering_controls", True)),
            show_region_points=bool(context_layer_settings.get("show_region_control_points", False)),
        )
        move_observation_traces_to_top(figure)
        current_plot_figure = figure
        st.plotly_chart(figure, width="stretch", config={"displaylogo": False, "scrollZoom": True})
        if is_pressure_map and pressure_reference_date:
            st.metric("Pressure Map Reference Date", format_map_date(pressure_reference_date))
        st.caption("Property Surface and Contours become available after Generate Map.")
    else:
        display_status = map_status(current_signature_for_display(generated), generated)
        measured_count = int(generated.get("measured_observation_count") or len(generated.get("included_observations", [])))
        engineering_control_count = int(generated.get("engineering_control_count") or 0)
        control_region_count = int(generated.get("control_region_count") or 0)
        status_cols = st.columns([1.2, 1, 1, 1, 1, 1])
        with status_cols[0]:
            render_status(display_status)
        status_cols[1].metric("Layer Scope", generated.get("layer_mapping_scope") or LAYER_SCOPE_SELECTED)
        status_cols[2].metric("Reservoir Layer", generated.get("reservoir_layer") or UNSPECIFIED_LAYER)
        status_cols[3].metric("Measured", f"{measured_count:,}")
        status_cols[4].metric("Controls", f"{engineering_control_count:,}")
        status_cols[5].metric("Regions", f"{control_region_count:,}")

        style = st.session_state.get("style_settings", {})
        layer_settings = st.session_state.get("layer_settings", {})
        plot_style = plot_style_with_overlay_settings(style, layer_settings, generated_surface=True)
        debug_cols = st.columns(3)
        show_debug_boundary = debug_cols[0].checkbox("Show Reservoir Boundary", value=True, key="debug_show_boundary")
        show_grid_bbox = debug_cols[1].checkbox("Show Grid Bounding Box", value=False, key="debug_show_grid_bbox")
        show_well_bbox = debug_cols[2].checkbox("Show Well Extent Bounding Box", value=False, key="debug_show_well_bbox")

        title = style.get("title_override") or generated.get("title")
        display_coordinate_unit = st.session_state.get("coordinate_unit", generated.get("coordinate_unit"))
        source_unit = generated.get("unit")
        display_grid = convert_grid_for_display(generated["grid_z"], source_unit, display_unit)
        display_property = generated["property_col"]
        surface_unit = display_unit
        included_display = convert_observations_for_display(
            generated["included_observations"],
            generated["property_col"],
            source_unit,
            display_unit,
        )
        excluded_display = convert_observations_for_display(
            generated["excluded_observations"],
            generated["property_col"],
            source_unit,
            display_unit,
        )
        if generated.get("grid_variance") is not None:
            display_mode = st.radio(
                "Map Display",
                ["Estimated Property", "Kriging Variance", "Kriging Standard Deviation"],
                horizontal=True,
                key="mapping_display_mode",
            )
            if display_mode == "Kriging Variance":
                display_grid = generated["grid_variance"]
                display_property = f"{generated['property_col']} Kriging Variance"
                surface_unit = f"{source_unit}^2" if source_unit else "property unit^2"
            elif display_mode == "Kriging Standard Deviation":
                display_grid = np.sqrt(np.maximum(generated["grid_variance"], 0.0))
                display_property = f"{generated['property_col']} Kriging Std Dev"
                surface_unit = source_unit

        try:
            plot_control_selection = control_selection_for_layer(
                generated.get("reservoir_layer"),
                generated.get("included_observations", pd.DataFrame()),
            )
        except ValueError:
            plot_control_selection = None
        display_controls = controls_display_frame(
            plot_control_selection,
            generated["property_col"],
            source_unit,
            display_unit,
        )
        display_regions = list(getattr(plot_control_selection, "regions", []) or [])
        if display_controls.empty:
            saved_controls = generated.get("engineering_controls", pd.DataFrame())
            if not isinstance(saved_controls, pd.DataFrame):
                saved_controls = pd.DataFrame(saved_controls)
            if not saved_controls.empty:
                display_controls = saved_controls.copy()
                if generated["property_col"] not in display_controls.columns and "Value" in display_controls.columns:
                    display_controls[generated["property_col"]] = pd.to_numeric(display_controls["Value"], errors="coerce")
                display_controls = convert_observations_for_display(
                    display_controls,
                    generated["property_col"],
                    source_unit,
                    display_unit,
                )
                if generated["property_col"] in display_controls.columns:
                    display_controls["Value"] = display_controls[generated["property_col"]]
                display_regions = list(generated.get("engineering_control_regions", []) or [])

        figure = build_map_figure(
            generated["grid_x"],
            generated["grid_y"],
            display_grid,
            included_display,
            excluded_display,
            generated["x_col"],
            generated["y_col"],
            generated["property_col"],
            well_col=generated.get("well_col"),
            hover_columns=generated.get("hover_columns", []),
            title=title,
            unit=display_unit,
            coordinate_unit=display_coordinate_unit,
            is_pressure_map=bool(generated.get("is_pressure_map")),
            map_reference_date=generated.get("map_reference_date"),
            measurement_date_col=generated.get("measurement_date_col"),
            map_reference_date_col=generated.get("map_reference_date_col"),
            style=plot_style,
            surface_label=display_property,
            surface_unit=surface_unit,
        )
        diagnostics = generated.get("mask_info", {})
        if show_grid_bbox:
            x_min, y_min, x_max, y_max = diagnostics.get("generated_grid_extent", (None, None, None, None))
            if x_min is not None:
                figure.add_shape(
                    type="rect",
                    x0=x_min,
                    y0=y_min,
                    x1=x_max,
                    y1=y_max,
                    line={"color": "#DC2626", "dash": "dash", "width": 1.5},
                    fillcolor="rgba(220, 38, 38, 0.03)",
                )
        if show_well_bbox:
            x_min, y_min, x_max, y_max = diagnostics.get("well_extent", (None, None, None, None))
            if x_min is not None:
                figure.add_shape(
                    type="rect",
                    x0=x_min,
                    y0=y_min,
                    x1=x_max,
                    y1=y_max,
                    line={"color": "#16A34A", "dash": "dot", "width": 1.5},
                    fillcolor="rgba(22, 163, 74, 0.03)",
                )

        add_geometry_overlays(
            figure,
            reservoir_boundary_layer=reservoir_boundary_layer,
            panel_layer=panel_layer,
            fault_layer=fault_layer,
            custom_layers=custom_layers,
            layer_settings=layer_settings,
            coordinate_unit=display_coordinate_unit,
            show_debug_boundary=show_debug_boundary,
        )
        add_control_region_overlays(
            figure,
            display_regions,
            coordinate_unit=display_coordinate_unit,
            visible=bool(layer_settings.get("show_control_regions", True)),
        )
        add_engineering_control_traces(
            figure,
            display_controls,
            x_col=generated["x_col"],
            y_col=generated["y_col"],
            property_col=generated["property_col"],
            unit=display_unit,
            show_manual=bool(layer_settings.get("show_engineering_controls", True)),
            show_region_points=bool(layer_settings.get("show_region_control_points", False)),
        )
        move_observation_traces_to_top(figure)
        current_plot_figure = figure
        st.plotly_chart(figure, width="stretch", config={"displaylogo": False, "scrollZoom": True})

        info_cols = st.columns(6)
        info_cols[0].metric("Display", display_property)
        info_cols[1].metric("Method", generated["method"])
        info_cols[2].metric("Grid", f"{generated['grid_parameters']['nx']} x {generated['grid_parameters']['ny']}")
        info_cols[3].metric("Coordinate Unit", coordinate_unit_symbol(display_coordinate_unit))
        info_cols[4].metric("Mask", generated["mask_info"]["mask_mode"])
        info_cols[5].metric("Valid Cells", f"{generated['mask_info']['valid_grid_cells']:,}")
        if generated.get("is_pressure_map") and generated.get("map_reference_date"):
            st.metric("Pressure Map Reference Date", format_map_date(generated.get("map_reference_date")))
        if generated.get("respect_compartments"):
            st.caption("Panel Interpolation Mode: Independent by Panel / Compartment.")
        elif generated.get("selected_panels"):
            st.caption("Panel Interpolation Mode: Combined Selected Panels.")
        if generated.get("control_warnings"):
            with st.expander("Engineering Control QC", expanded=False):
                for warning in generated.get("control_warnings", []):
                    st.warning(warning)

        with st.expander("Spatial Domain Diagnostics", expanded=False):
            st.write(f"Well extent: {extent_text(diagnostics.get('well_extent'))}")
            st.write(f"Reservoir extent: {extent_text(diagnostics.get('reservoir_extent'))}")
            panel_union_extent = (generated.get("geometry_references", {}) or {}).get("selected_panel_bounds")
            st.write(f"Selected panel union extent: {extent_text(panel_union_extent)}")
            st.write(f"Generated grid extent: {extent_text(diagnostics.get('generated_grid_extent'))}")
            st.write(f"Grid dimensions: {generated['grid_x'].shape[1]} x {generated['grid_x'].shape[0]}")
            if reservoir_boundary_layer is not None and reservoir_boundary_layer.polygon_features:
                geometry = polygon_union(reservoir_boundary_layer.polygon_features)
                st.write(f"Geometry type: {geometry.geom_type}")
                st.write(f"Number of polygons: {len(reservoir_boundary_layer.polygon_features)}")
                st.write(f"Boundary bounds: {extent_text(tuple(float(value) for value in geometry.bounds))}")
            st.write(f"Grid cells before mask: {diagnostics.get('grid_cells_before_mask', int(generated['grid_x'].size)):,}")
            st.write(f"Grid cells inside reservoir: {diagnostics.get('grid_cells_inside_reservoir', int(generated['grid_x'].size)):,}")
            st.write(
                f"Grid cells with interpolated finite values: "
                f"{diagnostics.get('finite_grid_cells_before_mask', int(np.isfinite(generated['grid_z']).sum())):,}"
            )
            st.write(f"Finite cells after mask: {diagnostics.get('valid_grid_cells', int(np.isfinite(generated['grid_z']).sum())):,}")

        method_details = generated.get("method_parameters", {})
        if generated["method"] == "IDW":
            st.caption(
                f"IDW power {method_details.get('power')}, neighbors {method_details.get('neighbors')}, "
                f"minimum neighbors {method_details.get('min_neighbors')}, search radius "
                f"{format_distance(method_details.get('search_radius'), display_coordinate_unit)}; "
                f"duplicate handling: {generated['duplicate_method']}."
            )
        if generated["mask_info"].get("max_distance") is not None:
            st.caption(
                "Maximum-distance mask used "
                f"{format_distance(generated['mask_info']['max_distance'], display_coordinate_unit)}."
            )
        if generated.get("grid_variance") is not None:
            valid_variance = generated["grid_variance"][np.isfinite(generated["grid_variance"])]
            if valid_variance.size:
                st.caption(
                    f"Kriging variance range: {valid_variance.min():.4g} to {valid_variance.max():.4g}; "
                    f"standard deviation range: {np.sqrt(valid_variance).min():.4g} to {np.sqrt(valid_variance).max():.4g}."
                )

with controls_col:
    with st.expander("Map Library", expanded=False):
        generated = st.session_state.get("generated_map")
        if generated:
            scenario_name = st.text_input(
                "Scenario Name",
                value=str(generated.get("title") or "Saved Map"),
                key="mapping_studio_scenario_name",
            )
            if st.button("Save Map Scenario", type="primary"):
                scenario = create_map_scenario(
                    scenario_name,
                    generated,
                    {
                        "style_settings": st.session_state.get("style_settings", {}),
                        "layer_settings": st.session_state.get("layer_settings", {}),
                        "filter_values": st.session_state.get("filter_values", {}),
                        "include_state": st.session_state.get("include_state", {}),
                        "crs": st.session_state.get("crs", {}),
                    },
                    st.session_state.get("geostatistics", {}).get("cross_validation"),
                )
                scenarios = list(st.session_state.get("map_scenarios", []))
                scenarios.append(scenario)
                st.session_state.map_scenarios = scenarios
                st.session_state.current_scenario_id = scenario["id"]
                mark_project_dirty()
                st.success("Map scenario saved.")
                st.rerun()
        else:
            st.caption("Generate a map before saving a scenario.")

        saved_scenarios = list(st.session_state.get("map_scenarios", []))
        if saved_scenarios:
            st.dataframe(scenario_summary_table(saved_scenarios), width="stretch", hide_index=True)
            labels = [str(item.get("name") or item.get("id")) for item in saved_scenarios]
            selected_label = st.selectbox("Selected Scenario", labels, key="mapping_studio_selected_scenario")
            selected_index = labels.index(selected_label)
            selected = saved_scenarios[selected_index]
            scenario_action_cols = st.columns(5)
            if scenario_action_cols[0].button("Open"):
                opened = scenario_to_generated_map(selected)
                opened_layer = layer_map_key(opened)
                st.session_state.generated_layer_maps = {opened_layer: opened}
                st.session_state.active_generated_layer = opened_layer
                st.session_state.generated_map = opened
                st.session_state.current_scenario_id = selected.get("id")
                st.success("Scenario opened.")
            rename_value = st.text_input(
                "Rename Selected Scenario",
                value=str(selected.get("name") or "Saved Map"),
                key="mapping_studio_rename_scenario",
            )
            if scenario_action_cols[1].button("Rename"):
                saved_scenarios[selected_index] = rename_scenario(selected, rename_value)
                st.session_state.map_scenarios = saved_scenarios
                mark_project_dirty()
                st.rerun()
            if scenario_action_cols[2].button("Duplicate"):
                saved_scenarios.append(duplicate_scenario(selected))
                st.session_state.map_scenarios = saved_scenarios
                mark_project_dirty()
                st.rerun()
            if scenario_action_cols[3].button("Delete"):
                deleted_id = selected.get("id")
                st.session_state.map_scenarios = [
                    scenario for scenario in saved_scenarios if scenario.get("id") != deleted_id
                ]
                if st.session_state.get("current_scenario_id") == deleted_id:
                    st.session_state.current_scenario_id = None
                    st.session_state.generated_map = None
                mark_project_dirty()
                st.rerun()
            if scenario_action_cols[4].button("Use In Comparison"):
                st.session_state.compare_map_a_id = selected.get("id")
                st.success("Selected as Map A in Map Comparison.")
        else:
            st.caption("No saved map scenarios yet.")

    with st.expander("Export", expanded=False):
        st.markdown("#### EXPORT")
        project_metadata = dict(st.session_state.get("project_metadata", {}) or {})
        current_for_export = filtered_with_include.copy()
        included_for_export = (
            current_for_export[current_for_export[INCLUDE_COLUMN]].copy()
            if INCLUDE_COLUMN in current_for_export
            else current_for_export
        )
        export_clean = drop_internal_columns(included_for_export)

        st.markdown("##### Grid / Engineering Data")
        observation_cols = st.columns(2)
        observation_cols[0].download_button(
            "Active Observations CSV",
            dataframe_to_csv_bytes(export_clean),
            file_name="active_map_observations.csv",
            mime="text/csv",
            width="stretch",
        )
        observation_cols[1].download_button(
            "Active Observations Excel",
            dataframe_to_excel_bytes(export_clean, "Active Observations"),
            file_name="active_map_observations.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            width="stretch",
        )

        generated = st.session_state.get("generated_map")
        if not generated:
            st.caption("Generate a map before exporting an interpolated grid, raster, ZMAP, image, or map metadata.")
        else:
            active_scenario = current_open_scenario()
            base_name = map_base_filename(generated, project_metadata=project_metadata)
            export_metadata = build_map_export_metadata(
                generated,
                project_metadata=project_metadata,
                scenario=active_scenario,
                nodata=GEOTIFF_NODATA,
            )
            controls_export = engineering_controls_export_dataframe(generated)
            regions_export = control_regions_export_dataframe(generated)
            validation_metrics = generated.get("validation_metrics", {})
            validation_export = pd.DataFrame([validation_metrics]) if validation_metrics else pd.DataFrame()

            metadata_summary = [
                f"Property: {export_metadata.get('Property')}",
                f"Stored / Original Unit: {export_metadata.get('Property_Unit') or 'unitless'}",
                f"Display Unit: {display_unit or 'unitless'}",
                f"Export Unit: original",
                f"CRS: {crs_display_name(generated.get('crs') or {'mode': export_metadata.get('CRS_Mode'), 'epsg': export_metadata.get('CRS_EPSG')})}",
            ]
            if generated.get("reservoir_layer"):
                metadata_summary.append(f"Layer: {generated.get('reservoir_layer')}")
            if generated.get("is_pressure_map") and generated.get("map_reference_date"):
                metadata_summary.append(f"Reference Date: {format_map_date(generated.get('map_reference_date'))}")
            st.caption(" | ".join(metadata_summary))

            include_nan = st.checkbox("Include masked cells in grid export", value=False)
            grid_export = grid_to_dataframe(
                generated["grid_x"],
                generated["grid_y"],
                generated["grid_z"],
                generated["property_col"],
                include_nan=include_nan,
                grid_variance=generated.get("grid_variance"),
                panel_grid=generated.get("panel_grid"),
            )
            grid_cols = st.columns(3)
            grid_cols[0].download_button(
                "CSV",
                dataframe_to_csv_bytes(grid_export),
                file_name=f"{base_name}.csv",
                mime="text/csv",
                width="stretch",
            )
            grid_cols[1].download_button(
                "XYZ ASCII",
                grid_to_xyz_ascii_bytes(grid_export),
                file_name=f"{base_name}.xyz",
                mime="text/plain",
                width="stretch",
            )
            grid_cols[2].download_button(
                "Excel",
                grid_to_excel_bytes(
                    grid_export,
                    export_metadata,
                    engineering_controls=controls_export,
                    control_regions=regions_export,
                    validation_df=validation_export,
                ),
                file_name=f"{base_name}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                width="stretch",
            )

            st.markdown("##### GIS Raster")
            if export_metadata.get("CRS_Mode") == "Local / Unknown XY":
                st.info("CRS is undefined. The GeoTIFF preserves local XY geometry but has no EPSG spatial reference.")
            try:
                geotiff_files = map_geotiff_export_files(
                    generated,
                    project_metadata=project_metadata,
                    scenario=active_scenario,
                )
                geotiff_cols = st.columns(min(3, max(1, len(geotiff_files))))
                for index, (filename, data, geotiff_metadata) in enumerate(geotiff_files):
                    label = str(geotiff_metadata.get("Export_Value") or "GeoTIFF")
                    geotiff_cols[index % len(geotiff_cols)].download_button(
                        f"{label} GeoTIFF",
                        data,
                        file_name=filename,
                        mime="image/tiff",
                        width="stretch",
                    )
            except Exception as exc:
                st.warning(f"GeoTIFF export unavailable: {exc}")

            st.markdown("##### ZMAP Grid ASCII")
            try:
                zmap_files = map_zmap_export_files(
                    generated,
                    project_metadata=project_metadata,
                    scenario=active_scenario,
                )
                for zmap_name, zmap_data, metadata_name, metadata_data, zmap_metadata in zmap_files:
                    zmap_cols = st.columns(2)
                    label = str(zmap_metadata.get("Export_Value") or "ZMAP")
                    zmap_cols[0].download_button(
                        f"{label} ZMAP",
                        zmap_data,
                        file_name=zmap_name,
                        mime="text/plain",
                        width="stretch",
                    )
                    zmap_cols[1].download_button(
                        f"{label} ZMAP Metadata JSON",
                        metadata_data,
                        file_name=metadata_name,
                        mime="application/json",
                        width="stretch",
                    )
            except Exception as exc:
                st.warning(f"ZMAP export unavailable: {exc}")

            st.markdown("##### Metadata")
            st.download_button(
                "Map Metadata JSON",
                metadata_to_json_bytes(export_metadata),
                file_name=f"{base_name}_metadata.json",
                mime="application/json",
                width="stretch",
            )

            st.markdown("##### Engineering Controls")
            if controls_export.empty and regions_export.empty:
                st.caption("No engineering controls or soft control regions are active in this map.")
            else:
                control_cols = st.columns(2)
                if not controls_export.empty:
                    control_cols[0].download_button(
                        "Controls CSV",
                        dataframe_to_csv_bytes(controls_export),
                        file_name=f"{base_name}_engineering_controls.csv",
                        mime="text/csv",
                        width="stretch",
                    )
                    control_cols[1].download_button(
                        "Controls Excel",
                        dataframe_to_excel_bytes(controls_export, "Engineering Controls"),
                        file_name=f"{base_name}_engineering_controls.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        width="stretch",
                    )
                if not regions_export.empty:
                    region_cols = st.columns(2)
                    region_cols[0].download_button(
                        "Control Regions CSV",
                        dataframe_to_csv_bytes(regions_export),
                        file_name=f"{base_name}_control_regions.csv",
                        mime="text/csv",
                        width="stretch",
                    )
                    region_cols[1].download_button(
                        "Control Regions Excel",
                        dataframe_to_excel_bytes(regions_export, "Control Regions"),
                        file_name=f"{base_name}_control_regions.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        width="stretch",
                    )

            st.markdown("##### Package")
            st.download_button(
                "Map Package ZIP",
                map_package_zip_bytes(
                    grid_export,
                    export_metadata,
                    figure=current_plot_figure,
                    validation_df=validation_export,
                    engineering_controls=controls_export,
                    control_regions=regions_export,
                ),
                file_name=f"{base_name}_package.zip",
                mime="application/zip",
                width="stretch",
            )

            if generated_layer_maps and len(generated_layer_maps) > 1:
                st.markdown("##### Batch Multi-Layer Export")
                batch_cols = st.columns(4)
                include_geotiff = batch_cols[0].checkbox("GeoTIFF", value=True, key="batch_export_geotiff")
                include_zmap = batch_cols[1].checkbox("ZMAP", value=True, key="batch_export_zmap")
                include_xyz = batch_cols[2].checkbox("CSV / XYZ", value=True, key="batch_export_xyz")
                include_excel = batch_cols[3].checkbox("Excel", value=False, key="batch_export_excel")
                selected_formats = {"metadata"}
                if include_geotiff:
                    selected_formats.add("geotiff")
                if include_zmap:
                    selected_formats.add("zmap")
                if include_xyz:
                    selected_formats.update({"csv", "xyz"})
                if include_excel:
                    selected_formats.add("excel")
                st.download_button(
                    "Export All Generated Layers ZIP",
                    batch_map_package_zip_bytes(
                        generated_layer_maps,
                        project_metadata=project_metadata,
                        include_formats=selected_formats,
                    ),
                    file_name=f"{base_name}_all_layers_export.zip",
                    mime="application/zip",
                    width="stretch",
                )

            st.markdown("##### Image")
            if current_plot_figure is not None:
                render_static_image_export_controls(current_plot_figure, base_name)
            else:
                st.caption("Image export is available after the map figure has rendered.")
