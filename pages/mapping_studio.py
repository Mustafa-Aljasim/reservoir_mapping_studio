"""Mapping Studio page."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from core.active_data import prepare_active_property_data
from core.column_mapper import normalize_column_mappings, numeric_property_candidates
from core.data_qc import build_qc_summary, descriptive_statistics, flag_outliers
from core.filtering import build_filter_column_list
from core.geometry.assignment import assign_points_to_polygons, outside_panel_count
from core.geometry.compartment import filter_dataframe_to_selected_panels, selected_panel_bounds
from core.geometry.masking import polygon_union
from core.geostatistics.variogram import (
    VARIOGRAM_RANGE_CONVENTION,
    compute_experimental_variogram,
    fit_candidate_models,
)
from core.interpolation.rbf import estimate_epsilon
from core.layer_mapping import (
    LAYER_SCOPE_ALL,
    LAYER_SCOPE_SELECTED,
    MAP_STATUS_UP_TO_DATE,
    UNSPECIFIED_LAYER,
    build_layer_model_signature,
    filter_values_excluding_semantics,
    generate_layer_map_collection,
    layer_column,
    map_status,
    prepare_layer_observations,
    reservoir_layer_values,
    resolve_domain_bounds,
    resolve_layer_method_parameters,
    select_generated_layer_map,
)
from core.map_context import build_map_metadata
from core.masking import auto_maximum_distance
from core.pressure_dates import format_map_date, summarize_measurement_dates
from core.plotting.geometry_layers import add_fault_layer, add_polygon_layer
from core.plotting.map_builder import build_map_figure, move_observation_traces_to_top
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
    dataframe_to_csv_bytes,
    dataframe_to_excel_bytes,
    drop_internal_columns,
    figure_to_image_bytes,
    grid_to_dataframe,
    grid_to_excel_bytes,
    map_package_zip_bytes,
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
    return build_filter_column_list(mappings, st.session_state.get("additional_filter_columns", []))


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


def render_static_image_exports(figure, base_name: str) -> None:
    with st.expander("Image Export", expanded=False):
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


def mask_controls(prepared: pd.DataFrame, coordinate_unit: str, has_reservoir_boundary: bool = False) -> dict[str, object]:
    unit_symbol = coordinate_unit_symbol(coordinate_unit)
    options = list(MASK_OPTIONS)
    if not has_reservoir_boundary:
        options = [option for option in options if "Reservoir Boundary" not in option]
    mode = st.radio("Mask", options, index=0)
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

    settings["show_surface"] = st.checkbox("Property Surface", value=bool(settings.get("show_surface", True)))
    settings["show_wells"] = st.checkbox("Wells", value=bool(settings.get("show_wells", True)))
    settings["show_excluded"] = st.checkbox("Excluded Wells", value=bool(settings.get("show_excluded", True)))
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
            active_for_layers = prepare_active_property_data(
                df,
                mappings,
                property_col,
                property_type,
                pressure_reference_date,
                filter_values_no_layer,
                selected_panels if panel_layer is not None else None,
            ).dataframe
            if selected_panels and panel_layer is not None and not mappings.get("panel"):
                active_for_layers = filter_dataframe_to_selected_panels(
                    active_for_layers,
                    x_col,
                    y_col,
                    panel_layer,
                    selected_panels,
                )
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
        mask_parameters = mask_controls(
            prepared,
            coordinate_unit,
            has_reservoir_boundary=reservoir_boundary_layer is not None and bool(reservoir_boundary_layer.polygon_features),
        )
        domain_options = ["Well Data Extent"]
        if reservoir_boundary_layer is not None and reservoir_boundary_layer.polygon_features:
            domain_options.append("Reservoir Boundary Extent")
        selected_panel_domain_bounds = selected_panel_bounds(panel_layer, selected_panels)
        if selected_panel_domain_bounds is not None:
            domain_options.append("Selected Panel Extent")
        current_domain = st.session_state.get("mapping_interpolation_domain")
        if current_domain not in domain_options:
            current_domain = domain_options[-1] if len(domain_options) > 1 else domain_options[0]
        interpolation_domain = st.radio(
            "Interpolation Domain",
            domain_options,
            index=domain_options.index(current_domain),
            horizontal=True,
            key="mapping_interpolation_domain",
        )
        domain_bounds = resolve_domain_bounds(
            interpolation_domain,
            reservoir_boundary_layer,
            panel_layer,
            selected_panels,
        )
        if interpolation_domain == "Reservoir Boundary Extent":
            st.caption("The full rectangular reservoir bounding box will be interpolated before any polygon mask is applied.")
        if interpolation_domain == "Selected Panel Extent":
            st.caption("The selected panel union bounds will be interpolated before panel or reservoir masks are applied.")
        if method in {"Linear", "Cubic"} and interpolation_domain == "Reservoir Boundary Extent":
            st.info("Linear and Cubic interpolation may remain NaN outside the convex hull of the observations.")
        st.caption(f"{len(prepared):,} finite included observation(s) will participate after duplicate handling.")

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
    if signature_prepared.empty:
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
    )


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

    if not generated:
        st.info("Configure the controls and generate a map to create the first surface.")
    else:
        display_status = map_status(current_signature_for_display(generated), generated)
        status_cols = st.columns([1.2, 1, 1, 1])
        with status_cols[0]:
            render_status(display_status)
        status_cols[1].metric("Layer Scope", generated.get("layer_mapping_scope") or LAYER_SCOPE_SELECTED)
        status_cols[2].metric("Reservoir Layer", generated.get("reservoir_layer") or UNSPECIFIED_LAYER)
        status_cols[3].metric("Included Wells", f"{len(generated.get('included_observations', [])):,}")

        style = st.session_state.get("style_settings", {})
        layer_settings = st.session_state.get("layer_settings", {})
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
            style={**style, **layer_settings},
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
        move_observation_traces_to_top(figure)
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

        with st.expander("Spatial Domain Diagnostics", expanded=False):
            st.write(f"Well extent: {extent_text(diagnostics.get('well_extent'))}")
            st.write(f"Reservoir extent: {extent_text(diagnostics.get('reservoir_extent'))}")
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

        render_static_image_exports(figure, f"{generated['property_col']}_{layer_map_key(generated)}")

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
        current_for_export = filtered_with_include.copy()
        included_for_export = (
            current_for_export[current_for_export[INCLUDE_COLUMN]].copy()
            if INCLUDE_COLUMN in current_for_export
            else current_for_export
        )
        export_clean = drop_internal_columns(included_for_export)
        export_cols = st.columns(2)
        with export_cols[0]:
            st.download_button(
                "Download Active Observations CSV",
                dataframe_to_csv_bytes(export_clean),
                file_name="active_map_observations.csv",
                mime="text/csv",
                width="stretch",
            )
        with export_cols[1]:
            st.download_button(
                "Download Active Observations Excel",
                dataframe_to_excel_bytes(export_clean, "Active Observations"),
                file_name="active_map_observations.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                width="stretch",
            )

        generated = st.session_state.get("generated_map")
        if generated:
            export_coordinate_unit = st.session_state.get("coordinate_unit", generated.get("coordinate_unit"))
            export_metadata = build_map_metadata(
                generated["property_col"],
                generated.get("unit"),
                generated["x_col"],
                generated["y_col"],
                export_coordinate_unit,
                generated["method"],
                generated["grid_parameters"],
                generated["method_parameters"],
                generated["mask_parameters"],
                generated["duplicate_method"],
                is_pressure_map=bool(generated.get("is_pressure_map")),
                map_reference_date=generated.get("map_reference_date"),
                geometry_context={
                    "reservoir_boundary_used": "Reservoir Boundary" in generated["mask_parameters"].get("mode", ""),
                    "panel_constraint_used": bool(generated.get("respect_compartments")),
                    "active_panels": ", ".join(str(value) for value in generated.get("selected_panels", [])),
                    "fault_layer_loaded": fault_layer is not None,
                    "custom_layer_count": len(custom_layers),
                    **(generated.get("geometry_references", {}) or {}),
                },
                property_type=generated.get("property_type") or ("Pressure" if generated.get("is_pressure_map") else "Generic"),
                crs=st.session_state.get("crs", {}),
                selected_panels=generated.get("selected_panels", []),
                selected_layers=generated.get("selected_layers", []),
                panel_interpolation_mode=generated.get("panel_interpolation_mode"),
                interpolation_domain=generated.get("interpolation_domain"),
                domain_bounds=generated.get("domain_bounds"),
                grid_x=generated.get("grid_x"),
                grid_y=generated.get("grid_y"),
                validation_metrics=generated.get("validation_metrics", {}),
                model_signature_hash=(generated.get("model_signature") or {}).get("hash", ""),
                reservoir_layer=generated.get("reservoir_layer"),
                layer_mapping_scope=generated.get("layer_mapping_scope"),
            )
            metadata_summary = [
                f"Property: {generated['property_col']}",
                f"Coordinate Unit: {coordinate_unit_symbol(export_coordinate_unit)}",
                f"Interpolation: {generated['method']}",
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
            grid_cols = st.columns(2)
            with grid_cols[0]:
                st.download_button(
                    "Download Grid CSV",
                    dataframe_to_csv_bytes(grid_export),
                    file_name="interpolated_grid.csv",
                    mime="text/csv",
                    width="stretch",
                )
                st.download_button(
                    "Download Metadata JSON",
                    metadata_to_json_bytes(export_metadata),
                    file_name="map_metadata.json",
                    mime="application/json",
                    width="stretch",
                )
            with grid_cols[1]:
                st.download_button(
                    "Download Grid Excel",
                    grid_to_excel_bytes(grid_export, export_metadata),
                    file_name="interpolated_grid.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    width="stretch",
                )
                st.download_button(
                    "Download Map Package ZIP",
                    map_package_zip_bytes(grid_export, export_metadata),
                    file_name="map_package.zip",
                    mime="application/zip",
                    width="stretch",
                )
        else:
            st.caption("Generate a map before exporting an interpolated grid.")
