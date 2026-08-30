"""Mapping Studio page."""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from core.column_mapper import numeric_property_candidates
from core.crs import EPSG_CRS_MODE
from core.data_qc import (
    build_qc_summary,
    descriptive_statistics,
    flag_outliers,
    prepare_interpolation_dataframe,
)
from core.geometry.assignment import assign_points_to_polygons, outside_panel_count
from core.filtering import build_filter_column_list
from core.grid import generate_grid
from core.geometry.compartment import compartment_interpolate
from core.geometry.masking import layer_keep_mask
from core.geometry.models import GeometryLayer
from core.geostatistics.variogram import compute_experimental_variogram, fit_candidate_models
from core.interpolation import InterpolationError, interpolate_surface_result
from core.interpolation.rbf import estimate_epsilon
from core.map_context import build_default_map_title, build_map_metadata
from core.masking import (
    apply_keep_mask,
    auto_maximum_distance,
    combine_masks,
    convex_hull_keep_mask,
    maximum_distance_keep_mask,
)
from core.pressure_dates import (
    format_map_date,
    summarize_measurement_dates,
    validate_date_column,
)
from core.plotting.map_builder import build_map_figure, move_observation_traces_to_top
from core.plotting.geometry_layers import add_fault_layer, add_polygon_layer
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
    get_current_filtered_data,
    mark_project_dirty,
    render_filter_controls,
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
    grid_to_geotiff_bytes,
    grid_to_xyz_ascii_bytes,
    map_package_zip_bytes,
    metadata_to_json_bytes,
)
from utils.units import coordinate_unit_symbol, format_distance
from utils.validators import is_phi_property, is_pressure_property, skewness_is_high


ensure_session_state()


@st.cache_data(show_spinner=False)
def compute_surface_cached(
    records: tuple[tuple[float, float, float], ...],
    method: str,
    method_parameters: dict,
    grid_parameters: dict,
    mask_parameters: dict,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None, dict]:
    data = np.asarray(records, dtype=float)
    if data.ndim != 2 or data.shape[1] != 3:
        raise ValueError("Interpolation records must contain X, Y, and Z values.")
    x = data[:, 0]
    y = data[:, 1]
    z = data[:, 2]
    grid_x, grid_y = generate_grid(
        x,
        y,
        nx=int(grid_parameters["nx"]),
        ny=int(grid_parameters["ny"]),
        buffer_fraction=float(grid_parameters["buffer_fraction"]),
    )
    surface_result = interpolate_surface_result(x, y, z, grid_x, grid_y, method, method_parameters)
    grid_z = surface_result.estimate
    grid_variance = surface_result.variance

    mask_mode = mask_parameters.get("mode", "Convex Hull")
    max_distance_value = mask_parameters.get("max_distance")
    hull_mask = distance_mask = None
    if mask_mode in {"Convex Hull", "Convex Hull + Maximum Distance"}:
        hull_mask = convex_hull_keep_mask(x, y, grid_x, grid_y)
    if mask_mode in {"Maximum Distance", "Reservoir Boundary + Maximum Distance", "Convex Hull + Maximum Distance"}:
        if max_distance_value is None:
            max_distance_value = auto_maximum_distance(x, y)
        distance_mask = maximum_distance_keep_mask(x, y, grid_x, grid_y, float(max_distance_value))

    if hull_mask is not None and distance_mask is not None:
        keep_mask = combine_masks(hull_mask, distance_mask)
    elif hull_mask is not None:
        keep_mask = hull_mask
    elif distance_mask is not None:
        keep_mask = distance_mask
    else:
        keep_mask = np.ones_like(grid_z, dtype=bool)

    masked_z = apply_keep_mask(grid_z, keep_mask)
    masked_variance = apply_keep_mask(grid_variance, keep_mask) if grid_variance is not None else None
    info = {
        "mask_mode": mask_mode,
        "max_distance": max_distance_value,
        "masked_cells": int((~keep_mask).sum()),
        "valid_grid_cells": int(np.isfinite(masked_z).sum()),
    }
    return grid_x, grid_y, masked_z, masked_variance, info


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
        fallback = auto_maximum_distance(prepared["X"], prepared["Y"])
        return fallback if fallback > 0 else 1.0
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
        export_figure = figure.to_plotly_json()
        import plotly.graph_objects as go

        export_figure = go.Figure(export_figure)
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


def pressure_reference_date_control(
    df: pd.DataFrame,
    mappings: dict[str, str | None],
) -> tuple[date | None, object | None]:
    reference_col = mappings.get("map_reference_date")
    validation = None
    detected_common_date = None
    available_dates: list[date] = []
    if reference_col and reference_col in df.columns:
        validation = validate_date_column(df[reference_col])
        available_dates = list(validation.unique_dates)
        detected_common_date = validation.common_date
        if detected_common_date:
            st.caption(f"Detected common Pressure Map Reference Date: {format_map_date(detected_common_date)}")
        elif validation.has_multiple_dates:
            st.caption(
                "Multiple Pressure Map Reference Dates are present in the active dataset. Select the date to use for this map."
            )
        if validation.failed_count:
            st.warning(f"{validation.failed_count} pressure map reference date value(s) could not be parsed.")

        if available_dates:
            current_reference_date = st.session_state.get("pressure_reference_date")
            if current_reference_date not in available_dates:
                current_reference_date = available_dates[0]
            if len(available_dates) == 1:
                selected_reference_date = available_dates[0]
            else:
                selected_reference_date = st.selectbox(
                    "Pressure Map Reference Date",
                    available_dates,
                    index=available_dates.index(current_reference_date),
                    format_func=format_map_date,
                    key="pressure_reference_date_input",
                    help="Select the prepared pressure-map reference date for the active interpolation. Original measurement dates remain metadata only.",
                )
            st.session_state.pressure_reference_date = selected_reference_date
            return selected_reference_date, validation

    current_reference_date = st.session_state.get("pressure_reference_date") or detected_common_date or date.today()
    selected_reference_date = st.date_input(
        "Pressure Map Reference Date",
        value=current_reference_date,
        key="pressure_reference_date_input",
        help="A single reference date represented by the supplied pressure values. No temporal extrapolation is performed.",
    )
    st.session_state.pressure_reference_date = selected_reference_date
    return selected_reference_date, validation


def method_parameter_controls(
    method: str,
    prepared: pd.DataFrame,
    coordinate_unit: str,
    grid_buffer_fraction: float,
) -> dict:
    unit_symbol = coordinate_unit_symbol(coordinate_unit)
    if method == "IDW":
        cols = st.columns(4)
        with cols[0]:
            power = st.number_input("Power", min_value=0.1, max_value=10.0, value=2.0, step=0.1)
        with cols[1]:
            neighbors = st.number_input("Neighbors", min_value=1, max_value=200, value=12, step=1)
        with cols[2]:
            min_neighbors = st.number_input("Minimum Neighbors", min_value=1, max_value=50, value=3, step=1)
        with cols[3]:
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

    if method == "Cubic":
        st.info("Cubic interpolation may overshoot observed property ranges and create smooth but potentially unrealistic reservoir values.")
        return {}

    if method == "RBF":
        cols = st.columns(3)
        with cols[0]:
            kernel_label = st.selectbox("Kernel", list(RBF_KERNELS), index=0)
        with cols[1]:
            smoothing = st.number_input("Smoothing", min_value=0.0, value=0.0, step=0.1)
        with cols[2]:
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

        cols = st.columns(4)
        with cols[0]:
            variogram_mode = st.radio("Variogram Parameters", ["Auto Fit", "Manual"], horizontal=True)
        best_fit = None
        fit_error = None
        if variogram_mode == "Auto Fit" and len(prepared) >= 5 and values.nunique() > 1:
            try:
                experimental = compute_experimental_variogram(prepared["X"], prepared["Y"], prepared["Z"], n_lags=10)
                fits = fit_candidate_models(experimental)
                if fits:
                    best_fit = fits[0]
                    st.caption(
                        f"Best numerical variogram fit: {best_fit.model}; range "
                        f"{format_distance(best_fit.range_value, coordinate_unit)}, "
                        f"variance {best_fit.variance:.4g}, nugget {best_fit.nugget:.4g}."
                    )
                    fit_error = best_fit.fit_error
            except ValueError as exc:
                st.warning(str(exc))
        with cols[1]:
            model = st.selectbox(
                "Model",
                ["Spherical", "Exponential", "Gaussian"],
                index=["Spherical", "Exponential", "Gaussian"].index(best_fit.model) if best_fit else 0,
            )
        with cols[2]:
            range_value = st.number_input(
                f"Range ({unit_symbol})",
                min_value=0.0001,
                value=float(best_fit.range_value if best_fit else default_range),
                step=max(default_range / 20.0, 1.0),
            )
        with cols[3]:
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


def grid_controls() -> dict:
    cols = st.columns(4)
    with cols[0]:
        preset = st.selectbox("Grid Resolution", list(GRID_PRESETS), index=1)
    if preset == "Custom":
        with cols[1]:
            nx = st.number_input("NX", min_value=10, max_value=500, value=150, step=10)
        with cols[2]:
            ny = st.number_input("NY", min_value=10, max_value=500, value=150, step=10)
    else:
        nx, ny = GRID_PRESETS[preset]
        cols[1].metric("NX", nx)
        cols[2].metric("NY", ny)
    with cols[3]:
        buffer_percent = st.number_input("Buffer (%)", min_value=0.0, max_value=50.0, value=3.0, step=0.5)
    return {"preset": preset, "nx": int(nx), "ny": int(ny), "buffer_fraction": float(buffer_percent) / 100.0}


def mask_controls(prepared: pd.DataFrame, coordinate_unit: str, has_reservoir_boundary: bool = False) -> dict:
    unit_symbol = coordinate_unit_symbol(coordinate_unit)
    cols = st.columns(3)
    with cols[0]:
        options = list(MASK_OPTIONS)
        if not has_reservoir_boundary:
            options = [option for option in options if "Reservoir Boundary" not in option]
        mode = st.radio("Mask", options, index=0)
    max_distance = None
    if "Maximum Distance" in mode:
        with cols[1]:
            distance_mode = st.radio("Maximum Distance Mode", ["Auto", "Manual"], horizontal=True)
        if distance_mode == "Manual":
            with cols[2]:
                max_distance = st.number_input(
                    f"Maximum Distance ({unit_symbol})",
                    min_value=0.0001,
                    value=1000.0,
                    step=100.0,
                )
        else:
            auto_distance = auto_maximum_distance(prepared["X"], prepared["Y"]) if not prepared.empty else 0.0
            cols[2].metric(f"Auto Distance ({unit_symbol})", f"{auto_distance:,.4g}")
            max_distance = auto_distance
    return {"mode": mode, "max_distance": max_distance, "distance_mode": locals().get("distance_mode")}


def render_layer_manager() -> dict:
    settings = dict(st.session_state.get("layer_settings", {}))
    geometry_layers = st.session_state.geometry_layers
    custom_layers = geometry_layers.get("custom", [])
    with st.expander("Layer Manager", expanded=True):
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
                0.5,
                6.0,
                float(settings.get("reservoir_boundary_width", 2.5)),
                0.5,
            )
        if geometry_layers.get("panels"):
            settings["show_panels"] = st.checkbox("Panel Boundaries", value=bool(settings.get("show_panels", True)))
            settings["show_panel_labels"] = st.checkbox(
                "Panel Labels",
                value=bool(settings.get("show_panel_labels", False)),
            )
            settings["panel_boundary_width"] = st.slider(
                "Panel Boundary Width",
                0.5,
                5.0,
                float(settings.get("panel_boundary_width", 1.5)),
                0.5,
            )
        if geometry_layers.get("faults"):
            settings["show_faults"] = st.checkbox("Faults", value=bool(settings.get("show_faults", True)))
            settings["show_fault_labels"] = st.checkbox(
                "Fault Names",
                value=bool(settings.get("show_fault_labels", False)),
            )
            settings["fault_line_width"] = st.slider(
                "Fault Line Width",
                0.5,
                6.0,
                float(settings.get("fault_line_width", 2.0)),
                0.5,
            )
        if custom_layers:
            settings["show_custom_layers"] = st.checkbox(
                "Custom Layers",
                value=bool(settings.get("show_custom_layers", True)),
            )
            settings["show_custom_labels"] = st.checkbox(
                "Custom Labels",
                value=bool(settings.get("show_custom_labels", False)),
            )
            settings["custom_line_width"] = st.slider(
                "Custom Layer Width",
                0.5,
                5.0,
                float(settings.get("custom_line_width", 1.5)),
                0.5,
            )
        settings["geometry_fill_opacity"] = st.slider(
            "Geometry Fill Opacity",
            0.0,
            0.25,
            float(settings.get("geometry_fill_opacity", 0.0)),
            0.01,
        )
    st.session_state.layer_settings = settings
    return settings


def apply_surface_masks(
    grid_z: np.ndarray,
    grid_variance: np.ndarray | None,
    prepared: pd.DataFrame,
    grid_x: np.ndarray,
    grid_y: np.ndarray,
    mask_parameters: dict,
    reservoir_layer: GeometryLayer | None = None,
    include_convex_and_distance: bool = True,
) -> tuple[np.ndarray, np.ndarray | None, dict]:
    mode = mask_parameters.get("mode", "Convex Hull")
    masks: list[np.ndarray] = []
    max_distance_value = mask_parameters.get("max_distance")
    if "Reservoir Boundary" in mode and reservoir_layer is not None:
        masks.append(layer_keep_mask(reservoir_layer, grid_x, grid_y))
    if include_convex_and_distance and "Convex Hull" in mode:
        masks.append(convex_hull_keep_mask(prepared["X"], prepared["Y"], grid_x, grid_y))
    if include_convex_and_distance and "Maximum Distance" in mode:
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
    }


st.title("Reservoir Mapping Studio")
st.subheader("Mapping Studio")

df = st.session_state.get("working_df")
if df is None:
    st.info("Load data in the Data Manager before opening the Mapping Studio.")
    st.stop()

mappings = st.session_state.get("column_mappings", {})
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

with st.sidebar:
    st.header("Project")
    coordinate_unit = coordinate_unit_input("mapping_studio_coordinate_unit")
    st.caption(
        f"Spatial distances use {coordinate_unit_symbol(coordinate_unit)}. "
        "Uploaded X/Y coordinates are not converted."
    )
    st.header("Filters")
    filtered_base = render_filter_controls(df, mappings, "mapping_studio")
    layer_settings = render_layer_manager()

property_options = numeric_property_candidates(df, mappings)
if not property_options:
    st.warning("No numeric property columns are available for mapping.")
    st.stop()

top_cols = st.columns([2, 1, 1, 1, 1])
with top_cols[0]:
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
with top_cols[1]:
    unit = unit_input("mapping_unit")
with top_cols[2]:
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
with top_cols[3]:
    inferred_type = "Pressure" if is_pressure_property(property_col) else "Generic"
    property_type = st.radio(
        "Property Type",
        ["Pressure", "Generic"],
        index=0 if inferred_type == "Pressure" else 1,
        horizontal=True,
        key=f"property_type_{property_col}",
        help="Auto-detection can be overridden when a pressure-like column name is ambiguous.",
    )
with top_cols[4]:
    st.metric("Filtered Rows", f"{len(filtered_base):,}")

is_pressure_map = property_type == "Pressure"
filtered = get_current_filtered_data()
filtered = flag_outliers(filtered, property_col)
pressure_reference_date = None
reference_validation = None
if is_pressure_map:
    st.info(
        "Pressure values should already be prepared/extrapolated to the map reference date before import. "
        "Reservoir Mapping Studio performs spatial interpolation only."
    )
    pressure_reference_date, reference_validation = pressure_reference_date_control(filtered, mappings)
    reference_col = mappings.get("map_reference_date")
    if reference_col and reference_col in filtered.columns and pressure_reference_date is not None:
        reference_mask = pd.to_datetime(filtered[reference_col], errors="coerce").dt.date.eq(pressure_reference_date)
        filtered = filtered.loc[reference_mask].copy()

filtered_with_include = attach_include_column(filtered)

property_values = pd.to_numeric(filtered_with_include[property_col], errors="coerce")

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

map_tab, interpolation_tab, style_tab, data_tab, export_tab = st.tabs(
    ["Map", "Interpolation", "Style", "Data / QC", "Export"]
)

with data_tab:
    st.markdown("#### Active Observations")
    display_columns = [INCLUDE_COLUMN]
    for column in [well_col, x_col, y_col, property_col]:
        if column and column not in display_columns:
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
    filtered_with_include = attach_include_column(filtered)

    included = filtered_with_include[filtered_with_include[INCLUDE_COLUMN]]
    stats = descriptive_statistics(pd.to_numeric(included[property_col], errors="coerce"))
    stat_cols = st.columns(4)
    stat_cols[0].metric("Observations", f"{stats['observations']:,}")
    stat_cols[1].metric("Mean", "" if stats["mean"] is None else f"{stats['mean']:.4g}")
    stat_cols[2].metric("Median", "" if stats["median"] is None else f"{stats['median']:.4g}")
    stat_cols[3].metric("Std Dev", "" if stats["std"] is None else f"{stats['std']:.4g}")
    stat_cols_2 = st.columns(5)
    stat_cols_2[0].metric("Minimum", "" if stats["minimum"] is None else f"{stats['minimum']:.4g}")
    stat_cols_2[1].metric("Maximum", "" if stats["maximum"] is None else f"{stats['maximum']:.4g}")
    stat_cols_2[2].metric("P10", "" if stats["p10"] is None else f"{stats['p10']:.4g}")
    stat_cols_2[3].metric("P50", "" if stats["p50"] is None else f"{stats['p50']:.4g}")
    stat_cols_2[4].metric("P90", "" if stats["p90"] is None else f"{stats['p90']:.4g}")

    qc = build_qc_summary(filtered_with_include, x_col, y_col, property_col, well_col)
    if qc.missing_property:
        st.warning(f"{qc.missing_property} record(s) have missing or non-numeric {property_col} values.")
    if qc.duplicate_xy_rows:
        st.warning(f"{qc.duplicate_xy_rows} record(s) share duplicate XY coordinates. Choose duplicate handling before interpolation.")
    if qc.outlier_count:
        st.warning(f"{qc.outlier_count} potential outlier(s) are flagged. They remain included unless you clear Include.")

    if panel_layer is not None and panel_layer.polygon_features:
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
        st.markdown("#### Pressure Date QC")
        if pressure_reference_date:
            st.metric("Pressure Map Reference Date", format_map_date(pressure_reference_date))
        measurement_col = mappings.get("measurement_date")
        if measurement_col and measurement_col in filtered_with_include.columns:
            measurement_summary = summarize_measurement_dates(filtered_with_include[measurement_col])
            pressure_date_cols = st.columns(3)
            pressure_date_cols[0].metric(
                "Earliest Original Measurement",
                format_map_date(measurement_summary.earliest) if measurement_summary.earliest else "",
            )
            pressure_date_cols[1].metric(
                "Latest Original Measurement",
                format_map_date(measurement_summary.latest) if measurement_summary.latest else "",
            )
            pressure_date_cols[2].metric(
                "Measurement-Date Span",
                "" if measurement_summary.span_days is None else f"{measurement_summary.span_days:,} days",
            )
            if measurement_summary.failed_count:
                st.warning(f"{measurement_summary.failed_count} original measurement date value(s) could not be parsed.")
        else:
            st.caption("No original pressure measurement date column is mapped.")

    with st.expander("Histogram", expanded=False):
        valid_values = pd.to_numeric(included[property_col], errors="coerce").dropna()
        if not valid_values.empty:
            histogram = px.histogram(valid_values, nbins=20, labels={"value": property_col})
            histogram.update_layout(showlegend=False, height=300, margin={"l": 20, "r": 20, "t": 20, "b": 20})
            st.plotly_chart(histogram, width="stretch")
        else:
            st.caption("No finite values are available for the histogram.")

with style_tab:
    style = dict(st.session_state.get("style_settings", {}))
    st.markdown("#### Map Styling")
    style_cols = st.columns(3)
    with style_cols[0]:
        style["color_scale"] = st.selectbox(
            "Color Scale",
            color_scale_options(),
            index=color_scale_options().index(style.get("color_scale", "Turbo")),
        )
        style["reverse_colors"] = st.checkbox("Reverse Colors", value=bool(style.get("reverse_colors", False)))
    with style_cols[1]:
        style["z_range_mode"] = st.radio("Property Range", ["Auto", "Manual"], index=0 if style.get("z_range_mode") == "Auto" else 1)
        if style["z_range_mode"] == "Manual":
            current_values = pd.to_numeric(filtered_with_include[property_col], errors="coerce").dropna()
            default_min = float(current_values.min()) if not current_values.empty else 0.0
            default_max = float(current_values.max()) if not current_values.empty else 1.0
            style["zmin"] = st.number_input("Minimum", value=float(style.get("zmin") or default_min))
            style["zmax"] = st.number_input("Maximum", value=float(style.get("zmax") or default_max))
    with style_cols[2]:
        style["height"] = st.slider("Map Height", min_value=450, max_value=1000, value=int(style.get("height", 720)), step=25)

    contour_cols = st.columns(4)
    with contour_cols[0]:
        style["contour_mode"] = st.radio(
            "Contours",
            ["Auto interval", "Manual interval"],
            index=0 if style.get("contour_mode") == "Auto interval" else 1,
        )
    with contour_cols[1]:
        if style["contour_mode"] == "Manual interval":
            style["contour_interval"] = st.number_input(
                "Interval",
                min_value=0.000001,
                value=float(style.get("contour_interval") or 10.0),
            )
    with contour_cols[2]:
        style["show_contour_lines"] = st.checkbox("Show Contour Lines", value=bool(style.get("show_contour_lines", True)))
        style["show_contour_labels"] = st.checkbox("Show Contour Labels", value=bool(style.get("show_contour_labels", False)))
    with contour_cols[3]:
        style["contour_line_width"] = st.slider(
            "Line Width",
            min_value=0.0,
            max_value=3.0,
            value=float(style.get("contour_line_width", 0.75)),
            step=0.25,
        )

    well_cols = st.columns(4)
    with well_cols[0]:
        style["show_wells"] = st.checkbox("Show Wells", value=bool(style.get("show_wells", True)))
        style["show_excluded"] = st.checkbox("Show Excluded Observations", value=bool(style.get("show_excluded", True)))
    with well_cols[1]:
        style["marker_size"] = st.slider("Marker Size", min_value=4, max_value=20, value=int(style.get("marker_size", 9)))
        style["marker_opacity"] = st.slider("Marker Opacity", min_value=0.1, max_value=1.0, value=float(style.get("marker_opacity", 0.9)))
    with well_cols[2]:
        style["marker_outline"] = st.checkbox("Marker Outline", value=bool(style.get("marker_outline", True)))
        label_options = ["None", "Property Value"]
        if well_col:
            label_options = ["None", "Well Name", "Property Value", "Well Name + Property Value"]
        current_label = style.get("well_label_mode", "None")
        if current_label not in label_options:
            current_label = "None"
        style["well_label_mode"] = st.selectbox("Well Labels", label_options, index=label_options.index(current_label))
    with well_cols[3]:
        style["label_text_size"] = st.slider("Label Text Size", min_value=8, max_value=18, value=int(style.get("label_text_size", 11)))

    style["title_override"] = st.text_input("Map Title Override", value=style.get("title_override", ""))
    st.session_state.style_settings = style

with interpolation_tab:
    st.markdown("#### Interpolation Setup")
    duplicate_method = st.selectbox("Duplicate Coordinate Handling", DUPLICATE_METHODS, index=0)
    prepared = prepare_interpolation_dataframe(
        filtered_with_include,
        x_col,
        y_col,
        property_col,
        include_col=INCLUDE_COLUMN,
        duplicate_method=duplicate_method,
    )
    grid_parameters = grid_controls()
    method = st.radio("Interpolation Method", INTERPOLATION_METHODS, horizontal=True, index=0)
    respect_compartments = False
    if panel_layer is not None and panel_layer.polygon_features:
        respect_compartments = st.checkbox(
            "Respect Panel / Compartment Boundaries",
            value=False,
            help="When enabled, interpolation is performed separately inside each active panel polygon.",
        )
        if respect_compartments:
            st.caption("Observations outside all active panel polygons are excluded from compartment-specific interpolation.")
    method_parameters = method_parameter_controls(
        method,
        prepared,
        coordinate_unit,
        float(grid_parameters["buffer_fraction"]),
    )
    mask_parameters = mask_controls(prepared, coordinate_unit, has_reservoir_boundary=reservoir_boundary_layer is not None)
    st.caption(f"{len(prepared):,} finite included observation(s) will participate after duplicate handling.")

    if st.button("GENERATE / UPDATE MAP", type="primary", width="stretch"):
        if prepared.empty:
            st.error("No finite included observations are available for interpolation.")
        else:
            records = tuple(
                tuple(float(value) for value in row)
                for row in prepared[["X", "Y", "Z"]].to_numpy()
            )
            try:
                with st.spinner("Generating interpolated reservoir map..."):
                    if respect_compartments and panel_layer is not None:
                        grid_x, grid_y = generate_grid(
                            prepared["X"],
                            prepared["Y"],
                            nx=int(grid_parameters["nx"]),
                            ny=int(grid_parameters["ny"]),
                            buffer_fraction=float(grid_parameters["buffer_fraction"]),
                        )
                        compartment_result = compartment_interpolate(
                            prepared,
                            panel_layer,
                            grid_x,
                            grid_y,
                            method,
                            method_parameters,
                            min_observations=max(3, int(method_parameters.get("min_neighbors", 3))),
                        )
                        grid_z, grid_variance, mask_info = apply_surface_masks(
                            compartment_result.surface,
                            compartment_result.variance,
                            prepared,
                            grid_x,
                            grid_y,
                            mask_parameters,
                            reservoir_boundary_layer,
                            include_convex_and_distance=True,
                        )
                        mask_info["panel_constraint"] = True
                        mask_info["compartment_warnings"] = compartment_result.warnings
                        panel_grid = compartment_result.panel_grid
                    else:
                        grid_x, grid_y, grid_z, grid_variance, mask_info = compute_surface_cached(
                            records,
                            method,
                            method_parameters,
                            grid_parameters,
                            mask_parameters,
                        )
                        if "Reservoir Boundary" in mask_parameters.get("mode", "") and reservoir_boundary_layer is not None:
                            grid_z, grid_variance, geometry_mask_info = apply_surface_masks(
                                grid_z,
                                grid_variance,
                                prepared,
                                grid_x,
                                grid_y,
                                mask_parameters,
                                reservoir_boundary_layer,
                                include_convex_and_distance=False,
                            )
                            mask_info["masked_cells"] = geometry_mask_info["masked_cells"]
                            mask_info["valid_grid_cells"] = geometry_mask_info["valid_grid_cells"]
                        mask_info["panel_constraint"] = False
                        panel_grid = None
                generated_included = filtered_with_include[filtered_with_include[INCLUDE_COLUMN]].copy()
                generated_excluded = filtered_with_include[~filtered_with_include[INCLUDE_COLUMN]].copy()
                title = build_default_map_title(
                    property_col,
                    st.session_state.get("filter_values", {}),
                    mappings,
                    is_pressure_map=is_pressure_map,
                    map_reference_date=pressure_reference_date,
                )
                export_metadata = build_map_metadata(
                    property_col,
                    unit,
                    x_col,
                    y_col,
                    coordinate_unit,
                    method,
                    grid_parameters,
                    method_parameters,
                    mask_parameters,
                    duplicate_method,
                    is_pressure_map=is_pressure_map,
                    map_reference_date=pressure_reference_date,
                    geometry_context={
                        "reservoir_boundary_used": "Reservoir Boundary" in mask_parameters.get("mode", ""),
                        "panel_constraint_used": respect_compartments,
                        "active_panels": ", ".join(feature.name for feature in panel_layer.polygon_features)
                        if panel_layer is not None
                        else "",
                        "fault_layer_loaded": fault_layer is not None,
                        "custom_layer_count": len(custom_layers),
                    },
                )
                st.session_state.generated_map = {
                    "grid_x": grid_x,
                    "grid_y": grid_y,
                    "grid_z": grid_z,
                    "grid_variance": grid_variance,
                    "panel_grid": panel_grid,
                    "included_observations": generated_included,
                    "excluded_observations": generated_excluded,
                    "property_col": property_col,
                    "unit": unit,
                    "x_col": x_col,
                    "y_col": y_col,
                    "well_col": well_col,
                    "coordinate_unit": coordinate_unit,
                    "is_pressure_map": is_pressure_map,
                    "map_reference_date": pressure_reference_date,
                    "measurement_date_col": mappings.get("measurement_date"),
                    "map_reference_date_col": mappings.get("map_reference_date"),
                    "method": method,
                    "method_parameters": method_parameters,
                    "grid_parameters": grid_parameters,
                    "mask_parameters": mask_parameters,
                    "mask_info": mask_info,
                    "respect_compartments": respect_compartments,
                    "geometry_context": {
                        "reservoir_boundary_loaded": reservoir_boundary_layer is not None,
                        "panel_layer_loaded": panel_layer is not None,
                        "fault_layer_loaded": fault_layer is not None,
                        "custom_layer_count": len(custom_layers),
                    },
                    "duplicate_method": duplicate_method,
                    "hover_columns": generated_hover_columns(mappings),
                    "title": title,
                    "export_metadata": export_metadata,
                }
                st.success("Map generated.")
                for warning in mask_info.get("compartment_warnings", []):
                    st.warning(warning)
            except InterpolationError as exc:
                st.error(str(exc))
            except ValueError as exc:
                st.error(str(exc))

    if not prepared.empty and prepared["Source_Count"].max() > 1:
        duplicate_count = int((prepared["Source_Count"] > 1).sum())
        st.caption(f"{duplicate_count} duplicate coordinate location(s) detected before interpolation.")

with map_tab:
    generated = st.session_state.get("generated_map")
    if not generated:
        st.info("Configure interpolation and click GENERATE / UPDATE MAP to create the first surface.")
    else:
        style = st.session_state.get("style_settings", {})
        layer_settings = st.session_state.get("layer_settings", {})
        title = style.get("title_override") or generated.get("title")
        display_coordinate_unit = st.session_state.get("coordinate_unit", generated.get("coordinate_unit"))
        display_grid = generated["grid_z"]
        display_property = generated["property_col"]
        display_unit = generated.get("unit")
        if generated.get("grid_variance") is not None:
            display_mode = st.radio(
                "Map Display",
                ["Estimated Property", "Kriging Variance", "Kriging Standard Deviation"],
                horizontal=True,
            )
            if display_mode == "Kriging Variance":
                display_grid = generated["grid_variance"]
                display_property = f"{generated['property_col']} Kriging Variance"
                display_unit = f"{display_unit}^2" if display_unit else "property unit^2"
            elif display_mode == "Kriging Standard Deviation":
                display_grid = np.sqrt(np.maximum(generated["grid_variance"], 0.0))
                display_property = f"{generated['property_col']} Kriging Std Dev"
        figure = build_map_figure(
            generated["grid_x"],
            generated["grid_y"],
            display_grid,
            generated["included_observations"],
            generated["excluded_observations"],
            generated["x_col"],
            generated["y_col"],
            generated["property_col"],
            well_col=generated.get("well_col"),
            hover_columns=generated.get("hover_columns", []),
            title=title,
            unit=generated.get("unit"),
            coordinate_unit=display_coordinate_unit,
            is_pressure_map=bool(generated.get("is_pressure_map")),
            map_reference_date=generated.get("map_reference_date"),
            measurement_date_col=generated.get("measurement_date_col"),
            map_reference_date_col=generated.get("map_reference_date_col"),
            style={**style, **layer_settings},
            surface_label=display_property,
            surface_unit=display_unit,
        )
        add_polygon_layer(
            figure,
            reservoir_boundary_layer,
            display_coordinate_unit,
            visible=bool(layer_settings.get("show_reservoir_boundary", True)),
            line_color="#0F172A",
            line_width=float(layer_settings.get("reservoir_boundary_width", 2.5)),
            fill_opacity=float(layer_settings.get("geometry_fill_opacity", 0.0)),
            name="Reservoir Boundary",
        )
        add_polygon_layer(
            figure,
            panel_layer,
            display_coordinate_unit,
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
                    display_coordinate_unit,
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
        move_observation_traces_to_top(figure)
        st.plotly_chart(figure, width="stretch", config={"displaylogo": False, "scrollZoom": True})

        info_cols = st.columns(7)
        info_cols[0].metric("Display", display_property)
        info_cols[1].metric("Method", generated["method"])
        info_cols[2].metric("Observations", f"{len(generated['included_observations']):,}")
        info_cols[3].metric("Grid", f"{generated['grid_parameters']['nx']} x {generated['grid_parameters']['ny']}")
        info_cols[4].metric("Coordinate Unit", coordinate_unit_symbol(display_coordinate_unit))
        info_cols[5].metric("Mask", generated["mask_info"]["mask_mode"])
        info_cols[6].metric("Valid Cells", f"{generated['mask_info']['valid_grid_cells']:,}")
        if generated.get("is_pressure_map") and generated.get("map_reference_date"):
            st.metric("Pressure Map Reference Date", format_map_date(generated.get("map_reference_date")))
        if generated.get("respect_compartments"):
            st.caption("Panel / compartment constraint: enabled. Each active panel was interpolated from its own observations.")

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

        st.markdown("#### Map Scenario")
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

        saved_scenarios = list(st.session_state.get("map_scenarios", []))
        if saved_scenarios:
            st.dataframe(scenario_summary_table(saved_scenarios), width="stretch", hide_index=True)
            labels = [str(item.get("name") or item.get("id")) for item in saved_scenarios]
            selected_label = st.selectbox("Selected Scenario", labels, key="mapping_studio_selected_scenario")
            selected_index = labels.index(selected_label)
            selected = saved_scenarios[selected_index]
            scenario_action_cols = st.columns(5)
            if scenario_action_cols[0].button("Open"):
                st.session_state.generated_map = scenario_to_generated_map(selected)
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
            st.info("No saved map scenarios yet. Save the current generated map to create one.")

with export_tab:
    st.markdown("#### Export")
    current_for_export = attach_include_column(get_current_filtered_data())
    included_for_export = current_for_export[current_for_export[INCLUDE_COLUMN]].copy()
    export_clean = drop_internal_columns(included_for_export)
    export_cols = st.columns(2)
    with export_cols[0]:
        st.download_button(
            "Download Filtered Observations CSV",
            dataframe_to_csv_bytes(export_clean),
            file_name="filtered_observations.csv",
            mime="text/csv",
            width="stretch",
        )
    with export_cols[1]:
        st.download_button(
            "Download Filtered Observations Excel",
            dataframe_to_excel_bytes(export_clean, "Filtered Observations"),
            file_name="filtered_observations.xlsx",
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
                "active_panels": ", ".join(feature.name for feature in panel_layer.polygon_features)
                if panel_layer is not None
                else "",
                "fault_layer_loaded": fault_layer is not None,
                "custom_layer_count": len(custom_layers),
            },
        )
        metadata_summary = [
            f"Property: {generated['property_col']}",
            f"Coordinate Unit: {coordinate_unit_symbol(export_coordinate_unit)}",
            f"Interpolation: {generated['method']}",
        ]
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
                "Download Interpolated Grid CSV",
                dataframe_to_csv_bytes(grid_export),
                file_name="interpolated_grid.csv",
                mime="text/csv",
                width="stretch",
            )
        with grid_cols[1]:
            st.download_button(
                "Download Interpolated Grid Excel",
                grid_to_excel_bytes(grid_export, export_metadata),
                file_name="interpolated_grid.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                width="stretch",
            )
    else:
        st.caption("Generate a map before exporting an interpolated grid.")
