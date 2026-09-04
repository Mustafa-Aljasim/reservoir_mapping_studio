"""Geostatistics Lab page."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from core.active_data import (
    PANEL_MODE_INDEPENDENT,
    build_model_signature,
    prepare_active_property_data,
    signatures_match,
)
from core.column_mapper import numeric_property_candidates
from core.data_qc import prepare_interpolation_dataframe
from core.geostatistics.comparison import compare_methods
from core.geostatistics.validation import leave_one_out_cross_validation
from core.geostatistics.variogram import (
    VARIOGRAM_RANGE_CONVENTION,
    compute_experimental_variogram,
    fit_candidate_models,
    semivariance_unit,
)
from core.geometry.compartment import assign_prepared_observations_to_panels, filter_dataframe_to_selected_panels
from core.plotting.validation_plot import (
    observed_vs_predicted_figure,
    residual_histogram_figure,
    residual_map_figure,
)
from core.plotting.variogram_plot import build_variogram_figure
from pages.shared import (
    attach_include_column,
    coordinate_unit_input,
    ensure_session_state,
    get_current_filtered_data,
    panel_interpolation_mode_control,
    panel_selection_control,
    pressure_reference_date_control,
    render_filter_controls,
    unit_input,
)
from utils.constants import DUPLICATE_METHODS, INCLUDE_COLUMN
from utils.export import dataframe_to_csv_bytes, dataframe_to_excel_bytes
from utils.units import coordinate_unit_symbol, format_distance
from utils.validators import is_pressure_property


ensure_session_state()

st.title("Reservoir Mapping Studio")
st.subheader("Geostatistics Lab")

df = st.session_state.get("working_df")
if df is None:
    st.info("Load data in the Data Manager before opening the Geostatistics Lab.")
    st.stop()

mappings = st.session_state.get("column_mappings", {})
x_col = mappings.get("x")
y_col = mappings.get("y")
well_col = mappings.get("well")
if not x_col or not y_col:
    st.warning("Map X and Y coordinate columns in the Data Manager before geostatistical analysis.")
    st.stop()

with st.sidebar:
    coordinate_unit = coordinate_unit_input("geostatistics_coordinate_unit")
    unit_symbol = coordinate_unit_symbol(coordinate_unit)
    st.header("Filters")
    render_filter_controls(df, mappings, "geostatistics_lab")
    panel_layer = st.session_state.geometry_layers.get("panels")
    if panel_layer is not None and panel_layer.polygon_features:
        st.header("Panels")
        selected_panels = panel_selection_control(panel_layer, "geostatistics_selected_panels")
        panel_interpolation_mode = panel_interpolation_mode_control(True, "geostatistics_panel_mode")
    else:
        selected_panels = []
        panel_interpolation_mode = panel_interpolation_mode_control(False, "geostatistics_panel_mode")

property_options = numeric_property_candidates(df, mappings)
if not property_options:
    st.warning("No numeric property columns are available.")
    st.stop()

top_cols = st.columns([2, 1, 1, 1])
with top_cols[0]:
    current_property = st.session_state.get("current_property")
    if current_property not in property_options:
        current_property = property_options[0]
    property_col = st.selectbox("Property", property_options, index=property_options.index(current_property))
    st.session_state.current_property = property_col
with top_cols[1]:
    property_unit = unit_input("geostatistics_property_unit")
with top_cols[2]:
    duplicate_method = st.selectbox("Duplicate Handling", DUPLICATE_METHODS, index=0)
with top_cols[3]:
    inferred_type = "Pressure" if is_pressure_property(property_col) else "Generic"
    property_type = st.radio(
        "Property Type",
        ["Pressure", "Generic"],
        index=0 if inferred_type == "Pressure" else 1,
        horizontal=True,
        key=f"geostatistics_property_type_{property_col}",
    )

pressure_reference_date = None
pre_pressure_filtered = get_current_filtered_data()
if property_type == "Pressure":
    pressure_reference_date, _ = pressure_reference_date_control(
        pre_pressure_filtered,
        mappings,
        "geostatistics_pressure_reference_date_input",
    )

active_data = prepare_active_property_data(
    df,
    mappings,
    property_col,
    property_type,
    pressure_reference_date,
    st.session_state.get("filter_values", {}),
    selected_panels if panel_layer is not None else None,
)
filtered = active_data.dataframe
if selected_panels and panel_layer is not None and not mappings.get("panel"):
    filtered = filter_dataframe_to_selected_panels(filtered, x_col, y_col, panel_layer, selected_panels)
filtered = attach_include_column(filtered)
pressure_reference_date = active_data.pressure_reference_date

prepared = prepare_interpolation_dataframe(
    filtered,
    x_col,
    y_col,
    property_col,
    include_col=INCLUDE_COLUMN,
    duplicate_method=duplicate_method,
    metadata_columns=[column for column in [well_col, mappings.get("panel")] if column],
)

if prepared.empty:
    st.warning("No finite included observations are available for geostatistical analysis.")
    st.stop()

st.caption(f"{len(prepared):,} finite included observation(s). Variogram lag distances use {unit_symbol}.")


def _current_variogram_context() -> dict[str, object]:
    return {
        "property": property_col,
        "property_type": property_type,
        "pressure_reference_date": None if pressure_reference_date is None else pressure_reference_date.isoformat(),
        "selected_panels": list(active_data.selected_panels),
        "panel_interpolation_mode": panel_interpolation_mode,
        "selected_layers": list(active_data.selected_layers),
        "duplicate_method": duplicate_method,
        "range_convention": VARIOGRAM_RANGE_CONVENTION,
    }


def _variogram_context_matches(settings: dict[str, object] | None) -> bool:
    if not settings:
        return False
    current = _current_variogram_context()
    return all(settings.get(key) == value for key, value in current.items())


def default_kriging_parameters() -> dict:
    values = pd.to_numeric(prepared["Z"], errors="coerce").dropna()
    x_values = pd.to_numeric(prepared["X"], errors="coerce").dropna()
    y_values = pd.to_numeric(prepared["Y"], errors="coerce").dropna()
    default_range = max(float(np.hypot(x_values.max() - x_values.min(), y_values.max() - y_values.min())) / 3.0, 1.0)
    default_variance = max(float(values.var(ddof=1)) if len(values) > 1 else 1.0, 1e-6)
    fit = st.session_state.geostatistics.get("variogram_fit")
    if fit and _variogram_context_matches(st.session_state.geostatistics.get("variogram_settings", {})):
        return {
            "variogram_model": fit.model,
            "variogram_range_convention": VARIOGRAM_RANGE_CONVENTION,
            "range": fit.range_value,
            "variance": fit.variance,
            "nugget": fit.nugget,
            "anisotropy_enabled": st.session_state.geostatistics.get("anisotropy_enabled", False),
            "anisotropy_angle": st.session_state.geostatistics.get("anisotropy_angle", 0.0),
            "anisotropy_ratio": st.session_state.geostatistics.get("anisotropy_ratio", 1.0),
        }
    return {
        "variogram_model": "Spherical",
        "variogram_range_convention": VARIOGRAM_RANGE_CONVENTION,
        "range": default_range,
        "variance": default_variance,
        "nugget": 0.0,
        "anisotropy_enabled": st.session_state.geostatistics.get("anisotropy_enabled", False),
        "anisotropy_angle": st.session_state.geostatistics.get("anisotropy_angle", 0.0),
        "anisotropy_ratio": st.session_state.geostatistics.get("anisotropy_ratio", 1.0),
    }


def prepared_panel_labels() -> list[object]:
    panel_layer = st.session_state.geometry_layers.get("panels")
    if panel_layer is None:
        return [None] * len(prepared)
    labels, warnings = assign_prepared_observations_to_panels(
        prepared,
        panel_layer,
        selected_panels=selected_panels,
        dataset_panel_col=mappings.get("panel"),
    )
    for warning in warnings:
        st.warning(warning)
    return labels.tolist()


variogram_tab, anisotropy_tab, validation_tab, residual_tab, comparison_tab = st.tabs(
    ["Variogram", "Anisotropy", "Validation", "Residuals", "Method Comparison"]
)

with variogram_tab:
    st.markdown("#### Experimental Variogram")
    distance_extent = float(
        np.hypot(prepared["X"].max() - prepared["X"].min(), prepared["Y"].max() - prepared["Y"].min())
    )
    cols = st.columns(3)
    with cols[0]:
        n_lags = st.number_input("Number of Lags", min_value=4, max_value=30, value=12, step=1)
    with cols[1]:
        max_lag = st.number_input(
            f"Maximum Lag ({unit_symbol})",
            min_value=0.0001,
            value=max(distance_extent * 0.75, 1.0),
        )
    with cols[2]:
        auto_fit = st.checkbox("Auto Fit Candidate Models", value=True)

    if st.button("FIT VARIOGRAM", type="primary"):
        try:
            with st.spinner("Calculating experimental variogram..."):
                experimental = compute_experimental_variogram(prepared["X"], prepared["Y"], prepared["Z"], int(n_lags), float(max_lag))
                fits = fit_candidate_models(experimental) if auto_fit else []
            st.session_state.geostatistics["experimental_variogram"] = experimental
            st.session_state.geostatistics["variogram_fits"] = fits
            st.session_state.geostatistics["variogram_fit"] = fits[0] if fits else None
            st.session_state.geostatistics["variogram_settings"] = {
                **_current_variogram_context(),
                "n_lags": int(n_lags),
                "max_lag": float(max_lag),
                "auto_fit": bool(auto_fit),
            }
            st.success("Variogram calculated.")
        except ValueError as exc:
            st.error(str(exc))

    experimental = st.session_state.geostatistics.get("experimental_variogram")
    fits = st.session_state.geostatistics.get("variogram_fits", [])
    selected_fit = st.session_state.geostatistics.get("variogram_fit")
    if experimental is not None:
        if not _variogram_context_matches(st.session_state.geostatistics.get("variogram_settings", {})):
            st.warning("Stored variogram results are stale for the current active data selection. Fit the variogram again.")
        if fits:
            fit_table = pd.DataFrame(
                [
                    {
                        "Model": fit.model,
                        f"Range ({unit_symbol})": fit.range_value,
                        f"Variance / Partial Sill ({semivariance_unit(property_unit)})": fit.variance,
                        "Nugget": fit.nugget,
                        "Fit Error": fit.fit_error,
                    }
                    for fit in fits
                ]
            )
            st.dataframe(fit_table, width="stretch", hide_index=True)
            st.caption(f"Best numerical variogram fit: {fits[0].model}. This is not a geological model selection.")
        st.plotly_chart(build_variogram_figure(experimental, selected_fit, coordinate_unit, property_unit), width="stretch")

with anisotropy_tab:
    st.markdown("#### Anisotropy")
    st.caption("Angle convention: 0 degrees = +X direction, 90 degrees = +Y direction, counterclockwise positive.")
    cols = st.columns(3)
    with cols[0]:
        st.session_state.geostatistics["anisotropy_enabled"] = st.checkbox(
            "Enable Anisotropy",
            value=bool(st.session_state.geostatistics.get("anisotropy_enabled", False)),
        )
    with cols[1]:
        st.session_state.geostatistics["anisotropy_angle"] = st.number_input(
            "Major Continuity Direction (degrees)",
            min_value=0.0,
            max_value=180.0,
            value=float(st.session_state.geostatistics.get("anisotropy_angle", 0.0)),
        )
    with cols[2]:
        st.session_state.geostatistics["anisotropy_ratio"] = st.number_input(
            "Anisotropy Ratio",
            min_value=0.01,
            max_value=1.0,
            value=float(st.session_state.geostatistics.get("anisotropy_ratio", 1.0)),
            step=0.05,
            help="Minor range divided by major range.",
        )
    center_x = float(prepared["X"].mean())
    center_y = float(prepared["Y"].mean())
    length = max(float(np.hypot(prepared["X"].max() - prepared["X"].min(), prepared["Y"].max() - prepared["Y"].min())) * 0.18, 1.0)
    angle_rad = np.deg2rad(float(st.session_state.geostatistics.get("anisotropy_angle", 0.0)))
    dx = np.cos(angle_rad) * length
    dy = np.sin(angle_rad) * length
    figure = go.Figure()
    figure.add_trace(go.Scatter(x=prepared["X"], y=prepared["Y"], mode="markers", name="Observations"))
    figure.add_trace(go.Scatter(x=[center_x - dx, center_x + dx], y=[center_y - dy, center_y + dy], mode="lines", name="Major direction"))
    figure.update_layout(template="plotly_white", height=420, xaxis_title=f"X ({unit_symbol})", yaxis_title=f"Y ({unit_symbol})")
    figure.update_yaxes(scaleanchor="x", scaleratio=1)
    st.plotly_chart(figure, width="stretch")

with validation_tab:
    st.markdown("#### Leave-One-Out Cross Validation")
    method = st.selectbox("Validation Method", ["IDW", "Ordinary Kriging"], index=0)
    respect_compartments = panel_interpolation_mode == PANEL_MODE_INDEPENDENT
    st.caption(f"Panel Interpolation Mode: {panel_interpolation_mode}.")
    if method == "IDW":
        params = {"power": 2.0, "neighbors": 12, "min_neighbors": 1}
    else:
        params = default_kriging_parameters()
        st.caption(
            f"Fixed variogram model LOOCV: {params['variogram_model']}, range "
            f"{format_distance(params['range'], coordinate_unit)}, variance {params['variance']:.4g}, nugget {params['nugget']:.4g}."
        )
    current_validation_signature = build_model_signature(
        property_column=property_col,
        property_type=property_type,
        pressure_reference_date=pressure_reference_date,
        selected_panels=active_data.selected_panels,
        selected_layers=active_data.selected_layers,
        panel_interpolation_mode=panel_interpolation_mode,
        filter_values=st.session_state.get("filter_values", {}),
        active_dataframe=filtered,
        duplicate_method=duplicate_method,
        interpolation_method=method,
        interpolation_parameters=params,
        variogram={
            "model": params.get("variogram_model"),
            "range": params.get("range"),
            "variance": params.get("variance"),
            "nugget": params.get("nugget"),
            "range_convention": params.get("variogram_range_convention"),
        }
        if method == "Ordinary Kriging"
        else {},
        anisotropy={
            "enabled": params.get("anisotropy_enabled", False),
            "angle": params.get("anisotropy_angle", 0.0),
            "ratio": params.get("anisotropy_ratio", 1.0),
        },
    )
    if st.button("RUN CROSS VALIDATION", type="primary"):
        well_names = (
            prepared[well_col].tolist()
            if well_col and well_col in prepared.columns
            else [f"Obs-{index + 1}" for index in range(len(prepared))]
        )
        panels = prepared_panel_labels() if respect_compartments else None
        with st.spinner("Running fixed-parameter leave-one-out cross validation..."):
            results, metrics = leave_one_out_cross_validation(
                prepared,
                method,
                params,
                well_names=well_names,
                panels=panels,
                respect_compartments=respect_compartments,
            )
        st.session_state.geostatistics["cross_validation"] = {
            "results": results,
            "metrics": metrics,
            "method": method,
            "signature": current_validation_signature,
        }
        st.success("Cross validation complete.")

    cv = st.session_state.geostatistics.get("cross_validation")
    if cv:
        cv_is_current = signatures_match(cv.get("signature"), current_validation_signature)
        if not cv_is_current:
            st.warning("Cross-validation results are stale for the current model definition. Run cross validation again.")
        else:
            metrics = cv["metrics"]
            metric_cols = st.columns(4)
            metric_cols[0].metric("RMSE", "" if metrics["RMSE"] is None else f"{metrics['RMSE']:.4g}")
            metric_cols[1].metric("MAE", "" if metrics["MAE"] is None else f"{metrics['MAE']:.4g}")
            metric_cols[2].metric("Bias", "" if metrics["Bias"] is None else f"{metrics['Bias']:.4g}")
            metric_cols[3].metric("R2", "" if metrics["R2"] is None else f"{metrics['R2']:.4g}")
            st.dataframe(cv["results"], width="stretch", hide_index=True)
            download_cols = st.columns(2)
            with download_cols[0]:
                st.download_button("Download Cross Validation CSV", dataframe_to_csv_bytes(cv["results"]), "cross_validation.csv", "text/csv")
            with download_cols[1]:
                st.download_button(
                    "Download Cross Validation Excel",
                    dataframe_to_excel_bytes(cv["results"], "Cross Validation"),
                    "cross_validation.xlsx",
                    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                )
            st.plotly_chart(observed_vs_predicted_figure(cv["results"], property_col), width="stretch")

with residual_tab:
    cv = st.session_state.geostatistics.get("cross_validation")
    if not cv:
        st.info("Run cross validation before reviewing residual diagnostics.")
    elif not signatures_match(cv.get("signature"), current_validation_signature):
        st.info("Current residual diagnostics are stale. Run cross validation again.")
    else:
        results = cv["results"]
        st.plotly_chart(residual_histogram_figure(results, property_col), width="stretch")
        st.plotly_chart(residual_map_figure(results, coordinate_unit), width="stretch")

with comparison_tab:
    st.markdown("#### Algorithm Comparison")
    selected_methods = st.multiselect("Methods", ["IDW", "Ordinary Kriging"], default=["IDW", "Ordinary Kriging"])
    comparison_respect_compartments = panel_interpolation_mode == PANEL_MODE_INDEPENDENT
    st.caption(f"Panel Interpolation Mode: {panel_interpolation_mode}.")
    comparison_params = {
        "IDW": {"power": 2.0, "neighbors": 12, "min_neighbors": 1},
        "Ordinary Kriging": default_kriging_parameters(),
    }
    current_comparison_signature = build_model_signature(
        property_column=property_col,
        property_type=property_type,
        pressure_reference_date=pressure_reference_date,
        selected_panels=active_data.selected_panels,
        selected_layers=active_data.selected_layers,
        panel_interpolation_mode=panel_interpolation_mode,
        filter_values=st.session_state.get("filter_values", {}),
        active_dataframe=filtered,
        duplicate_method=duplicate_method,
        interpolation_method="Method Comparison",
        interpolation_parameters={
            "methods": selected_methods,
            "method_parameters": comparison_params,
        },
    )
    if st.button("COMPARE METHODS", type="primary"):
        well_names = (
            prepared[well_col].tolist()
            if well_col and well_col in prepared.columns
            else [f"Obs-{index + 1}" for index in range(len(prepared))]
        )
        panels = prepared_panel_labels() if comparison_respect_compartments else None
        with st.spinner("Running method comparison..."):
            comparison = compare_methods(
                prepared,
                comparison_params,
                selected_methods,
                well_names=well_names,
                panels=panels,
                respect_compartments=comparison_respect_compartments,
            )
        st.session_state.geostatistics["method_comparison"] = {
            "results": comparison,
            "signature": current_comparison_signature,
        }
    comparison = st.session_state.geostatistics.get("method_comparison")
    if comparison is not None:
        comparison_results = comparison.get("results") if isinstance(comparison, dict) else comparison
        comparison_is_current = not isinstance(comparison, dict) or signatures_match(
            comparison.get("signature"),
            current_comparison_signature,
        )
        valid = pd.DataFrame()
        if not comparison_is_current:
            st.warning("Method-comparison results are stale for the current model definition. Compare methods again.")
        elif isinstance(comparison_results, pd.DataFrame):
            st.dataframe(comparison_results, width="stretch", hide_index=True)
            valid = comparison_results.dropna(subset=["RMSE"])
        else:
            st.caption("No method comparison table is available.")
        if not valid.empty:
            best = valid.sort_values("RMSE").iloc[0]
            st.caption(f"Lowest validation RMSE: {best['Method']}. The engineer retains control of map selection.")
