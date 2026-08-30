"""Geostatistics Lab page."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from shapely.geometry import Point

from core.column_mapper import numeric_property_candidates
from core.data_qc import prepare_interpolation_dataframe
from core.geostatistics.comparison import compare_methods
from core.geostatistics.validation import leave_one_out_cross_validation
from core.geostatistics.variogram import (
    compute_experimental_variogram,
    fit_candidate_models,
    semivariance_unit,
)
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
    render_filter_controls,
    unit_input,
)
from utils.constants import DUPLICATE_METHODS, INCLUDE_COLUMN
from utils.export import dataframe_to_csv_bytes, dataframe_to_excel_bytes
from utils.units import coordinate_unit_symbol, format_distance


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

filtered = attach_include_column(get_current_filtered_data())
property_options = numeric_property_candidates(df, mappings)
if not property_options:
    st.warning("No numeric property columns are available.")
    st.stop()

top_cols = st.columns([2, 1, 1])
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

prepared = prepare_interpolation_dataframe(
    filtered,
    x_col,
    y_col,
    property_col,
    include_col=INCLUDE_COLUMN,
    duplicate_method=duplicate_method,
)

if prepared.empty:
    st.warning("No finite included observations are available for geostatistical analysis.")
    st.stop()

st.caption(f"{len(prepared):,} finite included observation(s). Variogram lag distances use {unit_symbol}.")


def default_kriging_parameters() -> dict:
    values = pd.to_numeric(prepared["Z"], errors="coerce").dropna()
    x_values = pd.to_numeric(prepared["X"], errors="coerce").dropna()
    y_values = pd.to_numeric(prepared["Y"], errors="coerce").dropna()
    default_range = max(float(np.hypot(x_values.max() - x_values.min(), y_values.max() - y_values.min())) / 3.0, 1.0)
    default_variance = max(float(values.var(ddof=1)) if len(values) > 1 else 1.0, 1e-6)
    fit = st.session_state.geostatistics.get("variogram_fit")
    if fit:
        return {
            "variogram_model": fit.model,
            "range": fit.range_value,
            "variance": fit.variance,
            "nugget": fit.nugget,
            "anisotropy_enabled": st.session_state.geostatistics.get("anisotropy_enabled", False),
            "anisotropy_angle": st.session_state.geostatistics.get("anisotropy_angle", 0.0),
            "anisotropy_ratio": st.session_state.geostatistics.get("anisotropy_ratio", 1.0),
        }
    return {
        "variogram_model": "Spherical",
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
    labels: list[object] = []
    for _, row in prepared.iterrows():
        point = Point(float(row["X"]), float(row["Y"]))
        matches = [feature.name for feature in panel_layer.polygon_features if feature.geometry.covers(point)]
        labels.append(matches[0] if len(matches) == 1 else None)
    return labels


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
            st.success("Variogram calculated.")
        except ValueError as exc:
            st.error(str(exc))

    experimental = st.session_state.geostatistics.get("experimental_variogram")
    fits = st.session_state.geostatistics.get("variogram_fits", [])
    selected_fit = st.session_state.geostatistics.get("variogram_fit")
    if experimental is not None:
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
    respect_compartments = st.checkbox(
        "Respect Panel / Compartment Boundaries",
        value=False,
        disabled=st.session_state.geometry_layers.get("panels") is None,
    )
    if method == "IDW":
        params = {"power": 2.0, "neighbors": 12, "min_neighbors": 1}
    else:
        params = default_kriging_parameters()
        st.caption(
            f"Fixed variogram model LOOCV: {params['variogram_model']}, range "
            f"{format_distance(params['range'], coordinate_unit)}, variance {params['variance']:.4g}, nugget {params['nugget']:.4g}."
        )
    if st.button("RUN CROSS VALIDATION", type="primary"):
        well_names = (
            filtered[well_col].head(len(prepared)).tolist()
            if well_col and well_col in filtered.columns
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
        st.session_state.geostatistics["cross_validation"] = {"results": results, "metrics": metrics, "method": method}
        st.success("Cross validation complete.")

    cv = st.session_state.geostatistics.get("cross_validation")
    if cv:
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
    else:
        results = cv["results"]
        st.plotly_chart(residual_histogram_figure(results, property_col), width="stretch")
        st.plotly_chart(residual_map_figure(results, coordinate_unit), width="stretch")

with comparison_tab:
    st.markdown("#### Algorithm Comparison")
    selected_methods = st.multiselect("Methods", ["IDW", "Ordinary Kriging"], default=["IDW", "Ordinary Kriging"])
    comparison_respect_compartments = st.checkbox(
        "Respect Panel / Compartment Boundaries",
        value=False,
        disabled=st.session_state.geometry_layers.get("panels") is None,
        key="comparison_respect_compartments",
    )
    if st.button("COMPARE METHODS", type="primary"):
        params = {
            "IDW": {"power": 2.0, "neighbors": 12, "min_neighbors": 1},
            "Ordinary Kriging": default_kriging_parameters(),
        }
        well_names = (
            filtered[well_col].head(len(prepared)).tolist()
            if well_col and well_col in filtered.columns
            else [f"Obs-{index + 1}" for index in range(len(prepared))]
        )
        panels = prepared_panel_labels() if comparison_respect_compartments else None
        with st.spinner("Running method comparison..."):
            comparison = compare_methods(
                prepared,
                params,
                selected_methods,
                well_names=well_names,
                panels=panels,
                respect_compartments=comparison_respect_compartments,
            )
        st.session_state.geostatistics["method_comparison"] = comparison
    comparison = st.session_state.geostatistics.get("method_comparison")
    if comparison is not None:
        st.dataframe(comparison, width="stretch", hide_index=True)
        valid = comparison.dropna(subset=["RMSE"])
        if not valid.empty:
            best = valid.sort_values("RMSE").iloc[0]
            st.caption(f"Lowest validation RMSE: {best['Method']}. The engineer retains control of map selection.")
