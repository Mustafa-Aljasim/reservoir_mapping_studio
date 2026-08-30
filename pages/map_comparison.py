"""Saved map comparison and delta-map page."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from core.map_comparison import (
    calculate_delta,
    calculate_pressure_change,
    compatibility_report,
    grid_extent,
    pressure_date_order,
    symmetric_delta_range,
)
from core.plotting.map_builder import build_map_figure
from core.scenarios import create_map_scenario
from pages.shared import ensure_session_state, mark_project_dirty, project_display_name
from utils.export import (
    dataframe_to_csv_bytes,
    grid_to_dataframe,
    grid_to_excel_bytes,
    grid_to_geotiff_bytes,
    grid_to_xyz_ascii_bytes,
    metadata_to_json_bytes,
)
from utils.units import coordinate_unit_symbol


ensure_session_state()


def _scenario_labels(scenarios: list[dict[str, object]]) -> list[str]:
    return [f"{scenario.get('name')} | {scenario.get('property')} | {scenario.get('interpolation_method')}" for scenario in scenarios]


def _empty_observations(scenario: dict[str, object]) -> pd.DataFrame:
    return pd.DataFrame(columns=[scenario.get("x_col") or "X", scenario.get("y_col") or "Y", scenario.get("property") or "Value"])


def _scenario_figure(
    scenario: dict[str, object],
    z_range: tuple[float, float] | None = None,
    axis_range: tuple[float, float, float, float] | None = None,
) -> go.Figure:
    style = dict(scenario.get("style_settings", {}) or {})
    style.setdefault("color_scale", "Turbo")
    if z_range is not None:
        style["z_range_mode"] = "Manual"
        style["zmin"], style["zmax"] = z_range
    included = scenario.get("included_observations")
    excluded = scenario.get("excluded_observations")
    if not isinstance(included, pd.DataFrame):
        included = _empty_observations(scenario)
    if not isinstance(excluded, pd.DataFrame):
        excluded = _empty_observations(scenario)
    figure = build_map_figure(
        np.asarray(scenario["grid_x"], dtype=float),
        np.asarray(scenario["grid_y"], dtype=float),
        np.asarray(scenario["grid_z"], dtype=float),
        included,
        excluded,
        str(scenario.get("x_col") or "X"),
        str(scenario.get("y_col") or "Y"),
        str(scenario.get("property") or "Value"),
        well_col=scenario.get("well_col"),
        title=str(scenario.get("name") or "Saved Map"),
        unit=str(scenario.get("property_unit") or ""),
        coordinate_unit=str(scenario.get("coordinate_unit") or "meters"),
        is_pressure_map=bool(scenario.get("is_pressure_map", False)),
        map_reference_date=scenario.get("pressure_reference_date"),
        style=style,
    )
    if axis_range is not None:
        x_min, x_max, y_min, y_max = axis_range
        figure.update_xaxes(range=[x_min, x_max])
        figure.update_yaxes(range=[y_min, y_max])
    return figure


def _common_range(first: dict[str, object], second: dict[str, object]) -> tuple[float, float] | None:
    values = np.concatenate(
        [
            np.asarray(first["grid_z"], dtype=float).ravel(),
            np.asarray(second["grid_z"], dtype=float).ravel(),
        ]
    )
    valid = values[np.isfinite(values)]
    if valid.size == 0:
        return None
    return float(np.nanmin(valid)), float(np.nanmax(valid))


def _combined_extent(first: dict[str, object], second: dict[str, object]) -> tuple[float, float, float, float]:
    a = grid_extent(first)
    b = grid_extent(second)
    return min(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), max(a[3], b[3])


def _delta_figure(
    grid_x: np.ndarray,
    grid_y: np.ndarray,
    grid_z: np.ndarray,
    title: str,
    unit: str,
    coordinate_unit: str,
    z_range: tuple[float, float] | None,
) -> go.Figure:
    figure = go.Figure()
    zmin, zmax = z_range if z_range else (None, None)
    figure.add_trace(
        go.Contour(
            x=grid_x[0, :],
            y=grid_y[:, 0],
            z=grid_z,
            colorscale="RdBu",
            zmin=zmin,
            zmax=zmax,
            colorbar={"title": f"Delta ({unit})" if unit else "Delta"},
            contours={"coloring": "heatmap", "showlines": True},
            hovertemplate=(
                f"X ({coordinate_unit_symbol(coordinate_unit)}): %{{x:.3f}}<br>"
                f"Y ({coordinate_unit_symbol(coordinate_unit)}): %{{y:.3f}}<br>"
                f"Delta: %{{z:.4g}} {unit}<extra></extra>"
            ),
        )
    )
    figure.update_layout(template="plotly_white", height=620, title=title, margin={"l": 30, "r": 30, "t": 70, "b": 30})
    figure.update_xaxes(title_text=f"X ({coordinate_unit_symbol(coordinate_unit)})", zeroline=False)
    figure.update_yaxes(title_text=f"Y ({coordinate_unit_symbol(coordinate_unit)})", zeroline=False, scaleanchor="x", scaleratio=1)
    return figure


st.title("Reservoir Mapping Studio")
st.subheader("Map Comparison")
st.caption(f"Project: {project_display_name()}")

scenarios = list(st.session_state.get("map_scenarios", []))
if len(scenarios) < 2:
    st.info("Save at least two map scenarios before using Map Comparison.")
    st.stop()

labels = _scenario_labels(scenarios)
default_a = 0
stored_a = st.session_state.get("compare_map_a_id")
if stored_a:
    for index, scenario in enumerate(scenarios):
        if scenario.get("id") == stored_a:
            default_a = index
            break
default_b = 1 if len(scenarios) > 1 else 0

selector_cols = st.columns(3)
with selector_cols[0]:
    map_a_label = st.selectbox("Map A", labels, index=default_a)
with selector_cols[1]:
    map_b_label = st.selectbox("Map B", labels, index=default_b)
with selector_cols[2]:
    default_scale = "Common Range" if scenarios[labels.index(map_a_label)].get("property") == scenarios[labels.index(map_b_label)].get("property") else "Independent"
    color_scaling = st.radio("Color Scaling", ["Common Range", "Independent"], index=0 if default_scale == "Common Range" else 1)

map_a = scenarios[labels.index(map_a_label)]
map_b = scenarios[labels.index(map_b_label)]
report = compatibility_report(map_a, map_b)
for issue in report.issues:
    st.error(issue)
for warning in report.warnings:
    st.warning(warning)

axis_range = _combined_extent(map_a, map_b)
z_range = _common_range(map_a, map_b) if color_scaling == "Common Range" and map_a.get("property") == map_b.get("property") else None

map_cols = st.columns(2)
with map_cols[0]:
    st.plotly_chart(_scenario_figure(map_a, z_range, axis_range), width="stretch", config={"displaylogo": False, "scrollZoom": True})
with map_cols[1]:
    st.plotly_chart(_scenario_figure(map_b, z_range, axis_range), width="stretch", config={"displaylogo": False, "scrollZoom": True})

st.markdown("#### Delta / Pressure Change")
if not report.delta_allowed:
    st.info("Delta calculation is disabled until property, property unit, coordinate unit, and CRS are compatible.")
else:
    pressure_ready = bool(map_a.get("is_pressure_map") and map_b.get("is_pressure_map") and pressure_date_order(map_a, map_b))
    operation_options = ["Map B - Map A"]
    if pressure_ready:
        operation_options.insert(0, "Pressure Change = Later - Earlier")
    operation = st.radio("Operation", operation_options, horizontal=True)
    manual_range = st.checkbox("Manual Delta Color Range", value=False)
    if st.button("CALCULATE DELTA MAP", type="primary"):
        try:
            result = calculate_pressure_change(map_a, map_b) if operation.startswith("Pressure Change") else calculate_delta(map_a, map_b)
            metadata = dict(result.metadata)
            metadata["Created_Time"] = pd.Timestamp.utcnow().isoformat()
            st.session_state.delta_map = {
                "grid_x": result.grid_x,
                "grid_y": result.grid_y,
                "grid_z": result.grid_z,
                "metadata": metadata,
                "property": map_a.get("property"),
                "unit": map_a.get("property_unit") or "",
                "coordinate_unit": map_a.get("coordinate_unit"),
                "crs": map_a.get("crs", {}),
                "title": f"{operation}: {map_b.get('name')} vs {map_a.get('name')}",
            }
            st.success("Delta map calculated.")
        except ValueError as exc:
            st.error(str(exc))

    delta = st.session_state.get("delta_map")
    if delta:
        suggested = symmetric_delta_range(delta["grid_z"])
        if manual_range:
            range_cols = st.columns(2)
            with range_cols[0]:
                zmin = st.number_input("Delta Minimum", value=float(suggested[0] if suggested else -1.0))
            with range_cols[1]:
                zmax = st.number_input("Delta Maximum", value=float(suggested[1] if suggested else 1.0))
            delta_range = (zmin, zmax)
        else:
            delta_range = suggested
        figure = _delta_figure(
            delta["grid_x"],
            delta["grid_y"],
            delta["grid_z"],
            delta["title"],
            delta["unit"],
            delta["coordinate_unit"],
            delta_range,
        )
        st.plotly_chart(figure, width="stretch", config={"displaylogo": False, "scrollZoom": True})
        st.caption(f"Grid alignment: {delta['metadata'].get('Grid_Alignment')}")
        if delta["metadata"].get("Operation") == "Pressure Change = Later - Earlier":
            st.caption("Negative pressure change indicates pressure decline.")

        grid_df = grid_to_dataframe(delta["grid_x"], delta["grid_y"], delta["grid_z"], "Delta", include_nan=False)
        export_cols = st.columns(5)
        export_cols[0].download_button("Delta CSV", dataframe_to_csv_bytes(grid_df), "delta_map.csv", "text/csv")
        export_cols[1].download_button(
            "Delta Excel",
            grid_to_excel_bytes(grid_df, delta["metadata"]),
            "delta_map.xlsx",
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )
        export_cols[2].download_button("Delta Metadata JSON", metadata_to_json_bytes(delta["metadata"]), "delta_metadata.json", "application/json")
        export_cols[3].download_button("Delta XYZ ASCII", grid_to_xyz_ascii_bytes(grid_df), "delta_map.xyz", "text/plain")
        crs = delta.get("crs", {}) or {}
        if crs.get("mode") == "EPSG Code" and crs.get("epsg"):
            try:
                export_cols[4].download_button(
                    "Delta GeoTIFF",
                    grid_to_geotiff_bytes(delta["grid_x"], delta["grid_y"], delta["grid_z"], crs.get("epsg")),
                    "delta_map.tif",
                    "image/tiff",
                )
            except Exception as exc:
                export_cols[4].caption(f"GeoTIFF unavailable: {exc}")
        else:
            export_cols[4].caption("GeoTIFF requires EPSG CRS.")

        scenario_name = st.text_input("Delta Scenario Name", value=f"Delta - {map_b.get('name')} minus {map_a.get('name')}")
        if st.button("Save Delta As Scenario"):
            generated_delta = {
                "grid_x": delta["grid_x"],
                "grid_y": delta["grid_y"],
                "grid_z": delta["grid_z"],
                "grid_variance": None,
                "panel_grid": None,
                "included_observations": pd.DataFrame(),
                "excluded_observations": pd.DataFrame(),
                "property_col": f"Delta {delta['property']}",
                "unit": delta["unit"],
                "x_col": "X",
                "y_col": "Y",
                "well_col": None,
                "coordinate_unit": delta["coordinate_unit"],
                "is_pressure_map": False,
                "map_reference_date": None,
                "measurement_date_col": None,
                "map_reference_date_col": None,
                "method": "Delta",
                "method_parameters": {},
                "grid_parameters": {"nx": delta["grid_x"].shape[1], "ny": delta["grid_x"].shape[0], "buffer_fraction": 0.0},
                "mask_parameters": {"mode": "Inherited"},
                "mask_info": {"mask_mode": "Inherited", "valid_grid_cells": int(np.isfinite(delta["grid_z"]).sum())},
                "respect_compartments": False,
                "geometry_context": {},
                "duplicate_method": "",
                "hover_columns": [],
                "title": delta["title"],
                "export_metadata": delta["metadata"],
            }
            scenario = create_map_scenario(
                scenario_name,
                generated_delta,
                {
                    "style_settings": {"color_scale": "RdBu"},
                    "layer_settings": st.session_state.get("layer_settings", {}),
                    "filter_values": {},
                    "include_state": {},
                    "crs": delta.get("crs", {}),
                },
            )
            st.session_state.map_scenarios = scenarios + [scenario]
            st.session_state.current_scenario_id = scenario["id"]
            mark_project_dirty()
            st.success("Delta scenario saved.")
