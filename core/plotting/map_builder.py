"""Plotly contour-map construction."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from core.pressure_dates import format_map_date, measurement_age_days
from core.plotting.styling import format_numeric, property_display_name
from utils.units import axis_title, coordinate_unit_symbol


OBSERVATION_TRACE_NAMES = {
    "Included wells",
    "Excluded observations",
    "Engineering controls",
    "Region controls",
}


def move_observation_traces_to_top(figure: go.Figure) -> None:
    """Move well-marker traces after geometry overlays without changing trace contents."""

    base_traces = []
    observation_traces = []
    for trace in figure.data:
        if trace.name in OBSERVATION_TRACE_NAMES:
            observation_traces.append(trace)
        else:
            base_traces.append(trace)
    figure.data = tuple(base_traces + observation_traces)


def _build_hover_text(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    property_col: str,
    property_label: str,
    well_col: str | None,
    hover_columns: list[tuple[str, str]],
    coordinate_unit: str | None,
    is_pressure_map: bool = False,
    map_reference_date=None,
    measurement_date_col: str | None = None,
    map_reference_date_col: str | None = None,
) -> list[str]:
    hover_text: list[str] = []
    distance_unit = coordinate_unit_symbol(coordinate_unit)
    for _, row in df.iterrows():
        lines: list[str] = []
        if well_col and well_col in df.columns and pd.notna(row.get(well_col)):
            lines.append(f"Well: {row[well_col]}")
        lines.append(f"{property_label}: {format_numeric(row.get(property_col))}")
        if is_pressure_map:
            reference_value = map_reference_date
            if reference_value is None and map_reference_date_col and map_reference_date_col in df.columns:
                reference_value = row.get(map_reference_date_col)
            formatted_reference = format_map_date(reference_value)
            if formatted_reference:
                lines.append(f"Map Reference Date: {formatted_reference}")
            if measurement_date_col and measurement_date_col in df.columns and pd.notna(row.get(measurement_date_col)):
                measurement_value = row.get(measurement_date_col)
                lines.append(f"Original Measurement Date: {format_map_date(measurement_value)}")
                age_days = measurement_age_days(measurement_value, map_reference_date)
                if age_days is not None:
                    lines.append(f"Original Measurement Age: {age_days:,} days")
        lines.append(f"{x_col}: {format_numeric(row.get(x_col))} {distance_unit}")
        lines.append(f"{y_col}: {format_numeric(row.get(y_col))} {distance_unit}")
        excluded_hover_columns = {
            x_col,
            y_col,
            property_col,
            well_col,
            measurement_date_col,
            map_reference_date_col,
        }
        for label, column in hover_columns:
            if column in df.columns and column not in excluded_hover_columns:
                value = row.get(column)
                if pd.notna(value):
                    lines.append(f"{label}: {value}")
        hover_text.append("<br>".join(lines))
    return hover_text


def _label_text(
    df: pd.DataFrame,
    property_col: str,
    well_col: str | None,
    mode: str,
) -> list[str] | None:
    if mode == "None":
        return None
    labels: list[str] = []
    for _, row in df.iterrows():
        well = str(row.get(well_col, "")) if well_col and pd.notna(row.get(well_col)) else ""
        value = format_numeric(row.get(property_col))
        if mode == "Well Name":
            labels.append(well)
        elif mode == "Property Value":
            labels.append(value)
        else:
            labels.append(f"{well}<br>{value}" if well else value)
    return labels


def build_map_figure(
    grid_x: np.ndarray,
    grid_y: np.ndarray,
    grid_z: np.ndarray,
    included_observations: pd.DataFrame,
    excluded_observations: pd.DataFrame,
    x_col: str,
    y_col: str,
    property_col: str,
    well_col: str | None = None,
    hover_columns: list[tuple[str, str]] | None = None,
    title: str | None = None,
    unit: str | None = None,
    coordinate_unit: str | None = None,
    is_pressure_map: bool = False,
    map_reference_date=None,
    measurement_date_col: str | None = None,
    map_reference_date_col: str | None = None,
    style: dict | None = None,
    surface_label: str | None = None,
    surface_unit: str | None = None,
) -> go.Figure:
    """Build a Cartesian Plotly filled-contour map with well overlays."""

    style = style or {}
    hover_columns = hover_columns or []
    observation_label = property_display_name(property_col, unit)
    colorbar_title = property_display_name(surface_label or property_col, surface_unit if surface_unit is not None else unit)
    x_axis_title = axis_title(x_col, coordinate_unit)
    y_axis_title = axis_title(y_col, coordinate_unit)

    finite_z = np.asarray(grid_z, dtype=float)
    valid_z = finite_z[np.isfinite(finite_z)]
    z_range_mode = style.get("z_range_mode", "Auto")
    zmin = style.get("zmin") if z_range_mode == "Manual" else None
    zmax = style.get("zmax") if z_range_mode == "Manual" else None
    if z_range_mode == "Manual" and zmin is not None and zmax is not None and zmin >= zmax:
        zmin = zmax = None

    contours = {
        "coloring": "heatmap",
        "showlines": bool(style.get("show_contour_lines", True)),
        "showlabels": bool(style.get("show_contour_labels", False)),
    }
    if style.get("contour_mode", "Auto interval") == "Manual interval" and valid_z.size:
        interval = float(style.get("contour_interval") or 0)
        if interval > 0:
            start = float(zmin) if zmin is not None else float(np.nanmin(valid_z))
            end = float(zmax) if zmax is not None else float(np.nanmax(valid_z))
            contours.update({"start": start, "end": end, "size": interval})

    figure = go.Figure()
    figure.add_trace(
        go.Contour(
            x=grid_x[0, :],
            y=grid_y[:, 0],
            z=grid_z,
            colorscale=style.get("color_scale", "Turbo"),
            reversescale=bool(style.get("reverse_colors", False)),
            zmin=zmin,
            zmax=zmax,
            contours=contours,
            line={"width": float(style.get("contour_line_width", 0.75))},
            colorbar={"title": colorbar_title},
            visible=bool(style.get("show_surface", True)),
            hovertemplate=(
                f"{x_axis_title}: %{{x:.3f}}<br>"
                f"{y_axis_title}: %{{y:.3f}}<br>"
                f"{colorbar_title}: %{{z:.4g}}<extra></extra>"
            ),
            name=surface_label or property_col,
        )
    )

    if bool(style.get("show_wells", True)) and not included_observations.empty:
        label_mode = str(style.get("well_label_mode", "None"))
        labels = _label_text(included_observations, property_col, well_col, label_mode)
        figure.add_trace(
            go.Scatter(
                x=pd.to_numeric(included_observations[x_col], errors="coerce"),
                y=pd.to_numeric(included_observations[y_col], errors="coerce"),
                mode="markers+text" if labels else "markers",
                marker={
                    "size": int(style.get("marker_size", 9)),
                    "color": "#111827",
                    "opacity": float(style.get("marker_opacity", 0.9)),
                    "symbol": "circle",
                    "line": {
                        "width": 1.5 if bool(style.get("marker_outline", True)) else 0,
                        "color": "#FFFFFF",
                    },
                },
                text=labels,
                textposition="top center",
                textfont={"size": int(style.get("label_text_size", 11)), "color": "#111827"},
                hovertext=_build_hover_text(
                    included_observations,
                    x_col,
                    y_col,
                    property_col,
                    observation_label,
                    well_col,
                    hover_columns,
                    coordinate_unit,
                    is_pressure_map=is_pressure_map,
                    map_reference_date=map_reference_date,
                    measurement_date_col=measurement_date_col,
                    map_reference_date_col=map_reference_date_col,
                ),
                hoverinfo="text",
                name="Included wells",
            )
        )

    if bool(style.get("show_excluded", True)) and not excluded_observations.empty:
        figure.add_trace(
            go.Scatter(
                x=pd.to_numeric(excluded_observations[x_col], errors="coerce"),
                y=pd.to_numeric(excluded_observations[y_col], errors="coerce"),
                mode="markers",
                marker={
                    "size": max(int(style.get("marker_size", 9)), 10),
                    "color": "#6B7280",
                    "opacity": 0.8,
                    "symbol": "x",
                    "line": {"width": 2, "color": "#6B7280"},
                },
                hovertext=_build_hover_text(
                    excluded_observations,
                    x_col,
                    y_col,
                    property_col,
                    observation_label,
                    well_col,
                    hover_columns,
                    coordinate_unit,
                    is_pressure_map=is_pressure_map,
                    map_reference_date=map_reference_date,
                    measurement_date_col=measurement_date_col,
                    map_reference_date_col=map_reference_date_col,
                ),
                hoverinfo="text",
                name="Excluded observations",
            )
        )

    figure.update_layout(
        title=title or f"{property_col} Map",
        template="plotly_white",
        height=int(style.get("height", 720)),
        margin={"l": 30, "r": 30, "t": 70, "b": 30},
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.01, "xanchor": "left", "x": 0},
        hovermode="closest",
    )
    figure.update_xaxes(title_text=x_axis_title, zeroline=False)
    figure.update_yaxes(title_text=y_axis_title, zeroline=False, scaleanchor="x", scaleratio=1)
    return figure
