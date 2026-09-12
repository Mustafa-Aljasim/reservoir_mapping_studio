"""Plotly contour-map construction."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from core.pressure_dates import format_map_date, measurement_age_days
from core.plotting.map_style import MapStyle
from core.plotting.styling import format_numeric, property_display_name
from utils.units import axis_title, coordinate_unit_symbol


OBSERVATION_TRACE_NAMES = {
    "Raw measured points",
    "Included wells",
    "Excluded observations",
    "Engineering controls",
    "Region controls",
    "Control regions",
    "Control location preview",
}

RAW_POINT_TRACE_NAMES = {"Raw measured points", "Included wells"}
EXCLUDED_POINT_TRACE_NAMES = {"Excluded observations"}
ENGINEERING_CONTROL_TRACE_NAMES = {"Engineering controls", "Region controls", "Control location preview"}
CONTROL_REGION_TRACE_NAMES = {"Control regions"}


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


def map_figure_for_static_export(
    figure: go.Figure,
    *,
    include_raw_points: bool = True,
    include_excluded_observations: bool = True,
    include_engineering_controls: bool = True,
    include_control_regions: bool = True,
) -> go.Figure:
    """Return a copy of a map figure with export-only overlay visibility applied."""

    export_figure = go.Figure(figure.to_plotly_json())
    hidden_names: set[str] = set()
    if not include_raw_points:
        hidden_names.update(RAW_POINT_TRACE_NAMES)
    if not include_excluded_observations:
        hidden_names.update(EXCLUDED_POINT_TRACE_NAMES)
    if not include_engineering_controls:
        hidden_names.update(ENGINEERING_CONTROL_TRACE_NAMES)
    if not include_control_regions:
        hidden_names.update(CONTROL_REGION_TRACE_NAMES)
    if hidden_names:
        export_figure.data = tuple(trace for trace in export_figure.data if trace.name not in hidden_names)
    return export_figure


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
                age_days = measurement_age_days(measurement_value, reference_value)
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


def build_context_map_figure(
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
) -> go.Figure:
    """Build a Cartesian context map before a property surface exists."""

    map_style = MapStyle.from_settings(style, well_col=well_col, generated_surface=False)
    hover_columns = hover_columns or []
    observation_label = property_display_name(property_col, unit)
    x_axis_title = axis_title(x_col, coordinate_unit)
    y_axis_title = axis_title(y_col, coordinate_unit)
    figure = go.Figure()

    show_raw_points = map_style.show_wells
    if show_raw_points and not included_observations.empty:
        labels = map_style.visible_well_labels(
            included_observations,
            x_col=x_col,
            y_col=y_col,
            property_col=property_col,
            well_col=well_col,
            unit=unit,
        )
        has_visible_labels = bool(labels) and any(labels)
        figure.add_trace(
            go.Scatter(
                x=pd.to_numeric(included_observations[x_col], errors="coerce"),
                y=pd.to_numeric(included_observations[y_col], errors="coerce"),
                mode="markers+text" if has_visible_labels else "markers",
                marker={
                    "size": map_style.marker_size,
                    "color": "#0F766E",
                    "opacity": map_style.marker_opacity,
                    "symbol": "circle",
                    "line": {
                        "width": 1.5 if map_style.marker_outline else 0,
                        "color": "#FFFFFF",
                    },
                },
                text=labels,
                textposition=map_style.well_label_position,
                textfont={"size": map_style.well_label_font_size, "color": "#0F172A"},
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
                name="Raw measured points",
            )
        )

    if map_style.show_excluded and not excluded_observations.empty:
        figure.add_trace(
            go.Scatter(
                x=pd.to_numeric(excluded_observations[x_col], errors="coerce"),
                y=pd.to_numeric(excluded_observations[y_col], errors="coerce"),
                mode="markers",
                marker={
                    "size": max(map_style.marker_size, 10),
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
        title={"text": title or f"{property_col} Base Map", "font": {"size": map_style.title_font_size}},
        template="plotly_white",
        height=map_style.height,
        margin={"l": 30, "r": 30, "t": 70, "b": 30},
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.01, "xanchor": "left", "x": 0},
        hovermode="closest",
        font={"size": map_style.axis_font_size},
    )
    figure.update_xaxes(
        title_text=x_axis_title,
        title_font={"size": map_style.axis_font_size},
        tickfont={"size": map_style.axis_font_size},
        zeroline=False,
    )
    figure.update_yaxes(
        title_text=y_axis_title,
        title_font={"size": map_style.axis_font_size},
        tickfont={"size": map_style.axis_font_size},
        zeroline=False,
        scaleanchor="x",
        scaleratio=1,
    )
    return figure


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

    map_style = MapStyle.from_settings(style, well_col=well_col, generated_surface=True)
    hover_columns = hover_columns or []
    observation_label = property_display_name(property_col, unit)
    colorbar_title = property_display_name(surface_label or property_col, surface_unit if surface_unit is not None else unit)
    x_axis_title = axis_title(x_col, coordinate_unit)
    y_axis_title = axis_title(y_col, coordinate_unit)

    finite_z = np.asarray(grid_z, dtype=float)
    valid_z = finite_z[np.isfinite(finite_z)]
    z_range_mode = map_style.z_range_mode
    zmin = map_style.zmin if z_range_mode == "Manual" else None
    zmax = map_style.zmax if z_range_mode == "Manual" else None
    if z_range_mode == "Manual" and zmin is not None and zmax is not None and zmin >= zmax:
        zmin = zmax = None

    contours = {
        "coloring": "heatmap",
        "showlines": map_style.show_contour_lines,
        "showlabels": map_style.show_contour_labels,
        "labelfont": {"size": map_style.contour_label_font_size},
    }
    if map_style.contour_mode == "Manual interval" and valid_z.size:
        interval = float(map_style.contour_interval or 0)
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
            colorscale=map_style.color_scale,
            reversescale=map_style.reverse_colors,
            zmin=zmin,
            zmax=zmax,
            contours=contours,
            line={"width": map_style.contour_line_width if map_style.show_contour_lines else 0.0},
            colorbar={
                "title": {"text": colorbar_title, "font": {"size": map_style.colorbar_font_size}},
                "tickfont": {"size": map_style.colorbar_font_size},
            },
            visible=map_style.show_surface,
            hovertemplate=(
                f"{x_axis_title}: %{{x:.3f}}<br>"
                f"{y_axis_title}: %{{y:.3f}}<br>"
                f"{colorbar_title}: %{{z:.4g}}<extra></extra>"
            ),
            name=surface_label or property_col,
        )
    )

    if map_style.show_wells and not included_observations.empty:
        labels = map_style.visible_well_labels(
            included_observations,
            x_col=x_col,
            y_col=y_col,
            property_col=property_col,
            well_col=well_col,
            unit=unit,
        )
        has_visible_labels = bool(labels) and any(labels)
        figure.add_trace(
            go.Scatter(
                x=pd.to_numeric(included_observations[x_col], errors="coerce"),
                y=pd.to_numeric(included_observations[y_col], errors="coerce"),
                mode="markers+text" if has_visible_labels else "markers",
                marker={
                    "size": map_style.marker_size,
                    "color": "#111827",
                    "opacity": map_style.marker_opacity,
                    "symbol": "circle",
                    "line": {
                        "width": 1.5 if map_style.marker_outline else 0,
                        "color": "#FFFFFF",
                    },
                },
                text=labels,
                textposition=map_style.well_label_position,
                textfont={"size": map_style.well_label_font_size, "color": "#111827"},
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
                name="Raw measured points",
            )
        )

    if map_style.show_excluded and not excluded_observations.empty:
        figure.add_trace(
            go.Scatter(
                x=pd.to_numeric(excluded_observations[x_col], errors="coerce"),
                y=pd.to_numeric(excluded_observations[y_col], errors="coerce"),
                mode="markers",
                marker={
                    "size": max(map_style.marker_size, 10),
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
        title={"text": title or f"{property_col} Map", "font": {"size": map_style.title_font_size}},
        template="plotly_white",
        height=map_style.height,
        margin={"l": 30, "r": 30, "t": 70, "b": 30},
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.01, "xanchor": "left", "x": 0},
        hovermode="closest",
        font={"size": map_style.axis_font_size},
    )
    figure.update_xaxes(
        title_text=x_axis_title,
        title_font={"size": map_style.axis_font_size},
        tickfont={"size": map_style.axis_font_size},
        zeroline=False,
    )
    figure.update_yaxes(
        title_text=y_axis_title,
        title_font={"size": map_style.axis_font_size},
        tickfont={"size": map_style.axis_font_size},
        zeroline=False,
        scaleanchor="x",
        scaleratio=1,
    )
    return figure
