"""Plotly traces for reservoir geometry layers."""

from __future__ import annotations

import plotly.graph_objects as go

from core.geometry.models import GeometryLayer
from utils.units import format_area


def _iter_polygon_exteriors(geometry):
    if geometry.geom_type == "Polygon":
        yield geometry.exterior
    elif geometry.geom_type == "MultiPolygon":
        for polygon in geometry.geoms:
            yield polygon.exterior


def _iter_lines(geometry):
    if geometry.geom_type == "LineString":
        yield geometry
    elif geometry.geom_type == "MultiLineString":
        for line in geometry.geoms:
            yield line


def add_polygon_layer(
    figure: go.Figure,
    layer: GeometryLayer | None,
    coordinate_unit: str,
    visible: bool = True,
    show_labels: bool = False,
    line_color: str = "#111827",
    line_width: float = 2.0,
    fill_opacity: float = 0.0,
    name: str | None = None,
) -> None:
    if layer is None or not visible:
        return
    trace_name = name or layer.name
    fill = "toself" if fill_opacity > 0 else None
    fillcolor = f"rgba(17, 24, 39, {fill_opacity})"
    for feature in layer.polygon_features:
        hover = f"{feature.name}<br>Area: {format_area(feature.geometry.area, coordinate_unit)}"
        for ring in _iter_polygon_exteriors(feature.geometry):
            x, y = ring.xy
            figure.add_trace(
                go.Scatter(
                    x=list(x),
                    y=list(y),
                    mode="lines",
                    line={"color": line_color, "width": line_width},
                    fill=fill,
                    fillcolor=fillcolor,
                    hovertext=hover,
                    hoverinfo="text",
                    name=trace_name,
                    legendgroup=trace_name,
                    showlegend=False,
                )
            )
        if show_labels:
            point = feature.geometry.representative_point()
            figure.add_trace(
                go.Scatter(
                    x=[point.x],
                    y=[point.y],
                    mode="text",
                    text=[feature.name],
                    textfont={"size": 12, "color": line_color},
                    hoverinfo="skip",
                    name=f"{trace_name} labels",
                    legendgroup=trace_name,
                    showlegend=False,
                )
            )


def add_fault_layer(
    figure: go.Figure,
    layer: GeometryLayer | None,
    visible: bool = True,
    show_labels: bool = False,
    line_color: str = "#B91C1C",
    line_width: float = 2.0,
    line_dash: str = "dash",
    label_prefix: str = "Fault",
    name: str | None = None,
) -> None:
    if layer is None or not visible:
        return
    trace_name = name or layer.name
    for feature in layer.line_features:
        hover_lines = [f"{label_prefix}: {feature.name}"]
        for key, value in feature.attributes.items():
            if value not in (None, ""):
                hover_lines.append(f"{key}: {value}")
        for line in _iter_lines(feature.geometry):
            x, y = line.xy
            figure.add_trace(
                go.Scatter(
                    x=list(x),
                    y=list(y),
                    mode="lines",
                    line={"color": line_color, "width": line_width, "dash": line_dash},
                    hovertext="<br>".join(hover_lines),
                    hoverinfo="text",
                    name=trace_name,
                    legendgroup=trace_name,
                    showlegend=False,
                )
            )
        if show_labels:
            point = feature.geometry.interpolate(0.5, normalized=True)
            figure.add_trace(
                go.Scatter(
                    x=[point.x],
                    y=[point.y],
                    mode="text",
                    text=[feature.name],
                    textfont={"size": 11, "color": line_color},
                    hoverinfo="skip",
                    name=f"{trace_name} labels",
                    legendgroup=trace_name,
                    showlegend=False,
                )
            )
