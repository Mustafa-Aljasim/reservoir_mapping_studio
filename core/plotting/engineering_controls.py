"""Plotly overlays for engineering controls."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
from shapely.geometry.base import BaseGeometry

from core.engineering_controls import REGION_CONTROL_SOURCE, normalize_control_region
from core.pressure_dates import format_map_date
from core.plotting.styling import format_numeric
from utils.units import format_distance


ENGINEERING_CONTROL_TRACE = "Engineering controls"
REGION_CONTROL_TRACE = "Region controls"
CONTROL_REGION_TRACE = "Control regions"


def _control_hover(row, property_col: str, unit: str | None) -> str:
    control_id = row.get("Control_ID", "")
    source_type = row.get("Source_Type", "Manual Entry")
    value = row.get(property_col, row.get("Value"))
    lines = [
        f"Control: {control_id}",
        "Type: Engineering Control",
        f"Source: {source_type}",
        f"Value: {format_numeric(value)} {unit or ''}".strip(),
    ]
    if row.get("Reservoir_Layer"):
        lines.append(f"Layer: {row.get('Reservoir_Layer')}")
    elif row.get("Layer"):
        lines.append(f"Layer: {row.get('Layer')}")
    if row.get("Panel"):
        lines.append(f"Panel: {row.get('Panel')}")
    if row.get("Pressure_Map_Reference_Date"):
        lines.append(f"Reference Date: {format_map_date(row.get('Pressure_Map_Reference_Date'))}")
    if row.get("Comment"):
        lines.append(f"Comment: {row.get('Comment')}")
    return "<br>".join(lines)


def add_engineering_control_traces(
    figure: go.Figure,
    controls: pd.DataFrame | None,
    *,
    x_col: str,
    y_col: str,
    property_col: str,
    unit: str | None = "",
    show_manual: bool = True,
    show_region_points: bool = False,
    marker_size: int = 13,
) -> None:
    if controls is None or controls.empty:
        return
    working = controls.copy()
    if property_col not in working.columns and "Value" in working.columns:
        working[property_col] = working["Value"]
    if x_col not in working.columns and "X" in working.columns:
        working[x_col] = working["X"]
    if y_col not in working.columns and "Y" in working.columns:
        working[y_col] = working["Y"]

    source = working.get("Source_Type", pd.Series("", index=working.index)).astype(str).replace({"Soft Control Region": REGION_CONTROL_SOURCE})
    trace_groups = [
        (working[source != REGION_CONTROL_SOURCE], ENGINEERING_CONTROL_TRACE, "diamond", "#F97316", show_manual),
        (working[source == REGION_CONTROL_SOURCE], REGION_CONTROL_TRACE, "diamond-open", "#7C3AED", show_region_points),
    ]
    for frame, name, symbol, color, visible in trace_groups:
        if not visible or frame.empty:
            continue
        figure.add_trace(
            go.Scatter(
                x=pd.to_numeric(frame[x_col], errors="coerce"),
                y=pd.to_numeric(frame[y_col], errors="coerce"),
                mode="markers",
                marker={
                    "size": int(marker_size),
                    "color": color,
                    "symbol": symbol,
                    "line": {"width": 2, "color": "#111827"},
                },
                hovertext=[_control_hover(row, property_col, unit) for _, row in frame.iterrows()],
                hoverinfo="text",
                name=name,
            )
        )


def _polygon_rings(geometry: BaseGeometry):
    if geometry.geom_type == "Polygon":
        yield geometry.exterior
    elif geometry.geom_type == "MultiPolygon":
        for polygon in geometry.geoms:
            yield polygon.exterior


def add_control_region_overlays(
    figure: go.Figure,
    regions: list[dict[str, object]] | tuple[dict[str, object], ...] | None,
    *,
    coordinate_unit: str,
    visible: bool = True,
) -> None:
    if not visible:
        return
    for region in regions or []:
        normalized = normalize_control_region(region)
        geometry = normalized.get("Geometry")
        if not isinstance(geometry, BaseGeometry) or geometry.is_empty:
            continue
        hover_lines = [
            f"Region: {normalized.get('Region_Name') or normalized.get('Region_ID')}",
            f"Target: {format_numeric(normalized.get('Target_Value'))} {normalized.get('Property_Unit') or ''}".strip(),
            f"Layer: {normalized.get('Reservoir_Layer') or ''}",
            f"Generated Controls: {int(normalized.get('Generated_Control_Count') or 0)}",
            f"Spacing: {format_distance(normalized.get('Control_Point_Spacing'), coordinate_unit)}",
        ]
        if normalized.get("Region_Source"):
            hover_lines.append(f"Source: {normalized.get('Region_Source')}")
        if normalized.get("Panel"):
            hover_lines.append(f"Panel: {normalized.get('Panel')}")
        if normalized.get("Comment"):
            hover_lines.append(f"Comment: {normalized.get('Comment')}")
        for ring in _polygon_rings(geometry):
            x, y = ring.xy
            figure.add_trace(
                go.Scatter(
                    x=list(x),
                    y=list(y),
                    mode="lines",
                    line={"color": "#7C3AED", "width": 2},
                    fill="toself",
                    fillcolor="rgba(124, 58, 237, 0.12)",
                    hovertext="<br>".join(hover_lines),
                    hoverinfo="text",
                    name=CONTROL_REGION_TRACE,
                    legendgroup=CONTROL_REGION_TRACE,
                    showlegend=False,
                )
            )
