"""Plotly cross-validation diagnostics."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from utils.units import axis_title


def observed_vs_predicted_figure(results: pd.DataFrame, property_label: str) -> go.Figure:
    valid = results.dropna(subset=["Observed", "Predicted"])
    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=valid["Observed"],
            y=valid["Predicted"],
            mode="markers",
            text=valid.get("Well"),
            hovertemplate="Well: %{text}<br>Observed: %{x:.4g}<br>Predicted: %{y:.4g}<extra></extra>",
            name="Predictions",
        )
    )
    if not valid.empty:
        low = float(np.nanmin([valid["Observed"].min(), valid["Predicted"].min()]))
        high = float(np.nanmax([valid["Observed"].max(), valid["Predicted"].max()]))
        figure.add_trace(go.Scatter(x=[low, high], y=[low, high], mode="lines", name="1:1"))
    figure.update_layout(
        template="plotly_white",
        height=420,
        xaxis_title=f"Observed {property_label}",
        yaxis_title=f"Predicted {property_label}",
        margin={"l": 30, "r": 20, "t": 35, "b": 35},
    )
    figure.update_yaxes(scaleanchor="x", scaleratio=1)
    return figure


def residual_histogram_figure(results: pd.DataFrame, property_label: str) -> go.Figure:
    valid = results.dropna(subset=["Residual"])
    figure = go.Figure()
    figure.add_trace(go.Histogram(x=valid["Residual"], nbinsx=20, name="Residuals"))
    figure.update_layout(
        template="plotly_white",
        height=340,
        xaxis_title=f"Residual {property_label}",
        yaxis_title="Count",
        margin={"l": 30, "r": 20, "t": 35, "b": 35},
    )
    return figure


def residual_map_figure(results: pd.DataFrame, coordinate_unit: str) -> go.Figure:
    valid = results.dropna(subset=["Residual"])
    size = np.maximum(valid["Absolute_Error"].astype(float), 1.0)
    if len(size) and size.max() > 0:
        size = 8 + 22 * size / size.max()
    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=valid["X"],
            y=valid["Y"],
            mode="markers",
            marker={
                "size": size,
                "color": valid["Residual"],
                "colorscale": "RdBu",
                "cmid": 0,
                "line": {"width": 1, "color": "#111827"},
                "colorbar": {"title": "Residual"},
            },
            text=valid.get("Well"),
            customdata=np.column_stack([valid["Observed"], valid["Predicted"], valid["Residual"], valid["Absolute_Error"]]),
            hovertemplate=(
                "Well: %{text}<br>Observed: %{customdata[0]:.4g}<br>Predicted: %{customdata[1]:.4g}"
                "<br>Residual: %{customdata[2]:.4g}<br>Abs Error: %{customdata[3]:.4g}<extra></extra>"
            ),
            name="Residuals",
        )
    )
    figure.update_layout(
        template="plotly_white",
        height=520,
        xaxis_title=axis_title("X", coordinate_unit),
        yaxis_title=axis_title("Y", coordinate_unit),
        margin={"l": 30, "r": 20, "t": 35, "b": 35},
    )
    figure.update_yaxes(scaleanchor="x", scaleratio=1)
    return figure

