"""Plotly variogram diagnostics."""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go

from core.geostatistics.variogram import ExperimentalVariogram, VariogramFit, evaluate_variogram_model, semivariance_unit
from utils.units import coordinate_unit_symbol


def build_variogram_figure(
    experimental: ExperimentalVariogram,
    fit: VariogramFit | None,
    coordinate_unit: str,
    property_unit: str | None = None,
) -> go.Figure:
    unit_symbol = coordinate_unit_symbol(coordinate_unit)
    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=experimental.lag_distance,
            y=experimental.semivariance,
            mode="markers",
            marker={"size": np.maximum(experimental.pair_count, 1), "sizemode": "area", "sizeref": 2},
            text=[f"Pairs: {count}" for count in experimental.pair_count],
            name="Experimental",
        )
    )
    if fit is not None:
        h = np.linspace(0, experimental.max_lag, 200)
        gamma = evaluate_variogram_model(fit.model, h, fit.range_value, fit.variance, fit.nugget)
        figure.add_trace(go.Scatter(x=h, y=gamma, mode="lines", name=f"{fit.model} fit"))
    figure.update_layout(
        template="plotly_white",
        height=420,
        xaxis_title=f"Lag Distance ({unit_symbol})",
        yaxis_title=f"Semivariance ({semivariance_unit(property_unit)})",
        margin={"l": 30, "r": 20, "t": 35, "b": 35},
    )
    return figure

