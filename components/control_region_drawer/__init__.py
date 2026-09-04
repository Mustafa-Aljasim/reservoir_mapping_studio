"""Cartesian control-region drawing component wrapper."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import streamlit.components.v1 as components


_FRONTEND_DIR = Path(__file__).resolve().parent / "frontend"
_component = components.declare_component(
    "control_region_drawer",
    path=str(_FRONTEND_DIR),
)


def control_region_drawer(
    *,
    points: list[dict[str, Any]] | None = None,
    excluded_points: list[dict[str, Any]] | None = None,
    controls: list[dict[str, Any]] | None = None,
    regions: list[dict[str, Any]] | None = None,
    polygons: list[dict[str, Any]] | None = None,
    lines: list[dict[str, Any]] | None = None,
    bounds: dict[str, float] | None = None,
    vertices: list[dict[str, float]] | None = None,
    coordinate_unit: str = "",
    x_label: str = "X",
    y_label: str = "Y",
    height: int = 620,
    key: str | None = None,
) -> dict[str, Any] | None:
    """Render the drawer and return the latest drawing event."""

    return _component(
        points=points or [],
        excluded_points=excluded_points or [],
        controls=controls or [],
        regions=regions or [],
        polygons=polygons or [],
        lines=lines or [],
        bounds=bounds or {},
        vertices=vertices or [],
        coordinate_unit=coordinate_unit,
        x_label=x_label,
        y_label=y_label,
        height=int(height),
        default=None,
        key=key,
    )
