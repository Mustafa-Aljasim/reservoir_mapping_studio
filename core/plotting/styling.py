"""Formatting helpers for map labels and values."""

from __future__ import annotations

import math


def property_display_name(property_name: str, unit: str | None = None) -> str:
    unit = (unit or "").strip()
    return f"{property_name} ({unit})" if unit else property_name


def format_numeric(value, digits: int = 4) -> str:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return ""
    if not math.isfinite(numeric):
        return ""
    return f"{numeric:.{digits}g}"

