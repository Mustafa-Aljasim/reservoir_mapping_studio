"""Explicit display-unit conversion helpers."""

from __future__ import annotations

import numpy as np


PRESSURE_UNITS = ("psi", "bar", "kPa")
LENGTH_UNITS = ("m", "ft")
FRACTION_UNITS = ("fraction", "%")


def available_display_units(current_unit: str | None) -> list[str]:
    unit = (current_unit or "").strip()
    groups = [PRESSURE_UNITS, LENGTH_UNITS, FRACTION_UNITS]
    for group in groups:
        if unit in group:
            return list(group)
    return [unit] if unit else [""]


def convert_values(values, from_unit: str | None, to_unit: str | None):
    """Convert a numeric array for supported safe unit pairs."""

    source = (from_unit or "").strip()
    target = (to_unit or "").strip()
    array = np.asarray(values, dtype=float)
    if source == target or not target:
        return array

    pressure_to_kpa = {"psi": 6.894757293168, "bar": 100.0, "kPa": 1.0}
    if source in pressure_to_kpa and target in pressure_to_kpa:
        return array * pressure_to_kpa[source] / pressure_to_kpa[target]

    length_to_m = {"m": 1.0, "ft": 0.3048}
    if source in length_to_m and target in length_to_m:
        return array * length_to_m[source] / length_to_m[target]

    if source == "fraction" and target == "%":
        return array * 100.0
    if source == "%" and target == "fraction":
        return array / 100.0

    raise ValueError(f"Unsupported unit conversion: {source or 'unitless'} to {target or 'unitless'}.")


def conversion_label(from_unit: str | None, to_unit: str | None) -> str:
    source = (from_unit or "").strip() or "unitless"
    target = (to_unit or "").strip() or "unitless"
    return source if source == target else f"{source} to {target}"
