"""Centralized coordinate and distance unit helpers."""

from __future__ import annotations

from utils.constants import COORDINATE_UNIT_OPTIONS, DEFAULT_COORDINATE_UNIT


def coordinate_unit_options() -> list[str]:
    return list(COORDINATE_UNIT_OPTIONS)


def coordinate_unit_labels() -> list[str]:
    return [COORDINATE_UNIT_OPTIONS[key]["label"] for key in coordinate_unit_options()]


def coordinate_unit_key_from_label(label: str) -> str:
    for unit_key, metadata in COORDINATE_UNIT_OPTIONS.items():
        if metadata["label"] == label:
            return unit_key
    return DEFAULT_COORDINATE_UNIT


def coordinate_unit_label(unit_key: str | None) -> str:
    unit_key = unit_key if unit_key in COORDINATE_UNIT_OPTIONS else DEFAULT_COORDINATE_UNIT
    return COORDINATE_UNIT_OPTIONS[unit_key]["label"]


def coordinate_unit_symbol(unit_key: str | None) -> str:
    unit_key = unit_key if unit_key in COORDINATE_UNIT_OPTIONS else DEFAULT_COORDINATE_UNIT
    return COORDINATE_UNIT_OPTIONS[unit_key]["symbol"]


def format_distance(value: float | int | None, unit_key: str | None, precision: int = 4) -> str:
    if value is None:
        return "Not used"
    symbol = coordinate_unit_symbol(unit_key)
    return f"{float(value):,.{precision}g} {symbol}"


def area_unit_symbol(unit_key: str | None) -> str:
    symbol = coordinate_unit_symbol(unit_key)
    return f"{symbol}^2"


def format_area(value: float | int | None, unit_key: str | None, precision: int = 4) -> str:
    if value is None:
        return "Not available"
    return f"{float(value):,.{precision}g} {area_unit_symbol(unit_key)}"


def axis_title(column_name: str, unit_key: str | None) -> str:
    return f"{column_name} ({coordinate_unit_symbol(unit_key)})"


def distance_metadata(value: float | int | None, unit_key: str | None, label: str) -> dict[str, object]:
    return {
        f"{label}_Value": None if value is None else float(value),
        f"{label}_Unit": coordinate_unit_symbol(unit_key),
    }
