"""Column suggestion and property-candidate logic."""

from __future__ import annotations

import re
from collections.abc import Iterable

import pandas as pd
from pandas.api.types import is_numeric_dtype

from utils.constants import COLUMN_ALIASES, INTERNAL_ROW_ID, SEMANTIC_FIELDS


def normalize_name(name: str) -> str:
    return re.sub(r"[^A-Z0-9]+", "", str(name).upper())


def suggest_column(columns: Iterable[str], semantic_key: str) -> str | None:
    aliases = COLUMN_ALIASES.get(semantic_key, ())
    normalized_aliases = {normalize_name(alias) for alias in aliases}
    columns = list(columns)

    for column in columns:
        if normalize_name(column) in normalized_aliases:
            return column

    for column in columns:
        normalized = normalize_name(column)
        for alias in normalized_aliases:
            if len(alias) >= 3 and (alias in normalized or normalized in alias):
                return column
    return None


def suggest_mappings(df: pd.DataFrame) -> dict[str, str | None]:
    return {key: suggest_column(df.columns, key) for key in SEMANTIC_FIELDS}


def is_numeric_like(series: pd.Series, threshold: float = 0.8) -> bool:
    if is_numeric_dtype(series):
        return True
    if series.empty:
        return False
    converted = pd.to_numeric(series, errors="coerce")
    non_missing = series.notna().sum()
    if non_missing == 0:
        return False
    return converted.notna().sum() / non_missing >= threshold


def numeric_columns(df: pd.DataFrame) -> list[str]:
    return [
        column
        for column in df.columns
        if column != INTERNAL_ROW_ID and is_numeric_like(df[column])
    ]


def numeric_property_candidates(
    df: pd.DataFrame,
    mappings: dict[str, str | None] | None = None,
) -> list[str]:
    mappings = mappings or {}
    coordinate_columns = {mappings.get("x"), mappings.get("y"), INTERNAL_ROW_ID}
    return [column for column in numeric_columns(df) if column not in coordinate_columns]


def categorical_filter_candidates(
    df: pd.DataFrame,
    mappings: dict[str, str | None] | None = None,
    max_unique: int = 60,
) -> list[str]:
    mappings = mappings or {}
    excluded = {mappings.get("x"), mappings.get("y"), INTERNAL_ROW_ID}
    candidates: list[str] = []
    for column in df.columns:
        if column in excluded:
            continue
        unique_count = df[column].dropna().nunique()
        if 1 <= unique_count <= max_unique:
            candidates.append(column)
    return candidates

