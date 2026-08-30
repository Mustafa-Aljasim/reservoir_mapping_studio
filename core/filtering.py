"""Reusable metadata filtering logic."""

from __future__ import annotations

import pandas as pd

from utils.constants import INTERNAL_ROW_ID, METADATA_FILTER_ORDER, SEMANTIC_FIELDS


def mapped_filter_columns(mappings: dict[str, str | None]) -> list[tuple[str, str]]:
    columns: list[tuple[str, str]] = []
    seen: set[str] = set()
    for semantic_key in METADATA_FILTER_ORDER:
        column = mappings.get(semantic_key)
        if column and column not in seen:
            columns.append((SEMANTIC_FIELDS[semantic_key], column))
            seen.add(column)
    return columns


def apply_categorical_filters(
    df: pd.DataFrame,
    filter_values: dict[str, list[object]],
    skip_column: str | None = None,
) -> pd.DataFrame:
    output = df
    for column, selected in filter_values.items():
        if column == skip_column or column not in output.columns or not selected:
            continue
        output = output[output[column].isin(selected)]
    return output


def apply_filters(
    df: pd.DataFrame,
    mappings: dict[str, str | None],
    filter_values: dict[str, list[object]] | None = None,
) -> pd.DataFrame:
    output = apply_categorical_filters(df, filter_values or {})
    return output.copy()


def options_for_column(
    df: pd.DataFrame,
    column: str,
    filter_values: dict[str, list[object]],
    prior_columns: list[str],
) -> list[object]:
    active_filters = {
        filter_column: selected
        for filter_column, selected in filter_values.items()
        if filter_column in prior_columns
    }
    filtered = apply_categorical_filters(df, active_filters)
    if column not in filtered.columns:
        return []
    values = filtered[column].dropna().drop_duplicates().tolist()
    return sorted(values, key=lambda value: str(value))


def build_filter_column_list(
    mappings: dict[str, str | None],
    additional_columns: list[str] | None = None,
) -> list[tuple[str, str]]:
    columns = mapped_filter_columns(mappings)
    seen = {column for _, column in columns}
    for column in additional_columns or []:
        if column and column not in seen and column != INTERNAL_ROW_ID:
            columns.append((column, column))
            seen.add(column)
    return columns
