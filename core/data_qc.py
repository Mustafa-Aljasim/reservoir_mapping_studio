"""Data quality, duplicate-coordinate, and observation-preparation logic."""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Iterable
from typing import Literal

import numpy as np
import pandas as pd

from utils.constants import INCLUDE_COLUMN, OUTLIER_COLUMN
from utils.validators import finite_series

DuplicateMethod = Literal["Average", "Median", "Keep first", "Keep last"]


@dataclass(frozen=True)
class QCSummary:
    total_rows: int
    valid_xy_rows: int
    missing_x: int
    missing_y: int
    missing_property: int | None
    duplicate_xy_rows: int
    unique_wells: int | None
    property_min: float | None
    property_max: float | None
    property_mean: float | None
    property_median: float | None
    property_std: float | None
    outlier_count: int | None


def duplicate_coordinate_mask(df: pd.DataFrame, x_col: str, y_col: str) -> pd.Series:
    x = finite_series(df[x_col])
    y = finite_series(df[y_col])
    coordinates = pd.DataFrame({"x": x, "y": y}, index=df.index)
    finite = coordinates.notna().all(axis=1)
    duplicated = coordinates[finite].duplicated(subset=["x", "y"], keep=False)
    output = pd.Series(False, index=df.index)
    output.loc[duplicated.index] = duplicated
    return output


def duplicate_coordinate_rows(df: pd.DataFrame, x_col: str, y_col: str) -> pd.DataFrame:
    mask = duplicate_coordinate_mask(df, x_col, y_col)
    return df.loc[mask].copy()


def flag_outliers(
    df: pd.DataFrame,
    property_col: str,
    threshold: float = 3.5,
) -> pd.DataFrame:
    """Flag potential outliers using MAD, with IQR fallback for zero MAD."""

    output = df.copy()
    output[OUTLIER_COLUMN] = False
    values = finite_series(output[property_col])
    valid = values.dropna()
    if len(valid) < 4 or valid.nunique() < 3:
        return output

    median = float(valid.median())
    mad = float(np.median(np.abs(valid - median)))
    if mad > 0:
        modified_z = 0.6745 * (valid - median) / mad
        flags = modified_z.abs() > threshold
    else:
        q1 = float(valid.quantile(0.25))
        q3 = float(valid.quantile(0.75))
        iqr = q3 - q1
        if iqr == 0:
            return output
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        flags = (valid < lower) | (valid > upper)

    output.loc[flags.index, OUTLIER_COLUMN] = flags.astype(bool)
    return output


def build_qc_summary(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    property_col: str | None = None,
    well_col: str | None = None,
) -> QCSummary:
    x = finite_series(df[x_col])
    y = finite_series(df[y_col])
    valid_xy = x.notna() & y.notna()
    duplicate_rows = int(duplicate_coordinate_mask(df, x_col, y_col).sum())

    property_values = None
    outlier_count = None
    missing_property = None
    property_min = property_max = property_mean = property_median = property_std = None
    if property_col and property_col in df.columns:
        flagged = flag_outliers(df, property_col)
        property_values = finite_series(df[property_col])
        missing_property = int(property_values.isna().sum())
        valid_property = property_values.dropna()
        outlier_count = int(flagged[OUTLIER_COLUMN].sum())
        if not valid_property.empty:
            property_min = float(valid_property.min())
            property_max = float(valid_property.max())
            property_mean = float(valid_property.mean())
            property_median = float(valid_property.median())
            property_std = float(valid_property.std(ddof=1)) if len(valid_property) > 1 else 0.0

    return QCSummary(
        total_rows=int(len(df)),
        valid_xy_rows=int(valid_xy.sum()),
        missing_x=int(x.isna().sum()),
        missing_y=int(y.isna().sum()),
        missing_property=missing_property,
        duplicate_xy_rows=duplicate_rows,
        unique_wells=int(df[well_col].dropna().nunique()) if well_col and well_col in df.columns else None,
        property_min=property_min,
        property_max=property_max,
        property_mean=property_mean,
        property_median=property_median,
        property_std=property_std,
        outlier_count=outlier_count,
    )


def descriptive_statistics(values) -> dict[str, float | int | None]:
    series = pd.Series(values, dtype="float64").replace([np.inf, -np.inf], np.nan).dropna()
    if series.empty:
        return {
            "observations": 0,
            "mean": None,
            "median": None,
            "minimum": None,
            "maximum": None,
            "std": None,
            "p10": None,
            "p50": None,
            "p90": None,
        }
    return {
        "observations": int(series.count()),
        "mean": float(series.mean()),
        "median": float(series.median()),
        "minimum": float(series.min()),
        "maximum": float(series.max()),
        "std": float(series.std(ddof=1)) if len(series) > 1 else 0.0,
        "p10": float(series.quantile(0.10)),
        "p50": float(series.quantile(0.50)),
        "p90": float(series.quantile(0.90)),
    }


def _collapse_metadata(values: Iterable[object]) -> object:
    unique = []
    for value in values:
        if pd.isna(value):
            continue
        text = str(value)
        if text not in unique:
            unique.append(text)
    if not unique:
        return None
    if len(unique) == 1:
        return unique[0]
    return "|".join(unique)


def _source_row_ids(values: Iterable[object]) -> tuple[int, ...]:
    ids: list[int] = []
    for value in values:
        if pd.isna(value):
            continue
        try:
            ids.append(int(value))
        except (TypeError, ValueError):
            continue
    return tuple(ids)


def prepare_interpolation_dataframe(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    property_col: str,
    include_col: str = INCLUDE_COLUMN,
    duplicate_method: DuplicateMethod = "Average",
    metadata_columns: list[str] | None = None,
    row_id_col: str | None = None,
) -> pd.DataFrame:
    """Return finite, included observations aggregated by duplicate XY as needed.

    Metadata is optional and is collapsed only after filtering/date selection has
    already occurred. This prevents pressure snapshots from different map
    reference dates from being averaged together by duplicate XY handling.
    """

    working = df.copy()
    if include_col in working.columns:
        working = working[working[include_col].astype(bool)]

    prepared = pd.DataFrame(
        {
            "X": finite_series(working[x_col]),
            "Y": finite_series(working[y_col]),
            "Z": finite_series(working[property_col]),
        },
        index=working.index,
    )
    metadata_columns = [
        column
        for column in (metadata_columns or [])
        if column in working.columns and column not in {"X", "Y", "Z"}
    ]
    for column in metadata_columns:
        prepared[column] = working[column]
    if row_id_col and row_id_col in working.columns:
        prepared[row_id_col] = working[row_id_col]
    prepared = prepared.replace([np.inf, -np.inf], np.nan).dropna(subset=["X", "Y", "Z"])
    if prepared.empty:
        return prepared.reset_index(drop=True)

    aggregations = {"Source_Count": ("Z", "size")}
    for column in metadata_columns:
        aggregations[column] = (column, _collapse_metadata)
    if row_id_col and row_id_col in prepared.columns:
        aggregations["Source_Row_IDs"] = (row_id_col, _source_row_ids)

    if duplicate_method == "Average":
        grouped = prepared.groupby(["X", "Y"], as_index=False).agg(Z=("Z", "mean"), **aggregations)
    elif duplicate_method == "Median":
        grouped = prepared.groupby(["X", "Y"], as_index=False).agg(Z=("Z", "median"), **aggregations)
    elif duplicate_method == "Keep first":
        representatives = prepared.drop_duplicates(subset=["X", "Y"], keep="first")[["X", "Y", "Z"]].copy()
        metadata = prepared.groupby(["X", "Y"], as_index=False).agg(**aggregations)
        grouped = representatives.merge(metadata, on=["X", "Y"], how="left")
    elif duplicate_method == "Keep last":
        representatives = prepared.drop_duplicates(subset=["X", "Y"], keep="last")[["X", "Y", "Z"]].copy()
        metadata = prepared.groupby(["X", "Y"], as_index=False).agg(**aggregations)
        grouped = representatives.merge(metadata, on=["X", "Y"], how="left")
    else:
        raise ValueError(f"Unknown duplicate handling method: {duplicate_method}")

    return grouped.reset_index(drop=True)
