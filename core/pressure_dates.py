"""Pressure measurement and map-reference date helpers."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import pandas as pd


@dataclass(frozen=True)
class DateColumnValidation:
    parsed: pd.Series
    failed_count: int
    unique_dates: tuple[date, ...]

    @property
    def common_date(self) -> date | None:
        return self.unique_dates[0] if len(self.unique_dates) == 1 else None

    @property
    def has_multiple_dates(self) -> bool:
        return len(self.unique_dates) > 1


@dataclass(frozen=True)
class MeasurementDateSummary:
    failed_count: int
    earliest: date | None
    latest: date | None
    span_days: int | None


def parse_date_values(series: pd.Series) -> tuple[pd.Series, int]:
    parsed = pd.to_datetime(series, errors="coerce")
    failed_count = int(series.notna().sum() - parsed.notna().sum())
    return parsed, failed_count


def validate_date_column(series: pd.Series) -> DateColumnValidation:
    parsed, failed_count = parse_date_values(series)
    unique_dates = tuple(sorted(parsed.dropna().dt.date.unique()))
    return DateColumnValidation(parsed=parsed, failed_count=failed_count, unique_dates=unique_dates)


def summarize_measurement_dates(series: pd.Series) -> MeasurementDateSummary:
    parsed, failed_count = parse_date_values(series)
    valid_dates = parsed.dropna().dt.date
    if valid_dates.empty:
        return MeasurementDateSummary(failed_count=failed_count, earliest=None, latest=None, span_days=None)
    earliest = min(valid_dates)
    latest = max(valid_dates)
    return MeasurementDateSummary(
        failed_count=failed_count,
        earliest=earliest,
        latest=latest,
        span_days=(latest - earliest).days,
    )


def format_map_date(value: date | pd.Timestamp | str | None) -> str:
    if value is None or value == "":
        return ""
    parsed = pd.to_datetime(value, errors="coerce")
    if pd.isna(parsed):
        return str(value)
    return parsed.strftime("%d-%b-%Y")


def measurement_age_days(measurement_value, map_reference_date: date | None) -> int | None:
    if map_reference_date is None or pd.isna(measurement_value):
        return None
    parsed = pd.to_datetime(measurement_value, errors="coerce")
    if pd.isna(parsed):
        return None
    return (map_reference_date - parsed.date()).days

