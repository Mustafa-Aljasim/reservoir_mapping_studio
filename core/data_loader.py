"""Data loading and import-summary helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

import pandas as pd

from utils.constants import INTERNAL_ROW_ID


SUPPORTED_EXTENSIONS = {".csv", ".xlsx", ".xls"}


@dataclass(frozen=True)
class ImportSummary:
    rows: int
    columns: int
    numeric_columns: int
    text_columns: int


def _rewind(file: BinaryIO) -> None:
    if hasattr(file, "seek"):
        file.seek(0)


def get_file_extension(filename: str) -> str:
    return Path(filename).suffix.lower()


def get_excel_sheet_names(file: BinaryIO) -> list[str]:
    _rewind(file)
    workbook = pd.ExcelFile(file)
    _rewind(file)
    return workbook.sheet_names


def load_uploaded_dataframe(file: BinaryIO, filename: str, sheet_name: str | None = None) -> pd.DataFrame:
    """Load a CSV or Excel file into a DataFrame without mutating the file."""

    extension = get_file_extension(filename)
    if extension not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Unsupported file type: {extension}")

    _rewind(file)
    if extension == ".csv":
        df = pd.read_csv(file)
    else:
        df = pd.read_excel(file, sheet_name=sheet_name)
    _rewind(file)
    return normalize_columns(df)


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with string column names and no all-empty rows."""

    output = df.copy()
    output.columns = [str(column).strip() for column in output.columns]
    output = output.dropna(how="all").reset_index(drop=True)
    return output


def add_internal_row_id(df: pd.DataFrame) -> pd.DataFrame:
    """Attach a stable row id used for inclusion/exclusion state."""

    output = df.copy()
    if INTERNAL_ROW_ID in output.columns:
        output = output.drop(columns=[INTERNAL_ROW_ID])
    output.insert(0, INTERNAL_ROW_ID, range(1, len(output) + 1))
    return output


def summarize_dataframe(df: pd.DataFrame) -> ImportSummary:
    numeric_columns = int(df.select_dtypes(include="number").shape[1])
    text_columns = int(df.select_dtypes(include=["object", "string", "category"]).shape[1])
    return ImportSummary(
        rows=int(df.shape[0]),
        columns=int(df.shape[1]),
        numeric_columns=numeric_columns,
        text_columns=text_columns,
    )

