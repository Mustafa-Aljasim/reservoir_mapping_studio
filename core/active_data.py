"""Shared active-property preparation and model-signature helpers."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
import hashlib
import json
from typing import Any

import numpy as np
import pandas as pd

from core.filtering import apply_filters
from core.pressure_dates import validate_date_column
from utils.constants import INCLUDE_COLUMN, INTERNAL_ROW_ID
from utils.validators import is_pressure_property


PROPERTY_TYPE_PRESSURE = "Pressure"
PROPERTY_TYPE_GENERIC = "Generic"

PANEL_MODE_COMBINED = "Combined Selected Panels"
PANEL_MODE_INDEPENDENT = "Independent by Panel / Compartment"
PANEL_INTERPOLATION_MODES = (PANEL_MODE_COMBINED, PANEL_MODE_INDEPENDENT)


@dataclass(frozen=True)
class ActivePropertyData:
    dataframe: pd.DataFrame
    property_type: str
    pressure_reference_date: date | None
    pressure_reference_date_col: str | None
    selected_panels: tuple[str, ...]
    selected_layers: tuple[str, ...]
    warnings: tuple[str, ...] = ()


def normalize_property_type(property_column: str | None, property_type: str | None = None) -> str:
    if property_type in {PROPERTY_TYPE_PRESSURE, PROPERTY_TYPE_GENERIC}:
        return str(property_type)
    return PROPERTY_TYPE_PRESSURE if is_pressure_property(property_column) else PROPERTY_TYPE_GENERIC


def parse_reference_date(value: Any) -> date | None:
    if value in (None, ""):
        return None
    parsed = pd.to_datetime(value, errors="coerce")
    if pd.isna(parsed):
        return None
    return parsed.date()


def pressure_reference_dates(df: pd.DataFrame, mappings: dict[str, str | None]) -> tuple[date, ...]:
    reference_col = mappings.get("map_reference_date")
    if not reference_col or reference_col not in df.columns:
        return ()
    return validate_date_column(df[reference_col]).unique_dates


def resolve_pressure_reference_date(
    df: pd.DataFrame,
    mappings: dict[str, str | None],
    requested: Any = None,
) -> date | None:
    requested_date = parse_reference_date(requested)
    if requested_date is not None:
        return requested_date
    available = pressure_reference_dates(df, mappings)
    if available:
        return available[0]
    return None


def selected_filter_values(
    mappings: dict[str, str | None],
    filter_values: dict[str, list[object]] | None,
    semantic_key: str,
) -> tuple[str, ...]:
    column = mappings.get(semantic_key)
    if not column:
        return ()
    selected = (filter_values or {}).get(column, [])
    return tuple(str(value) for value in selected if value not in (None, ""))


def panel_mode_from_legacy(
    panel_interpolation_mode: str | None = None,
    respect_compartments: bool | None = None,
) -> str:
    if panel_interpolation_mode in PANEL_INTERPOLATION_MODES:
        return str(panel_interpolation_mode)
    return PANEL_MODE_INDEPENDENT if bool(respect_compartments) else PANEL_MODE_COMBINED


def respect_compartments_from_mode(panel_interpolation_mode: str | None) -> bool:
    return panel_mode_from_legacy(panel_interpolation_mode) == PANEL_MODE_INDEPENDENT


def _filter_selected_panel_column(
    df: pd.DataFrame,
    mappings: dict[str, str | None],
    selected_panels: tuple[str, ...],
) -> pd.DataFrame:
    panel_col = mappings.get("panel")
    if not selected_panels or not panel_col or panel_col not in df.columns:
        return df.copy()
    selected = set(selected_panels)
    return df[df[panel_col].astype(str).isin(selected)].copy()


def _filter_selected_layer_column(
    df: pd.DataFrame,
    mappings: dict[str, str | None],
    selected_layers: tuple[str, ...],
) -> pd.DataFrame:
    layer_col = mappings.get("layer")
    if not selected_layers or not layer_col or layer_col not in df.columns:
        return df.copy()
    selected = set(selected_layers)
    return df[df[layer_col].astype(str).isin(selected)].copy()


def filter_pressure_reference_date(
    df: pd.DataFrame,
    mappings: dict[str, str | None],
    property_type: str,
    pressure_reference_date: Any = None,
) -> pd.DataFrame:
    if normalize_property_type(None, property_type) != PROPERTY_TYPE_PRESSURE:
        return df.copy()
    reference_col = mappings.get("map_reference_date")
    selected_date = parse_reference_date(pressure_reference_date)
    if not selected_date or not reference_col or reference_col not in df.columns:
        return df.copy()
    reference_mask = pd.to_datetime(df[reference_col], errors="coerce").dt.date.eq(selected_date)
    return df.loc[reference_mask].copy()


def prepare_active_property_data(
    dataframe: pd.DataFrame,
    mappings: dict[str, str | None],
    property_column: str,
    property_type: str | None = None,
    pressure_reference_date: Any = None,
    filter_values: dict[str, list[object]] | None = None,
    selected_panels: list[object] | tuple[object, ...] | None = None,
    selected_layers: list[object] | tuple[object, ...] | None = None,
) -> ActivePropertyData:
    """Apply the V1 computational data sequence before duplicate handling.

    The pressure map reference date is an engineering snapshot selector. The
    original measurement date is deliberately ignored here and remains QC-only.
    """

    resolved_property_type = normalize_property_type(property_column, property_type)
    filtered = apply_filters(dataframe, mappings, filter_values or {})
    panel_selection_supplied = selected_panels is not None
    selected_panel_tuple = tuple(str(value) for value in (selected_panels or ()) if value not in (None, ""))
    if panel_selection_supplied and not selected_panel_tuple:
        filtered = filtered.iloc[0:0].copy()
    elif selected_panel_tuple:
        filtered = _filter_selected_panel_column(filtered, mappings, selected_panel_tuple)
    layer_selection_supplied = selected_layers is not None
    selected_layer_tuple = tuple(str(value) for value in (selected_layers or ()) if value not in (None, ""))
    if layer_selection_supplied and not selected_layer_tuple:
        filtered = filtered.iloc[0:0].copy()
    elif selected_layer_tuple:
        filtered = _filter_selected_layer_column(filtered, mappings, selected_layer_tuple)

    resolved_reference_date = None
    if resolved_property_type == PROPERTY_TYPE_PRESSURE:
        resolved_reference_date = resolve_pressure_reference_date(filtered, mappings, pressure_reference_date)
        filtered = filter_pressure_reference_date(
            filtered,
            mappings,
            resolved_property_type,
            resolved_reference_date,
        )

    return ActivePropertyData(
        dataframe=filtered.reset_index(drop=True),
        property_type=resolved_property_type,
        pressure_reference_date=resolved_reference_date,
        pressure_reference_date_col=mappings.get("map_reference_date"),
        selected_panels=selected_panel_tuple
        if panel_selection_supplied
        else selected_filter_values(mappings, filter_values, "panel"),
        selected_layers=selected_layer_tuple
        if layer_selection_supplied
        else selected_filter_values(mappings, filter_values, "layer"),
    )


def active_observation_ids(df: pd.DataFrame, include_col: str = INCLUDE_COLUMN) -> dict[str, list[int]]:
    if INTERNAL_ROW_ID not in df.columns:
        return {"included": [], "excluded": []}
    if include_col in df.columns:
        included_mask = df[include_col].astype(bool)
    else:
        included_mask = pd.Series(True, index=df.index)
    included = [int(value) for value in df.loc[included_mask, INTERNAL_ROW_ID].dropna().tolist()]
    excluded = [int(value) for value in df.loc[~included_mask, INTERNAL_ROW_ID].dropna().tolist()]
    return {"included": included, "excluded": excluded}


def _stable_json_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _stable_json_value(value[key]) for key in sorted(value, key=str)}
    if isinstance(value, (list, tuple, set)):
        return [_stable_json_value(item) for item in value]
    if isinstance(value, (date, pd.Timestamp)):
        return parse_reference_date(value).isoformat() if parse_reference_date(value) else str(value)
    if isinstance(value, np.ndarray):
        return _stable_json_value(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    return value


def signature_hash(signature: dict[str, Any]) -> str:
    payload = json.dumps(_stable_json_value(signature), sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def build_model_signature(
    *,
    property_column: str,
    property_type: str,
    pressure_reference_date: Any = None,
    selected_panels: tuple[str, ...] | list[str] = (),
    selected_layers: tuple[str, ...] | list[str] = (),
    panel_interpolation_mode: str | None = None,
    filter_values: dict[str, list[object]] | None = None,
    active_dataframe: pd.DataFrame | None = None,
    duplicate_method: str | None = None,
    interpolation_method: str | None = None,
    interpolation_parameters: dict[str, Any] | None = None,
    layer_mapping_scope: str | None = None,
    interpolation_domain: str | None = None,
    domain_bounds: Any = None,
    mask_parameters: dict[str, Any] | None = None,
    variogram: dict[str, Any] | None = None,
    anisotropy: dict[str, Any] | None = None,
) -> dict[str, Any]:
    ids = active_observation_ids(active_dataframe) if active_dataframe is not None else {"included": [], "excluded": []}
    signature = {
        "signature_version": "1.0",
        "property": property_column,
        "property_type": normalize_property_type(property_column, property_type),
        "pressure_reference_date": None
        if parse_reference_date(pressure_reference_date) is None
        else parse_reference_date(pressure_reference_date).isoformat(),
        "selected_panels": list(selected_panels or ()),
        "panel_interpolation_mode": panel_mode_from_legacy(panel_interpolation_mode),
        "selected_layers": list(selected_layers or ()),
        "active_filters": filter_values or {},
        "included_observation_ids": ids["included"],
        "excluded_observation_ids": ids["excluded"],
        "duplicate_method": duplicate_method,
        "interpolation_method": interpolation_method,
        "interpolation_parameters": interpolation_parameters or {},
        "layer_mapping_scope": layer_mapping_scope,
        "interpolation_domain": interpolation_domain,
        "domain_bounds": domain_bounds,
        "mask_parameters": mask_parameters or {},
        "variogram": variogram or {},
        "anisotropy": anisotropy or {},
        "control_points": [],
    }
    signature["hash"] = signature_hash(signature)
    return signature


def signatures_match(first: dict[str, Any] | None, second: dict[str, Any] | None) -> bool:
    if not first or not second:
        return False
    return str(first.get("hash") or signature_hash(first)) == str(second.get("hash") or signature_hash(second))
