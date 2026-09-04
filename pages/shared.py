"""Shared Streamlit state and UI helpers."""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from core.active_data import (
    PANEL_INTERPOLATION_MODES,
    PANEL_MODE_COMBINED,
    panel_mode_from_legacy,
)
from core.column_mapper import categorical_filter_candidates, suggest_mappings
from core.data_loader import add_internal_row_id, normalize_columns
from core.filtering import (
    apply_filters,
    build_filter_column_list,
    options_for_column,
)
from utils.constants import (
    COLOR_SCALES,
    DEFAULT_COORDINATE_UNIT,
    INCLUDE_COLUMN,
    INTERNAL_ROW_ID,
    UNIT_PRESETS,
)
from core.crs import LOCAL_CRS_MODE, local_crs
from core.geometry.compartment import panel_feature_names
from core.layer_mapping import LAYER_MAPPING_SCOPES, LAYER_SCOPE_ALL, LAYER_SCOPE_SELECTED
from core.pressure_dates import format_map_date, validate_date_column
from utils.units import coordinate_unit_key_from_label, coordinate_unit_label, coordinate_unit_labels


ROOT = Path(__file__).resolve().parents[1]
SAMPLE_DATA_PATH = ROOT / "data" / "sample_reservoir_data.csv"


def ensure_session_state() -> None:
    defaults: dict[str, Any] = {
        "project_metadata": {
            "name": "Untitled Project",
            "description": "",
            "engineer": "",
            "field": "",
            "notes": "",
        },
        "project_dirty": False,
        "crs": local_crs().to_dict(),
        "map_scenarios": [],
        "current_scenario_id": None,
        "original_df": None,
        "working_df": None,
        "source_name": None,
        "source_key": None,
        "column_mappings": {},
        "additional_filter_columns": [],
        "filter_values": {},
        "coordinate_unit": DEFAULT_COORDINATE_UNIT,
        "pressure_reference_date": None,
        "selected_panels": [],
        "panel_interpolation_mode": PANEL_MODE_COMBINED,
        "layer_mapping_scope": "Selected Layer",
        "selected_reservoir_layer": None,
        "active_generated_layer": None,
        "generated_layer_maps": {},
        "generated_layer_statuses": [],
        "generated_layer_batch_signature": {},
        "engineering_control_points": [],
        "engineering_control_regions": [],
        "selected_control_region_well_ids": [],
        "geometry_layers": {
            "reservoir_boundary": None,
            "panels": None,
            "faults": None,
            "custom": [],
        },
        "layer_settings": {
            "show_surface": True,
            "show_wells": True,
            "show_excluded": True,
            "show_reservoir_boundary": True,
            "show_panels": True,
            "show_panel_labels": False,
            "show_faults": True,
            "show_fault_labels": False,
            "show_custom_layers": True,
            "show_custom_labels": False,
            "show_engineering_controls": True,
            "show_control_regions": True,
            "show_region_control_points": False,
            "reservoir_boundary_width": 2.5,
            "panel_boundary_width": 1.5,
            "fault_line_width": 2.0,
            "custom_line_width": 1.5,
            "geometry_fill_opacity": 0.0,
        },
        "geostatistics": {
            "variogram_fit": None,
            "variogram_fits": [],
            "experimental_variogram": None,
            "cross_validation": None,
            "method_comparison": None,
            "variogram_settings": {},
            "cross_validation_signature": {},
            "method_comparison_signature": {},
        },
        "current_property": None,
        "property_unit": "",
        "include_state": {},
        "generated_map": None,
        "style_settings": {
            "color_scale": "Turbo",
            "reverse_colors": False,
            "z_range_mode": "Auto",
            "zmin": None,
            "zmax": None,
            "contour_mode": "Auto interval",
            "contour_interval": None,
            "show_contour_lines": True,
            "show_contour_labels": False,
            "contour_line_width": 0.75,
            "show_wells": True,
            "marker_size": 9,
            "marker_opacity": 0.9,
            "marker_outline": True,
            "well_label_mode": "None",
            "label_text_size": 11,
            "show_excluded": True,
            "title_override": "",
            "height": 720,
        },
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def set_active_dataframe(df: pd.DataFrame, source_name: str, source_key: str | None = None) -> None:
    clean = normalize_columns(df)
    st.session_state.original_df = clean.copy()
    st.session_state.working_df = add_internal_row_id(clean)
    st.session_state.source_name = source_name
    st.session_state.source_key = source_key or source_name
    st.session_state.column_mappings = suggest_mappings(st.session_state.working_df)
    st.session_state.additional_filter_columns = []
    st.session_state.filter_values = {}
    st.session_state.pressure_reference_date = None
    st.session_state.selected_panels = []
    st.session_state.panel_interpolation_mode = PANEL_MODE_COMBINED
    st.session_state.layer_mapping_scope = "Selected Layer"
    st.session_state.selected_reservoir_layer = None
    st.session_state.active_generated_layer = None
    st.session_state.generated_layer_maps = {}
    st.session_state.generated_layer_statuses = []
    st.session_state.generated_layer_batch_signature = {}
    st.session_state.engineering_control_points = []
    st.session_state.engineering_control_regions = []
    st.session_state.selected_control_region_well_ids = []
    st.session_state.current_property = None
    st.session_state.include_state = {}
    st.session_state.generated_map = None
    mark_project_dirty()


def geometry_layers() -> dict:
    ensure_session_state()
    return st.session_state.geometry_layers


def mark_project_dirty() -> None:
    st.session_state.project_dirty = True


def mark_project_saved() -> None:
    st.session_state.project_dirty = False


def project_display_name() -> str:
    metadata = st.session_state.get("project_metadata", {})
    name = str(metadata.get("name") or "Untitled Project")
    return f"{name} *" if st.session_state.get("project_dirty") else name


def reset_workspace_for_new_project(metadata: dict[str, str]) -> None:
    st.session_state.project_metadata = {
        "name": metadata.get("name") or "Untitled Project",
        "description": metadata.get("description", ""),
        "engineer": metadata.get("engineer", ""),
        "field": metadata.get("field", ""),
        "notes": metadata.get("notes", ""),
    }
    st.session_state.crs = local_crs().to_dict()
    st.session_state.original_df = None
    st.session_state.working_df = None
    st.session_state.source_name = None
    st.session_state.source_key = None
    st.session_state.column_mappings = {}
    st.session_state.additional_filter_columns = []
    st.session_state.filter_values = {}
    st.session_state.pressure_reference_date = None
    st.session_state.selected_panels = []
    st.session_state.panel_interpolation_mode = PANEL_MODE_COMBINED
    st.session_state.layer_mapping_scope = "Selected Layer"
    st.session_state.selected_reservoir_layer = None
    st.session_state.active_generated_layer = None
    st.session_state.generated_layer_maps = {}
    st.session_state.generated_layer_statuses = []
    st.session_state.generated_layer_batch_signature = {}
    st.session_state.engineering_control_points = []
    st.session_state.engineering_control_regions = []
    st.session_state.selected_control_region_well_ids = []
    st.session_state.geometry_layers = {
        "reservoir_boundary": None,
        "panels": None,
        "faults": None,
        "custom": [],
    }
    st.session_state.geostatistics = {
        "variogram_fit": None,
        "variogram_fits": [],
        "experimental_variogram": None,
        "cross_validation": None,
        "method_comparison": None,
        "variogram_settings": {},
        "cross_validation_signature": {},
    }
    st.session_state.current_property = None
    st.session_state.property_unit = ""
    st.session_state.include_state = {}
    st.session_state.generated_map = None
    st.session_state.map_scenarios = []
    st.session_state.current_scenario_id = None
    mark_project_dirty()


def workflow_status() -> dict[str, bool]:
    geometry = st.session_state.get("geometry_layers", {})
    geostatistics = st.session_state.get("geostatistics", {})
    return {
        "Data": st.session_state.get("working_df") is not None,
        "Geometry": bool(geometry.get("reservoir_boundary") or geometry.get("panels") or geometry.get("faults")),
        "Map": st.session_state.get("generated_map") is not None,
        "Validation": bool(geostatistics.get("cross_validation")),
        "Saved": not bool(st.session_state.get("project_dirty", False)),
    }


def load_sample_dataset() -> None:
    sample = pd.read_csv(SAMPLE_DATA_PATH)
    set_active_dataframe(sample, "Bundled sample reservoir data", str(SAMPLE_DATA_PATH))


def selectbox_with_none(
    label: str,
    options: list[str],
    current: str | None,
    suggested: str | None,
    key: str,
    required: bool = False,
) -> str | None:
    display_options = options if required else ["None"] + options
    desired = current or suggested
    if desired not in display_options:
        desired = options[0] if required and options else "None"
    index = display_options.index(desired) if desired in display_options else 0
    selected = st.selectbox(label, display_options, index=index, key=key)
    return selected if selected != "None" else None


def render_filter_controls(
    df: pd.DataFrame,
    mappings: dict[str, str | None],
    key_prefix: str,
    include_additional: bool = True,
    exclude_semantic_keys: tuple[str, ...] | list[str] | set[str] = (),
) -> pd.DataFrame:
    additional = st.session_state.get("additional_filter_columns", []) if include_additional else []
    filter_columns = build_filter_column_list(mappings, additional, exclude_semantic_keys)
    excluded_columns = {
        mappings.get(key)
        for key in (exclude_semantic_keys or ())
        if mappings.get(key)
    }

    if not filter_columns:
        st.caption("No mapped metadata filters are available.")
        return df.copy()

    filter_values = {
        column: selected
        for column, selected in dict(st.session_state.get("filter_values", {})).items()
        if column not in excluded_columns
    }
    prior_columns: list[str] = []
    for label, column in filter_columns:
        if column not in df.columns:
            continue
        options = options_for_column(df, column, filter_values, prior_columns)
        selected = [value for value in filter_values.get(column, []) if value in options]
        chosen = st.multiselect(
            label,
            options=options,
            default=selected,
            key=f"{key_prefix}_filter_{column}",
        )
        filter_values[column] = chosen
        prior_columns.append(column)

    st.session_state.filter_values = {
        column: selected for column, selected in filter_values.items() if selected
    }

    return apply_filters(
        df,
        mappings,
        st.session_state.filter_values,
    )


def get_current_filtered_data() -> pd.DataFrame:
    df = st.session_state.get("working_df")
    if df is None:
        return pd.DataFrame()
    return apply_filters(
        df,
        st.session_state.get("column_mappings", {}),
        st.session_state.get("filter_values", {}),
    )


def sync_include_state(df: pd.DataFrame) -> None:
    state = dict(st.session_state.get("include_state", {}))
    if INTERNAL_ROW_ID not in df.columns:
        st.session_state.include_state = state
        return
    for row_id in df[INTERNAL_ROW_ID].tolist():
        state.setdefault(int(row_id), True)
    st.session_state.include_state = state


def attach_include_column(df: pd.DataFrame) -> pd.DataFrame:
    sync_include_state(df)
    output = df.copy()
    if INTERNAL_ROW_ID in output.columns:
        output[INCLUDE_COLUMN] = output[INTERNAL_ROW_ID].map(
            lambda row_id: bool(st.session_state.include_state.get(int(row_id), True))
        )
    else:
        output[INCLUDE_COLUMN] = True
    return output


def update_include_state_from_editor(edited_df: pd.DataFrame) -> None:
    if INTERNAL_ROW_ID not in edited_df.columns or INCLUDE_COLUMN not in edited_df.columns:
        return
    state = dict(st.session_state.get("include_state", {}))
    for _, row in edited_df.iterrows():
        state[int(row[INTERNAL_ROW_ID])] = bool(row[INCLUDE_COLUMN])
    st.session_state.include_state = state


def filter_candidate_options(df: pd.DataFrame, mappings: dict[str, str | None]) -> list[str]:
    mapped = {column for column in mappings.values() if column}
    candidates = []
    for column in categorical_filter_candidates(df, mappings):
        if column in mapped:
            continue
        candidates.append(column)
    return candidates


def unit_input(key: str = "property_unit_input") -> str:
    current = st.session_state.get("property_unit", "")
    unit_options = list(UNIT_PRESETS)
    if current not in unit_options:
        unit_options.append(current)
    selected = st.selectbox(
        "Property Unit",
        unit_options,
        index=unit_options.index(current) if current in unit_options else 0,
        key=f"{key}_preset",
    )
    custom = st.text_input("Custom Unit", value=selected, key=f"{key}_custom")
    st.session_state.property_unit = custom.strip()
    return st.session_state.property_unit


def coordinate_unit_input(key: str = "coordinate_unit_input") -> str:
    current_key = st.session_state.get("coordinate_unit", DEFAULT_COORDINATE_UNIT)
    labels = coordinate_unit_labels()
    current_label = coordinate_unit_label(current_key)
    selected_label = st.selectbox(
        "Coordinate / Distance Unit",
        labels,
        index=labels.index(current_label) if current_label in labels else 0,
        key=key,
        help="Describes the numeric units of the uploaded X/Y coordinates. Coordinates are not converted.",
    )
    st.session_state.coordinate_unit = coordinate_unit_key_from_label(selected_label)
    return st.session_state.coordinate_unit


def pressure_reference_date_control(
    df: pd.DataFrame,
    mappings: dict[str, str | None],
    key: str = "pressure_reference_date_input",
) -> tuple[date | None, object | None]:
    reference_col = mappings.get("map_reference_date")
    validation = None
    detected_common_date = None
    available_dates: list[date] = []
    if reference_col and reference_col in df.columns:
        validation = validate_date_column(df[reference_col])
        available_dates = list(validation.unique_dates)
        detected_common_date = validation.common_date
        if detected_common_date:
            st.caption(f"Detected common Pressure Map Reference Date: {format_map_date(detected_common_date)}")
        elif validation.has_multiple_dates:
            st.caption(
                "Multiple Pressure Map Reference Dates are present. Select the prepared pressure snapshot to use."
            )
        if validation.failed_count:
            st.warning(f"{validation.failed_count} pressure map reference date value(s) could not be parsed.")

        if available_dates:
            current_reference_date = st.session_state.get("pressure_reference_date")
            if current_reference_date not in available_dates:
                current_reference_date = available_dates[0]
            selected_reference_date = available_dates[0]
            if len(available_dates) > 1:
                selected_reference_date = st.selectbox(
                    "Pressure Map Reference Date",
                    available_dates,
                    index=available_dates.index(current_reference_date),
                    format_func=format_map_date,
                    key=key,
                    help="Select the prepared pressure-map reference date for this spatial interpolation. Original measurement dates remain metadata only.",
                )
            st.session_state.pressure_reference_date = selected_reference_date
            return selected_reference_date, validation

    current_reference_date = st.session_state.get("pressure_reference_date") or detected_common_date or date.today()
    selected_reference_date = st.date_input(
        "Pressure Map Reference Date",
        value=current_reference_date,
        key=key,
        help="A single reference date represented by the supplied pressure values. No temporal extrapolation is performed.",
    )
    st.session_state.pressure_reference_date = selected_reference_date
    return selected_reference_date, validation


def panel_selection_control(panel_layer, key: str = "selected_panels") -> list[str]:
    options = panel_feature_names(panel_layer)
    if not options:
        st.session_state.selected_panels = []
        st.session_state[key] = []
        return []
    if key in st.session_state:
        current = [value for value in st.session_state.get(key, []) if value in options]
    else:
        current = [value for value in st.session_state.get("selected_panels", []) if value in options]
    if not current and key not in st.session_state:
        current = options
    selected = st.multiselect(
        "Panel Selection",
        options,
        default=current,
        key=key,
        help="Select which panel observations participate in the active model.",
    )
    st.session_state.selected_panels = selected
    return selected


def panel_interpolation_mode_control(
    has_panel_layer: bool,
    key: str = "panel_interpolation_mode",
) -> str:
    if not has_panel_layer:
        st.session_state.panel_interpolation_mode = PANEL_MODE_COMBINED
        return PANEL_MODE_COMBINED
    current = panel_mode_from_legacy(st.session_state.get("panel_interpolation_mode"), False)
    selected = st.radio(
        "Panel Interpolation Mode",
        PANEL_INTERPOLATION_MODES,
        index=PANEL_INTERPOLATION_MODES.index(current),
        horizontal=True,
        key=key,
        help="Combined pools selected panels into one interpolation model. Independent treats selected panels as separated compartments.",
    )
    st.session_state.panel_interpolation_mode = selected
    return selected


def layer_mapping_scope_control(has_layer_column: bool, key: str = "layer_mapping_scope") -> str:
    if not has_layer_column:
        st.session_state.layer_mapping_scope = LAYER_SCOPE_SELECTED
        return LAYER_SCOPE_SELECTED
    current = st.session_state.get("layer_mapping_scope", LAYER_SCOPE_SELECTED)
    if current not in LAYER_MAPPING_SCOPES:
        current = LAYER_SCOPE_SELECTED
    selected = st.radio(
        "Layer Mapping Scope",
        LAYER_MAPPING_SCOPES,
        index=LAYER_MAPPING_SCOPES.index(current),
        horizontal=True,
        key=key,
        help="Selected Layer generates one map. All Layers generates one independent map per reservoir layer.",
    )
    st.session_state.layer_mapping_scope = selected
    return selected


def reservoir_layer_control(options: list[str], key: str = "selected_reservoir_layer") -> str | None:
    if not options:
        st.session_state.selected_reservoir_layer = None
        return None
    current = st.session_state.get("selected_reservoir_layer")
    if current not in options:
        current = options[0]
    selected = st.selectbox(
        "Reservoir Layer",
        options,
        index=options.index(current),
        key=key,
    )
    st.session_state.selected_reservoir_layer = selected
    return selected


def color_scale_options() -> list[str]:
    return list(COLOR_SCALES)
