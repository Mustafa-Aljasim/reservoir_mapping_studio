from datetime import date

import numpy as np
import pandas as pd

from core.filtering import apply_filters
from core.map_comparison import calculate_delta
from core.map_context import build_default_map_title
from core.pressure_dates import (
    format_map_date,
    measurement_age_days,
    summarize_measurement_dates,
    validate_date_column,
)
from core.project_io import load_project_archive, save_project_archive
from core.scenarios import create_map_scenario


def test_measurement_dates_do_not_filter_observations():
    df = pd.DataFrame(
        {
            "Well": ["P-01", "P-02", "P-03"],
            "Reservoir": ["R1", "R1", "R1"],
            "Measurement_Date": ["2024-01-01", "2025-01-01", "2025-06-01"],
            "Pressure": [3210, 3195, 3178],
        }
    )
    filtered = apply_filters(df, {"measurement_date": "Measurement_Date"}, {"Reservoir": ["R1"]})
    assert len(filtered) == 3


def test_common_map_reference_date_is_resolved():
    validation = validate_date_column(pd.Series(["2026-01-01", "2026-01-01", "2026-01-01"]))
    assert validation.common_date == date(2026, 1, 1)
    assert not validation.has_multiple_dates


def test_multiple_map_reference_dates_are_valid_and_selectable():
    validation = validate_date_column(pd.Series(["2026-01-01", "2026-02-01"]))
    assert validation.common_date is None
    assert validation.has_multiple_dates
    assert list(validation.unique_dates) == [date(2026, 1, 1), date(2026, 2, 1)]


def test_pressure_reference_date_selection_filters_only_selected_reference_date():
    df = pd.DataFrame(
        {
            "Well": ["P01", "P02", "P01", "P02"],
            "X": [0, 1, 0, 1],
            "Y": [0, 0, 1, 1],
            "Measurement_Date": ["2024-03-01", "2024-08-10", "2024-03-01", "2024-08-10"],
            "Map_Reference_Date": ["2025-01-01", "2025-01-01", "2026-01-01", "2026-01-01"],
            "Pressure": [3210, 3180, 3010, 2970],
        }
    )
    selected_2025 = df[df["Map_Reference_Date"] == "2025-01-01"].copy()
    selected_2026 = df[df["Map_Reference_Date"] == "2026-01-01"].copy()
    assert len(selected_2025) == 2
    assert len(selected_2026) == 2
    assert set(selected_2025["Pressure"]) == {3210, 3180}
    assert set(selected_2026["Pressure"]) == {3010, 2970}
    assert selected_2025["Measurement_Date"].nunique() == 2


def test_measurement_date_does_not_affect_pressure_reference_date_selection():
    df = pd.DataFrame(
        {
            "Well": ["P01", "P02"],
            "Measurement_Date": ["2024-01-01", "2025-06-01"],
            "Map_Reference_Date": ["2025-01-01", "2026-01-01"],
            "Pressure": [3200, 2940],
        }
    )
    assert df[df["Map_Reference_Date"] == "2025-01-01"]["Pressure"].iloc[0] == 3200
    assert df[df["Map_Reference_Date"] == "2026-01-01"]["Pressure"].iloc[0] == 2940


def test_pressure_scenarios_remain_available_across_reference_dates_and_project_reload():
    def generated_map(name: str, date_value: str, value: float):
        return {
            "grid_x": np.array([[0.0, 1.0], [0.0, 1.0]], dtype=float),
            "grid_y": np.array([[0.0, 0.0], [1.0, 1.0]], dtype=float),
            "grid_z": np.array([[value, value], [value, value]], dtype=float),
            "property_col": "Pressure",
            "method": "IDW",
            "title": name,
            "unit": "psi",
            "coordinate_unit": "m",
            "is_pressure_map": True,
            "map_reference_date": date_value,
            "measurement_date_col": "Measurement_Date",
            "map_reference_date_col": "Map_Reference_Date",
            "x_col": "X",
            "y_col": "Y",
            "well_col": "Well",
            "method_parameters": {"power": 2.0, "neighbors": 2, "min_neighbors": 1},
            "grid_parameters": {"nx": 2, "ny": 2, "buffer_fraction": 0.03},
            "mask_parameters": {"mode": "No Mask"},
            "mask_info": {"mask_mode": "No Mask", "valid_grid_cells": 4},
            "respect_compartments": False,
            "geometry_context": {},
            "duplicate_method": "Average",
            "hover_columns": ["Well"],
            "export_metadata": {"property": "Pressure"},
            "included_observations": pd.DataFrame({"X": [0.0, 1.0], "Y": [0.0, 1.0], "Pressure": [value, value]}),
            "excluded_observations": pd.DataFrame(columns=["X", "Y", "Pressure"]),
        }

    scenario_2025 = create_map_scenario("Pressure 2025", generated_map("Pressure 2025", "2025-01-01", 3000.0))
    scenario_2026 = create_map_scenario("Pressure 2026", generated_map("Pressure 2026", "2026-01-01", 2800.0))
    scenarios = [scenario_2025, scenario_2026]

    assert {scenario["name"] for scenario in scenarios} == {"Pressure 2025", "Pressure 2026"}
    assert scenario_2025["pressure_reference_date"] == "2025-01-01"
    assert scenario_2026["pressure_reference_date"] == "2026-01-01"

    map_a = generated_map("Pressure 2025", "2025-01-01", 3000.0)
    map_b = generated_map("Pressure 2026", "2026-01-01", 2800.0)
    delta = calculate_delta(map_a, map_b)
    assert np.allclose(delta.grid_z, -200.0)

    state = {
        "project_metadata": {"name": "Ref Date Project"},
        "column_mappings": {"x": "X", "y": "Y", "property": "Pressure"},
        "filter_values": {},
        "coordinate_unit": "m",
        "property_unit": "psi",
        "pressure_reference_date": "2025-01-01",
        "crs": {"mode": "Local / Unknown XY"},
        "include_state": {},
        "layer_settings": {},
        "style_settings": {},
        "current_property": "Pressure",
        "geometry_layers": {"reservoir_boundary": None, "panels": None, "faults": None, "custom": []},
        "map_scenarios": scenarios,
        "current_scenario_id": scenario_2025["id"],
        "working_df": pd.DataFrame({"X": [0, 1], "Y": [0, 1], "Pressure": [3000, 2800]}),
    }
    restored = load_project_archive(save_project_archive(state))
    assert [scenario["name"] for scenario in restored["map_scenarios"]] == ["Pressure 2025", "Pressure 2026"]


def test_invalid_measurement_date_is_reported_but_data_can_remain():
    df = pd.DataFrame(
        {
            "Measurement_Date": ["2025-01-01", "not-a-date", None],
            "Pressure": [3000, 3010, 3020],
        }
    )
    summary = summarize_measurement_dates(df["Measurement_Date"])
    filtered = apply_filters(df, {}, {})
    assert summary.failed_count == 1
    assert len(filtered) == 3


def test_measurement_age_is_informational():
    assert measurement_age_days("2025-03-15", date(2026, 1, 1)) == 292


def test_pressure_title_contains_reference_date_but_generic_title_does_not():
    pressure_title = build_default_map_title(
        "Pressure",
        {"Reservoir": ["R1"]},
        {"reservoir": "Reservoir"},
        is_pressure_map=True,
        map_reference_date=date(2026, 1, 1),
    )
    generic_title = build_default_map_title(
        "Net_Sand",
        {"Reservoir": ["R1"]},
        {"reservoir": "Reservoir"},
        is_pressure_map=False,
        map_reference_date=date(2026, 1, 1),
    )
    assert "Reference Date: 01-Jan-2026" in pressure_title
    assert "Reference Date" not in generic_title
    assert format_map_date(date(2026, 1, 1)) == "01-Jan-2026"

