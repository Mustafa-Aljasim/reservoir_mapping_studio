from __future__ import annotations

import numpy as np
import pandas as pd

from core.map_comparison import calculate_delta, calculate_pressure_change
from core.project_io import load_project_archive, save_project_archive
from core.scenarios import create_map_scenario


def _example_generated_map(name: str, pressure_reference_date: str, value: float):
    return {
        "grid_x": np.array([[0.0, 1.0], [0.0, 1.0]], dtype=float),
        "grid_y": np.array([[0.0, 0.0], [1.0, 1.0]], dtype=float),
        "grid_z": np.array([[value, value], [value, value]], dtype=float),
        "property_col": "Pressure",
        "method": "Ordinary Kriging",
        "title": name,
        "unit": "psi",
        "coordinate_unit": "m",
        "is_pressure_map": True,
        "map_reference_date": pressure_reference_date,
        "measurement_date_col": "Original_Measurement_Date",
        "map_reference_date_col": "Pressure_Map_Reference_Date",
        "x_col": "X",
        "y_col": "Y",
        "well_col": "Well",
        "method_parameters": {"variogram_model": "Spherical"},
        "grid_parameters": {"nx": 2, "ny": 2},
        "mask_parameters": {"mode": "No Mask"},
        "mask_info": {"mask_mode": "No Mask"},
        "geometry_context": {"panel_boundaries": False},
        "duplicate_method": "Average",
        "respect_compartments": False,
        "hover_columns": ["Well"],
        "export_metadata": {"property": "Pressure"},
        "included_observations": pd.DataFrame({"X": [0.0, 1.0], "Y": [0.0, 1.0], "Pressure": [value, value]}),
        "excluded_observations": pd.DataFrame(columns=["X", "Y", "Pressure"]),
    }


def test_project_archive_round_trip_preserves_project_state():
    working_df = pd.DataFrame(
        {
            "X": [0.0, 1.0, 2.0],
            "Y": [0.0, 1.0, 2.0],
            "Pressure": [3000.0, 3100.0, 3050.0],
            "Well": ["A", "B", "C"],
            "Pressure_Map_Reference_Date": ["2025-01-01", "2025-01-01", "2025-01-01"],
        }
    )
    scenario = create_map_scenario("Pressure 2025", _example_generated_map("Pressure 2025", "2025-01-01", 3000.0), {
        "style_settings": {"color_scale": "Turbo"},
        "layer_settings": {"show_panels": False},
        "filter_values": {"Well": ["A", "B"]},
        "include_state": {0: True, 1: True, 2: False},
        "crs": {"mode": "Local / Unknown XY"},
    })

    state = {
        "project_metadata": {"name": "Zubair Pressure Study", "description": "Prototype", "engineer": "Test Engineer"},
        "source_name": "sample_data.csv",
        "column_mappings": {"x": "X", "y": "Y", "property": "Pressure", "well": "Well"},
        "additional_filter_columns": ["Well"],
        "filter_values": {"Well": ["A", "B"]},
        "coordinate_unit": "m",
        "property_unit": "psi",
        "pressure_reference_date": "2025-01-01",
        "crs": {"mode": "Local / Unknown XY"},
        "include_state": {0: True, 1: True, 2: False},
        "layer_settings": {"show_panels": False},
        "style_settings": {"color_scale": "Turbo"},
        "current_property": "Pressure",
        "geometry_layers": {"reservoir_boundary": None, "panels": None, "faults": None, "custom": []},
        "map_scenarios": [scenario],
        "current_scenario_id": scenario["id"],
        "working_df": working_df,
    }

    restored = load_project_archive(save_project_archive(state))

    assert restored["project_metadata"]["name"] == "Zubair Pressure Study"
    assert restored["coordinate_unit"] == "m"
    assert restored["property_unit"] == "psi"
    assert restored["column_mappings"]["well"] == "Well"
    assert restored["filter_values"]["Well"] == ["A", "B"]
    assert restored["current_scenario_id"] == scenario["id"]
    assert len(restored["map_scenarios"]) == 1
    assert restored["map_scenarios"][0]["name"] == "Pressure 2025"
    assert restored["working_df"].loc[0, "Pressure"] == 3000.0


def test_delta_map_and_pressure_change_are_correct():
    map_a = {
        "name": "Pressure 2025",
        "property": "Pressure",
        "property_unit": "psi",
        "coordinate_unit": "m",
        "crs": {"mode": "Local / Unknown XY"},
        "grid_x": np.array([[0.0, 1.0], [0.0, 1.0]], dtype=float),
        "grid_y": np.array([[0.0, 0.0], [1.0, 1.0]], dtype=float),
        "grid_z": np.array([[3000.0, 3000.0], [3000.0, 3000.0]], dtype=float),
        "pressure_reference_date": "2025-01-01",
    }
    map_b = {
        "name": "Pressure 2026",
        "property": "Pressure",
        "property_unit": "psi",
        "coordinate_unit": "m",
        "crs": {"mode": "Local / Unknown XY"},
        "grid_x": np.array([[0.0, 1.0], [0.0, 1.0]], dtype=float),
        "grid_y": np.array([[0.0, 0.0], [1.0, 1.0]], dtype=float),
        "grid_z": np.array([[2800.0, 2800.0], [2800.0, 2800.0]], dtype=float),
        "pressure_reference_date": "2026-01-01",
    }

    delta = calculate_delta(map_a, map_b)
    pressure_change = calculate_pressure_change(map_a, map_b)

    assert np.allclose(delta.grid_z, -200.0)
    assert np.allclose(pressure_change.grid_z, -200.0)
    assert pressure_change.metadata["Date_A"] == "2025-01-01"
    assert pressure_change.metadata["Date_B"] == "2026-01-01"
    assert pressure_change.metadata["Operation"] == "Pressure Change = Later - Earlier"
