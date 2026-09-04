from __future__ import annotations

import json
import zipfile
from io import BytesIO

import numpy as np
import pandas as pd

from core.active_data import PANEL_MODE_COMBINED, PANEL_MODE_INDEPENDENT, build_model_signature
from core.geostatistics.variogram import (
    VARIOGRAM_RANGE_CONVENTION,
    ExperimentalVariogram,
    VariogramFit,
)
from core.map_comparison import calculate_delta, calculate_pressure_change, pressure_change_allowed
from core.project_io import load_project_archive, save_project_archive
from core.scenarios import create_map_scenario
from utils.constants import INCLUDE_COLUMN, INTERNAL_ROW_ID


def _example_generated_map(name: str, pressure_reference_date: str, value: float):
    included = pd.DataFrame(
        {
            INTERNAL_ROW_ID: [10, 11],
            INCLUDE_COLUMN: [True, True],
            "X": [0.0, 1.0],
            "Y": [0.0, 1.0],
            "Pressure": [value, value],
            "Panel": ["A", "B"],
        }
    )
    signature = build_model_signature(
        property_column="Pressure",
        property_type="Pressure",
        pressure_reference_date=pressure_reference_date,
        selected_panels=["A", "B"],
        selected_layers=["L1"],
        panel_interpolation_mode=PANEL_MODE_COMBINED,
        filter_values={"Layer": ["L1"]},
        active_dataframe=included,
        duplicate_method="Average",
        interpolation_method="Ordinary Kriging",
        interpolation_parameters={
            "variogram_model": "Spherical",
            "variogram_range_convention": VARIOGRAM_RANGE_CONVENTION,
            "range": 10.0,
            "variance": 50.0,
            "nugget": 0.0,
        },
    )
    return {
        "grid_x": np.array([[0.0, 1.0], [0.0, 1.0]], dtype=float),
        "grid_y": np.array([[0.0, 0.0], [1.0, 1.0]], dtype=float),
        "grid_z": np.array([[value, value], [value, value]], dtype=float),
        "grid_variance": np.array([[1.0, 2.0], [3.0, 4.0]], dtype=float),
        "property_col": "Pressure",
        "property_type": "Pressure",
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
        "method_parameters": {
            "variogram_model": "Spherical",
            "variogram_range_convention": VARIOGRAM_RANGE_CONVENTION,
            "range": 10.0,
            "variance": 50.0,
            "nugget": 0.0,
        },
        "grid_parameters": {"nx": 2, "ny": 2},
        "mask_parameters": {"mode": "No Mask"},
        "mask_info": {"mask_mode": "No Mask"},
        "geometry_context": {"panel_boundaries": False},
        "geometry_references": {
            "reservoir_boundary_name": "R1 Boundary",
            "selected_panel_names": ["A", "B"],
            "selected_panel_bounds": [0.0, 0.0, 2.0, 1.0],
        },
        "duplicate_method": "Average",
        "respect_compartments": False,
        "panel_interpolation_mode": PANEL_MODE_COMBINED,
        "selected_panels": ["A", "B"],
        "selected_layers": ["L1"],
        "interpolation_domain": "Selected Panel Extent",
        "domain_bounds": [0.0, 0.0, 2.0, 1.0],
        "hover_columns": ["Well"],
        "export_metadata": {"property": "Pressure", "Variogram_Range_Convention": VARIOGRAM_RANGE_CONVENTION},
        "included_observations": included,
        "excluded_observations": pd.DataFrame(columns=[INTERNAL_ROW_ID, INCLUDE_COLUMN, "X", "Y", "Pressure"]),
        "model_signature": signature,
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
    assert pressure_change.metadata["Operation"] == "Pressure Change 01-Jan-2026 minus 01-Jan-2025"
    assert pressure_change.metadata["Delta_P_Convention"] == "Later pressure map minus earlier pressure map."


def test_pressure_change_disabled_for_same_reference_date_but_generic_delta_still_works():
    map_a = {
        "name": "Pressure A",
        "property": "Pressure",
        "property_unit": "psi",
        "coordinate_unit": "m",
        "crs": {"mode": "Local / Unknown XY"},
        "grid_x": np.array([[0.0, 1.0], [0.0, 1.0]], dtype=float),
        "grid_y": np.array([[0.0, 0.0], [1.0, 1.0]], dtype=float),
        "grid_z": np.full((2, 2), 3000.0),
        "pressure_reference_date": "2026-01-01",
        "property_type": "Pressure",
    }
    map_b = {**map_a, "name": "Pressure B", "grid_z": np.full((2, 2), 2990.0)}

    allowed, message = pressure_change_allowed(map_a, map_b)
    generic_delta = calculate_delta(map_a, map_b)

    assert not allowed
    assert "different Pressure Map Reference Dates" in message
    assert np.allclose(generic_delta.grid_z, -10.0)


def test_pressure_change_orders_dates_regardless_map_dropdown_order():
    later = {
        "name": "Later",
        "property": "Pressure",
        "property_type": "Pressure",
        "property_unit": "psi",
        "coordinate_unit": "m",
        "crs": {"mode": "Local / Unknown XY"},
        "grid_x": np.array([[0.0, 1.0], [0.0, 1.0]], dtype=float),
        "grid_y": np.array([[0.0, 0.0], [1.0, 1.0]], dtype=float),
        "grid_z": np.full((2, 2), 2900.0),
        "pressure_reference_date": "2026-01-01",
    }
    earlier = {**later, "name": "Earlier", "grid_z": np.full((2, 2), 3000.0), "pressure_reference_date": "2025-01-01"}

    result = calculate_pressure_change(later, earlier)

    assert np.allclose(result.grid_z, -100.0)
    assert result.metadata["Source_Map_Earlier"] == "Earlier"
    assert result.metadata["Source_Map_Later"] == "Later"
    assert result.metadata["Pressure_Change_Label"] == "Pressure Change 01-Jan-2026 minus 01-Jan-2025"


def test_scenario_snapshot_is_immutable_after_generated_map_mutation():
    generated = _example_generated_map("Pressure 2025", "2025-01-01", 3000.0)
    scenario = create_map_scenario(
        "Pressure 2025",
        generated,
        {
            "style_settings": {"color_scale": "Turbo"},
            "layer_settings": {"show_panels": True},
            "filter_values": {"Layer": ["L1"]},
            "include_state": {10: True, 11: True},
            "crs": {"mode": "Local / Unknown XY"},
        },
        validation={"metrics": {"RMSE": 12.3}, "signature": generated["model_signature"]},
    )

    generated["grid_z"][0, 0] = 9999.0
    generated["grid_variance"][0, 0] = 9999.0
    generated["method_parameters"]["range"] = 500.0
    generated["included_observations"].loc[0, "Pressure"] = 1111.0

    assert scenario["grid_z"][0, 0] == 3000.0
    assert scenario["grid_variance"][0, 0] == 1.0
    assert scenario["grid_stddev"][1, 1] == 2.0
    assert scenario["interpolation_parameters"]["range"] == 10.0
    assert scenario["included_observations"].loc[0, "Pressure"] == 3000.0
    assert scenario["selected_panels"] == ["A", "B"]
    assert scenario["panel_interpolation_mode"] == PANEL_MODE_COMBINED
    assert scenario["validation_metrics"]["RMSE"] == 12.3


def test_scenario_and_geostatistics_new_state_round_trip_through_project_archive():
    generated = _example_generated_map("Pressure 2025", "2025-01-01", 3000.0)
    scenario = create_map_scenario("Pressure 2025", generated)
    experimental = ExperimentalVariogram(
        lag_distance=np.array([1.0, 2.0, 3.0], dtype=float),
        semivariance=np.array([10.0, 20.0, 30.0], dtype=float),
        pair_count=np.array([2, 3, 4], dtype=int),
        max_lag=3.0,
        n_lags=3,
    )
    fit = VariogramFit("Gaussian", 25.0, 100.0, 5.0, 0.12)
    state = {
        "project_metadata": {"name": "Reproducible Project"},
        "source_name": "sample.csv",
        "column_mappings": {
            "x": "X",
            "y": "Y",
            "property": "Pressure",
            "well": "Well",
            "panel": "Panel",
            "map_reference_date": "Pressure_Map_Reference_Date",
        },
        "filter_values": {"Layer": ["L1"]},
        "coordinate_unit": "m",
        "property_unit": "psi",
        "pressure_reference_date": "2025-01-01",
        "selected_panels": ["A", "B"],
        "panel_interpolation_mode": PANEL_MODE_INDEPENDENT,
        "crs": {"mode": "Local / Unknown XY"},
        "include_state": {10: True, 11: True},
        "layer_settings": {"show_panels": True},
        "style_settings": {"color_scale": "Turbo"},
        "current_property": "Pressure",
        "geostatistics": {
            "experimental_variogram": experimental,
            "variogram_fit": fit,
            "variogram_fits": [fit],
            "variogram_settings": {
                "property": "Pressure",
                "property_type": "Pressure",
                "pressure_reference_date": "2025-01-01",
                "selected_panels": ["A", "B"],
                "panel_interpolation_mode": PANEL_MODE_INDEPENDENT,
                "selected_layers": ["L1"],
                "duplicate_method": "Average",
                "range_convention": VARIOGRAM_RANGE_CONVENTION,
            },
            "anisotropy_enabled": True,
            "anisotropy_angle": 35.0,
            "anisotropy_ratio": 0.45,
            "cross_validation_signature": generated["model_signature"],
        },
        "geometry_layers": {"reservoir_boundary": None, "panels": None, "faults": None, "custom": []},
        "map_scenarios": [scenario],
        "current_scenario_id": scenario["id"],
        "working_df": pd.DataFrame(
            {
                "X": [0.0, 1.0],
                "Y": [0.0, 1.0],
                "Pressure": [3000.0, 3000.0],
                "Panel": ["A", "B"],
                "Pressure_Map_Reference_Date": ["2025-01-01", "2025-01-01"],
            }
        ),
    }

    restored = load_project_archive(save_project_archive(state))
    restored_scenario = restored["map_scenarios"][0]
    restored_geo = restored["geostatistics"]

    assert restored["selected_panels"] == ["A", "B"]
    assert restored["panel_interpolation_mode"] == PANEL_MODE_INDEPENDENT
    assert restored_scenario["selected_panels"] == ["A", "B"]
    assert restored_scenario["panel_interpolation_mode"] == PANEL_MODE_COMBINED
    assert np.array_equal(restored_scenario["grid_z"], scenario["grid_z"])
    assert np.array_equal(restored_scenario["grid_stddev"], scenario["grid_stddev"])
    assert restored_scenario["interpolation_parameters"]["variogram_range_convention"] == VARIOGRAM_RANGE_CONVENTION
    assert restored_geo["variogram_fit"].model == "Gaussian"
    assert restored_geo["variogram_fit"].range_value == 25.0
    assert restored_geo["experimental_variogram"].n_lags == 3
    assert restored_geo["variogram_settings"]["range_convention"] == VARIOGRAM_RANGE_CONVENTION
    assert restored_geo["anisotropy_enabled"]
    assert restored_geo["anisotropy_angle"] == 35.0
    assert restored_geo["anisotropy_ratio"] == 0.45
    assert restored_geo["cross_validation_signature"]["hash"] == generated["model_signature"]["hash"]


def test_generated_layer_maps_round_trip_through_project_archive():
    upper = _example_generated_map("Upper Pressure", "2025-01-01", 3000.0)
    upper["reservoir_layer"] = "Upper"
    upper["layer_mapping_scope"] = "All Layers"
    upper["selected_layers"] = ["Upper"]
    lower = _example_generated_map("Lower Pressure", "2025-01-01", 2500.0)
    lower["reservoir_layer"] = "Lower"
    lower["layer_mapping_scope"] = "All Layers"
    lower["selected_layers"] = ["Lower"]
    state = {
        "project_metadata": {"name": "Layer Workspace"},
        "source_name": "sample.csv",
        "column_mappings": {"x": "X", "y": "Y", "property": "Pressure", "well": "Well", "layer": "Layer"},
        "coordinate_unit": "m",
        "property_unit": "psi",
        "layer_mapping_scope": "All Layers",
        "selected_reservoir_layer": "Upper",
        "active_generated_layer": "Lower",
        "generated_layer_maps": {"Upper": upper, "Lower": lower},
        "generated_layer_statuses": [
            {"layer": "Upper", "status": "Generated", "observations": 2, "message": ""},
            {"layer": "Lower", "status": "Generated", "observations": 2, "message": ""},
        ],
        "generated_layer_batch_signature": {"hash": "batch-hash"},
        "geometry_layers": {"reservoir_boundary": None, "panels": None, "faults": None, "custom": []},
        "map_scenarios": [],
        "working_df": pd.DataFrame(
            {
                "X": [0.0, 1.0],
                "Y": [0.0, 1.0],
                "Layer": ["Upper", "Lower"],
                "Pressure": [3000.0, 2500.0],
            }
        ),
    }

    restored = load_project_archive(save_project_archive(state))

    assert restored["layer_mapping_scope"] == "All Layers"
    assert restored["selected_reservoir_layer"] == "Upper"
    assert restored["active_generated_layer"] == "Lower"
    assert restored["generated_layer_batch_signature"]["hash"] == "batch-hash"
    assert set(restored["generated_layer_maps"]) == {"Upper", "Lower"}
    assert restored["generated_layer_maps"]["Lower"]["reservoir_layer"] == "Lower"
    assert np.array_equal(restored["generated_layer_maps"]["Upper"]["grid_z"], upper["grid_z"])
    assert restored["generated_layer_statuses"][0]["layer"] == "Upper"


def test_legacy_project_zone_mapping_promotes_to_layer_without_dropping_source_column():
    data = pd.DataFrame(
        {
            "X": [0.0],
            "Y": [0.0],
            "Zone": ["Upper"],
            "Pressure": [3000.0],
        }
    )
    manifest = {
        "project_schema_version": "1.1",
        "source_name": "legacy.csv",
        "column_mappings": {"x": "X", "y": "Y", "zone": "Zone"},
        "working_data_path": "data/working_data.csv",
    }
    buffer = BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("project.json", json.dumps(manifest).encode("utf-8"))
        archive.writestr("data/working_data.csv", data.to_csv(index=False).encode("utf-8"))

    restored = load_project_archive(buffer.getvalue())

    assert restored["column_mappings"]["layer"] == "Zone"
    assert "zone" not in restored["column_mappings"]
    assert "Zone" in restored["working_df"].columns
    assert restored["working_df"].loc[0, "Zone"] == "Upper"
