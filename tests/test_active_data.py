from datetime import date

import numpy as np
import pandas as pd

from core.active_data import (
    PANEL_MODE_COMBINED,
    PANEL_MODE_INDEPENDENT,
    build_model_signature,
    prepare_active_property_data,
    signatures_match,
)
from core.data_qc import prepare_interpolation_dataframe
from core.geostatistics.validation import leave_one_out_cross_validation
from core.geostatistics.variogram import compute_experimental_variogram
from utils.constants import INCLUDE_COLUMN, INTERNAL_ROW_ID


def _multi_date_pressure_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            INTERNAL_ROW_ID: [1, 2, 3, 4, 5, 6],
            "Well": ["P01", "P02", "P03", "P01", "P02", "P03"],
            "X": [0.0, 1.0, 0.0, 0.0, 1.0, 0.0],
            "Y": [0.0, 0.0, 1.0, 0.0, 0.0, 1.0],
            "Panel": ["North", "North", "Central", "North", "North", "Central"],
            "Layer": ["L1", "L1", "L1", "L1", "L1", "L1"],
            "Measurement_Date": [
                "2024-03-01",
                "2024-08-10",
                "2024-05-20",
                "2024-03-01",
                "2024-08-10",
                "2024-05-20",
            ],
            "Map_Reference_Date": [
                "2025-01-01",
                "2025-01-01",
                "2025-01-01",
                "2026-01-01",
                "2026-01-01",
                "2026-01-01",
            ],
            "Pressure": [3210.0, 3180.0, 3190.0, 3010.0, 2970.0, 2990.0],
        }
    )


def test_active_pressure_data_uses_only_selected_reference_date_and_not_measurement_date():
    df = _multi_date_pressure_frame()
    mappings = {
        "x": "X",
        "y": "Y",
        "well": "Well",
        "panel": "Panel",
        "layer": "Layer",
        "measurement_date": "Measurement_Date",
        "map_reference_date": "Map_Reference_Date",
    }

    active_2025 = prepare_active_property_data(df, mappings, "Pressure", "Pressure", date(2025, 1, 1))
    active_2026 = prepare_active_property_data(df, mappings, "Pressure", "Pressure", date(2026, 1, 1))

    assert set(active_2025.dataframe["Pressure"]) == {3210.0, 3180.0, 3190.0}
    assert set(active_2026.dataframe["Pressure"]) == {3010.0, 2970.0, 2990.0}
    assert active_2025.dataframe["Measurement_Date"].nunique() == 3
    assert active_2025.pressure_reference_date == date(2025, 1, 1)


def test_duplicate_xy_aggregation_happens_after_pressure_reference_date_filtering():
    df = _multi_date_pressure_frame()
    mappings = {"x": "X", "y": "Y", "map_reference_date": "Map_Reference_Date", "well": "Well"}
    active = prepare_active_property_data(df, mappings, "Pressure", "Pressure", "2025-01-01")
    active_with_include = active.dataframe.copy()
    active_with_include[INCLUDE_COLUMN] = True

    prepared = prepare_interpolation_dataframe(
        active_with_include,
        "X",
        "Y",
        "Pressure",
        duplicate_method="Average",
        metadata_columns=["Well", "Map_Reference_Date"],
        row_id_col=INTERNAL_ROW_ID,
    )

    value_at_p01 = prepared.loc[(prepared["X"] == 0.0) & (prepared["Y"] == 0.0), "Z"].iloc[0]
    assert value_at_p01 == 3210.0
    assert 3110.0 not in prepared["Z"].tolist()
    assert set(prepared["Map_Reference_Date"]) == {"2025-01-01"}


def test_geostatistics_and_loocv_use_selected_pressure_snapshot_only():
    df = _multi_date_pressure_frame()
    mappings = {"x": "X", "y": "Y", "map_reference_date": "Map_Reference_Date", "well": "Well"}
    active = prepare_active_property_data(df, mappings, "Pressure", "Pressure", "2026-01-01")
    active_with_include = active.dataframe.copy()
    active_with_include[INCLUDE_COLUMN] = True
    prepared = prepare_interpolation_dataframe(active_with_include, "X", "Y", "Pressure", duplicate_method="Average")

    experimental = compute_experimental_variogram(prepared["X"], prepared["Y"], prepared["Z"], n_lags=4, max_lag=2.0)
    results, metrics = leave_one_out_cross_validation(
        prepared,
        "IDW",
        {"power": 2.0, "neighbors": 3, "min_neighbors": 1},
    )

    assert prepared["Z"].max() < 3050.0
    assert experimental.pair_count.sum() > 0
    assert results["Observed"].max() < 3050.0
    assert metrics["Count"] == 3


def test_active_observation_export_source_is_selected_reference_date_only():
    df = _multi_date_pressure_frame()
    mappings = {"x": "X", "y": "Y", "map_reference_date": "Map_Reference_Date"}
    active = prepare_active_property_data(df, mappings, "Pressure", "Pressure", "2025-01-01")

    export_frame = active.dataframe.copy()

    assert len(export_frame) == 3
    assert export_frame["Map_Reference_Date"].eq("2025-01-01").all()
    assert not export_frame["Map_Reference_Date"].eq("2026-01-01").any()


def test_model_signature_invalidates_validation_when_relevant_inputs_change():
    df = _multi_date_pressure_frame().iloc[:3].copy()
    df[INCLUDE_COLUMN] = [True, True, False]

    baseline = build_model_signature(
        property_column="Pressure",
        property_type="Pressure",
        pressure_reference_date="2025-01-01",
        selected_panels=["North"],
        selected_layers=["L1"],
        panel_interpolation_mode=PANEL_MODE_COMBINED,
        filter_values={"Layer": ["L1"]},
        active_dataframe=df,
        duplicate_method="Average",
        interpolation_method="IDW",
        interpolation_parameters={"power": 2.0},
        variogram={},
        anisotropy={"enabled": False},
    )

    for changed in [
        {"pressure_reference_date": "2026-01-01"},
        {"selected_panels": ["Central"]},
        {"panel_interpolation_mode": PANEL_MODE_INDEPENDENT},
        {"interpolation_parameters": {"power": 3.0}},
        {"variogram": {"model": "Spherical", "range": 10.0}},
    ]:
        params = {
            "property_column": "Pressure",
            "property_type": "Pressure",
            "pressure_reference_date": "2025-01-01",
            "selected_panels": ["North"],
            "selected_layers": ["L1"],
            "panel_interpolation_mode": PANEL_MODE_COMBINED,
            "filter_values": {"Layer": ["L1"]},
            "active_dataframe": df,
            "duplicate_method": "Average",
            "interpolation_method": "IDW",
            "interpolation_parameters": {"power": 2.0},
            "variogram": {},
            "anisotropy": {"enabled": False},
        }
        params.update(changed)
        assert not signatures_match(baseline, build_model_signature(**params))

    changed_include = df.copy()
    changed_include[INCLUDE_COLUMN] = [True, True, True]
    assert not signatures_match(
        baseline,
        build_model_signature(
            property_column="Pressure",
            property_type="Pressure",
            pressure_reference_date="2025-01-01",
            selected_panels=["North"],
            selected_layers=["L1"],
            panel_interpolation_mode=PANEL_MODE_COMBINED,
            filter_values={"Layer": ["L1"]},
            active_dataframe=changed_include,
            duplicate_method="Average",
            interpolation_method="IDW",
            interpolation_parameters={"power": 2.0},
            variogram={},
            anisotropy={"enabled": False},
        ),
    )

