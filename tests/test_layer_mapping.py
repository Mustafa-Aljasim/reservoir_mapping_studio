from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
from shapely.geometry import Polygon

from core.active_data import PANEL_MODE_COMBINED, PANEL_MODE_INDEPENDENT
from core.column_mapper import normalize_column_mappings, suggest_mappings
from core.filtering import build_filter_column_list
from core.geometry.models import GeometryFeature, GeometryLayer
from core.layer_mapping import (
    LAYER_SCOPE_ALL,
    LAYER_SCOPE_SELECTED,
    MAP_STATUS_STALE,
    MAP_STATUS_UP_TO_DATE,
    build_layer_model_signature,
    generate_layer_map_collection,
    map_status,
    prepare_layer_observations,
    resolve_layer_method_parameters,
    select_generated_layer_map,
)
from utils.constants import INTERNAL_ROW_ID, SEMANTIC_FIELDS


MAPPINGS = {
    "x": "X",
    "y": "Y",
    "well": "Well",
    "panel": "Panel",
    "layer": "Layer",
    "measurement_date": "Measurement_Date",
    "map_reference_date": "Map_Reference_Date",
}
GRID = {"preset": "Test", "nx": 9, "ny": 7, "buffer_fraction": 0.0}
MASK = {"mode": "No Mask", "max_distance": None}


def _panel_layer() -> GeometryLayer:
    return GeometryLayer(
        name="Panels",
        layer_type="Panel / Compartment",
        features=[
            GeometryFeature(Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]), "Panel / Compartment", "A"),
            GeometryFeature(Polygon([(1, 0), (2, 0), (2, 1), (1, 1)]), "Panel / Compartment", "B"),
        ],
    )


def _layered_frame() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    row_id = 1
    panel_points = {
        "A": [(0.2, 0.2), (0.8, 0.2), (0.5, 0.8)],
        "B": [(1.2, 0.2), (1.8, 0.2), (1.5, 0.8)],
    }
    layer_panel_values = {
        ("Upper", "A"): [3300.0, 3310.0, 3290.0],
        ("Upper", "B"): [3100.0, 3110.0, 3090.0],
        ("Lower", "A"): [2700.0, 2710.0, 2690.0],
        ("Lower", "B"): [2500.0, 2510.0, 2490.0],
        ("Thin", "A"): [1800.0, 1810.0],
    }
    for (layer, panel), values in layer_panel_values.items():
        for index, value in enumerate(values):
            x, y = panel_points[panel][index]
            rows.append(
                {
                    INTERNAL_ROW_ID: row_id,
                    "Well": f"{layer[:1]}{panel}{index + 1}",
                    "X": x,
                    "Y": y,
                    "Panel": panel,
                    "Layer": layer,
                    "Measurement_Date": "2025-06-01",
                    "Map_Reference_Date": "2026-01-01",
                    "Pressure": value,
                }
            )
            row_id += 1
    return pd.DataFrame(rows)


def _generate(
    *,
    layer_scope: str,
    selected_layer: str | None = None,
    panel_layer: GeometryLayer | None = None,
    selected_panels: list[str] | None = None,
    panel_mode: str = PANEL_MODE_COMBINED,
    idw_params: dict[str, object] | None = None,
    dataframe: pd.DataFrame | None = None,
    pressure_reference_date=date(2026, 1, 1),
):
    return generate_layer_map_collection(
        dataframe=_layered_frame() if dataframe is None else dataframe,
        mappings=MAPPINGS,
        property_column="Pressure",
        property_type="Pressure",
        property_unit="psi",
        pressure_reference_date=pressure_reference_date,
        is_pressure_map=True,
        filter_values={},
        selected_panels=selected_panels or [],
        layer_scope=layer_scope,
        selected_layer=selected_layer,
        panel_interpolation_mode=panel_mode,
        include_state={},
        duplicate_method="Average",
        interpolation_method="IDW",
        interpolation_parameters=idw_params
        or {"power": 2.0, "neighbors": 6, "min_neighbors": 1, "search_radius": None},
        grid_parameters=GRID,
        mask_parameters=MASK,
        interpolation_domain="Well Data Extent",
        coordinate_unit="meters",
        crs={"mode": "Local / Unknown XY"},
        panel_layer=panel_layer,
        x_col="X",
        y_col="Y",
        well_col="Well",
        hover_columns=[("Layer", "Layer")],
    )


def test_zone_semantic_removed_but_source_zone_can_map_to_layer():
    df = pd.DataFrame({"X": [0.0], "Y": [0.0], "Zone": ["Upper"], "Pressure": [3000.0]})

    suggested = suggest_mappings(df)
    normalized = normalize_column_mappings({"x": "X", "y": "Y", "zone": "Zone"})
    filters = build_filter_column_list(
        {"x": "X", "y": "Y", "reservoir": "Reservoir", "layer": "Zone"},
        exclude_semantic_keys=("layer",),
    )

    assert "zone" not in SEMANTIC_FIELDS
    assert "zone" not in suggested
    assert suggested["layer"] == "Zone"
    assert normalized["layer"] == "Zone"
    assert "zone" not in normalized
    assert ("Layer", "Zone") not in filters
    assert "Zone" in df.columns


def test_selected_layer_generates_one_map_without_mixing_layers():
    collection = _generate(layer_scope=LAYER_SCOPE_SELECTED, selected_layer="Upper")

    assert list(collection.maps) == ["Upper"]
    generated = collection.maps["Upper"]
    assert generated["reservoir_layer"] == "Upper"
    assert generated["selected_layers"] == ["Upper"]
    assert set(generated["included_observations"]["Layer"]) == {"Upper"}
    assert generated["included_observations"]["Pressure"].min() > 3000.0


def test_all_layers_generates_independent_map_per_layer():
    collection = _generate(layer_scope=LAYER_SCOPE_ALL)

    assert set(collection.maps) == {"Lower", "Thin", "Upper"}
    for layer_name, generated in collection.maps.items():
        assert generated["reservoir_layer"] == layer_name
        assert generated["selected_layers"] == [layer_name]
        assert set(generated["included_observations"]["Layer"]) == {layer_name}
    assert collection.maps["Upper"]["included_observations"]["Pressure"].mean() > 3000.0
    assert collection.maps["Lower"]["included_observations"]["Pressure"].mean() < 2800.0


def test_switching_generated_layer_selects_existing_map_without_recompute():
    collection = _generate(layer_scope=LAYER_SCOPE_ALL)

    selected = select_generated_layer_map(collection.maps, "Lower")

    assert selected is collection.maps["Lower"]
    assert set(collection.maps) == {"Lower", "Thin", "Upper"}


def test_insufficient_layer_reports_status_and_continues_other_layers():
    collection = _generate(
        layer_scope=LAYER_SCOPE_ALL,
        idw_params={"power": 2.0, "neighbors": 6, "min_neighbors": 3, "search_radius": None},
    )
    statuses = {status.layer: status for status in collection.statuses}

    assert set(collection.maps) == {"Lower", "Upper"}
    assert statuses["Thin"].status == "Insufficient data"
    assert "insufficient data" in statuses["Thin"].message


def test_all_layers_combined_panels_pool_selected_panel_observations_per_layer():
    collection = _generate(
        layer_scope=LAYER_SCOPE_ALL,
        panel_layer=_panel_layer(),
        selected_panels=["A", "B"],
        panel_mode=PANEL_MODE_COMBINED,
    )

    generated = collection.maps["Upper"]
    assert not generated["respect_compartments"]
    assert generated["panel_grid"] is None
    assert generated["selected_panels"] == ["A", "B"]
    assert set(generated["included_observations"]["Panel"]) == {"A", "B"}
    assert set(generated["included_observations"]["Layer"]) == {"Upper"}


def test_all_layers_independent_panels_respect_compartments_per_layer():
    collection = _generate(
        layer_scope=LAYER_SCOPE_ALL,
        panel_layer=_panel_layer(),
        selected_panels=["A", "B"],
        panel_mode=PANEL_MODE_INDEPENDENT,
    )

    generated = collection.maps["Upper"]
    finite_panel_labels = {
        value for value in generated["panel_grid"].ravel().tolist() if value not in (None, "")
    }

    assert generated["respect_compartments"]
    assert set(collection.maps) == {"Lower", "Upper"}
    assert {"A", "B"}.issubset(finite_panel_labels)
    assert set(generated["included_observations"]["Layer"]) == {"Upper"}


def test_map_status_changes_for_computational_inputs_but_not_visual_style():
    df = _layered_frame()
    params = {"power": 2.0, "neighbors": 6, "min_neighbors": 1, "search_radius": None}
    collection = _generate(layer_scope=LAYER_SCOPE_SELECTED, selected_layer="Upper", idw_params=params, dataframe=df)
    generated = collection.maps["Upper"]

    filtered, prepared = prepare_layer_observations(
        dataframe=df,
        mappings=MAPPINGS,
        property_column="Pressure",
        property_type="Pressure",
        pressure_reference_date=date(2026, 1, 1),
        filter_values={},
        selected_panels=[],
        reservoir_layer="Upper",
        panel_layer=None,
        include_state={},
        duplicate_method="Average",
        x_col="X",
        y_col="Y",
        well_col="Well",
    )
    resolved = resolve_layer_method_parameters("IDW", params, prepared)
    current = build_layer_model_signature(
        property_column="Pressure",
        property_type="Pressure",
        pressure_reference_date=date(2026, 1, 1),
        selected_panels=[],
        reservoir_layer="Upper",
        layer_mapping_scope=LAYER_SCOPE_SELECTED,
        panel_interpolation_mode=PANEL_MODE_COMBINED,
        filter_values={},
        active_dataframe=filtered,
        duplicate_method="Average",
        interpolation_method="IDW",
        interpolation_parameters=resolved,
        grid_parameters=GRID,
        interpolation_domain="Well Data Extent",
        domain_bounds=None,
        mask_parameters=MASK,
    )
    changed_grid = build_layer_model_signature(
        property_column="Pressure",
        property_type="Pressure",
        pressure_reference_date=date(2026, 1, 1),
        selected_panels=[],
        reservoir_layer="Upper",
        layer_mapping_scope=LAYER_SCOPE_SELECTED,
        panel_interpolation_mode=PANEL_MODE_COMBINED,
        filter_values={},
        active_dataframe=filtered,
        duplicate_method="Average",
        interpolation_method="IDW",
        interpolation_parameters=resolved,
        grid_parameters={**GRID, "nx": 11},
        interpolation_domain="Well Data Extent",
        domain_bounds=None,
        mask_parameters=MASK,
    )
    visually_changed = {**generated, "title": "Styled differently"}

    assert map_status(current, generated) == MAP_STATUS_UP_TO_DATE
    assert map_status(current, visually_changed) == MAP_STATUS_UP_TO_DATE
    assert map_status(changed_grid, generated) == MAP_STATUS_STALE


def test_pressure_reference_date_and_layer_filters_are_applied_together():
    df_2026 = _layered_frame()
    old_upper = df_2026[df_2026["Layer"] == "Upper"].copy()
    old_upper[INTERNAL_ROW_ID] = old_upper[INTERNAL_ROW_ID] + 100
    old_upper["Map_Reference_Date"] = "2025-01-01"
    old_upper["Pressure"] = old_upper["Pressure"] + 500.0
    df = pd.concat([df_2026, old_upper], ignore_index=True)

    collection = _generate(
        layer_scope=LAYER_SCOPE_SELECTED,
        selected_layer="Upper",
        dataframe=df,
        pressure_reference_date=date(2025, 1, 1),
    )
    included = collection.maps["Upper"]["included_observations"]

    assert set(included["Layer"]) == {"Upper"}
    assert set(included["Map_Reference_Date"]) == {"2025-01-01"}
    assert included["Pressure"].min() > 3500.0
