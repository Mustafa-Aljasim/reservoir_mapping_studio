from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest
from shapely.geometry import Polygon

from core.active_data import PANEL_MODE_COMBINED, PANEL_MODE_INDEPENDENT, prepare_active_property_data
from core.column_mapper import normalize_column_mappings, suggest_mappings
from core.filtering import build_filter_column_list
from core.geometry.models import GeometryFeature, GeometryLayer
from core.layer_mapping import (
    DOMAIN_RESERVOIR_BOUNDARY_EXTENT,
    DOMAIN_SELECTED_PANEL_UNION_EXTENT,
    DOMAIN_WELL_DATA_EXTENT,
    LAYER_SCOPE_ALL,
    LAYER_SCOPE_SELECTED,
    MAP_STATUS_STALE,
    MAP_STATUS_UP_TO_DATE,
    build_layer_model_signature,
    coerce_interpolation_domain_selection,
    generate_layer_map_collection,
    interpolation_domain_options,
    map_status,
    reservoir_layer_values,
    prepare_layer_observations,
    resolve_layer_method_parameters,
    select_generated_layer_map,
)
from core.project_io import load_project_archive, save_project_archive
from core.scenarios import create_map_scenario, scenario_to_generated_map
from utils.export import map_zmap_export_files, parse_zmap_grid_ascii_bytes
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


def _boundary_layer() -> GeometryLayer:
    return GeometryLayer(
        name="Reservoir Boundary",
        layer_type="Reservoir Boundary",
        features=[
            GeometryFeature(Polygon([(0, 0), (4, 0), (4, 4), (0, 4)]), "Reservoir Boundary", "R1"),
        ],
    )


def _three_panel_layer() -> GeometryLayer:
    return GeometryLayer(
        name="Panels",
        layer_type="Panel / Compartment",
        features=[
            GeometryFeature(Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]), "Panel / Compartment", "A"),
            GeometryFeature(Polygon([(1, 0), (2, 0), (2, 1), (1, 1)]), "Panel / Compartment", "B"),
            GeometryFeature(Polygon([(2, 0), (3, 0), (3, 1), (2, 1)]), "Panel / Compartment", "C"),
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


def _domain_frame() -> pd.DataFrame:
    points = [
        (1.0, 1.0, 10.0),
        (3.0, 1.0, 20.0),
        (1.0, 3.0, 30.0),
        (3.0, 3.0, 40.0),
        (2.0, 2.0, 25.0),
        (2.8, 2.2, 32.0),
    ]
    rows = []
    for index, (x_value, y_value, pressure) in enumerate(points, start=1):
        rows.append(
            {
                INTERNAL_ROW_ID: index,
                "Well": f"W{index}",
                "X": x_value,
                "Y": y_value,
                "Panel": "A",
                "Layer": "Upper",
                "Measurement_Date": "2025-06-01",
                "Map_Reference_Date": "2026-01-01",
                "Pressure": pressure,
            }
        )
    return pd.DataFrame(rows)


def _three_panel_frame() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    row_id = 1
    panel_points = {
        "A": [(0.2, 0.2), (0.8, 0.2), (0.5, 0.8)],
        "B": [(1.2, 0.2), (1.8, 0.2), (1.5, 0.8)],
        "C": [(2.2, 0.2), (2.8, 0.2), (2.5, 0.8)],
    }
    panel_values = {"A": 100.0, "B": 500.0, "C": 900.0}
    for panel, points in panel_points.items():
        for index, (x_value, y_value) in enumerate(points, start=1):
            rows.append(
                {
                    INTERNAL_ROW_ID: row_id,
                    "Well": f"{panel}{index}",
                    "X": x_value,
                    "Y": y_value,
                    "Panel": panel,
                    "Layer": "Upper",
                    "Measurement_Date": "2025-06-01",
                    "Map_Reference_Date": "2026-01-01",
                    "Pressure": panel_values[panel],
                }
            )
            row_id += 1
    return pd.DataFrame(rows)


def _generate(
    *,
    layer_scope: str,
    selected_layer: str | None = None,
    mappings: dict[str, str | None] | None = None,
    panel_layer: GeometryLayer | None = None,
    reservoir_boundary_layer: GeometryLayer | None = None,
    selected_panels: list[str] | None = None,
    panel_mode: str = PANEL_MODE_COMBINED,
    method: str = "IDW",
    idw_params: dict[str, object] | None = None,
    method_parameters: dict[str, object] | None = None,
    dataframe: pd.DataFrame | None = None,
    pressure_reference_date=date(2026, 1, 1),
    grid_parameters: dict[str, object] | None = None,
    mask_parameters: dict[str, object] | None = None,
    interpolation_domain: str = DOMAIN_WELL_DATA_EXTENT,
):
    return generate_layer_map_collection(
        dataframe=_layered_frame() if dataframe is None else dataframe,
        mappings=MAPPINGS if mappings is None else mappings,
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
        interpolation_method=method,
        interpolation_parameters=method_parameters
        or idw_params
        or {"power": 2.0, "neighbors": 6, "min_neighbors": 1, "search_radius": None},
        grid_parameters=GRID if grid_parameters is None else grid_parameters,
        mask_parameters=MASK if mask_parameters is None else mask_parameters,
        interpolation_domain=interpolation_domain,
        coordinate_unit="meters",
        crs={"mode": "Local / Unknown XY"},
        panel_layer=panel_layer,
        reservoir_boundary_layer=reservoir_boundary_layer,
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


def test_dynamic_interpolation_domain_options_keep_session_value_valid():
    well_only = interpolation_domain_options(has_reservoir_boundary=False, has_selected_panel_union=False)
    with_boundary = interpolation_domain_options(has_reservoir_boundary=True, has_selected_panel_union=False)
    with_boundary_and_panels = interpolation_domain_options(
        has_reservoir_boundary=True,
        has_selected_panel_union=True,
    )

    assert well_only == [DOMAIN_WELL_DATA_EXTENT]
    assert with_boundary == [DOMAIN_WELL_DATA_EXTENT, DOMAIN_RESERVOIR_BOUNDARY_EXTENT]
    assert with_boundary_and_panels == [
        DOMAIN_WELL_DATA_EXTENT,
        DOMAIN_RESERVOIR_BOUNDARY_EXTENT,
        DOMAIN_SELECTED_PANEL_UNION_EXTENT,
    ]
    assert coerce_interpolation_domain_selection(
        "Selected Panel Extent",
        with_boundary_and_panels,
    ) == DOMAIN_SELECTED_PANEL_UNION_EXTENT
    assert coerce_interpolation_domain_selection(
        DOMAIN_SELECTED_PANEL_UNION_EXTENT,
        with_boundary,
    ) == DOMAIN_WELL_DATA_EXTENT
    assert coerce_interpolation_domain_selection(
        "not a real domain",
        with_boundary_and_panels,
    ) == DOMAIN_WELL_DATA_EXTENT


def test_data_panel_column_does_not_filter_layers_without_loaded_panel_geometry():
    collection = _generate(layer_scope=LAYER_SCOPE_ALL, panel_layer=None, selected_panels=[])

    assert set(collection.maps) == {"Lower", "Thin", "Upper"}
    for layer_name, generated in collection.maps.items():
        assert set(generated["included_observations"]["Layer"]) == {layer_name}


def test_layer_availability_ignores_spatial_panel_geometry_without_dataset_panel_mapping():
    mappings_without_panel = {**MAPPINGS, "panel": None}
    remote_panel_layer = GeometryLayer(
        name="Remote Panels",
        layer_type="Panel / Compartment",
        features=[
            GeometryFeature(Polygon([(10, 10), (11, 10), (11, 11), (10, 11)]), "Panel / Compartment", "Remote"),
        ],
    )
    active_for_layers = prepare_active_property_data(
        _layered_frame(),
        mappings_without_panel,
        "Pressure",
        "Pressure",
        date(2026, 1, 1),
        {},
        None,
    ).dataframe

    collection = _generate(
        layer_scope=LAYER_SCOPE_ALL,
        mappings=mappings_without_panel,
        panel_layer=remote_panel_layer,
        selected_panels=["Remote"],
    )
    statuses = {status.layer: status for status in collection.statuses}

    assert reservoir_layer_values(active_for_layers, mappings_without_panel) == ["Lower", "Thin", "Upper"]
    assert set(statuses) == {"Lower", "Thin", "Upper"}
    assert collection.maps == {}
    assert all("insufficient data" in status.message.lower() for status in statuses.values())


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


def test_all_layers_combined_panel_filter_keeps_panel_and_layer_dimensions_separate():
    collection = _generate(
        layer_scope=LAYER_SCOPE_ALL,
        panel_layer=_panel_layer(),
        selected_panels=["A"],
        panel_mode=PANEL_MODE_COMBINED,
        idw_params={"power": 2.0, "neighbors": 6, "min_neighbors": 3, "search_radius": None},
    )
    statuses = {status.layer: status for status in collection.statuses}

    assert set(collection.maps) == {"Lower", "Upper"}
    assert statuses["Thin"].status == "Insufficient data"
    for layer_name, generated in collection.maps.items():
        assert not generated["respect_compartments"]
        assert generated["panel_grid"] is None
        assert set(generated["included_observations"]["Panel"]) == {"A"}
        assert set(generated["included_observations"]["Layer"]) == {layer_name}


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


@pytest.mark.parametrize(
    ("method", "method_parameters"),
    [
        ("IDW", {"power": 2.0, "neighbors": 6, "min_neighbors": 1, "search_radius": None}),
        ("RBF", {"kernel": "linear", "smoothing": 0.0, "neighbors": None, "epsilon": None}),
    ],
)
def test_reservoir_boundary_extent_drives_grid_for_global_methods(method, method_parameters):
    collection = _generate(
        layer_scope=LAYER_SCOPE_SELECTED,
        selected_layer="Upper",
        dataframe=_domain_frame(),
        reservoir_boundary_layer=_boundary_layer(),
        method=method,
        method_parameters=method_parameters,
        grid_parameters={"preset": "Test", "nx": 9, "ny": 9, "buffer_fraction": 0.0},
        mask_parameters={"mode": "No Mask", "max_distance": None},
        interpolation_domain=DOMAIN_RESERVOIR_BOUNDARY_EXTENT,
    )
    generated = collection.maps["Upper"]
    zmap_name, zmap_data, _, _, zmap_metadata = map_zmap_export_files(generated)[0]
    parsed_zmap = parse_zmap_grid_ascii_bytes(zmap_data)

    assert zmap_name.endswith(".zmap")
    assert tuple(generated["domain_bounds"]) == pytest.approx((0.0, 0.0, 4.0, 4.0))
    assert generated["mask_info"]["well_extent"] == pytest.approx((1.0, 1.0, 3.0, 3.0))
    assert float(generated["grid_x"].min()) == pytest.approx(0.0)
    assert float(generated["grid_x"].max()) == pytest.approx(4.0)
    assert float(generated["grid_y"].min()) == pytest.approx(0.0)
    assert float(generated["grid_y"].max()) == pytest.approx(4.0)
    assert np.isfinite(generated["grid_z"]).all()
    assert generated["export_metadata"]["Interpolation_Domain_Type"] == DOMAIN_RESERVOIR_BOUNDARY_EXTENT
    assert zmap_metadata["Domain_Geometry_Source"] == "Reservoir Boundary"
    assert parsed_zmap["x_min"] == pytest.approx(0.0)
    assert parsed_zmap["x_max"] == pytest.approx(4.0)
    assert parsed_zmap["y_min"] == pytest.approx(0.0)
    assert parsed_zmap["y_max"] == pytest.approx(4.0)


def test_reservoir_boundary_extent_drives_kriging_estimate_and_variance_grid():
    pytest.importorskip("gstools")
    collection = _generate(
        layer_scope=LAYER_SCOPE_SELECTED,
        selected_layer="Upper",
        dataframe=_domain_frame(),
        reservoir_boundary_layer=_boundary_layer(),
        method="Ordinary Kriging",
        method_parameters={"variogram_model": "Spherical", "range": 5.0, "variance": 100.0, "nugget": 0.0},
        grid_parameters={"preset": "Test", "nx": 9, "ny": 9, "buffer_fraction": 0.0},
        mask_parameters={"mode": "No Mask", "max_distance": None},
        interpolation_domain=DOMAIN_RESERVOIR_BOUNDARY_EXTENT,
    )
    generated = collection.maps["Upper"]

    assert tuple(generated["domain_bounds"]) == pytest.approx((0.0, 0.0, 4.0, 4.0))
    assert generated["grid_variance"] is not None
    assert generated["grid_variance"].shape == generated["grid_z"].shape
    assert np.isfinite(generated["grid_z"]).all()
    assert np.isfinite(generated["grid_variance"]).all()


def test_selected_panel_union_extent_combined_mode_has_no_internal_panel_gap():
    collection = _generate(
        layer_scope=LAYER_SCOPE_SELECTED,
        selected_layer="Upper",
        dataframe=_three_panel_frame(),
        panel_layer=_three_panel_layer(),
        selected_panels=["A", "B", "C"],
        panel_mode=PANEL_MODE_COMBINED,
        grid_parameters={"preset": "Test", "nx": 13, "ny": 5, "buffer_fraction": 0.0},
        mask_parameters={"mode": "Selected Panel Union", "max_distance": None},
        interpolation_domain=DOMAIN_SELECTED_PANEL_UNION_EXTENT,
    )
    generated = collection.maps["Upper"]
    x_axis = generated["grid_x"][0]
    y_axis = generated["grid_y"][:, 0]
    mid_row = int(np.where(np.isclose(y_axis, 0.5))[0][0])
    internal_cols = [int(np.where(np.isclose(x_axis, x_value))[0][0]) for x_value in (1.0, 2.0)]

    assert list(collection.maps) == ["Upper"]
    assert generated["panel_grid"] is None
    assert generated["domain_bounds"] == pytest.approx((0.0, 0.0, 3.0, 1.0))
    assert generated["export_metadata"]["Domain_Geometry_Source"] == "Selected Panel Union"
    assert len(generated["included_observations"]) == 9
    assert np.isfinite(generated["grid_z"]).all()
    assert all(np.isfinite(generated["grid_z"][mid_row, col]) for col in internal_cols)


def test_selected_panel_union_extent_independent_mode_keeps_one_layer_map_and_compartment_values():
    collection = _generate(
        layer_scope=LAYER_SCOPE_SELECTED,
        selected_layer="Upper",
        dataframe=_three_panel_frame(),
        panel_layer=_three_panel_layer(),
        selected_panels=["A", "B", "C"],
        panel_mode=PANEL_MODE_INDEPENDENT,
        grid_parameters={"preset": "Test", "nx": 13, "ny": 5, "buffer_fraction": 0.0},
        mask_parameters={"mode": "Selected Panel Union", "max_distance": None},
        interpolation_domain=DOMAIN_SELECTED_PANEL_UNION_EXTENT,
    )
    generated = collection.maps["Upper"]
    panel_grid = generated["panel_grid"]

    assert list(collection.maps) == ["Upper"]
    assert generated["respect_compartments"]
    assert panel_grid is not None
    assert {value for value in panel_grid.ravel().tolist() if value not in (None, "")} == {"A", "B", "C"}
    assert np.nanmax(generated["grid_z"][panel_grid == "A"]) < 150.0
    assert np.nanmin(generated["grid_z"][panel_grid == "B"]) > 450.0
    assert np.nanmax(generated["grid_z"][panel_grid == "B"]) < 550.0
    assert np.nanmin(generated["grid_z"][panel_grid == "C"]) > 850.0


def test_project_and_scenario_preserve_selected_spatial_domain():
    collection = _generate(
        layer_scope=LAYER_SCOPE_SELECTED,
        selected_layer="Upper",
        dataframe=_three_panel_frame(),
        panel_layer=_three_panel_layer(),
        selected_panels=["A", "B", "C"],
        panel_mode=PANEL_MODE_COMBINED,
        grid_parameters={"preset": "Test", "nx": 13, "ny": 5, "buffer_fraction": 0.0},
        mask_parameters={"mode": "Selected Panel Union", "max_distance": None},
        interpolation_domain=DOMAIN_SELECTED_PANEL_UNION_EXTENT,
    )
    generated = collection.maps["Upper"]
    scenario = create_map_scenario("Upper selected panel union", generated)
    generated["interpolation_domain"] = DOMAIN_WELL_DATA_EXTENT
    generated["domain_bounds"] = None
    generated["export_metadata"]["Interpolation_Domain_Type"] = DOMAIN_WELL_DATA_EXTENT

    opened = scenario_to_generated_map(scenario)
    restored = load_project_archive(
        save_project_archive(
            {
                "project_metadata": {"name": "Spatial Domain"},
                "source_name": "spatial.csv",
                "column_mappings": MAPPINGS,
                "coordinate_unit": "meters",
                "property_unit": "psi",
                "selected_panels": ["A", "B", "C"],
                "panel_interpolation_mode": PANEL_MODE_COMBINED,
                "mapping_interpolation_domain": DOMAIN_SELECTED_PANEL_UNION_EXTENT,
                "geometry_layers": {
                    "reservoir_boundary": None,
                    "panels": _three_panel_layer(),
                    "faults": None,
                    "custom": [],
                },
                "generated_layer_maps": {"Upper": opened},
                "map_scenarios": [scenario],
                "working_df": _three_panel_frame(),
            }
        )
    )

    assert opened["interpolation_domain"] == DOMAIN_SELECTED_PANEL_UNION_EXTENT
    assert opened["domain_bounds"] == pytest.approx((0.0, 0.0, 3.0, 1.0))
    assert opened["export_metadata"]["Interpolation_Domain_Type"] == DOMAIN_SELECTED_PANEL_UNION_EXTENT
    assert restored["mapping_interpolation_domain"] == DOMAIN_SELECTED_PANEL_UNION_EXTENT
    assert restored["selected_panels"] == ["A", "B", "C"]
    assert restored["generated_layer_maps"]["Upper"]["interpolation_domain"] == DOMAIN_SELECTED_PANEL_UNION_EXTENT
    assert restored["map_scenarios"][0]["interpolation_domain"] == DOMAIN_SELECTED_PANEL_UNION_EXTENT


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
