from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest
from shapely.geometry import Point, Polygon

from core.active_data import PANEL_MODE_COMBINED, PANEL_MODE_INDEPENDENT
from core.data_qc import prepare_interpolation_dataframe
from core.engineering_controls import (
    DRAWN_CONTROL_REGION_SOURCE,
    clip_control_region_to_active_domain,
    create_control_point,
    create_control_region,
    create_region_from_wells,
    engineering_controls_for_context,
    polygon_from_vertices,
    validate_control_region_parameters,
)
from core.geometry.models import GeometryFeature, GeometryLayer
from core.geostatistics.validation import leave_one_out_cross_validation
from core.geostatistics.variogram import VARIOGRAM_RANGE_CONVENTION
from core.layer_mapping import (
    LAYER_SCOPE_ALL,
    LAYER_SCOPE_SELECTED,
    build_layer_model_signature,
    generate_layer_map_collection,
    prepare_layer_observations,
    resolve_layer_method_parameters,
)
from core.plotting.map_builder import build_context_map_figure
from core.project_io import load_project_archive, save_project_archive
from core.scenarios import create_map_scenario, scenario_to_generated_map
from utils.constants import INCLUDE_COLUMN, INTERNAL_ROW_ID


MAPPINGS = {
    "x": "X",
    "y": "Y",
    "well": "Well",
    "panel": "Panel",
    "layer": "Layer",
    "map_reference_date": "Map_Reference_Date",
}
GRID = {"preset": "Test", "nx": 3, "ny": 3, "buffer_fraction": 0.0}
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


def _reservoir_boundary() -> GeometryLayer:
    return GeometryLayer(
        name="Reservoir",
        layer_type="Reservoir Boundary",
        features=[GeometryFeature(Polygon([(0, 0), (2, 0), (2, 1), (0, 1)]), "Reservoir Boundary", "R1")],
    )


def _layer_frame() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    row_id = 1
    for layer, base in {"Upper": 3000.0, "Lower": 2500.0}.items():
        for x, y, panel, offset in [
            (0.0, 0.0, "A", 0.0),
            (1.0, 0.0, "A", 10.0),
            (0.0, 1.0, "A", 20.0),
            (1.0, 1.0, "A", 30.0),
            (1.5, 0.5, "B", 40.0),
        ]:
            rows.append(
                {
                    INTERNAL_ROW_ID: row_id,
                    "Well": f"{layer}-{row_id}",
                    "X": x,
                    "Y": y,
                    "Panel": panel,
                    "Layer": layer,
                    "Pressure": base + offset,
                    "Map_Reference_Date": "2026-01-01",
                }
            )
            row_id += 1
    return pd.DataFrame(rows)


def _generate(
    dataframe: pd.DataFrame,
    *,
    controls: list[dict[str, object]] | None = None,
    regions: list[dict[str, object]] | None = None,
    method: str = "IDW",
    method_parameters: dict[str, object] | None = None,
    layer_scope: str = LAYER_SCOPE_SELECTED,
    selected_layer: str | None = "Upper",
    panel_layer: GeometryLayer | None = None,
    selected_panels: list[str] | None = None,
    panel_mode: str = PANEL_MODE_COMBINED,
):
    return generate_layer_map_collection(
        dataframe=dataframe,
        mappings=MAPPINGS,
        property_column="Pressure",
        property_type="Pressure",
        property_unit="psi",
        pressure_reference_date=date(2026, 1, 1),
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
        or {"power": 2.0, "neighbors": 10, "min_neighbors": 1, "search_radius": None},
        grid_parameters=GRID,
        mask_parameters=MASK,
        interpolation_domain="Well Data Extent",
        coordinate_unit="meters",
        crs={"mode": "Local / Unknown XY"},
        panel_layer=panel_layer,
        reservoir_boundary_layer=_reservoir_boundary(),
        control_points=controls or [],
        control_regions=regions or [],
        x_col="X",
        y_col="Y",
        well_col="Well",
    )


def test_manual_control_conditions_interpolation_without_mutating_measured_dataframe():
    dataframe = _layer_frame()
    original_columns = set(dataframe.columns)
    control = create_control_point(
        x=0.75,
        y=0.5,
        property_name="Pressure",
        value=4200.0,
        property_unit="psi",
        reservoir_layer="Upper",
        panel="A",
        pressure_reference_date=date(2026, 1, 1),
    )

    collection = _generate(dataframe, controls=[control])
    generated = collection.maps["Upper"]

    assert generated["grid_z"][1, 1] == 4200.0
    assert generated["engineering_control_count"] == 1
    assert generated["measured_observation_count"] == len(generated["included_observations"])
    assert set(dataframe.columns) == original_columns
    assert "Control_ID" not in dataframe.columns


def test_control_scoping_prevents_property_layer_panel_and_date_leakage():
    dataframe = _layer_frame()
    controls = [
        create_control_point(x=0.5, y=0.5, property_name="Pressure", value=3100.0, property_unit="psi", reservoir_layer="Upper", panel="A", pressure_reference_date="2026-01-01", control_id="CP-M"),
        create_control_point(x=0.5, y=0.5, property_name="Phi", value=0.2, property_unit="frac", reservoir_layer="Upper", panel="A", pressure_reference_date="2026-01-01", control_id="CP-PROP"),
        create_control_point(x=0.5, y=0.5, property_name="Pressure", value=3100.0, property_unit="psi", reservoir_layer="Lower", panel="A", pressure_reference_date="2026-01-01", control_id="CP-LAYER"),
        create_control_point(x=0.5, y=0.5, property_name="Pressure", value=3100.0, property_unit="psi", reservoir_layer="Upper", panel="B", pressure_reference_date="2026-01-01", control_id="CP-PANEL"),
        create_control_point(x=0.5, y=0.5, property_name="Pressure", value=3100.0, property_unit="psi", reservoir_layer="Upper", panel="A", pressure_reference_date="2025-01-01", control_id="CP-DATE"),
    ]

    selection = engineering_controls_for_context(
        control_points=controls,
        control_regions=[],
        measured_dataframe=dataframe[dataframe["Layer"] == "Upper"],
        mappings=MAPPINGS,
        property_col="Pressure",
        property_type="Pressure",
        property_unit="psi",
        pressure_reference_date=date(2026, 1, 1),
        reservoir_layer="Upper",
        selected_panels=["A"],
        panel_interpolation_mode=PANEL_MODE_COMBINED,
        x_col="X",
        y_col="Y",
        panel_layer=_panel_layer(),
        reservoir_boundary_layer=_reservoir_boundary(),
    )

    assert selection.dataframe["Control_ID"].tolist() == ["CP-M"]


def test_all_layers_receive_only_matching_layer_controls():
    dataframe = _layer_frame()
    controls = [
        create_control_point(x=0.5, y=0.5, property_name="Pressure", value=4200.0, property_unit="psi", reservoir_layer="Upper", panel="A", pressure_reference_date="2026-01-01", control_id="CP-U"),
        create_control_point(x=0.5, y=0.5, property_name="Pressure", value=2100.0, property_unit="psi", reservoir_layer="Lower", panel="A", pressure_reference_date="2026-01-01", control_id="CP-L"),
    ]

    collection = _generate(dataframe, controls=controls, layer_scope=LAYER_SCOPE_ALL, selected_layer=None)

    assert set(collection.maps) == {"Lower", "Upper"}
    assert collection.maps["Upper"]["engineering_controls"]["Control_ID"].tolist() == ["CP-U"]
    assert collection.maps["Lower"]["engineering_controls"]["Control_ID"].tolist() == ["CP-L"]


def test_ordinary_kriging_auto_fit_uses_measured_observations_only():
    dataframe = _layer_frame()
    controls = [
        create_control_point(x=0.25, y=0.75, property_name="Pressure", value=9000.0, property_unit="psi", reservoir_layer="Upper", panel="A", pressure_reference_date="2026-01-01", control_id="CP-HIGH"),
    ]
    params = {
        "variogram_mode": "Auto Fit",
        "variogram_model": "Spherical",
        "variogram_range_convention": VARIOGRAM_RANGE_CONVENTION,
        "range": 1.0,
        "variance": 1.0,
        "nugget": 0.0,
    }
    filtered, measured_prepared = prepare_layer_observations(
        dataframe=dataframe,
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
    expected_parameters = resolve_layer_method_parameters("Ordinary Kriging", params, measured_prepared)

    collection = _generate(dataframe, controls=controls, method="Ordinary Kriging", method_parameters=params)
    generated = collection.maps["Upper"]

    assert generated["engineering_control_count"] == 1
    assert generated["conditioning_observation_count"] == len(measured_prepared) + 1
    assert generated["method_parameters"] == expected_parameters
    assert "CP-HIGH" in generated["model_signature"]["engineering_controls"]["manual_control_points"][0]["id"]
    assert filtered["Pressure"].max() < 9000.0


def test_soft_control_region_generates_points_inside_region_and_splits_by_panel():
    region = create_control_region(
        region_name="Cross-panel support",
        geometry=Polygon([(0, 0), (2, 0), (2, 1), (0, 1)]),
        property_name="Pressure",
        target_value=3150.0,
        property_unit="psi",
        reservoir_layer="Upper",
        pressure_reference_date="2026-01-01",
        control_point_spacing=0.5,
    )

    selection = engineering_controls_for_context(
        control_points=[],
        control_regions=[region],
        measured_dataframe=_layer_frame()[lambda frame: frame["Layer"] == "Upper"],
        mappings=MAPPINGS,
        property_col="Pressure",
        property_type="Pressure",
        property_unit="psi",
        pressure_reference_date=date(2026, 1, 1),
        reservoir_layer="Upper",
        selected_panels=["A", "B"],
        panel_interpolation_mode=PANEL_MODE_INDEPENDENT,
        x_col="X",
        y_col="Y",
        panel_layer=_panel_layer(),
        reservoir_boundary_layer=_reservoir_boundary(),
    )

    panels = set(selection.dataframe["Panel"].dropna().astype(str))
    assert {"A", "B"}.issubset(panels)
    assert len(selection.region_points) == len(selection.dataframe)
    for control in selection.region_points:
        assert region["Geometry"].covers(Point(float(control["X"]), float(control["Y"])))


def test_control_region_from_selected_wells_honors_convex_hull_and_buffer():
    wells = pd.DataFrame({"X": [0.0, 1.0, 0.0], "Y": [0.0, 0.0, 1.0]})

    hull = create_region_from_wells(wells, "X", "Y", buffer_distance=0.25)

    assert hull.area > Polygon([(0, 0), (1, 0), (0, 1)]).area
    assert hull.geom_type == "Polygon"


def test_drawn_control_region_preserves_cartesian_vertices_and_repairs_simple_topology():
    drawn = polygon_from_vertices(
        [
            {"x": 10.0, "y": 20.0},
            {"x": 11.0, "y": 20.0},
            {"x": 11.0, "y": 21.0},
            {"x": 10.0, "y": 21.0},
        ]
    )
    repaired = polygon_from_vertices([(0.0, 0.0), (1.0, 1.0), (0.0, 1.0), (1.0, 0.0)])

    assert drawn.vertices[0] == (10.0, 20.0)
    assert drawn.geometry.area == pytest.approx(1.0)
    assert repaired.geometry.is_valid
    assert repaired.warnings


def test_drawn_control_region_rejects_nonfinite_and_invalid_parameters():
    with pytest.raises(ValueError, match="non-finite"):
        polygon_from_vertices([(0.0, 0.0), (np.nan, 1.0), (1.0, 0.0)])
    with pytest.raises(ValueError, match="Target Value"):
        validate_control_region_parameters(target_value=np.nan, spacing=1.0)
    with pytest.raises(ValueError, match="Pressure Map Reference Date"):
        validate_control_region_parameters(target_value=3000.0, spacing=1.0, property_type="Pressure")


def test_drawn_control_region_clips_outside_reservoir_and_selected_panel_domain():
    drawn = polygon_from_vertices([(-0.5, 0.0), (1.5, 0.0), (1.5, 1.0), (-0.5, 1.0)])

    clipped = clip_control_region_to_active_domain(
        drawn.geometry,
        reservoir_boundary_layer=_reservoir_boundary(),
        panel_layer=_panel_layer(),
        selected_panels=["A"],
    )

    assert clipped.clipped is True
    assert clipped.geometry.bounds == pytest.approx((0.0, 0.0, 1.0, 1.0))
    assert any("outside the reservoir domain" in warning for warning in clipped.warnings)
    assert any("outside the selected panel domain" in warning for warning in clipped.warnings)


def test_base_context_map_shows_raw_measured_points_before_interpolation_with_pressure_hover():
    frame = _layer_frame()[lambda item: item["Layer"] == "Upper"].copy()
    included = frame[frame["Panel"] == "A"].copy()
    excluded = frame[frame["Panel"] == "B"].copy()

    figure = build_context_map_figure(
        included,
        excluded,
        "X",
        "Y",
        "Pressure",
        well_col="Well",
        hover_columns=[("Reservoir Layer", "Layer"), ("Panel", "Panel")],
        unit="psi",
        coordinate_unit="meters",
        is_pressure_map=True,
        map_reference_date=date(2026, 1, 1),
        map_reference_date_col="Map_Reference_Date",
        style={"show_raw_points": True, "well_label_mode": "Well Name"},
    )

    names = [trace.name for trace in figure.data]
    raw_trace = next(trace for trace in figure.data if trace.name == "Raw measured points")
    assert "Raw measured points" in names
    assert "Excluded observations" in names
    assert "Map Reference Date" in raw_trace.hovertext[0]
    assert "Reservoir Layer: Upper" in raw_trace.hovertext[0]
    assert "Panel: A" in raw_trace.hovertext[0]


def test_drawn_control_region_combined_panels_conditions_selected_panel_union():
    region = create_control_region(
        region_name="Drawn union",
        geometry=Polygon([(0, 0), (2, 0), (2, 1), (0, 1)]),
        property_name="Pressure",
        target_value=3150.0,
        property_unit="psi",
        reservoir_layer="Upper",
        pressure_reference_date="2026-01-01",
        control_point_spacing=0.5,
        region_source=DRAWN_CONTROL_REGION_SOURCE,
    )

    selection = engineering_controls_for_context(
        control_points=[],
        control_regions=[region],
        measured_dataframe=_layer_frame()[lambda frame: frame["Layer"] == "Upper"],
        mappings=MAPPINGS,
        property_col="Pressure",
        property_type="Pressure",
        property_unit="psi",
        pressure_reference_date=date(2026, 1, 1),
        reservoir_layer="Upper",
        selected_panels=["A", "B"],
        panel_interpolation_mode=PANEL_MODE_COMBINED,
        x_col="X",
        y_col="Y",
        panel_layer=_panel_layer(),
        reservoir_boundary_layer=_reservoir_boundary(),
    )

    assert {"A", "B"}.issubset(set(selection.dataframe["Panel"].dropna().astype(str)))
    reservoir_geometry = _reservoir_boundary().features[0].geometry
    assert all(
        reservoir_geometry.covers(Point(float(row["X"]), float(row["Y"])))
        for _, row in selection.dataframe.iterrows()
    )


def test_drawn_control_region_layer_and_pressure_date_scope_are_required():
    regions = [
        create_control_region(
            region_name="Good drawn",
            geometry=Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]),
            property_name="Pressure",
            target_value=3100.0,
            property_unit="psi",
            reservoir_layer="Upper",
            pressure_reference_date="2026-01-01",
            control_point_spacing=0.5,
            region_source=DRAWN_CONTROL_REGION_SOURCE,
            region_id="CR-GOOD",
        ),
        create_control_region(
            region_name="Wrong layer",
            geometry=Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]),
            property_name="Pressure",
            target_value=3100.0,
            property_unit="psi",
            reservoir_layer="Lower",
            pressure_reference_date="2026-01-01",
            control_point_spacing=0.5,
            region_source=DRAWN_CONTROL_REGION_SOURCE,
            region_id="CR-LAYER",
        ),
        create_control_region(
            region_name="Wrong date",
            geometry=Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]),
            property_name="Pressure",
            target_value=3100.0,
            property_unit="psi",
            reservoir_layer="Upper",
            pressure_reference_date="2025-01-01",
            control_point_spacing=0.5,
            region_source=DRAWN_CONTROL_REGION_SOURCE,
            region_id="CR-DATE",
        ),
    ]

    selection = engineering_controls_for_context(
        control_points=[],
        control_regions=regions,
        measured_dataframe=_layer_frame()[lambda frame: frame["Layer"] == "Upper"],
        mappings=MAPPINGS,
        property_col="Pressure",
        property_type="Pressure",
        property_unit="psi",
        pressure_reference_date=date(2026, 1, 1),
        reservoir_layer="Upper",
        selected_panels=[],
        panel_interpolation_mode=PANEL_MODE_COMBINED,
        x_col="X",
        y_col="Y",
        panel_layer=None,
        reservoir_boundary_layer=_reservoir_boundary(),
    )

    assert [region["Region_ID"] for region in selection.regions] == ["CR-GOOD"]


@pytest.mark.parametrize(
    ("method", "parameters"),
    [
        ("IDW", {"power": 2.0, "neighbors": 10, "min_neighbors": 1, "search_radius": None}),
        ("RBF", {"kernel": "linear", "smoothing": 0.0, "neighbors": None, "epsilon": None}),
        (
            "Ordinary Kriging",
            {
                "variogram_mode": "Manual",
                "variogram_model": "Spherical",
                "variogram_range_convention": VARIOGRAM_RANGE_CONVENTION,
                "range": 2.0,
                "variance": 10000.0,
                "nugget": 0.0,
                "max_neighbors": None,
                "search_radius": None,
            },
        ),
    ],
)
def test_drawn_control_region_influences_idw_rbf_and_ordinary_kriging(method, parameters):
    region = create_control_region(
        region_name="High support",
        geometry=Polygon([(0.5, 0.25), (1.0, 0.25), (1.0, 0.75), (0.5, 0.75)]),
        property_name="Pressure",
        target_value=4200.0,
        property_unit="psi",
        reservoir_layer="Upper",
        pressure_reference_date="2026-01-01",
        control_point_spacing=0.5,
        region_source=DRAWN_CONTROL_REGION_SOURCE,
    )

    baseline = _generate(_layer_frame(), method=method, method_parameters=parameters).maps["Upper"]["grid_z"][1, 1]
    controlled = _generate(_layer_frame(), regions=[region], method=method, method_parameters=parameters).maps["Upper"]["grid_z"][1, 1]

    assert np.isfinite(controlled)
    assert abs(float(controlled) - float(baseline)) > 1e-6


def test_control_conflict_with_measured_observation_warns_and_uses_measured_precedence():
    dataframe = _layer_frame()
    control = create_control_point(
        x=0.0,
        y=0.0,
        property_name="Pressure",
        value=9999.0,
        property_unit="psi",
        reservoir_layer="Upper",
        panel="A",
        pressure_reference_date="2026-01-01",
        control_id="CP-CONFLICT",
    )

    selection = engineering_controls_for_context(
        control_points=[control],
        control_regions=[],
        measured_dataframe=dataframe[dataframe["Layer"] == "Upper"],
        mappings=MAPPINGS,
        property_col="Pressure",
        property_type="Pressure",
        property_unit="psi",
        pressure_reference_date=date(2026, 1, 1),
        reservoir_layer="Upper",
        selected_panels=["A"],
        panel_interpolation_mode=PANEL_MODE_COMBINED,
        x_col="X",
        y_col="Y",
        panel_layer=_panel_layer(),
        reservoir_boundary_layer=_reservoir_boundary(),
    )

    assert selection.dataframe.empty
    assert any("measured observation kept" in warning for warning in selection.warnings)
    assert selection.signature_state["excluded_control_points"][0]["id"] == "CP-CONFLICT"


def test_overlapping_soft_regions_with_different_targets_raise_qc_warning():
    regions = [
        create_control_region(
            region_name="R1",
            geometry=Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]),
            property_name="Pressure",
            target_value=3000.0,
            property_unit="psi",
            reservoir_layer="Upper",
            pressure_reference_date="2026-01-01",
            control_point_spacing=1.0,
            region_id="CR-001",
        ),
        create_control_region(
            region_name="R2",
            geometry=Polygon([(0.5, 0), (1.5, 0), (1.5, 1), (0.5, 1)]),
            property_name="Pressure",
            target_value=3500.0,
            property_unit="psi",
            reservoir_layer="Upper",
            pressure_reference_date="2026-01-01",
            control_point_spacing=1.0,
            region_id="CR-002",
        ),
    ]

    selection = engineering_controls_for_context(
        control_points=[],
        control_regions=regions,
        measured_dataframe=_layer_frame()[lambda frame: frame["Layer"] == "Upper"],
        mappings=MAPPINGS,
        property_col="Pressure",
        property_type="Pressure",
        property_unit="psi",
        pressure_reference_date=date(2026, 1, 1),
        reservoir_layer="Upper",
        selected_panels=[],
        panel_interpolation_mode=PANEL_MODE_COMBINED,
        x_col="X",
        y_col="Y",
        panel_layer=None,
        reservoir_boundary_layer=_reservoir_boundary(),
    )

    assert any("Overlapping control regions" in warning for warning in selection.warnings)


def test_validation_uses_controls_as_conditioning_points_not_targets_or_metrics():
    measured = pd.DataFrame({"X": [0.0, 10.0, 20.0], "Y": [0.0, 0.0, 0.0], "Z": [0.0, 0.0, 0.0]})
    controls = pd.DataFrame({"X": [1.0], "Y": [0.0], "Z": [100.0]})

    without_controls, _ = leave_one_out_cross_validation(
        measured,
        "IDW",
        {"power": 2.0, "neighbors": 3, "min_neighbors": 1},
    )
    with_controls, metrics = leave_one_out_cross_validation(
        measured,
        "IDW",
        {"power": 2.0, "neighbors": 3, "min_neighbors": 1},
        conditioning_points=controls,
    )

    assert metrics["Count"] == 3
    assert with_controls.loc[0, "Predicted"] > without_controls.loc[0, "Predicted"]
    assert len(with_controls) == len(measured)


def test_control_state_changes_layer_model_signature():
    dataframe = _layer_frame()
    filtered, prepared = prepare_layer_observations(
        dataframe=dataframe,
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
    params = {"power": 2.0, "neighbors": 10, "min_neighbors": 1, "search_radius": None}
    control = create_control_point(
        x=0.5,
        y=0.5,
        property_name="Pressure",
        value=4200.0,
        property_unit="psi",
        reservoir_layer="Upper",
        panel="A",
        pressure_reference_date="2026-01-01",
    )
    control_selection = engineering_controls_for_context(
        control_points=[control],
        control_regions=[],
        measured_dataframe=filtered,
        mappings=MAPPINGS,
        property_col="Pressure",
        property_type="Pressure",
        property_unit="psi",
        pressure_reference_date=date(2026, 1, 1),
        reservoir_layer="Upper",
        selected_panels=[],
        panel_interpolation_mode=PANEL_MODE_COMBINED,
        x_col="X",
        y_col="Y",
    )

    baseline = build_layer_model_signature(
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
        interpolation_parameters=params,
        grid_parameters=GRID,
        interpolation_domain="Well Data Extent",
        domain_bounds=None,
        mask_parameters=MASK,
    )
    with_control = build_layer_model_signature(
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
        interpolation_parameters=params,
        grid_parameters=GRID,
        interpolation_domain="Well Data Extent",
        domain_bounds=None,
        mask_parameters=MASK,
        control_state=control_selection.signature_state,
    )

    assert not prepared.empty
    assert baseline["hash"] != with_control["hash"]


def test_project_and_scenario_persistence_restore_control_definitions():
    control = create_control_point(
        x=0.5,
        y=0.5,
        property_name="Pressure",
        value=4200.0,
        property_unit="psi",
        reservoir_layer="Upper",
        panel="A",
        pressure_reference_date="2026-01-01",
        active=False,
        comment="shut-in boundary support",
    )
    region = create_control_region(
        region_name="Support region",
        geometry=Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]),
        property_name="Pressure",
        target_value=3500.0,
        property_unit="psi",
        reservoir_layer="Upper",
        panel="A",
        pressure_reference_date="2026-01-01",
        control_point_spacing=0.25,
        region_source=DRAWN_CONTROL_REGION_SOURCE,
    )
    generated = _generate(_layer_frame(), controls=[{**control, "Active": True}], regions=[region]).maps["Upper"]
    scenario = create_map_scenario("Controlled Upper", generated)
    generated["engineering_control_points"][0]["Value"] = 1.0
    generated["engineering_control_regions"][0]["Target_Value"] = 1.0

    opened = scenario_to_generated_map(scenario)
    state = {
        "project_metadata": {"name": "Controls"},
        "source_name": "controls.csv",
        "column_mappings": MAPPINGS,
        "coordinate_unit": "meters",
        "property_unit": "psi",
        "engineering_control_points": [control],
        "engineering_control_regions": [region],
        "geometry_layers": {"reservoir_boundary": None, "panels": None, "faults": None, "custom": []},
        "map_scenarios": [scenario],
        "working_df": _layer_frame(),
    }

    restored = load_project_archive(save_project_archive(state))

    assert opened["engineering_control_points"][0]["Value"] == 4200.0
    assert opened["engineering_control_regions"][0]["Target_Value"] == 3500.0
    assert opened["engineering_control_regions"][0]["Region_Source"] == DRAWN_CONTROL_REGION_SOURCE
    assert restored["engineering_control_points"][0]["Active"] is False
    assert restored["engineering_control_regions"][0]["Geometry"].area == region["Geometry"].area
    assert restored["engineering_control_regions"][0]["Region_Source"] == DRAWN_CONTROL_REGION_SOURCE
    assert len(restored["map_scenarios"][0]["engineering_control_points"]) == 1
    assert restored["map_scenarios"][0]["control_region_count"] == 1
    assert restored["map_scenarios"][0]["engineering_control_count"] == len(scenario["engineering_controls"])
