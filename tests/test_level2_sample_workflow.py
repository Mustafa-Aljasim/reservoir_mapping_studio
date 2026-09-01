from pathlib import Path

import numpy as np
import pandas as pd
from shapely.geometry import Point
from shapely.geometry import MultiPolygon, Polygon

from core.data_qc import prepare_interpolation_dataframe
from core.geometry.assignment import assign_points_to_polygons, outside_panel_count
from core.geometry.compartment import compartment_interpolate
from core.geometry.loader import load_geojson_bytes
from core.geometry.masking import layer_keep_mask
from core.geometry.models import GeometryFeature, GeometryLayer
from core.geometry.validation import validate_geometry_layer
from core.grid import generate_grid
from core.interpolation import interpolate_surface_result
from core.masking import apply_keep_mask
from utils.validators import non_collinear_points


ROOT = Path(__file__).resolve().parents[1]


def _load_layer(filename: str, layer_type: str, name_attribute: str | None = None):
    return load_geojson_bytes(
        (ROOT / "data" / filename).read_bytes(),
        layer_type,
        layer_type,
        name_attribute,
        filename,
    )


def test_level2_sample_geometry_and_interpolation_workflow():
    df = pd.read_csv(ROOT / "data" / "sample_reservoir_data.csv")
    boundary, boundary_report = validate_geometry_layer(
        _load_layer("sample_reservoir_boundary.geojson", "Reservoir Boundary", "Name"),
        (
            float(df["EASTING"].min()),
            float(df["NORTHING"].min()),
            float(df["EASTING"].max()),
            float(df["NORTHING"].max()),
        ),
    )
    panels, panel_report = validate_geometry_layer(_load_layer("sample_panels.geojson", "Panel / Compartment", "Panel"))
    faults, fault_report = validate_geometry_layer(_load_layer("sample_faults.geojson", "Fault", "Fault_Name"))

    assert boundary_report.valid_features == 1
    assert panel_report.valid_features == 3
    assert fault_report.valid_features == 3

    assignments = assign_points_to_polygons(df, "EASTING", "NORTHING", panels, "Panel")
    assert outside_panel_count(assignments) == 2
    assert int((assignments["Assignment_Status"] == "Match").sum()) >= 39

    prepared = prepare_interpolation_dataframe(df, "EASTING", "NORTHING", "Pressure", duplicate_method="Average")
    assert len(prepared) == 38
    grid_x, grid_y = generate_grid(prepared["X"], prepared["Y"], nx=14, ny=14, buffer_fraction=0.08)

    method_params = {
        "IDW": {"power": 2.0, "neighbors": 8, "min_neighbors": 1, "search_radius": None},
        "Linear": {},
        "Cubic": {},
        "RBF": {"kernel": "linear", "smoothing": 0.0, "neighbors": None, "epsilon": None},
        "Ordinary Kriging": {"variogram_model": "Spherical", "range": 5000.0, "variance": 60000.0, "nugget": 100.0},
    }
    for method, params in method_params.items():
        result = interpolate_surface_result(prepared["X"], prepared["Y"], prepared["Z"], grid_x, grid_y, method, params)
        assert result.estimate.shape == grid_x.shape
        assert np.isfinite(result.estimate).any()
        if method == "Ordinary Kriging":
            assert result.variance is not None
            assert result.variance.shape == grid_x.shape

    keep = layer_keep_mask(boundary, grid_x, grid_y)
    assert not keep.all()
    masked = apply_keep_mask(np.ones_like(grid_x, dtype=float), keep)
    assert np.isnan(masked[~keep]).all()

    panel_labels = []
    for _, row in prepared.iterrows():
        point = Point(float(row["X"]), float(row["Y"]))
        matches = [feature.name for feature in panels.polygon_features if feature.geometry.covers(point)]
        panel_labels.append(matches[0] if len(matches) == 1 else None)
    for panel in ("North", "Central", "South"):
        panel_points = prepared[[label == panel for label in panel_labels]]
        assert non_collinear_points(panel_points["X"], panel_points["Y"])

    idw_compartment = compartment_interpolate(
        prepared,
        panels,
        grid_x,
        grid_y,
        "IDW",
        {"power": 2.0, "neighbors": 8, "min_neighbors": 1, "search_radius": None},
    )
    north_mean = float(np.nanmean(idw_compartment.surface[idw_compartment.panel_grid == "North"]))
    central_mean = float(np.nanmean(idw_compartment.surface[idw_compartment.panel_grid == "Central"]))
    south_mean = float(np.nanmean(idw_compartment.surface[idw_compartment.panel_grid == "South"]))
    assert north_mean > central_mean > south_mean

    kriging_compartment = compartment_interpolate(
        prepared,
        panels,
        grid_x,
        grid_y,
        "Ordinary Kriging",
        {"variogram_model": "Spherical", "range": 5000.0, "variance": 60000.0, "nugget": 100.0},
    )
    assert kriging_compartment.variance is not None
    assert np.isfinite(kriging_compartment.surface).any()


def test_reservoir_domain_interpolates_outside_well_extent_and_masks_multipolygon():
    observations = pd.DataFrame(
        {"X": [4.0, 6.0, 4.0, 6.0, 5.0], "Y": [4.0, 4.0, 6.0, 6.0, 5.0], "Z": [10.0, 12.0, 14.0, 16.0, 13.0]}
    )
    boundary = GeometryLayer(
        "Boundary",
        "Reservoir Boundary",
        [
            GeometryFeature(
                MultiPolygon(
                    [
                        Polygon([(0, 0), (10, 0), (5, 10)]),
                        Polygon([(12, 0), (14, 0), (14, 2), (12, 2)]),
                    ]
                ),
                "Reservoir Boundary",
                "Boundary",
            )
        ],
    )
    grid_x, grid_y = generate_grid(
        observations["X"], observations["Y"], nx=11, ny=11, buffer_fraction=0, bounds=(0, 0, 10, 10)
    )
    assert (grid_x.min(), grid_y.min(), grid_x.max(), grid_y.max()) == (0, 0, 10, 10)

    for method, parameters in (
        ("RBF", {"kernel": "thin_plate_spline", "smoothing": 0.0, "neighbors": None}),
        ("IDW", {"power": 2.0, "neighbors": 5, "min_neighbors": 1, "search_radius": None}),
        ("Ordinary Kriging", {"variogram_model": "Spherical", "range": 10.0, "variance": 10.0, "nugget": 0.0}),
    ):
        result = interpolate_surface_result(observations["X"], observations["Y"], observations["Z"], grid_x, grid_y, method, parameters)
        outside_wells = (grid_x < 4) | (grid_x > 6) | (grid_y < 4) | (grid_y > 6)
        assert np.isfinite(result.estimate[outside_wells]).any()
        if method == "Ordinary Kriging":
            assert result.variance is not None
            assert np.isfinite(result.variance[outside_wells]).any()

    keep = layer_keep_mask(boundary, grid_x, grid_y)
    assert keep[5, 5]
    assert not keep[10, 10]
    assert layer_keep_mask(boundary, np.array([[13.0]]), np.array([[1.0]])).item()
