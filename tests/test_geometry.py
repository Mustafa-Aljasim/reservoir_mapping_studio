import json

import numpy as np
import pandas as pd
from shapely.geometry import Polygon

from core.geometry.assignment import assign_points_to_polygons, outside_panel_count
from core.geometry.compartment import compartment_interpolate
from core.geometry.loader import load_geojson_bytes
from core.geometry.masking import polygon_keep_mask
from core.geometry.models import GeometryFeature, GeometryLayer
from core.geometry.validation import detect_polygon_overlaps, validate_geometry_layer


def _two_panel_layer() -> GeometryLayer:
    return GeometryLayer(
        name="Panels",
        layer_type="Panel / Compartment",
        features=[
            GeometryFeature(Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]), "Panel / Compartment", "A"),
            GeometryFeature(Polygon([(1, 0), (2, 0), (2, 1), (1, 1)]), "Panel / Compartment", "B"),
        ],
    )


def test_polygon_mask_keeps_inside_and_masks_outside():
    layer = GeometryLayer(
        name="Boundary",
        layer_type="Reservoir Boundary",
        features=[GeometryFeature(Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]), "Reservoir Boundary", "Boundary")],
    )
    grid_x = np.array([[0.5, 1.5]])
    grid_y = np.array([[0.5, 0.5]])
    keep = polygon_keep_mask(layer.features, grid_x, grid_y)
    assert keep[0, 0]
    assert not keep[0, 1]


def test_point_assignment_and_outside_flag():
    df = pd.DataFrame({"X": [0.5, 1.5, 3.0], "Y": [0.5, 0.5, 0.5], "Panel": ["A", "B", "A"]})
    assignments = assign_points_to_polygons(df, "X", "Y", _two_panel_layer(), "Panel")
    assert assignments.loc[0, "Assignment_Status"] == "Match"
    assert assignments.loc[1, "Assignment_Status"] == "Match"
    assert assignments.loc[2, "Assignment_Status"] == "Outside"
    assert outside_panel_count(assignments) == 1


def test_overlapping_polygons_are_detected():
    layer = GeometryLayer(
        name="Overlap",
        layer_type="Panel / Compartment",
        features=[
            GeometryFeature(Polygon([(0, 0), (2, 0), (2, 1), (0, 1)]), "Panel / Compartment", "A"),
            GeometryFeature(Polygon([(1, 0), (3, 0), (3, 1), (1, 1)]), "Panel / Compartment", "B"),
        ],
    )
    overlaps = detect_polygon_overlaps(layer)
    assert len(overlaps) == 1
    assert overlaps[0].area > 0


def test_geojson_loader_and_validation():
    payload = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {"Panel": "A"},
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[(0, 0), (1, 0), (1, 1), (0, 1), (0, 0)]],
                },
            }
        ],
    }
    layer = load_geojson_bytes(json.dumps(payload).encode("utf-8"), "Panel / Compartment", "Panels", "Panel")
    validated, report = validate_geometry_layer(layer)
    assert validated.feature_count == 1
    assert validated.features[0].name == "A"
    assert report.valid_features == 1


def test_compartment_interpolation_uses_own_observations():
    prepared = pd.DataFrame(
        {
            "X": [0.2, 0.8, 0.5, 1.2, 1.8, 1.5],
            "Y": [0.2, 0.2, 0.8, 0.2, 0.2, 0.8],
            "Z": [3300, 3310, 3290, 2700, 2710, 2690],
        }
    )
    grid_x = np.array([[0.5, 1.5]])
    grid_y = np.array([[0.5, 0.5]])
    result = compartment_interpolate(
        prepared,
        _two_panel_layer(),
        grid_x,
        grid_y,
        "IDW",
        {"power": 2, "neighbors": 3, "min_neighbors": 1, "search_radius": None},
    )
    assert result.surface[0, 0] > 3200
    assert result.surface[0, 1] < 2800
    assert result.panel_grid[0, 0] == "A"
    assert result.panel_grid[0, 1] == "B"

