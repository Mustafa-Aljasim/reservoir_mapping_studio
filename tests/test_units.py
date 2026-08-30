import numpy as np

from core.map_context import build_map_metadata
from core.masking import auto_maximum_distance
from utils.units import coordinate_unit_symbol, distance_metadata, format_distance


def test_coordinate_unit_symbols_are_resolved():
    assert coordinate_unit_symbol("meters") == "m"
    assert coordinate_unit_symbol("feet") == "ft"


def test_format_distance_includes_unit_without_changing_value():
    assert format_distance(2500, "meters") == "2,500 m"
    assert format_distance(2500, "feet") == "2,500 ft"


def test_distance_metadata_carries_value_and_unit_separately():
    metadata = distance_metadata(1800, "feet", "Maximum_Distance")
    assert metadata["Maximum_Distance_Value"] == 1800.0
    assert metadata["Maximum_Distance_Unit"] == "ft"


def test_auto_max_distance_is_numeric_and_unit_independent():
    x = [0.0, 100.0, 200.0]
    y = [0.0, 0.0, 0.0]
    meters_value = auto_maximum_distance(x, y)
    feet_label = format_distance(meters_value, "feet")
    assert isinstance(meters_value, float)
    assert np.isclose(meters_value, 250.0)
    assert feet_label == "250 ft"


def test_export_metadata_includes_distance_units():
    metadata = build_map_metadata(
        "Pressure",
        "psi",
        "EASTING",
        "NORTHING",
        "meters",
        "IDW",
        {"nx": 150, "ny": 150, "buffer_fraction": 0.03},
        {"power": 2, "neighbors": 12, "search_radius": 2500.0, "epsilon": 750.0},
        {"mode": "Maximum Distance", "max_distance": 1800.0},
        "Average",
    )
    assert metadata["Search_Radius_Value"] == 2500.0
    assert metadata["Search_Radius_Unit"] == "m"
    assert metadata["RBF_Epsilon_Value"] == 750.0
    assert metadata["RBF_Epsilon_Unit"] == "m"
    assert metadata["Maximum_Distance_Value"] == 1800.0
    assert metadata["Maximum_Distance_Unit"] == "m"
