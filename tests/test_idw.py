import numpy as np

from core.interpolation.idw import idw_interpolate


def test_exact_value_returned_at_observation_location():
    grid_x = np.array([[1.0]])
    grid_y = np.array([[1.0]])
    result = idw_interpolate(
        [0.0, 1.0, 2.0],
        [0.0, 1.0, 0.0],
        [10.0, 25.0, 30.0],
        grid_x,
        grid_y,
        min_neighbors=1,
    )
    assert result[0, 0] == 25.0


def test_different_power_values_change_weighting():
    grid_x = np.array([[2.0]])
    grid_y = np.array([[0.0]])
    low_power = idw_interpolate(
        [0.0, 10.0, 10.0],
        [0.0, 0.0, 5.0],
        [0.0, 10.0, 10.0],
        grid_x,
        grid_y,
        power=1.0,
        min_neighbors=1,
    )
    high_power = idw_interpolate(
        [0.0, 10.0, 10.0],
        [0.0, 0.0, 5.0],
        [0.0, 10.0, 10.0],
        grid_x,
        grid_y,
        power=4.0,
        min_neighbors=1,
    )
    assert high_power[0, 0] < low_power[0, 0]


def test_nearest_neighbor_selection_uses_requested_neighbor_count():
    grid_x = np.array([[1.0]])
    grid_y = np.array([[0.0]])
    result = idw_interpolate(
        [0.0, 10.0, 20.0],
        [0.0, 0.0, 0.0],
        [100.0, 0.0, 0.0],
        grid_x,
        grid_y,
        neighbors=1,
        min_neighbors=1,
    )
    assert np.isclose(result[0, 0], 100.0)


def test_minimum_neighbor_rule_returns_nan():
    grid_x = np.array([[1.0]])
    grid_y = np.array([[1.0]])
    result = idw_interpolate(
        [0.0, 2.0],
        [0.0, 2.0],
        [10.0, 20.0],
        grid_x,
        grid_y,
        neighbors=2,
        min_neighbors=3,
    )
    assert np.isnan(result[0, 0])


def test_nan_input_values_are_ignored():
    grid_x = np.array([[2.0]])
    grid_y = np.array([[0.0]])
    result = idw_interpolate(
        [0.0, 1.0, 2.0],
        [0.0, 0.0, 0.0],
        [10.0, np.nan, 30.0],
        grid_x,
        grid_y,
        min_neighbors=1,
    )
    assert result[0, 0] == 30.0

