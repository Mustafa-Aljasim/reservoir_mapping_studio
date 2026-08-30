import numpy as np

from core.grid import coordinate_limits, generate_grid


def test_grid_dimensions_are_correct():
    grid_x, grid_y = generate_grid([0, 10], [100, 200], nx=5, ny=4, buffer_fraction=0)
    assert grid_x.shape == (4, 5)
    assert grid_y.shape == (4, 5)


def test_grid_coordinate_limits_match_extent_without_buffer():
    grid_x, grid_y = generate_grid([0, 10], [100, 200], nx=5, ny=4, buffer_fraction=0)
    assert np.isclose(grid_x.min(), 0)
    assert np.isclose(grid_x.max(), 10)
    assert np.isclose(grid_y.min(), 100)
    assert np.isclose(grid_y.max(), 200)


def test_coordinate_limits_expand_with_fractional_buffer():
    low, high = coordinate_limits([0, 100], buffer_fraction=0.05)
    assert np.isclose(low, -5)
    assert np.isclose(high, 105)

