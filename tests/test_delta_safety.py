import numpy as np
import pytest

from core.map_comparison import calculate_delta


def _scenario(name, grid_x, grid_y, grid_z):
    return {
        "name": name,
        "property": "Pressure",
        "property_unit": "psi",
        "coordinate_unit": "m",
        "crs": {"mode": "Local / Unknown XY"},
        "grid_x": np.asarray(grid_x, dtype=float),
        "grid_y": np.asarray(grid_y, dtype=float),
        "grid_z": np.asarray(grid_z, dtype=float),
        "pressure_reference_date": "2025-01-01",
        "selected_panels": ["A"],
        "panel_interpolation_mode": "Combined Selected Panels",
        "selected_layers": ["L1"],
    }


def _mesh(start, stop, count):
    axis = np.linspace(start, stop, count)
    return np.meshgrid(axis, axis)


def test_delta_identical_grids_uses_only_both_finite_cells():
    grid_x, grid_y = _mesh(0.0, 1.0, 2)
    map_a = _scenario("A", grid_x, grid_y, [[1.0, np.nan], [3.0, 4.0]])
    map_b = _scenario("B", grid_x, grid_y, [[2.0, 5.0], [np.nan, 8.0]])

    delta = calculate_delta(map_a, map_b)

    assert np.isfinite(delta.grid_z).sum() == 2
    assert delta.grid_z[0, 0] == 1.0
    assert delta.grid_z[1, 1] == 4.0
    assert np.isnan(delta.grid_z[0, 1])
    assert np.isnan(delta.grid_z[1, 0])
    assert delta.metadata["Alignment_Resampling_Method"] == "none"


def test_delta_different_grid_resolution_linear_resamples_without_nearest_fill():
    a_x, a_y = _mesh(0.0, 2.0, 3)
    b_x, b_y = _mesh(0.0, 2.0, 5)
    map_a = _scenario("A", a_x, a_y, a_x + a_y)
    map_b = _scenario("B", b_x, b_y, 2.0 * b_x + 3.0 * b_y)

    delta = calculate_delta(map_a, map_b)

    assert np.isfinite(delta.grid_z).sum() == 9
    assert np.allclose(delta.grid_z, a_x + 2.0 * a_y)
    assert delta.metadata["Alignment_Resampling_Method"] == "linear"


def test_delta_partial_overlap_keeps_only_intersection_support():
    a_x, a_y = _mesh(0.0, 2.0, 3)
    b_x, b_y = _mesh(1.0, 3.0, 3)
    map_a = _scenario("A", a_x, a_y, np.ones_like(a_x))
    map_b = _scenario("B", b_x, b_y, np.full_like(b_x, 5.0))

    delta = calculate_delta(map_a, map_b)

    expected_support = (a_x >= 1.0) & (a_y >= 1.0)
    assert np.array_equal(np.isfinite(delta.grid_z), expected_support)
    assert np.all(delta.grid_z[expected_support] == 4.0)


def test_delta_preserves_internal_nan_hole_after_alignment():
    a_x, a_y = _mesh(0.0, 4.0, 5)
    b_x, b_y = _mesh(0.0, 4.0, 3)
    b_z = np.full_like(b_x, 10.0)
    b_z[1, 1] = np.nan
    map_a = _scenario("A", a_x, a_y, np.ones_like(a_x))
    map_b = _scenario("B", b_x, b_y, b_z)

    delta = calculate_delta(map_a, map_b)

    assert np.isnan(delta.grid_z[2, 2])
    assert np.isfinite(delta.grid_z).sum() < delta.grid_z.size
    assert "both finite" in delta.metadata["Mask_Intersection_Rule"]


def test_delta_raises_when_grids_have_no_overlapping_valid_support():
    a_x, a_y = _mesh(0.0, 1.0, 2)
    b_x, b_y = _mesh(10.0, 11.0, 2)
    map_a = _scenario("A", a_x, a_y, np.ones_like(a_x))
    map_b = _scenario("B", b_x, b_y, np.ones_like(b_x))

    with pytest.raises(ValueError, match="no overlapping valid support"):
        calculate_delta(map_a, map_b)
