from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from core.geostatistics.kriging import universal_kriging_interpolate
from core.interpolation import interpolate_surface_result
from core.plotting.map_builder import build_map_figure, map_figure_for_static_export
from core.scenarios import create_map_scenario, safe_scenario_to_generated_map, scenario_display_label


def _grid(n: int = 5):
    return np.meshgrid(np.linspace(0.0, 1.0, n), np.linspace(0.0, 1.0, n))


def _observations(x, y, z) -> pd.DataFrame:
    return pd.DataFrame({"X": x, "Y": y, "Z": z, "Well": [f"W{i}" for i in range(len(z))], "Include": True})


def _generated_map(title: str, value: float, method: str = "IDW") -> dict[str, object]:
    grid_x, grid_y = _grid(2)
    observations = _observations([0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [value, value, value])
    return {
        "grid_x": grid_x,
        "grid_y": grid_y,
        "grid_z": np.full(grid_x.shape, value, dtype=float),
        "grid_variance": np.full(grid_x.shape, 2.0, dtype=float) if "Kriging" in method else None,
        "included_observations": observations,
        "excluded_observations": observations.iloc[0:0].copy(),
        "property_col": "Pressure",
        "property_type": "Pressure",
        "unit": "psi",
        "x_col": "X",
        "y_col": "Y",
        "well_col": "Well",
        "coordinate_unit": "meters",
        "crs": {"mode": "Local / Unknown XY"},
        "is_pressure_map": True,
        "map_reference_date": "2025-01-01",
        "method": method,
        "method_parameters": {"power": 2.0, "neighbors": 3, "min_neighbors": 1} if method == "IDW" else {},
        "grid_parameters": {"nx": 2, "ny": 2, "buffer_fraction": 0.0},
        "mask_parameters": {"mode": "No Mask"},
        "mask_info": {"mask_mode": "No Mask", "valid_grid_cells": 4},
        "duplicate_method": "Average",
        "title": title,
        "model_signature": {"hash": title},
    }


def test_scenario_snapshot_switching_uses_ids_and_never_recomputes(monkeypatch):
    def fail_if_interpolated(*args, **kwargs):
        raise AssertionError("scenario display must not interpolate")

    monkeypatch.setattr("core.interpolation.interpolate_surface_result", fail_if_interpolated)
    scenario_a = create_map_scenario("Pressure 2025", _generated_map("A", 3000.0, "IDW"))
    scenario_b = create_map_scenario("Pressure 2026", _generated_map("B", 2800.0, "Ordinary Kriging"))
    scenario_c = create_map_scenario("PHI Upper", {**_generated_map("C", 0.22, "RBF"), "property_col": "PHI", "unit": "fraction", "is_pressure_map": False, "reservoir_layer": "Upper"})

    sequence = [scenario_a, scenario_b, scenario_a, None, scenario_b, scenario_c]
    workspace = _generated_map("Current Workspace", 2500.0, "IDW")
    for scenario in sequence:
        generated, errors = (workspace, []) if scenario is None else safe_scenario_to_generated_map(scenario)
        assert not errors
        assert generated is not None
        if scenario is not None:
            assert np.array_equal(generated["grid_z"], scenario["grid_z"])
            assert scenario_display_label(scenario)


def test_contour_lines_off_preserves_filled_surface_and_hides_labels():
    grid_x, grid_y = _grid(4)
    observations = pd.DataFrame({"X": [0.0, 1.0, 0.0], "Y": [0.0, 0.0, 1.0], "Pressure": [10.0, 20.0, 30.0]})
    figure = build_map_figure(
        grid_x,
        grid_y,
        grid_x + grid_y,
        observations,
        observations.iloc[0:0].copy(),
        "X",
        "Y",
        "Pressure",
        style={"show_surface": True, "show_contour_lines": False, "show_contour_labels": True},
    )
    surface = figure.data[0]
    assert surface.visible is True
    assert surface.contours.showlines is False
    assert surface.contours.showlabels is False
    assert surface.line.width == 0.0

    with_lines = build_map_figure(
        grid_x,
        grid_y,
        grid_x + grid_y,
        observations,
        observations.iloc[0:0].copy(),
        "X",
        "Y",
        "Pressure",
        style={"show_surface": True, "show_contour_lines": True, "show_contour_labels": True, "contour_line_width": 1.25},
    )
    assert with_lines.data[0].contours.showlines is True
    assert with_lines.data[0].contours.showlabels is True
    assert with_lines.data[0].line.width == 1.25


def test_static_export_can_remove_raw_points_and_controls():
    grid_x, grid_y = _grid(4)
    observations = pd.DataFrame({"X": [0.0, 1.0, 0.0], "Y": [0.0, 0.0, 1.0], "Pressure": [10.0, 20.0, 30.0]})
    figure = build_map_figure(
        grid_x,
        grid_y,
        grid_x + grid_y,
        observations,
        observations.iloc[0:0].copy(),
        "X",
        "Y",
        "Pressure",
        style={"show_surface": True, "show_wells": True},
    )
    figure.add_trace(go.Scatter(x=[0.5], y=[0.5], mode="markers", name="Engineering controls"))
    figure.add_trace(go.Scatter(x=[0.25], y=[0.25], mode="markers", name="Region controls"))
    figure.add_trace(go.Scatter(x=[0.2, 0.8], y=[0.2, 0.2], mode="lines", name="Control regions"))

    export_figure = map_figure_for_static_export(
        figure,
        include_raw_points=False,
        include_excluded_observations=False,
        include_engineering_controls=False,
        include_control_regions=False,
    )

    names = {trace.name for trace in export_figure.data}
    assert "Pressure" in names
    assert "Raw measured points" not in names
    assert "Engineering controls" not in names
    assert "Region controls" not in names
    assert "Control regions" not in names


def test_saved_scenario_string_reference_date_builds_hover_text_without_crashing():
    grid_x, grid_y = _grid(3)
    observations = pd.DataFrame(
        {
            "X": [0.0, 1.0, 0.0],
            "Y": [0.0, 0.0, 1.0],
            "Pressure": [3000.0, 3020.0, 3010.0],
            "Measurement_Date": ["2025-03-15", "2025-04-01", "2025-05-01"],
            "Map_Reference_Date": ["2026-01-01", "2026-01-01", "2026-01-01"],
        }
    )
    figure = build_map_figure(
        grid_x,
        grid_y,
        grid_x + grid_y,
        observations,
        observations.iloc[0:0].copy(),
        "X",
        "Y",
        "Pressure",
        is_pressure_map=True,
        map_reference_date="2026-01-01",
        measurement_date_col="Measurement_Date",
        map_reference_date_col="Map_Reference_Date",
    )
    assert "Original Measurement Age" in figure.data[1].hovertext[0]


def test_natural_neighbor_is_exact_at_conditioning_points_and_conservative_outside_support():
    grid_x, grid_y = np.meshgrid(np.array([-0.25, 0.0, 0.5, 1.0, 1.25]), np.array([0.0, 0.5, 1.0]))
    x = np.array([0.0, 1.0, 0.0, 1.0, 0.5])
    y = np.array([0.0, 0.0, 1.0, 1.0, 0.5])
    z = x + 2.0 * y
    result = interpolate_surface_result(x, y, z, grid_x, grid_y, "Natural Neighbor", {})
    assert result.estimate[0, 1] == z[0]
    assert result.estimate[0, 3] == z[1]
    assert np.isfinite(result.estimate[1, 2])
    assert np.isnan(result.estimate[0, 0])
    assert np.isnan(result.estimate[0, 4])


def test_minimum_curvature_honors_smooth_conditioning_points():
    grid_x, grid_y = _grid(3)
    x = np.array([0.0, 1.0, 0.0, 1.0, 0.5])
    y = np.array([0.0, 0.0, 1.0, 1.0, 0.5])
    z = 10.0 + 2.0 * x + 3.0 * y
    result = interpolate_surface_result(x, y, z, grid_x, grid_y, "Minimum Curvature", {"smoothing": 0.0})
    assert np.isfinite(result.estimate).all()
    assert np.isclose(result.estimate[0, 0], 10.0)
    assert np.isclose(result.estimate[-1, -1], 15.0)
    assert "minimum-bending" in result.metadata["approach"]


def test_convergent_interpolation_reduces_conditioning_rmse_and_reports_metadata():
    grid_x, grid_y = _grid(6)
    x = np.array([0.08, 0.92, 0.15, 0.85, 0.5])
    y = np.array([0.12, 0.18, 0.88, 0.82, 0.47])
    z = np.array([100.0, 130.0, 170.0, 210.0, 155.0])
    result = interpolate_surface_result(
        x,
        y,
        z,
        grid_x,
        grid_y,
        "Convergent Interpolation",
        {"max_iterations": 8, "convergence_tolerance": 0.01, "relaxation_factor": 0.7, "neighbors": 4},
    )
    metadata = result.metadata
    assert metadata["iterations"] >= 1
    assert metadata["final_rmse"] <= metadata["initial_rmse"]
    assert len(metadata["rmse_history"]) == metadata["iterations"] + 1


def test_universal_kriging_handles_xy_trend_and_returns_uncertainty():
    x = np.array([0.0, 1.0, 0.0, 1.0, 0.5, 1.5, 1.5])
    y = np.array([0.0, 0.0, 1.0, 1.0, 0.5, 0.2, 1.3])
    z = 100.0 + 20.0 * x - 10.0 * y + np.array([0.0, 1.0, -1.0, 0.5, 0.0, -0.5, 1.5])
    grid_x, grid_y = _grid(4)
    result = universal_kriging_interpolate(
        x,
        y,
        z,
        grid_x,
        grid_y,
        {"trend_model": "Linear XY", "variogram_model": "Spherical", "range": 2.0, "variance": 20.0, "nugget": 0.0},
    )
    assert result.estimate.shape == grid_x.shape
    assert result.variance.shape == grid_x.shape
    assert np.isfinite(result.estimate).any()
    assert np.nanmin(result.variance) >= 0.0
