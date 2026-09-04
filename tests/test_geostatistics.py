import numpy as np
import pandas as pd
import pytest

from core.geostatistics.kriging import ordinary_kriging_interpolate
from core.geostatistics.validation import leave_one_out_cross_validation, validation_metrics
from core.geostatistics.variogram import (
    VARIOGRAM_RANGE_CONVENTION,
    compute_experimental_variogram,
    evaluate_variogram_model,
    fit_candidate_models,
    gstools_len_scale_from_practical_range,
)


def _sample_points():
    x = np.array([0.0, 1.0, 0.0, 1.0, 0.5, 1.5])
    y = np.array([0.0, 0.0, 1.0, 1.0, 0.5, 1.5])
    z = np.array([10.0, 12.0, 13.0, 15.0, 12.0, 18.0])
    return x, y, z


def test_experimental_variogram_and_candidate_fit():
    x, y, z = _sample_points()
    experimental = compute_experimental_variogram(x, y, z, n_lags=5)
    fits = fit_candidate_models(experimental)
    assert experimental.lag_distance.shape == (5,)
    assert experimental.pair_count.sum() > 0
    assert fits
    assert fits[0].range_value > 0


def test_constant_property_variogram_is_invalid():
    x, y, _ = _sample_points()
    try:
        compute_experimental_variogram(x, y, np.ones_like(x), n_lags=5)
    except ValueError as exc:
        assert "identical" in str(exc)
    else:
        raise AssertionError("Constant property variogram should fail.")


def test_ordinary_kriging_returns_estimate_and_uncertainty():
    x, y, z = _sample_points()
    grid_x, grid_y = np.meshgrid(np.linspace(0, 1, 3), np.linspace(0, 1, 3))
    result = ordinary_kriging_interpolate(
        x,
        y,
        z,
        grid_x,
        grid_y,
        {"variogram_model": "Spherical", "range": 1.5, "variance": 8.0, "nugget": 0.0},
    )
    assert result.estimate.shape == grid_x.shape
    assert result.variance.shape == grid_x.shape
    assert np.isfinite(result.estimate).any()
    assert np.nanmin(result.variance) >= 0


@pytest.mark.parametrize(
    ("model_name", "class_name"),
    [
        ("Spherical", "Spherical"),
        ("Exponential", "Exponential"),
        ("Gaussian", "Gaussian"),
    ],
)
def test_user_facing_practical_range_matches_gstools_variogram(model_name, class_name):
    gs = pytest.importorskip("gstools")
    practical_range = 10.0
    variance = 4.0
    nugget = 1.0
    distances = np.array([0.0, 2.5, 5.0, 10.0, 15.0], dtype=float)

    len_scale = gstools_len_scale_from_practical_range(model_name, practical_range)
    gstools_model = getattr(gs, class_name)(dim=2, var=variance, len_scale=len_scale, nugget=nugget)
    displayed = evaluate_variogram_model(model_name, distances, practical_range, variance, nugget)

    assert VARIOGRAM_RANGE_CONVENTION == "Practical Range"
    assert np.allclose(displayed, gstools_model.variogram(distances), rtol=1e-8, atol=1e-10)


def test_auto_fit_range_parameters_are_kriging_ready_under_same_convention():
    gs = pytest.importorskip("gstools")
    x, y, z = _sample_points()
    experimental = compute_experimental_variogram(x, y, z, n_lags=5)
    fit = fit_candidate_models(experimental)[0]
    distances = experimental.lag_distance[np.isfinite(experimental.semivariance)]

    len_scale = gstools_len_scale_from_practical_range(fit.model, fit.range_value)
    gstools_model = getattr(gs, fit.model.replace(" ", ""))(
        dim=2,
        var=fit.variance,
        len_scale=len_scale,
        nugget=fit.nugget,
    )

    assert np.allclose(
        evaluate_variogram_model(fit.model, distances, fit.range_value, fit.variance, fit.nugget),
        gstools_model.variogram(distances),
        rtol=1e-8,
        atol=1e-10,
    )


def test_validation_metrics_and_residual_sign():
    results = pd.DataFrame({"Observed": [10.0, 12.0], "Predicted": [9.0, 14.0]})
    results["Residual"] = results["Observed"] - results["Predicted"]
    results["Absolute_Error"] = results["Residual"].abs()
    results["Squared_Error"] = results["Residual"] ** 2
    metrics = validation_metrics(results)
    assert metrics["Bias"] == -0.5
    assert np.isclose(metrics["RMSE"], np.sqrt(2.5))


def test_loocv_excludes_target_observation_for_idw():
    prepared = pd.DataFrame({"X": [0.0, 10.0, 20.0], "Y": [0.0, 0.0, 0.0], "Z": [100.0, 0.0, 0.0]})
    results, metrics = leave_one_out_cross_validation(
        prepared,
        "IDW",
        {"power": 2.0, "neighbors": 2, "min_neighbors": 1},
    )
    assert results.loc[0, "Predicted"] < 100.0
    assert results.loc[0, "Residual"] == results.loc[0, "Observed"] - results.loc[0, "Predicted"]
    assert metrics["Count"] == 3


def test_compartment_aware_loocv_uses_only_same_panel_and_flags_outside():
    prepared = pd.DataFrame(
        {
            "X": [0.0, 1.0, 0.5, 10.0, 11.0, 10.5, 25.0],
            "Y": [0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0],
            "Z": [3300.0, 3310.0, 3290.0, 2700.0, 2710.0, 2690.0, 3000.0],
        }
    )
    panels = ["A", "A", "A", "B", "B", "B", None]
    results, _ = leave_one_out_cross_validation(
        prepared,
        "IDW",
        {"power": 2.0, "neighbors": 3, "min_neighbors": 1},
        panels=panels,
        respect_compartments=True,
    )
    assert results.loc[0, "Predicted"] > 3200.0
    assert results.loc[3, "Predicted"] < 2800.0
    assert np.isnan(results.loc[6, "Predicted"])
