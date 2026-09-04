"""Experimental variogram and simple model fitting."""

from __future__ import annotations

from dataclasses import dataclass
import warnings

import numpy as np
from scipy.optimize import OptimizeWarning, curve_fit
from scipy.spatial.distance import pdist


VARIOGRAM_MODELS = ("Spherical", "Exponential", "Gaussian")
VARIOGRAM_RANGE_CONVENTION = "Practical Range"
PRACTICAL_RANGE_EXPONENT = 3.0


@dataclass(frozen=True)
class ExperimentalVariogram:
    lag_distance: np.ndarray
    semivariance: np.ndarray
    pair_count: np.ndarray
    max_lag: float
    n_lags: int


@dataclass(frozen=True)
class VariogramFit:
    model: str
    range_value: float
    variance: float
    nugget: float
    fit_error: float


def _clean_xyz(x, y, z) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x_array = np.asarray(x, dtype=float)
    y_array = np.asarray(y, dtype=float)
    z_array = np.asarray(z, dtype=float)
    mask = np.isfinite(x_array) & np.isfinite(y_array) & np.isfinite(z_array)
    return x_array[mask], y_array[mask], z_array[mask]


def _semivariances(z: np.ndarray) -> np.ndarray:
    differences = pdist(z[:, np.newaxis], metric="euclidean")
    return 0.5 * differences**2


def compute_experimental_variogram(
    x,
    y,
    z,
    n_lags: int = 12,
    max_lag: float | None = None,
) -> ExperimentalVariogram:
    x_array, y_array, z_array = _clean_xyz(x, y, z)
    if len(z_array) < 3:
        raise ValueError("At least 3 finite observations are required for an experimental variogram.")
    if np.nanvar(z_array) == 0:
        raise ValueError("All active property values are identical. A geostatistical variogram cannot be meaningfully fitted.")
    if n_lags < 2:
        raise ValueError("Number of variogram lags must be at least 2.")

    points = np.column_stack([x_array, y_array])
    distances = pdist(points, metric="euclidean")
    semivariances = _semivariances(z_array)
    finite = np.isfinite(distances) & np.isfinite(semivariances) & (distances > 0)
    distances = distances[finite]
    semivariances = semivariances[finite]
    if distances.size == 0:
        raise ValueError("No finite pair distances are available for variogram calculation.")

    resolved_max_lag = float(max_lag) if max_lag and max_lag > 0 else float(np.nanmax(distances) * 0.75)
    bin_edges = np.linspace(0.0, resolved_max_lag, int(n_lags) + 1)
    lag_distance: list[float] = []
    semivariance: list[float] = []
    pair_count: list[int] = []
    for start, end in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (distances > start) & (distances <= end)
        lag_distance.append(float((start + end) / 2.0))
        pair_count.append(int(mask.sum()))
        semivariance.append(float(np.nanmean(semivariances[mask])) if mask.any() else np.nan)

    return ExperimentalVariogram(
        lag_distance=np.asarray(lag_distance, dtype=float),
        semivariance=np.asarray(semivariance, dtype=float),
        pair_count=np.asarray(pair_count, dtype=int),
        max_lag=resolved_max_lag,
        n_lags=int(n_lags),
    )


def spherical_model(h, range_value, variance, nugget):
    h = np.asarray(h, dtype=float)
    hr = np.divide(h, range_value, out=np.zeros_like(h), where=range_value != 0)
    gamma = np.where(hr < 1.0, nugget + variance * (1.5 * hr - 0.5 * hr**3), nugget + variance)
    return gamma


def exponential_model(h, range_value, variance, nugget):
    """Exponential semivariogram using user-facing practical range.

    Practical range is the distance where this model reaches approximately
    95 percent of the partial sill.
    """

    h = np.asarray(h, dtype=float)
    return nugget + variance * (1.0 - np.exp(-PRACTICAL_RANGE_EXPONENT * h / max(range_value, 1e-12)))


def gaussian_model(h, range_value, variance, nugget):
    """Gaussian semivariogram using user-facing practical range.

    Practical range is the distance where this model reaches approximately
    95 percent of the partial sill.
    """

    h = np.asarray(h, dtype=float)
    return nugget + variance * (
        1.0 - np.exp(-PRACTICAL_RANGE_EXPONENT * (h / max(range_value, 1e-12)) ** 2)
    )


MODEL_FUNCTIONS = {
    "Spherical": spherical_model,
    "Exponential": exponential_model,
    "Gaussian": gaussian_model,
}


def evaluate_variogram_model(model: str, lag_distance, range_value: float, variance: float, nugget: float) -> np.ndarray:
    if model not in MODEL_FUNCTIONS:
        raise ValueError(f"Unsupported variogram model: {model}")
    return MODEL_FUNCTIONS[model](lag_distance, range_value, variance, nugget)


def gstools_len_scale_from_practical_range(model: str, practical_range: float) -> float:
    """Convert user-facing practical range to GSTools ``len_scale``.

    V1 uses practical range in the UI, fitted parameters, saved scenarios, and
    exports. GSTools expects model-specific length scales, so Ordinary Kriging
    must convert explicitly before constructing the covariance model.
    """

    value = float(max(practical_range, 1e-12))
    if model == "Spherical":
        return value
    if model == "Exponential":
        return value / PRACTICAL_RANGE_EXPONENT
    if model == "Gaussian":
        return value * float(np.sqrt(np.pi / (4.0 * PRACTICAL_RANGE_EXPONENT)))
    raise ValueError(f"Unsupported variogram model: {model}")


def practical_range_from_gstools_len_scale(model: str, len_scale: float) -> float:
    value = float(max(len_scale, 1e-12))
    if model == "Spherical":
        return value
    if model == "Exponential":
        return value * PRACTICAL_RANGE_EXPONENT
    if model == "Gaussian":
        return value * float(np.sqrt((4.0 * PRACTICAL_RANGE_EXPONENT) / np.pi))
    raise ValueError(f"Unsupported variogram model: {model}")


def fit_variogram_model(experimental: ExperimentalVariogram, model: str) -> VariogramFit:
    if model not in MODEL_FUNCTIONS:
        raise ValueError(f"Unsupported variogram model: {model}")
    valid = np.isfinite(experimental.lag_distance) & np.isfinite(experimental.semivariance) & (experimental.pair_count > 0)
    h = experimental.lag_distance[valid]
    gamma = experimental.semivariance[valid]
    counts = experimental.pair_count[valid]
    if len(h) < 3:
        raise ValueError("At least 3 populated lag bins are required to fit a variogram model.")

    initial_range = max(float(experimental.max_lag) * 0.6, 1e-6)
    initial_variance = max(float(np.nanmax(gamma) - np.nanmin(gamma)), 1e-6)
    initial_nugget = max(float(np.nanmin(gamma)) * 0.25, 0.0)
    sigma = 1.0 / np.sqrt(np.maximum(counts, 1))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", OptimizeWarning)
        popt, _ = curve_fit(
            MODEL_FUNCTIONS[model],
            h,
            gamma,
            p0=(initial_range, initial_variance, initial_nugget),
            bounds=([1e-12, 1e-12, 0.0], [np.inf, np.inf, np.inf]),
            sigma=sigma,
            maxfev=10000,
        )
    fitted = MODEL_FUNCTIONS[model](h, *popt)
    denom = float(np.nanvar(gamma)) if float(np.nanvar(gamma)) > 0 else 1.0
    fit_error = float(np.nanmean((fitted - gamma) ** 2) / denom)
    return VariogramFit(
        model=model,
        range_value=float(popt[0]),
        variance=float(popt[1]),
        nugget=float(popt[2]),
        fit_error=fit_error,
    )


def fit_candidate_models(experimental: ExperimentalVariogram) -> list[VariogramFit]:
    fits: list[VariogramFit] = []
    for model in VARIOGRAM_MODELS:
        try:
            fits.append(fit_variogram_model(experimental, model))
        except ValueError:
            continue
    return sorted(fits, key=lambda fit: fit.fit_error)


def semivariance_unit(property_unit: str | None) -> str:
    return f"{property_unit}^2" if property_unit else "property unit^2"
