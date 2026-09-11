"""Ordinary Kriging using GSTools."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree

from core.geostatistics.variogram import gstools_len_scale_from_practical_range
from utils.validators import non_collinear_points


@dataclass(frozen=True)
class KrigingResult:
    estimate: np.ndarray
    variance: np.ndarray

    @property
    def stddev(self) -> np.ndarray:
        return np.sqrt(np.maximum(self.variance, 0.0))


def _clean_xyz(x, y, z) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x_array = np.asarray(x, dtype=float)
    y_array = np.asarray(y, dtype=float)
    z_array = np.asarray(z, dtype=float)
    mask = np.isfinite(x_array) & np.isfinite(y_array) & np.isfinite(z_array)
    return x_array[mask], y_array[mask], z_array[mask]


def _gstools_model(
    model_name: str,
    range_value: float,
    variance: float,
    nugget: float,
    anisotropy_enabled: bool = False,
    anisotropy_angle: float = 0.0,
    anisotropy_ratio: float = 1.0,
):
    import gstools as gs

    models = {
        "Spherical": gs.Spherical,
        "Exponential": gs.Exponential,
        "Gaussian": gs.Gaussian,
    }
    model_class = models.get(model_name)
    if model_class is None:
        raise ValueError(f"Unsupported variogram model for kriging: {model_name}")
    kwargs = {
        "dim": 2,
        "var": float(max(variance, 1e-12)),
        "len_scale": gstools_len_scale_from_practical_range(model_name, float(max(range_value, 1e-12))),
        "nugget": float(max(nugget, 0.0)),
    }
    if anisotropy_enabled:
        kwargs["anis"] = float(max(min(anisotropy_ratio, 1.0), 1e-6))
        kwargs["angles"] = float(np.deg2rad(anisotropy_angle))
    return model_class(**kwargs)


def _krige_points(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    query_x: np.ndarray,
    query_y: np.ndarray,
    parameters: dict,
    *,
    universal: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    model = _gstools_model(
        parameters.get("variogram_model", "Spherical"),
        float(parameters.get("range", parameters.get("range_value", 1.0))),
        float(parameters.get("variance", parameters.get("sill", np.nanvar(z) or 1.0))),
        float(parameters.get("nugget", 0.0)),
        bool(parameters.get("anisotropy_enabled", False)),
        float(parameters.get("anisotropy_angle", 0.0)),
        float(parameters.get("anisotropy_ratio", 1.0)),
    )
    import gstools as gs

    if universal:
        trend_model = str(parameters.get("trend_model", "Linear XY"))
        if trend_model == "Quadratic XY":
            drift_functions = "quadratic"
        else:
            drift_functions = "linear"
        ok = gs.krige.Universal(model, cond_pos=[x, y], cond_val=z, drift_functions=drift_functions)
    else:
        ok = gs.krige.Ordinary(model, cond_pos=[x, y], cond_val=z)
    estimate, variance = ok((query_x, query_y), mesh_type="unstructured", return_var=True)
    return np.asarray(estimate, dtype=float), np.asarray(variance, dtype=float)


def _kriging_interpolate(
    x,
    y,
    z,
    grid_x: np.ndarray,
    grid_y: np.ndarray,
    parameters: dict | None = None,
    *,
    universal: bool = False,
) -> KrigingResult:
    parameters = parameters or {}
    x_array, y_array, z_array = _clean_xyz(x, y, z)
    if len(z_array) < 3:
        name = "Universal Kriging" if universal else "Ordinary Kriging"
        raise ValueError(f"At least 3 finite observations are required for {name}.")
    if not non_collinear_points(x_array, y_array):
        name = "Universal Kriging" if universal else "Ordinary Kriging"
        raise ValueError(f"At least 3 non-collinear observations are required for {name}.")
    if np.nanvar(z_array) == 0:
        name = "Universal Kriging" if universal else "Ordinary Kriging"
        raise ValueError(f"All active property values are identical. {name} cannot estimate a meaningful variogram.")

    query = np.column_stack([grid_x.ravel(), grid_y.ravel()])
    max_neighbors = parameters.get("max_neighbors")
    search_radius = parameters.get("search_radius")
    use_local = (max_neighbors is not None and int(max_neighbors) < len(z_array)) or search_radius is not None

    if not use_local:
        estimate, variance = _krige_points(
            x_array,
            y_array,
            z_array,
            query[:, 0],
            query[:, 1],
            parameters,
            universal=universal,
        )
        return KrigingResult(estimate.reshape(grid_x.shape), variance.reshape(grid_x.shape))

    tree = cKDTree(np.column_stack([x_array, y_array]))
    estimates = np.full(len(query), np.nan, dtype=float)
    variances = np.full(len(query), np.nan, dtype=float)
    max_neighbors = int(max_neighbors) if max_neighbors is not None else len(z_array)
    max_neighbors = max(3, min(max_neighbors, len(z_array)))
    upper_bound = float(search_radius) if search_radius is not None else np.inf
    for index, point in enumerate(query):
        distances, indices = tree.query(point, k=max_neighbors, distance_upper_bound=upper_bound)
        indices = np.atleast_1d(indices)
        valid_indices = indices[np.isfinite(np.atleast_1d(distances)) & (indices < len(z_array))]
        if len(valid_indices) < 3:
            continue
        try:
            estimate, variance = _krige_points(
                x_array[valid_indices],
                y_array[valid_indices],
                z_array[valid_indices],
                np.asarray([point[0]]),
                np.asarray([point[1]]),
                parameters,
                universal=universal,
            )
        except ValueError:
            continue
        estimates[index] = estimate[0]
        variances[index] = variance[0]
    return KrigingResult(estimates.reshape(grid_x.shape), variances.reshape(grid_x.shape))


def ordinary_kriging_interpolate(
    x,
    y,
    z,
    grid_x: np.ndarray,
    grid_y: np.ndarray,
    parameters: dict | None = None,
) -> KrigingResult:
    """Return Ordinary Kriging estimate and variance surfaces."""

    return _kriging_interpolate(x, y, z, grid_x, grid_y, parameters, universal=False)


def universal_kriging_interpolate(
    x,
    y,
    z,
    grid_x: np.ndarray,
    grid_y: np.ndarray,
    parameters: dict | None = None,
) -> KrigingResult:
    """Return Universal Kriging estimate and variance surfaces with XY drift."""

    return _kriging_interpolate(x, y, z, grid_x, grid_y, parameters, universal=True)


def kriging_predict_point(x, y, z, point_x: float, point_y: float, parameters: dict | None = None) -> tuple[float, float]:
    grid_x = np.asarray([[point_x]], dtype=float)
    grid_y = np.asarray([[point_y]], dtype=float)
    result = ordinary_kriging_interpolate(x, y, z, grid_x, grid_y, parameters or {})
    return float(result.estimate[0, 0]), float(result.variance[0, 0])
