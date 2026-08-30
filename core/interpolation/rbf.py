"""Radial Basis Function interpolation."""

from __future__ import annotations

import numpy as np
from scipy.interpolate import RBFInterpolator
from scipy.spatial import cKDTree

from utils.validators import non_collinear_points

RBF_KERNELS_REQUIRING_EPSILON = {"multiquadric", "inverse_multiquadric", "inverse_quadratic", "gaussian"}


def _clean_xyz(x, y, z) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x_array = np.asarray(x, dtype=float)
    y_array = np.asarray(y, dtype=float)
    z_array = np.asarray(z, dtype=float)
    mask = np.isfinite(x_array) & np.isfinite(y_array) & np.isfinite(z_array)
    return x_array[mask], y_array[mask], z_array[mask]


def estimate_epsilon(x, y) -> float:
    points = np.unique(np.column_stack([np.asarray(x, dtype=float), np.asarray(y, dtype=float)]), axis=0)
    points = points[np.isfinite(points).all(axis=1)]
    if len(points) < 2:
        return 1.0
    tree = cKDTree(points)
    distances, _ = tree.query(points, k=min(2, len(points)))
    nearest = distances[:, 1] if distances.ndim == 2 else distances
    nearest = nearest[np.isfinite(nearest) & (nearest > 0)]
    if nearest.size == 0:
        return 1.0
    return float(np.median(nearest))


def rbf_interpolate(
    x,
    y,
    z,
    grid_x: np.ndarray,
    grid_y: np.ndarray,
    kernel: str = "thin_plate_spline",
    smoothing: float = 0.0,
    neighbors: int | None = None,
    epsilon: float | None = None,
) -> np.ndarray:
    """Interpolate Z values with scipy.interpolate.RBFInterpolator."""

    x_array, y_array, z_array = _clean_xyz(x, y, z)
    if len(z_array) < 3:
        raise ValueError("At least 3 finite observations are required for RBF interpolation.")
    if kernel in {"thin_plate_spline", "cubic"} and not non_collinear_points(x_array, y_array):
        raise ValueError(f"RBF kernel '{kernel}' requires non-collinear observations.")
    if smoothing < 0:
        raise ValueError("RBF smoothing must be zero or greater.")

    points = np.column_stack([x_array, y_array])
    query_points = np.column_stack([grid_x.ravel(), grid_y.ravel()])
    if neighbors is not None:
        neighbors = int(neighbors)
        if neighbors < 1:
            raise ValueError("RBF neighbors must be at least 1 when supplied.")
        neighbors = min(neighbors, len(z_array))

    if kernel in RBF_KERNELS_REQUIRING_EPSILON and (epsilon is None or float(epsilon) <= 0):
        epsilon = estimate_epsilon(x_array, y_array)

    try:
        interpolator = RBFInterpolator(
            points,
            z_array,
            kernel=kernel,
            smoothing=float(smoothing),
            neighbors=neighbors,
            epsilon=epsilon,
        )
        grid_z = interpolator(query_points).reshape(grid_x.shape)
    except Exception as exc:
        raise ValueError(
            "RBF interpolation failed with the selected parameters. "
            "Try increasing smoothing, changing the kernel, or limiting neighbors."
        ) from exc
    return np.asarray(grid_z, dtype=float)

