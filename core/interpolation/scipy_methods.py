"""SciPy griddata interpolation methods."""

from __future__ import annotations

import warnings

import numpy as np
from scipy.interpolate import griddata
from scipy.spatial import QhullError

from utils.validators import non_collinear_points


def _clean_xyz(x, y, z) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x_array = np.asarray(x, dtype=float)
    y_array = np.asarray(y, dtype=float)
    z_array = np.asarray(z, dtype=float)
    mask = np.isfinite(x_array) & np.isfinite(y_array) & np.isfinite(z_array)
    return x_array[mask], y_array[mask], z_array[mask]


def griddata_interpolate(
    x,
    y,
    z,
    grid_x: np.ndarray,
    grid_y: np.ndarray,
    method: str = "linear",
) -> np.ndarray:
    """Interpolate with scipy.interpolate.griddata using linear or cubic mode."""

    if method not in {"linear", "cubic"}:
        raise ValueError("SciPy griddata method must be 'linear' or 'cubic'.")
    x_array, y_array, z_array = _clean_xyz(x, y, z)
    if len(z_array) < 3:
        raise ValueError(f"At least 3 finite observations are required for {method.title()} interpolation.")
    if not non_collinear_points(x_array, y_array):
        raise ValueError(f"At least 3 non-collinear observations are required for {method.title()} interpolation.")

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            grid_z = griddata(
                np.column_stack([x_array, y_array]),
                z_array,
                (grid_x, grid_y),
                method=method,
            )
    except QhullError as exc:
        raise ValueError(
            f"{method.title()} interpolation could not triangulate the observations. "
            "Check for duplicate or collinear coordinates."
        ) from exc
    return np.asarray(grid_z, dtype=float)

