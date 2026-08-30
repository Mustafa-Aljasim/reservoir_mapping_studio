"""Inverse Distance Weighting interpolation."""

from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree


def _clean_xyz(x, y, z) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x_array = np.asarray(x, dtype=float)
    y_array = np.asarray(y, dtype=float)
    z_array = np.asarray(z, dtype=float)
    mask = np.isfinite(x_array) & np.isfinite(y_array) & np.isfinite(z_array)
    return x_array[mask], y_array[mask], z_array[mask]


def idw_interpolate(
    x,
    y,
    z,
    grid_x: np.ndarray,
    grid_y: np.ndarray,
    power: float = 2.0,
    neighbors: int = 12,
    search_radius: float | None = None,
    min_neighbors: int = 3,
) -> np.ndarray:
    """Interpolate Z values to grid cells using cKDTree-backed IDW."""

    x_array, y_array, z_array = _clean_xyz(x, y, z)
    if len(z_array) == 0:
        raise ValueError("At least one finite observation is required for IDW interpolation.")
    if power <= 0:
        raise ValueError("IDW power must be greater than zero.")
    if neighbors < 1:
        raise ValueError("Number of IDW neighbors must be at least 1.")
    if min_neighbors < 1:
        raise ValueError("Minimum neighbors must be at least 1.")

    k = min(int(neighbors), len(z_array))
    tree = cKDTree(np.column_stack([x_array, y_array]))
    query_points = np.column_stack([grid_x.ravel(), grid_y.ravel()])
    upper_bound = np.inf if search_radius is None else float(search_radius)
    if upper_bound <= 0:
        raise ValueError("Search radius must be greater than zero when supplied.")

    distances, indices = tree.query(query_points, k=k, distance_upper_bound=upper_bound)
    if k == 1:
        distances = distances[:, np.newaxis]
        indices = indices[:, np.newaxis]

    valid_neighbor = np.isfinite(distances) & (indices < len(z_array))
    result = np.full(len(query_points), np.nan, dtype=float)

    exact_rows = np.any(valid_neighbor & (distances == 0), axis=1)
    if np.any(exact_rows):
        exact_indices = np.argmax(valid_neighbor[exact_rows] & (distances[exact_rows] == 0), axis=1)
        source_indices = indices[exact_rows, exact_indices]
        result[exact_rows] = z_array[source_indices]

    remaining = ~exact_rows
    if np.any(remaining):
        row_valid = valid_neighbor[remaining]
        row_distances = distances[remaining]
        row_indices = indices[remaining]
        valid_counts = row_valid.sum(axis=1)
        enough = valid_counts >= min_neighbors

        if np.any(enough):
            selected_distances = row_distances[enough]
            selected_indices = row_indices[enough]
            selected_valid = row_valid[enough]

            weights = np.zeros_like(selected_distances, dtype=float)
            weights[selected_valid] = 1.0 / np.power(selected_distances[selected_valid], power)
            weighted_values = np.zeros_like(selected_distances, dtype=float)
            weighted_values[selected_valid] = z_array[selected_indices[selected_valid]]

            denominator = weights.sum(axis=1)
            interpolated = np.full(len(denominator), np.nan, dtype=float)
            nonzero = denominator > 0
            interpolated[nonzero] = (weights[nonzero] * weighted_values[nonzero]).sum(axis=1) / denominator[nonzero]

            remaining_indices = np.where(remaining)[0]
            result[remaining_indices[enough]] = interpolated

    return result.reshape(grid_x.shape)

