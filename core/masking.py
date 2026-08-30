"""Masking utilities for interpolated surfaces."""

from __future__ import annotations

import numpy as np
from scipy.spatial import Delaunay, QhullError, cKDTree


def valid_points(x, y) -> np.ndarray:
    points = np.column_stack([np.asarray(x, dtype=float), np.asarray(y, dtype=float)])
    return points[np.isfinite(points).all(axis=1)]


def convex_hull_keep_mask(x, y, grid_x: np.ndarray, grid_y: np.ndarray) -> np.ndarray:
    """Return True for grid cells inside the observation convex hull."""

    points = np.unique(valid_points(x, y), axis=0)
    if len(points) < 3:
        return np.ones_like(grid_x, dtype=bool)
    query = np.column_stack([grid_x.ravel(), grid_y.ravel()])
    try:
        hull = Delaunay(points)
        keep = hull.find_simplex(query) >= 0
    except QhullError:
        keep = np.ones(len(query), dtype=bool)
    return keep.reshape(grid_x.shape)


def nearest_distance_grid(x, y, grid_x: np.ndarray, grid_y: np.ndarray) -> np.ndarray:
    points = valid_points(x, y)
    if len(points) == 0:
        return np.full_like(grid_x, np.nan, dtype=float)
    tree = cKDTree(points)
    distances, _ = tree.query(np.column_stack([grid_x.ravel(), grid_y.ravel()]), k=1)
    return distances.reshape(grid_x.shape)


def maximum_distance_keep_mask(
    x,
    y,
    grid_x: np.ndarray,
    grid_y: np.ndarray,
    max_distance: float,
) -> np.ndarray:
    distances = nearest_distance_grid(x, y, grid_x, grid_y)
    return distances <= float(max_distance)


def auto_maximum_distance(x, y, multiplier: float = 2.5) -> float:
    points = np.unique(valid_points(x, y), axis=0)
    if len(points) < 2:
        return 0.0
    tree = cKDTree(points)
    distances, _ = tree.query(points, k=min(2, len(points)))
    nearest = distances[:, 1] if distances.ndim == 2 else distances
    nearest = nearest[np.isfinite(nearest) & (nearest > 0)]
    if nearest.size == 0:
        return 0.0
    return float(np.mean(nearest) * multiplier)


def apply_keep_mask(grid_z: np.ndarray, keep_mask: np.ndarray) -> np.ndarray:
    masked = np.array(grid_z, dtype=float, copy=True)
    masked[~keep_mask] = np.nan
    return masked


def combine_masks(*masks: np.ndarray | None) -> np.ndarray:
    valid_masks = [mask for mask in masks if mask is not None]
    if not valid_masks:
        raise ValueError("At least one mask is required.")
    combined = np.ones_like(valid_masks[0], dtype=bool)
    for mask in valid_masks:
        combined &= mask
    return combined

