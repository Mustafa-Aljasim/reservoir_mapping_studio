"""Geometry-aware masking helpers."""

from __future__ import annotations

import numpy as np
from shapely.geometry import Point
from shapely.ops import unary_union

from core.geometry.models import GeometryFeature, GeometryLayer


def _vectorized_covers(geometry, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    try:
        from shapely import covers, points

        return np.asarray(covers(geometry, points(xs, ys)), dtype=bool)
    except Exception:  # pragma: no cover - compatibility fallback
        return np.asarray([geometry.covers(Point(x, y)) for x, y in zip(xs, ys)], dtype=bool)


def polygon_union(features: list[GeometryFeature]):
    polygons = [feature.geometry for feature in features if feature.geometry.geom_type in {"Polygon", "MultiPolygon"}]
    if not polygons:
        return None
    return unary_union(polygons)


def polygon_keep_mask(features: list[GeometryFeature], grid_x: np.ndarray, grid_y: np.ndarray) -> np.ndarray:
    """Return True for grid cells covered by at least one polygon feature."""

    geometry = polygon_union(features)
    if geometry is None or geometry.is_empty:
        return np.ones_like(grid_x, dtype=bool)
    keep = _vectorized_covers(geometry, grid_x.ravel(), grid_y.ravel())
    return keep.reshape(grid_x.shape)


def layer_keep_mask(layer: GeometryLayer | None, grid_x: np.ndarray, grid_y: np.ndarray) -> np.ndarray:
    if layer is None or not layer.polygon_features:
        return np.ones_like(grid_x, dtype=bool)
    return polygon_keep_mask(layer.polygon_features, grid_x, grid_y)
