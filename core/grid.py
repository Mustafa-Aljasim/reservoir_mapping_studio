"""Grid generation utilities."""

from __future__ import annotations

import numpy as np


def _valid_array(values) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    return array[np.isfinite(array)]


def coordinate_limits(values, buffer_fraction: float = 0.03) -> tuple[float, float]:
    """Return min/max coordinate limits expanded by a fractional buffer."""

    valid = _valid_array(values)
    if valid.size == 0:
        raise ValueError("At least one valid coordinate is required to build a grid.")

    low = float(np.min(valid))
    high = float(np.max(valid))
    span = high - low
    if span == 0:
        fallback = max(abs(low) * 0.01, 1.0)
        low -= fallback
        high += fallback
        span = high - low
    buffer = max(float(buffer_fraction), 0.0) * span
    return low - buffer, high + buffer


def generate_grid(
    x,
    y,
    nx: int = 150,
    ny: int = 150,
    buffer_fraction: float = 0.03,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate a Cartesian mesh grid over the observation extent."""

    if nx < 2 or ny < 2:
        raise ValueError("Grid dimensions must be at least 2 by 2.")

    x_min, x_max = coordinate_limits(x, buffer_fraction)
    y_min, y_max = coordinate_limits(y, buffer_fraction)
    grid_x, grid_y = np.meshgrid(
        np.linspace(x_min, x_max, int(nx)),
        np.linspace(y_min, y_max, int(ny)),
    )
    return grid_x, grid_y

