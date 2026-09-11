"""Interpolation dispatch layer for deterministic surface generation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from core.geostatistics.kriging import ordinary_kriging_interpolate, universal_kriging_interpolate
from core.interpolation.advanced import (
    convergent_interpolate,
    minimum_curvature_interpolate,
    natural_neighbor_interpolate,
)
from core.interpolation.idw import idw_interpolate
from core.interpolation.rbf import rbf_interpolate
from core.interpolation.scipy_methods import griddata_interpolate


class InterpolationError(RuntimeError):
    """Raised when an interpolation method cannot generate a surface."""


@dataclass(frozen=True)
class SurfaceResult:
    estimate: np.ndarray
    variance: np.ndarray | None = None
    metadata: dict[str, object] | None = None

    @property
    def stddev(self) -> np.ndarray | None:
        if self.variance is None:
            return None
        return np.sqrt(np.maximum(self.variance, 0.0))


def interpolate_surface(
    x,
    y,
    z,
    grid_x: np.ndarray,
    grid_y: np.ndarray,
    method: str,
    method_parameters: dict | None = None,
) -> np.ndarray:
    """Universal X/Y/Z interpolation entrypoint.

    The method names are intentionally simple so Level 2 geostatistical methods
    can be added without redesigning the Streamlit workflow.
    """

    return interpolate_surface_result(x, y, z, grid_x, grid_y, method, method_parameters).estimate


def interpolate_surface_result(
    x,
    y,
    z,
    grid_x: np.ndarray,
    grid_y: np.ndarray,
    method: str,
    method_parameters: dict | None = None,
) -> SurfaceResult:
    parameters = method_parameters or {}
    method_key = method.strip().lower().replace(" ", "_")
    try:
        if method_key == "idw":
            return SurfaceResult(
                idw_interpolate(
                    x,
                    y,
                    z,
                    grid_x,
                    grid_y,
                    power=float(parameters.get("power", 2.0)),
                    neighbors=int(parameters.get("neighbors", 12)),
                    search_radius=parameters.get("search_radius"),
                    min_neighbors=int(parameters.get("min_neighbors", 3)),
                )
            )
        if method_key in {"linear", "cubic"}:
            return SurfaceResult(griddata_interpolate(x, y, z, grid_x, grid_y, method=method_key))
        if method_key == "natural_neighbor":
            return SurfaceResult(
                natural_neighbor_interpolate(
                    x,
                    y,
                    z,
                    grid_x,
                    grid_y,
                ),
                metadata={
                    "support": "Convex hull of conditioning points",
                    "extrapolation": "No extrapolation outside natural-neighbor support.",
                },
            )
        if method_key == "minimum_curvature":
            return SurfaceResult(
                minimum_curvature_interpolate(
                    x,
                    y,
                    z,
                    grid_x,
                    grid_y,
                    smoothing=float(parameters.get("smoothing", 0.0)),
                ),
                metadata={
                    "approach": "Thin-plate spline biharmonic minimum-bending-energy surface",
                    "extrapolation": "Global smooth surface over the selected computational domain before masks.",
                },
            )
        if method_key == "convergent_interpolation":
            result = convergent_interpolate(x, y, z, grid_x, grid_y, parameters)
            return SurfaceResult(result.estimate, metadata=result.metadata)
        if method_key == "rbf":
            return SurfaceResult(
                rbf_interpolate(
                    x,
                    y,
                    z,
                    grid_x,
                    grid_y,
                    kernel=str(parameters.get("kernel", "thin_plate_spline")),
                    smoothing=float(parameters.get("smoothing", 0.0)),
                    neighbors=parameters.get("neighbors"),
                    epsilon=parameters.get("epsilon"),
                )
            )
        if method_key in {"ordinary_kriging", "kriging"}:
            kriging_result = ordinary_kriging_interpolate(x, y, z, grid_x, grid_y, parameters)
            return SurfaceResult(kriging_result.estimate, kriging_result.variance)
        if method_key == "universal_kriging":
            kriging_result = universal_kriging_interpolate(x, y, z, grid_x, grid_y, parameters)
            return SurfaceResult(
                kriging_result.estimate,
                kriging_result.variance,
                metadata={"trend_model": parameters.get("trend_model", "Linear XY")},
            )
    except ValueError as exc:
        raise InterpolationError(str(exc)) from exc
    except Exception as exc:  # pragma: no cover - defensive UI boundary
        raise InterpolationError(f"{method} interpolation failed: {exc}") from exc

    raise InterpolationError(f"Unsupported interpolation method: {method}")
