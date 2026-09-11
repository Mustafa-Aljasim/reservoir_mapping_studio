"""Advanced interpolation methods for reservoir map surfaces."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.interpolate import RBFInterpolator, RegularGridInterpolator
from scipy.ndimage import gaussian_filter
from scipy.spatial import Delaunay, QhullError, Voronoi, cKDTree
from shapely.geometry import MultiPoint, Point, Polygon

from core.interpolation.idw import idw_interpolate
from utils.validators import non_collinear_points


@dataclass(frozen=True)
class ConvergentResult:
    estimate: np.ndarray
    metadata: dict[str, object]


def _clean_xyz(x, y, z) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x_array = np.asarray(x, dtype=float)
    y_array = np.asarray(y, dtype=float)
    z_array = np.asarray(z, dtype=float)
    mask = np.isfinite(x_array) & np.isfinite(y_array) & np.isfinite(z_array)
    return x_array[mask], y_array[mask], z_array[mask]


def _validate_surface_points(x, y, z, method_name: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x_array, y_array, z_array = _clean_xyz(x, y, z)
    if len(z_array) < 3:
        raise ValueError(f"At least 3 finite observations are required for {method_name}.")
    if not non_collinear_points(x_array, y_array):
        raise ValueError(f"At least 3 non-collinear observations are required for {method_name}.")
    return x_array, y_array, z_array


def _voronoi_finite_polygons_2d(vor: Voronoi, radius: float | None = None) -> tuple[list[list[int]], np.ndarray]:
    """Reconstruct finite Voronoi regions for 2-D input points."""

    if vor.points.shape[1] != 2:
        raise ValueError("Voronoi input must be 2-D.")
    if radius is None:
        radius = float(np.ptp(vor.points, axis=0).max() * 2.0)

    new_regions: list[list[int]] = []
    new_vertices = vor.vertices.tolist()
    center = vor.points.mean(axis=0)
    all_ridges: dict[int, list[tuple[int, int, int]]] = {}
    for (point_a, point_b), (vertex_a, vertex_b) in zip(vor.ridge_points, vor.ridge_vertices):
        all_ridges.setdefault(point_a, []).append((point_b, vertex_a, vertex_b))
        all_ridges.setdefault(point_b, []).append((point_a, vertex_a, vertex_b))

    for point_index, region_index in enumerate(vor.point_region):
        vertices = vor.regions[region_index]
        if all(vertex >= 0 for vertex in vertices):
            new_regions.append(vertices)
            continue

        ridges = all_ridges[point_index]
        new_region = [vertex for vertex in vertices if vertex >= 0]
        for neighbor_index, vertex_a, vertex_b in ridges:
            if vertex_a >= 0 and vertex_b >= 0:
                continue
            vertex = vertex_a if vertex_a >= 0 else vertex_b
            tangent = vor.points[neighbor_index] - vor.points[point_index]
            tangent /= np.linalg.norm(tangent)
            normal = np.array([-tangent[1], tangent[0]])
            midpoint = vor.points[[point_index, neighbor_index]].mean(axis=0)
            direction = np.sign(np.dot(midpoint - center, normal)) * normal
            far_point = vor.vertices[vertex] + direction * radius
            new_vertices.append(far_point.tolist())
            new_region.append(len(new_vertices) - 1)

        region_vertices = np.asarray([new_vertices[vertex] for vertex in new_region])
        centroid = region_vertices.mean(axis=0)
        angles = np.arctan2(region_vertices[:, 1] - centroid[1], region_vertices[:, 0] - centroid[0])
        new_regions.append([vertex for _, vertex in sorted(zip(angles, new_region))])

    return new_regions, np.asarray(new_vertices)


def _bounded_voronoi_cells(points: np.ndarray, clip_geometry) -> list[object]:
    span = float(max(np.ptp(points[:, 0]), np.ptp(points[:, 1]), 1.0))
    regions, vertices = _voronoi_finite_polygons_2d(Voronoi(points), radius=span * 4.0)
    cells = []
    for region in regions:
        polygon = Polygon(vertices[region]).buffer(0)
        cells.append(polygon.intersection(clip_geometry) if not polygon.is_empty else polygon)
    return cells


def natural_neighbor_interpolate(x, y, z, grid_x: np.ndarray, grid_y: np.ndarray) -> np.ndarray:
    """Interpolate with bounded Sibson natural-neighbor area weights.

    Values are returned only inside the conditioning-point convex hull. For each
    query node, the inserted query-cell area stolen from the original Voronoi
    cells provides the interpolation weights.
    """

    x_array, y_array, z_array = _validate_surface_points(x, y, z, "Natural Neighbor interpolation")
    points = np.column_stack([x_array, y_array])
    try:
        Delaunay(points)
    except QhullError as exc:
        raise ValueError("Natural Neighbor interpolation could not triangulate the observations.") from exc

    hull = MultiPoint(points).convex_hull
    original_cells = _bounded_voronoi_cells(points, hull)
    tree = cKDTree(points)
    query = np.column_stack([grid_x.ravel(), grid_y.ravel()])
    result = np.full(len(query), np.nan, dtype=float)
    tolerance = max(float(np.ptp(points[:, 0])), float(np.ptp(points[:, 1])), 1.0) * 1e-10

    for index, query_point in enumerate(query):
        distance, source_index = tree.query(query_point, k=1)
        if distance <= tolerance:
            result[index] = z_array[int(source_index)]
            continue
        point = Point(float(query_point[0]), float(query_point[1]))
        if not hull.covers(point):
            continue
        inserted_points = np.vstack([points, query_point])
        try:
            inserted_cells = _bounded_voronoi_cells(inserted_points, hull)
        except QhullError:
            continue
        query_cell = inserted_cells[-1]
        if query_cell.is_empty or query_cell.area <= 0:
            continue
        weights = np.asarray([query_cell.intersection(cell).area for cell in original_cells], dtype=float)
        total = float(weights.sum())
        if total > 0:
            result[index] = float(np.dot(weights, z_array) / total)
    return result.reshape(grid_x.shape)


def minimum_curvature_interpolate(
    x,
    y,
    z,
    grid_x: np.ndarray,
    grid_y: np.ndarray,
    smoothing: float = 0.0,
) -> np.ndarray:
    """Interpolate with a thin-plate spline minimum-bending-energy surface."""

    x_array, y_array, z_array = _validate_surface_points(x, y, z, "Minimum Curvature interpolation")
    if smoothing < 0:
        raise ValueError("Minimum Curvature smoothing must be zero or greater.")
    interpolator = RBFInterpolator(
        np.column_stack([x_array, y_array]),
        z_array,
        kernel="thin_plate_spline",
        smoothing=float(smoothing),
    )
    query = np.column_stack([grid_x.ravel(), grid_y.ravel()])
    return np.asarray(interpolator(query), dtype=float).reshape(grid_x.shape)


def _moving_average_surface(x, y, z, grid_x, grid_y, neighbors: int, search_radius: float | None) -> np.ndarray:
    x_array, y_array, z_array = _clean_xyz(x, y, z)
    points = np.column_stack([x_array, y_array])
    query = np.column_stack([grid_x.ravel(), grid_y.ravel()])
    tree = cKDTree(points)
    k = max(1, min(int(neighbors), len(z_array)))
    upper_bound = np.inf if search_radius is None else float(search_radius)
    distances, indices = tree.query(query, k=k, distance_upper_bound=upper_bound)
    distances = np.atleast_2d(distances).T if k == 1 else distances
    indices = np.atleast_2d(indices).T if k == 1 else indices
    valid = np.isfinite(distances) & (indices < len(z_array))
    output = np.full(len(query), np.nan, dtype=float)
    for row_index, row_valid in enumerate(valid):
        if row_valid.any():
            output[row_index] = float(np.mean(z_array[indices[row_index, row_valid]]))
    return output.reshape(grid_x.shape)


def _sample_grid(grid_x: np.ndarray, grid_y: np.ndarray, surface: np.ndarray, x, y) -> np.ndarray:
    x_axis = np.asarray(grid_x[0, :], dtype=float)
    y_axis = np.asarray(grid_y[:, 0], dtype=float)
    interpolator = RegularGridInterpolator(
        (y_axis, x_axis),
        np.asarray(surface, dtype=float),
        bounds_error=False,
        fill_value=np.nan,
    )
    return np.asarray(interpolator(np.column_stack([np.asarray(y, dtype=float), np.asarray(x, dtype=float)])), dtype=float)


def _rmse(values: np.ndarray) -> float:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return float("nan")
    return float(np.sqrt(np.mean(finite**2)))


def convergent_interpolate(
    x,
    y,
    z,
    grid_x: np.ndarray,
    grid_y: np.ndarray,
    parameters: dict | None = None,
) -> ConvergentResult:
    """Iteratively correct a smooth surface using interpolated residuals."""

    params = parameters or {}
    x_array, y_array, z_array = _validate_surface_points(x, y, z, "Convergent Interpolation")
    neighbors = int(params.get("neighbors", params.get("neighbor_count", 12)) or 12)
    max_iterations = max(1, int(params.get("max_iterations", 25) or 25))
    tolerance = max(0.0, float(params.get("convergence_tolerance", params.get("tolerance", 1.0)) or 0.0))
    relaxation = float(params.get("relaxation", params.get("relaxation_factor", 0.7)) or 0.7)
    relaxation = min(max(relaxation, 0.0), 1.0)
    search_radius = params.get("search_radius")
    smoothing = max(0.0, float(params.get("smoothing", 0.0) or 0.0))
    initial_method = str(params.get("initial_surface_method", "IDW"))

    if initial_method == "Moving Average":
        surface = _moving_average_surface(x_array, y_array, z_array, grid_x, grid_y, neighbors, search_radius)
    else:
        surface = idw_interpolate(
            x_array,
            y_array,
            z_array,
            grid_x,
            grid_y,
            power=float(params.get("initial_idw_power", 2.0)),
            neighbors=neighbors,
            search_radius=search_radius,
            min_neighbors=1,
        )
    if smoothing > 0:
        surface = gaussian_filter(surface, sigma=smoothing, mode="nearest")

    estimated = _sample_grid(grid_x, grid_y, surface, x_array, y_array)
    initial_rmse = _rmse(z_array - estimated)
    previous_rmse = initial_rmse
    rmse_history = [initial_rmse]
    converged = bool(np.isfinite(initial_rmse) and initial_rmse <= tolerance)
    iterations = 0

    for iteration in range(1, max_iterations + 1):
        residuals = z_array - _sample_grid(grid_x, grid_y, surface, x_array, y_array)
        residuals = np.where(np.isfinite(residuals), residuals, 0.0)
        correction = idw_interpolate(
            x_array,
            y_array,
            residuals,
            grid_x,
            grid_y,
            power=2.0,
            neighbors=neighbors,
            search_radius=search_radius,
            min_neighbors=1,
        )
        correction = np.where(np.isfinite(correction), correction, 0.0)
        if smoothing > 0:
            correction = gaussian_filter(correction, sigma=smoothing, mode="nearest")
        surface = surface + relaxation * correction
        current_rmse = _rmse(z_array - _sample_grid(grid_x, grid_y, surface, x_array, y_array))
        rmse_history.append(current_rmse)
        iterations = iteration
        improvement = previous_rmse - current_rmse if np.isfinite(previous_rmse) and np.isfinite(current_rmse) else float("nan")
        if np.isfinite(current_rmse) and current_rmse <= tolerance:
            converged = True
            break
        if iteration > 1 and np.isfinite(improvement) and improvement >= 0 and improvement < tolerance:
            break
        previous_rmse = current_rmse

    final_rmse = rmse_history[-1] if rmse_history else float("nan")
    return ConvergentResult(
        estimate=np.asarray(surface, dtype=float),
        metadata={
            "method": "Convergent Interpolation",
            "iterations": iterations,
            "initial_rmse": initial_rmse,
            "final_rmse": final_rmse,
            "tolerance": tolerance,
            "converged": bool(converged),
            "rmse_history": rmse_history,
            "mathematics": "Initial smooth surface plus iterative IDW residual correction with relaxation.",
        },
    )
