"""Validation and summary helpers for imported reservoir geometry."""

from __future__ import annotations

from dataclasses import replace

from shapely import make_valid
from shapely.geometry import box
from shapely.geometry.base import BaseGeometry
from shapely.ops import unary_union

from core.geometry.models import (
    GeometryFeature,
    GeometryLayer,
    GeometryOverlap,
    GeometryValidationIssue,
    GeometryValidationReport,
)


def _well_extent_geometry(extent: tuple[float, float, float, float] | None, buffer_fraction: float = 0.05):
    if extent is None:
        return None
    min_x, min_y, max_x, max_y = extent
    x_span = max(max_x - min_x, 1.0)
    y_span = max(max_y - min_y, 1.0)
    return box(
        min_x - x_span * buffer_fraction,
        min_y - y_span * buffer_fraction,
        max_x + x_span * buffer_fraction,
        max_y + y_span * buffer_fraction,
    )


def repair_geometry(geometry: BaseGeometry) -> tuple[BaseGeometry, bool]:
    if geometry.is_valid:
        return geometry, False
    repaired = make_valid(geometry)
    if repaired.is_empty or not repaired.is_valid:
        repaired = geometry.buffer(0)
    return repaired, bool(not repaired.equals_exact(geometry, tolerance=0.0))


def validate_geometry_layer(
    layer: GeometryLayer,
    well_extent: tuple[float, float, float, float] | None = None,
    min_area: float = 1e-9,
) -> tuple[GeometryLayer, GeometryValidationReport]:
    issues: list[GeometryValidationIssue] = []
    repaired_features = 0
    invalid_features = 0
    empty_features = 0
    outside_extent_features = 0
    valid_features: list[GeometryFeature] = []
    extent_geometry = _well_extent_geometry(well_extent)

    allowed_geometry_types = {
        "Reservoir Boundary": {"Polygon", "MultiPolygon"},
        "Panel / Compartment": {"Polygon", "MultiPolygon"},
        "Fault": {"LineString", "MultiLineString"},
        "Custom": {"Polygon", "MultiPolygon", "LineString", "MultiLineString"},
    }
    allowed = allowed_geometry_types.get(layer.layer_type)
    if allowed is None:
        raise ValueError(f"Unsupported layer type '{layer.layer_type}'.")

    for feature in layer.features:
        geometry = feature.geometry
        if geometry.is_empty:
            empty_features += 1
            issues.append(GeometryValidationIssue(feature.name, "error", "Geometry is empty."))
            continue
        if geometry.geom_type not in allowed:
            raise ValueError(
                f"{layer.layer_type} geometry requires {', '.join(sorted(allowed))}, but feature '{feature.name}' is {geometry.geom_type}."
            )
        if not geometry.is_valid:
            repaired, changed = repair_geometry(geometry)
            if changed and repaired.is_valid and not repaired.is_empty:
                geometry = repaired
                repaired_features += 1
                issues.append(GeometryValidationIssue(feature.name, "warning", "Invalid geometry was repaired."))
            else:
                invalid_features += 1
                issues.append(GeometryValidationIssue(feature.name, "error", "Geometry is invalid and could not be repaired."))
                continue
        if geometry.geom_type in {"Polygon", "MultiPolygon"} and geometry.area <= min_area:
            invalid_features += 1
            issues.append(GeometryValidationIssue(feature.name, "error", "Polygon has zero or very small area."))
            continue
        if extent_geometry is not None and not geometry.intersects(extent_geometry):
            outside_extent_features += 1
            issues.append(GeometryValidationIssue(feature.name, "warning", "Geometry is outside the active well-data extent."))
        valid_features.append(replace(feature, geometry=geometry))

    validated_layer = replace(layer, features=valid_features)
    overlaps = detect_polygon_overlaps(validated_layer)
    report = GeometryValidationReport(
        total_features=len(layer.features),
        valid_features=len(valid_features),
        repaired_features=repaired_features,
        invalid_features=invalid_features,
        empty_features=empty_features,
        outside_extent_features=outside_extent_features,
        issues=issues,
        overlaps=overlaps,
    )
    return validated_layer, report


def detect_polygon_overlaps(layer: GeometryLayer, tolerance: float = 1e-9) -> list[GeometryOverlap]:
    polygons = layer.polygon_features
    overlaps: list[GeometryOverlap] = []
    for index, first in enumerate(polygons):
        for second in polygons[index + 1 :]:
            intersection = first.geometry.intersection(second.geometry)
            if not intersection.is_empty and intersection.area > tolerance:
                overlaps.append(GeometryOverlap(first.name, second.name, float(intersection.area)))
    return overlaps


def layer_total_area(layer: GeometryLayer) -> float:
    polygons = [feature.geometry for feature in layer.polygon_features]
    if not polygons:
        return 0.0
    return float(unary_union(polygons).area)

