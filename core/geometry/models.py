"""Data models for imported spatial geometry."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from shapely.geometry.base import BaseGeometry


@dataclass
class GeometryFeature:
    """One spatial feature with engineering metadata."""

    geometry: BaseGeometry
    layer_type: str
    name: str
    attributes: dict[str, Any] = field(default_factory=dict)


@dataclass
class GeometryLayer:
    """A collection of imported reservoir geometry features."""

    name: str
    layer_type: str
    features: list[GeometryFeature]
    source_name: str = ""
    name_attribute: str | None = None
    extraction_dir: str | None = None
    source_features: list[GeometryFeature] = field(default_factory=list)
    polygonization_report: dict[str, Any] = field(default_factory=dict)

    @property
    def is_loaded(self) -> bool:
        return bool(self.features)

    @property
    def feature_count(self) -> int:
        return len(self.features)

    @property
    def polygon_features(self) -> list[GeometryFeature]:
        return [feature for feature in self.features if feature.geometry.geom_type in {"Polygon", "MultiPolygon"}]

    @property
    def line_features(self) -> list[GeometryFeature]:
        return [feature for feature in self.features if feature.geometry.geom_type in {"LineString", "MultiLineString"}]


@dataclass
class GeometryValidationIssue:
    feature_name: str
    severity: str
    message: str


@dataclass
class GeometryOverlap:
    first: str
    second: str
    area: float


@dataclass
class GeometryValidationReport:
    total_features: int
    valid_features: int
    repaired_features: int
    invalid_features: int
    empty_features: int
    outside_extent_features: int
    issues: list[GeometryValidationIssue] = field(default_factory=list)
    overlaps: list[GeometryOverlap] = field(default_factory=list)

