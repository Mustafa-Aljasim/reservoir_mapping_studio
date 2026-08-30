"""Geometry import helpers for GeoJSON, zipped shapefiles, and CSV polygons."""

from __future__ import annotations

import json
import tempfile
import zipfile
from pathlib import Path
from typing import Any

import pandas as pd
from shapely.geometry import LineString, Polygon, shape
from shapely.geometry.base import BaseGeometry

from core.geometry.models import GeometryFeature, GeometryLayer


POLYGON_LAYER_TYPES = {"Reservoir Boundary", "Panel / Compartment", "Custom"}
LINE_LAYER_TYPES = {"Fault", "Custom"}
NAME_ATTRIBUTES = ("Panel", "Panel_Name", "Compartment", "Compartment_ID", "Block", "Zone", "Name", "ID")


def suggest_geometry_name_attribute(attributes: list[str]) -> str | None:
    normalized = {attribute.upper(): attribute for attribute in attributes}
    for candidate in NAME_ATTRIBUTES:
        key = candidate.upper()
        if key in normalized:
            return normalized[key]
    return attributes[0] if attributes else None


def geojson_property_names(data: bytes) -> list[str]:
    payload = json.loads(data.decode("utf-8-sig"))
    raw_features = payload.get("features", []) if payload.get("type") == "FeatureCollection" else [payload]
    names: list[str] = []
    for raw_feature in raw_features:
        properties = raw_feature.get("properties", {}) if raw_feature.get("type") == "Feature" else {}
        for name in properties:
            if name not in names:
                names.append(str(name))
    return names


def _feature_name(properties: dict[str, Any], fallback: str, name_attribute: str | None) -> str:
    if name_attribute and properties.get(name_attribute) not in (None, ""):
        return str(properties[name_attribute])
    for candidate in NAME_ATTRIBUTES:
        if properties.get(candidate) not in (None, ""):
            return str(properties[candidate])
    return fallback


def _explode_supported_geometries(geometry: BaseGeometry) -> list[BaseGeometry]:
    if geometry.geom_type in {"Polygon", "LineString"}:
        return [geometry]
    if geometry.geom_type in {"MultiPolygon", "MultiLineString", "GeometryCollection"}:
        return [part for part in geometry.geoms if part.geom_type in {"Polygon", "LineString"}]
    return [geometry]


def load_geojson_bytes(
    data: bytes,
    layer_type: str,
    layer_name: str,
    name_attribute: str | None = None,
    source_name: str = "",
) -> GeometryLayer:
    payload = json.loads(data.decode("utf-8-sig"))
    raw_features = payload.get("features", []) if payload.get("type") == "FeatureCollection" else [payload]
    features: list[GeometryFeature] = []
    for index, raw_feature in enumerate(raw_features, start=1):
        properties = dict(raw_feature.get("properties", {})) if raw_feature.get("type") == "Feature" else {}
        geometry_payload = raw_feature.get("geometry") if raw_feature.get("type") == "Feature" else raw_feature
        if not geometry_payload:
            continue
        geometry = shape(geometry_payload)
        for part_index, part in enumerate(_explode_supported_geometries(geometry), start=1):
            fallback = f"{layer_type} {index}" if part_index == 1 else f"{layer_type} {index}.{part_index}"
            features.append(
                GeometryFeature(
                    geometry=part,
                    layer_type=layer_type,
                    name=_feature_name(properties, fallback, name_attribute),
                    attributes=properties,
                )
            )
    return GeometryLayer(
        name=layer_name,
        layer_type=layer_type,
        features=features,
        source_name=source_name,
        name_attribute=name_attribute,
    )


def load_geometry_csv(
    df: pd.DataFrame,
    layer_type: str,
    layer_name: str,
    x_col: str = "X",
    y_col: str = "Y",
    group_col: str | None = None,
    source_name: str = "",
) -> GeometryLayer:
    if x_col not in df.columns or y_col not in df.columns:
        raise ValueError("CSV geometry requires X and Y coordinate columns.")
    group_col = group_col if group_col and group_col in df.columns else None
    grouped = [(layer_name, df)] if group_col is None else list(df.groupby(group_col, dropna=False))

    features: list[GeometryFeature] = []
    for group_value, group in grouped:
        coordinates = (
            group[[x_col, y_col]]
            .apply(pd.to_numeric, errors="coerce")
            .dropna()
            .drop_duplicates()
            .to_numpy(dtype=float)
            .tolist()
        )
        if len(coordinates) < 2:
            continue
        attributes = {group_col: group_value} if group_col else {}
        name = str(group_value) if group_col else layer_name
        if layer_type in POLYGON_LAYER_TYPES:
            if coordinates[0] != coordinates[-1]:
                coordinates.append(coordinates[0])
            if len(coordinates) < 4:
                continue
            geometry = Polygon(coordinates)
        elif layer_type in LINE_LAYER_TYPES:
            geometry = LineString(coordinates)
        else:
            raise ValueError(f"Unsupported CSV geometry layer type: {layer_type}")
        features.append(GeometryFeature(geometry=geometry, layer_type=layer_type, name=name, attributes=attributes))

    return GeometryLayer(
        name=layer_name,
        layer_type=layer_type,
        features=features,
        source_name=source_name,
        name_attribute=group_col,
    )


def load_zipped_shapefile_bytes(
    data: bytes,
    layer_type: str,
    layer_name: str,
    name_attribute: str | None = None,
    source_name: str = "",
) -> GeometryLayer:
    try:
        import shapefile  # type: ignore
    except ImportError as exc:  # pragma: no cover - depends on optional runtime dependency
        raise ImportError("Install pyshp to load zipped shapefiles.") from exc

    with tempfile.TemporaryDirectory() as tmp_dir:
        zip_path = Path(tmp_dir) / "geometry.zip"
        zip_path.write_bytes(data)
        with zipfile.ZipFile(zip_path) as archive:
            archive.extractall(tmp_dir)
        shp_files = list(Path(tmp_dir).glob("*.shp"))
        if not shp_files:
            raise ValueError("Zipped shapefile did not contain a .shp file.")
        reader = shapefile.Reader(str(shp_files[0]))
        field_names = [field[0] for field in reader.fields[1:]]
        features: list[GeometryFeature] = []
        for index, shape_record in enumerate(reader.iterShapeRecords(), start=1):
            properties = dict(zip(field_names, shape_record.record))
            geometry = shape(shape_record.shape.__geo_interface__)
            for part in _explode_supported_geometries(geometry):
                features.append(
                    GeometryFeature(
                        geometry=part,
                        layer_type=layer_type,
                        name=_feature_name(properties, f"{layer_type} {index}", name_attribute),
                        attributes=properties,
                    )
                )
    return GeometryLayer(layer_name, layer_type, features, source_name, name_attribute)
