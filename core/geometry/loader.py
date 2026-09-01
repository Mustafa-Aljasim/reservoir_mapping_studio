"""Geometry import helpers for GeoJSON, zipped shapefiles, and CSV polygons."""

from __future__ import annotations

import io
import json
import shutil
import tempfile
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any

import pandas as pd
from shapely.geometry import LineString, MultiLineString, Polygon, shape
from shapely.geometry.base import BaseGeometry
from shapely.ops import polygonize

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


def _line_segments_from_geometry(geometry: BaseGeometry) -> list[BaseGeometry]:
    if geometry.geom_type == "LineString":
        return [geometry]
    if geometry.geom_type == "MultiLineString":
        return list(geometry.geoms)
    if geometry.geom_type == "GeometryCollection":
        return [part for part in geometry.geoms if part.geom_type in {"LineString", "MultiLineString"}]
    return []


def detect_polygonizable_linework(features: list[GeometryFeature]) -> dict[str, int | bool]:
    total_segments = 0
    closed_segments = 0
    open_segments = 0
    for feature in features:
        segments = _line_segments_from_geometry(feature.geometry)
        total_segments += len(segments)
        closed_segments += sum(1 for segment in segments if segment.is_closed)
        open_segments += sum(1 for segment in segments if not segment.is_closed)
    return {
        "has_line_geometry": total_segments > 0,
        "total_line_segments": total_segments,
        "closed_line_segments": closed_segments,
        "open_line_segments": open_segments,
        "has_polygonizable_closed_lines": closed_segments > 0,
    }


def _polygonize_closed_linework(layer: GeometryLayer) -> GeometryLayer:
    source_features = list(layer.features)
    polygon_features: list[GeometryFeature] = []
    total_unclosed = 0
    total_source_lines = 0
    for feature in source_features:
        segments = _line_segments_from_geometry(feature.geometry)
        if not segments:
            continue
        total_source_lines += len(segments)
        closed_segments = [segment for segment in segments if segment.is_closed]
        open_segments = [segment for segment in segments if not segment.is_closed]
        total_unclosed += len(open_segments)
        polygonized = list(polygonize(closed_segments)) if closed_segments else []
        for polygon in polygonized:
            if polygon.is_empty or not polygon.is_valid or polygon.area <= 0:
                continue
            polygon_features.append(
                GeometryFeature(
                    geometry=polygon,
                    layer_type=layer.layer_type,
                    name=feature.name,
                    attributes=dict(feature.attributes),
                )
            )

    if not polygon_features:
        raise ValueError(
            "The uploaded linework does not form closed polygon boundaries. Use this dataset as a Custom/Fault layer or convert it to polygons in GIS software."
        )

    status = "Complete" if total_unclosed == 0 else "Partial"
    report = {
        "source_geometry": "LineString / MultiLineString",
        "polygons_created": len(polygon_features),
        "unclosed_line_segments": total_unclosed,
        "polygonization_status": status,
        "total_source_lines": total_source_lines,
    }
    return GeometryLayer(
        name=layer.name,
        layer_type=layer.layer_type,
        features=polygon_features,
        source_name=layer.source_name,
        name_attribute=layer.name_attribute,
        extraction_dir=layer.extraction_dir,
        source_features=source_features,
        polygonization_report=report,
    )


def load_geojson_bytes(
    data: bytes,
    layer_type: str,
    layer_name: str,
    name_attribute: str | None = None,
    source_name: str = "",
    convert_closed_linework: bool = False,
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
    layer = GeometryLayer(
        name=layer_name,
        layer_type=layer_type,
        features=features,
        source_name=source_name,
        name_attribute=name_attribute,
    )
    if layer_type in {"Reservoir Boundary", "Panel / Compartment"} and detect_polygonizable_linework(features).get("has_line_geometry"):
        if convert_closed_linework:
            return _polygonize_closed_linework(layer)
        layer.polygonization_report = {
            "source_geometry": "LineString / MultiLineString",
            "polygons_created": 0,
            "unclosed_line_segments": detect_polygonizable_linework(features)["open_line_segments"],
            "polygonization_status": "Detected",
            "total_source_lines": detect_polygonizable_linework(features)["total_line_segments"],
        }
        return layer
    return layer


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


def _safe_zip_member_name(name: str) -> str:
    pure_name = name.replace("\\", "/")
    if not pure_name or pure_name.startswith("/"):
        raise ValueError("ZIP contains an invalid shapefile path.")
    candidate = PurePosixPath(pure_name)
    if candidate.is_absolute() or ".." in candidate.parts:
        raise ValueError("ZIP contains a path traversal entry and cannot be extracted safely.")
    return candidate.as_posix()


def _has_required_shapefile_members(zip_file: zipfile.ZipFile, shapefile_path: str) -> bool:
    base_name = shapefile_path[:-4]
    suffixes = (".shp", ".shx", ".dbf")
    return all(any(info.filename.replace("\\", "/") == f"{base_name}{suffix}" for info in zip_file.infolist()) for suffix in suffixes)


def get_zipped_shapefile_candidates(data: bytes) -> list[str]:
    try:
        import shapefile  # type: ignore
    except ImportError as exc:  # pragma: no cover - depends on optional runtime dependency
        raise ImportError("Install pyshp to load zipped shapefiles.") from exc

    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        candidates: list[str] = []
        seen: set[str] = set()
        for info in archive.infolist():
            if info.is_dir():
                continue
            relative_name = _safe_zip_member_name(info.filename)
            if relative_name.lower().endswith(".shp"):
                if relative_name not in seen and _has_required_shapefile_members(archive, relative_name):
                    candidates.append(relative_name)
                    seen.add(relative_name)
        if not candidates:
            shp_names = [
                _safe_zip_member_name(info.filename)
                for info in archive.infolist()
                if not info.is_dir() and info.filename.lower().endswith(".shp")
            ]
            if not shp_names:
                raise ValueError("Zipped shapefile did not contain a .shp file.")
            raise ValueError(
                "Zipped shapefile is missing required .shx/.dbf companion files. "
                "A valid shapefile ZIP must include matching .shp, .shx, and .dbf files."
            )
        return sorted(candidates)


def extract_zipped_shapefile_dir(data: bytes) -> Path:
    temp_dir = Path(tempfile.mkdtemp(prefix="rms_geometry_"))
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        for info in archive.infolist():
            if info.is_dir():
                continue
            relative_name = _safe_zip_member_name(info.filename)
            target = temp_dir / relative_name
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(info, "r") as source, open(target, "wb") as destination:
                shutil.copyfileobj(source, destination)
    return temp_dir


def read_zipped_shapefile_fields(data: bytes, shp_name: str) -> list[str]:
    try:
        import shapefile  # type: ignore
    except ImportError as exc:  # pragma: no cover - depends on optional runtime dependency
        raise ImportError("Install pyshp to load zipped shapefiles.") from exc

    temp_dir = extract_zipped_shapefile_dir(data)
    shp_path = temp_dir / shp_name
    if not shp_path.exists():
        raise ValueError(f"Selected shapefile {shp_name} was not found in the uploaded ZIP.")
    reader = shapefile.Reader(str(shp_path))
    return [field[0] for field in reader.fields[1:]]


def read_zipped_shapefile_geometry_types(data: bytes, shp_name: str) -> dict[str, int | bool]:
    try:
        import shapefile  # type: ignore
    except ImportError as exc:  # pragma: no cover - depends on optional runtime dependency
        raise ImportError("Install pyshp to load zipped shapefiles.") from exc

    temp_dir = extract_zipped_shapefile_dir(data)
    shp_path = temp_dir / shp_name
    if not shp_path.exists():
        raise ValueError(f"Selected shapefile {shp_name} was not found in the uploaded ZIP.")

    total_segments = 0
    closed_segments = 0
    open_segments = 0
    reader = shapefile.Reader(str(shp_path))
    for shape_record in reader.iterShapeRecords():
        geometry = shape(shape_record.shape.__geo_interface__)
        segments = _line_segments_from_geometry(geometry)
        if not segments:
            continue
        total_segments += len(segments)
        closed_segments += sum(1 for segment in segments if segment.is_closed)
        open_segments += sum(1 for segment in segments if not segment.is_closed)

    return {
        "has_line_geometry": total_segments > 0,
        "total_line_segments": total_segments,
        "closed_line_segments": closed_segments,
        "open_line_segments": open_segments,
        "has_polygonizable_closed_lines": closed_segments > 0,
    }


def _validate_geometry_type(layer: GeometryLayer) -> None:
    allowed_geometry_types = {
        "Reservoir Boundary": {"Polygon", "MultiPolygon"},
        "Panel / Compartment": {"Polygon", "MultiPolygon"},
        "Fault": {"LineString", "MultiLineString"},
        "Custom": {"Polygon", "MultiPolygon", "LineString", "MultiLineString"},
    }
    allowed = allowed_geometry_types.get(layer.layer_type, None)
    if allowed is None:
        return
    for feature in layer.features:
        feature_type = feature.geometry.geom_type
        if feature_type not in allowed:
            expected = ", ".join(sorted(allowed))
            raise ValueError(
                f"{layer.layer_type} geometry requires {expected}, but feature '{feature.name}' is {feature_type}."
            )


def load_zipped_shapefile_bytes(
    data: bytes,
    layer_type: str,
    layer_name: str,
    name_attribute: str | None = None,
    source_name: str = "",
    selected_shp_name: str | None = None,
    convert_closed_linework: bool = False,
) -> GeometryLayer:
    try:
        import shapefile  # type: ignore
    except ImportError as exc:  # pragma: no cover - depends on optional runtime dependency
        raise ImportError("Install pyshp to load zipped shapefiles.") from exc

    candidates = get_zipped_shapefile_candidates(data)
    if selected_shp_name is not None and selected_shp_name not in candidates:
        raise ValueError(f"Selected shapefile {selected_shp_name} is not present in the uploaded ZIP.")
    if len(candidates) > 1 and selected_shp_name is None:
        raise ValueError(
            "Multiple shapefiles were found in the ZIP. Select the correct one using the 'Shapefile in ZIP' dropdown."
        )
    selected = selected_shp_name or candidates[0]

    temp_dir = extract_zipped_shapefile_dir(data)
    shp_path = temp_dir / selected
    if not shp_path.exists():
        raise ValueError(f"Selected shapefile {selected} could not be found after extraction.")
    required = (shp_path.with_suffix(".shx"), shp_path.with_suffix(".dbf"))
    missing = [suffix.name for suffix in required if not suffix.exists()]
    if missing:
        raise ValueError(
            f"Selected shapefile is missing required companion files: {', '.join(missing)}. "
            "A valid shapefile ZIP must include matching .shp, .shx, and .dbf files."
        )

    reader = shapefile.Reader(str(shp_path))
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

    layer = GeometryLayer(
        layer_name,
        layer_type,
        features,
        source_name,
        name_attribute,
        str(temp_dir),
    )

    if layer_type in {"Reservoir Boundary", "Panel / Compartment"} and any(
        feature.geometry.geom_type in {"LineString", "MultiLineString"} for feature in features
    ):
        if convert_closed_linework:
            return _polygonize_closed_linework(layer)
        layer.polygonization_report = {
            "source_geometry": "LineString / MultiLineString",
            "polygons_created": 0,
            "unclosed_line_segments": sum(
                1 for feature in features for segment in _line_segments_from_geometry(feature.geometry) if not segment.is_closed
            ),
            "polygonization_status": "Detected",
            "total_source_lines": sum(len(_line_segments_from_geometry(feature.geometry)) for feature in features),
        }
        return layer

    _validate_geometry_type(layer)
    return layer
