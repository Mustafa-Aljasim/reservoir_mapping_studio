import io
import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import shapefile
from shapely.geometry import LineString, Polygon

from core.geometry.assignment import assign_points_to_polygons, outside_panel_count
from core.geometry.compartment import compartment_interpolate
from core.geometry.loader import (
    _polygonize_closed_linework,
    get_zipped_shapefile_candidates,
    load_geojson_bytes,
    load_zipped_shapefile_bytes,
)
from core.geometry.masking import polygon_keep_mask
from core.geometry.models import GeometryFeature, GeometryLayer
from core.geometry.validation import detect_polygon_overlaps, validate_geometry_layer


def _two_panel_layer() -> GeometryLayer:
    return GeometryLayer(
        name="Panels",
        layer_type="Panel / Compartment",
        features=[
            GeometryFeature(Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]), "Panel / Compartment", "A"),
            GeometryFeature(Polygon([(1, 0), (2, 0), (2, 1), (1, 1)]), "Panel / Compartment", "B"),
        ],
    )


def test_polygon_mask_keeps_inside_and_masks_outside():
    layer = GeometryLayer(
        name="Boundary",
        layer_type="Reservoir Boundary",
        features=[GeometryFeature(Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]), "Reservoir Boundary", "Boundary")],
    )
    grid_x = np.array([[0.5, 1.5]])
    grid_y = np.array([[0.5, 0.5]])
    keep = polygon_keep_mask(layer.features, grid_x, grid_y)
    assert keep[0, 0]
    assert not keep[0, 1]


def test_point_assignment_and_outside_flag():
    df = pd.DataFrame({"X": [0.5, 1.5, 3.0], "Y": [0.5, 0.5, 0.5], "Panel": ["A", "B", "A"]})
    assignments = assign_points_to_polygons(df, "X", "Y", _two_panel_layer(), "Panel")
    assert assignments.loc[0, "Assignment_Status"] == "Match"
    assert assignments.loc[1, "Assignment_Status"] == "Match"
    assert assignments.loc[2, "Assignment_Status"] == "Outside"
    assert outside_panel_count(assignments) == 1


def test_overlapping_polygons_are_detected():
    layer = GeometryLayer(
        name="Overlap",
        layer_type="Panel / Compartment",
        features=[
            GeometryFeature(Polygon([(0, 0), (2, 0), (2, 1), (0, 1)]), "Panel / Compartment", "A"),
            GeometryFeature(Polygon([(1, 0), (3, 0), (3, 1), (1, 1)]), "Panel / Compartment", "B"),
        ],
    )
    overlaps = detect_polygon_overlaps(layer)
    assert len(overlaps) == 1
    assert overlaps[0].area > 0


def test_geojson_loader_and_validation():
    payload = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {"Panel": "A"},
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[(0, 0), (1, 0), (1, 1), (0, 1), (0, 0)]],
                },
            }
        ],
    }
    layer = load_geojson_bytes(json.dumps(payload).encode("utf-8"), "Panel / Compartment", "Panels", "Panel")
    validated, report = validate_geometry_layer(layer)
    assert validated.feature_count == 1
    assert validated.features[0].name == "A"
    assert report.valid_features == 1


def test_compartment_interpolation_uses_own_observations():
    prepared = pd.DataFrame(
        {
            "X": [0.2, 0.8, 0.5, 1.2, 1.8, 1.5],
            "Y": [0.2, 0.2, 0.8, 0.2, 0.2, 0.8],
            "Z": [3300, 3310, 3290, 2700, 2710, 2690],
        }
    )
    grid_x = np.array([[0.5, 1.5]])
    grid_y = np.array([[0.5, 0.5]])
    result = compartment_interpolate(
        prepared,
        _two_panel_layer(),
        grid_x,
        grid_y,
        "IDW",
        {"power": 2, "neighbors": 3, "min_neighbors": 1, "search_radius": None},
    )
    assert result.surface[0, 0] > 3200
    assert result.surface[0, 1] < 2800
    assert result.panel_grid[0, 0] == "A"
    assert result.panel_grid[0, 1] == "B"


def _write_shapefile_bundle(tmp_path: Path, name: str, records: list[tuple[str, list[tuple[float, float]]]]) -> bytes:
    shp_path = tmp_path / f"{name}.shp"
    writer = shapefile.Writer(str(shp_path))
    writer.shapeType = shapefile.POLYGON
    writer.field("Name", "C")
    writer.field("Panel", "C")
    for feature_name, polygon in records:
        coords = polygon
        writer.poly([coords])
        writer.record(feature_name, feature_name)
    writer.close()

    prj_path = tmp_path / f"{name}.prj"
    prj_path.write_text('GEOGCS["WGS 84",DATUM["WGS_1984",SPHEROID["WGS 84",6378137,298.257223563]],PRIMEM["Greenwich",0],UNIT["degree",0.0174532925199433]]', encoding="utf-8")
    (tmp_path / f"{name}.cpg").write_text("UTF-8", encoding="utf-8")
    return b""


def _zip_files(files: dict[str, bytes], archive_name: str = "geometry.zip") -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for arc_name, payload in files.items():
            zf.writestr(arc_name, payload)
    return buffer.getvalue()


def test_zipped_shapefile_loads_with_prj_and_cpg(tmp_path):
    base = tmp_path / "ROO Development Areas - Panels"
    base.parent.mkdir(parents=True, exist_ok=True)
    writer = shapefile.Writer(str(base))
    writer.shapeType = shapefile.POLYGON
    writer.field("Panel", "C")
    writer.poly([[(0, 0), (1, 0), (1, 1), (0, 1), (0, 0)]])
    writer.record("A")
    writer.close()
    (tmp_path / "ROO Development Areas - Panels.prj").write_text('GEOGCS["WGS 84",DATUM["WGS_1984",SPHEROID["WGS 84",6378137,298.257223563]],PRIMEM["Greenwich",0],UNIT["degree",0.0174532925199433]]', encoding="utf-8")
    (tmp_path / "ROO Development Areas - Panels.cpg").write_text("UTF-8", encoding="utf-8")

    payload = _zip_files({
        "ROO Development Areas - Panels.shp": (tmp_path / "ROO Development Areas - Panels.shp").read_bytes(),
        "ROO Development Areas - Panels.shx": (tmp_path / "ROO Development Areas - Panels.shx").read_bytes(),
        "ROO Development Areas - Panels.dbf": (tmp_path / "ROO Development Areas - Panels.dbf").read_bytes(),
        "ROO Development Areas - Panels.prj": (tmp_path / "ROO Development Areas - Panels.prj").read_bytes(),
        "ROO Development Areas - Panels.cpg": (tmp_path / "ROO Development Areas - Panels.cpg").read_bytes(),
    })

    layer = load_zipped_shapefile_bytes(payload, "Panel / Compartment", "Panels", "Panel")
    assert layer.feature_count == 1
    assert layer.features[0].name == "A"


def test_zipped_shapefile_without_cpg_still_loads(tmp_path):
    base = tmp_path / "example_panels"
    writer = shapefile.Writer(str(base))
    writer.shapeType = shapefile.POLYGON
    writer.field("Panel", "C")
    writer.poly([[(0, 0), (2, 0), (2, 2), (0, 2), (0, 0)]])
    writer.record("B")
    writer.close()
    (tmp_path / "example_panels.prj").write_text("EPSG:4326", encoding="utf-8")

    payload = _zip_files({
        "example_panels.shp": (tmp_path / "example_panels.shp").read_bytes(),
        "example_panels.shx": (tmp_path / "example_panels.shx").read_bytes(),
        "example_panels.dbf": (tmp_path / "example_panels.dbf").read_bytes(),
        "example_panels.prj": (tmp_path / "example_panels.prj").read_bytes(),
    })

    layer = load_zipped_shapefile_bytes(payload, "Panel / Compartment", "Panels", "Panel")
    assert layer.feature_count == 1


def test_zipped_shapefile_missing_dbf_raises_meaningful_validation_error(tmp_path):
    base = tmp_path / "broken"
    writer = shapefile.Writer(str(base))
    writer.shapeType = shapefile.POLYGON
    writer.field("Name", "C")
    writer.poly([[(0, 0), (1, 0), (1, 1), (0, 1), (0, 0)]])
    writer.record("A")
    writer.close()

    payload = _zip_files({
        "broken.shp": (tmp_path / "broken.shp").read_bytes(),
        "broken.shx": (tmp_path / "broken.shx").read_bytes(),
    })

    try:
        load_zipped_shapefile_bytes(payload, "Panel / Compartment", "Panels")
        raise AssertionError("Expected validation error")
    except ValueError as exc:
        assert ".dbf" in str(exc)


def test_multiple_shapefiles_in_zip_return_candidates(tmp_path):
    for name in ("first", "second"):
        writer = shapefile.Writer(str(tmp_path / name))
        writer.shapeType = shapefile.POLYGON
        writer.field("Name", "C")
        writer.poly([[(0, 0), (1, 0), (1, 1), (0, 1), (0, 0)]])
        writer.record(name)
        writer.close()

    payload = _zip_files({
        "first.shp": (tmp_path / "first.shp").read_bytes(),
        "first.shx": (tmp_path / "first.shx").read_bytes(),
        "first.dbf": (tmp_path / "first.dbf").read_bytes(),
        "second.shp": (tmp_path / "second.shp").read_bytes(),
        "second.shx": (tmp_path / "second.shx").read_bytes(),
        "second.dbf": (tmp_path / "second.dbf").read_bytes(),
    })

    candidates = get_zipped_shapefile_candidates(payload)
    assert candidates == ["first.shp", "second.shp"]


def test_nested_shapefile_folder_inside_zip_loads(tmp_path):
    base = tmp_path / "nested" / "area"
    base.parent.mkdir(parents=True, exist_ok=True)
    writer = shapefile.Writer(str(base))
    writer.shapeType = shapefile.POLYGON
    writer.field("Name", "C")
    writer.poly([[(0, 0), (1, 0), (1, 1), (0, 1), (0, 0)]])
    writer.record("Nested")
    writer.close()
    (tmp_path / "nested" / "area.prj").write_text("EPSG:4326", encoding="utf-8")

    payload = _zip_files({
        "nested/area.shp": (tmp_path / "nested" / "area.shp").read_bytes(),
        "nested/area.shx": (tmp_path / "nested" / "area.shx").read_bytes(),
        "nested/area.dbf": (tmp_path / "nested" / "area.dbf").read_bytes(),
        "nested/area.prj": (tmp_path / "nested" / "area.prj").read_bytes(),
    })

    layer = load_zipped_shapefile_bytes(payload, "Reservoir Boundary", "Boundary")
    assert layer.feature_count == 1


def test_geometry_type_validation_for_panel_and_fault():
    layered = GeometryLayer(
        "Panels",
        "Panel / Compartment",
        [GeometryFeature(Polygon([(0, 0), (1, 0), (1, 1), (0, 1), (0, 0)]), "Panel / Compartment", "A")],
    )
    validated, report = validate_geometry_layer(layered)
    assert report.valid_features == 1

    fault_layer = GeometryLayer(
        "Faults",
        "Fault",
        [GeometryFeature(LineString([(0, 0), (1, 1)]), "Fault", "F1")],
    )
    validated_fault, fault_report = validate_geometry_layer(fault_layer)
    assert fault_report.valid_features == 1
    assert validated_fault.features[0].geometry.geom_type == "LineString"

    bad_fault = GeometryLayer(
        "Faults",
        "Fault",
        [GeometryFeature(Polygon([(0, 0), (1, 0), (1, 1), (0, 1), (0, 0)]), "Fault", "Bad")],
    )
    try:
        validate_geometry_layer(bad_fault)
        raise AssertionError("Expected type validation error")
    except ValueError as exc:
        assert "Fault" in str(exc)


def test_closed_lines_polygonize_into_panel_polygons():
    layer = GeometryLayer(
        "Panels",
        "Panel / Compartment",
        [GeometryFeature(LineString([(0, 0), (0, 1), (1, 1), (1, 0), (0, 0)]), "Panel / Compartment", "A", {"Panel": "A"})],
    )
    converted = _polygonize_closed_linework(layer)
    assert converted.feature_count == 1
    assert converted.features[0].geometry.geom_type == "Polygon"


def test_polygonization_handles_multiple_closed_lines_and_preserves_attributes():
    feature = GeometryFeature(
        LineString([(0, 0), (0, 1), (1, 1), (1, 0), (0, 0)]),
        "Panel / Compartment",
        "Alpha",
        {"Panel": "Alpha"},
    )
    another = GeometryFeature(
        LineString([(2, 0), (2, 1), (3, 1), (3, 0), (2, 0)]),
        "Panel / Compartment",
        "Beta",
        {"Panel": "Beta"},
    )
    layer = GeometryLayer("Panels", "Panel / Compartment", [feature, another])
    converted = _polygonize_closed_linework(layer)
    assert converted.feature_count == 2
    assert {item.name for item in converted.features} == {"Alpha", "Beta"}


def test_zipped_linework_can_polygonize_during_panel_load(tmp_path):
    base = tmp_path / "ROO Development Areas - Panels"
    writer = shapefile.Writer(str(base))
    writer.shapeType = shapefile.POLYLINE
    writer.field("Panel", "C")
    writer.line([[(0, 0), (0, 1), (1, 1), (1, 0), (0, 0)]])
    writer.record("A")
    writer.close()
    (tmp_path / "ROO Development Areas - Panels.prj").write_text("EPSG:4326", encoding="utf-8")

    payload = _zip_files({
        "ROO Development Areas - Panels.shp": (tmp_path / "ROO Development Areas - Panels.shp").read_bytes(),
        "ROO Development Areas - Panels.shx": (tmp_path / "ROO Development Areas - Panels.shx").read_bytes(),
        "ROO Development Areas - Panels.dbf": (tmp_path / "ROO Development Areas - Panels.dbf").read_bytes(),
        "ROO Development Areas - Panels.prj": (tmp_path / "ROO Development Areas - Panels.prj").read_bytes(),
    })

    layer = load_zipped_shapefile_bytes(
        payload,
        "Panel / Compartment",
        "Panels",
        "Panel",
        "ROO Development Areas - Panels.shp",
        selected_shp_name="ROO Development Areas - Panels.shp",
        convert_closed_linework=True,
    )
    assert layer.feature_count == 1
    assert layer.features[0].geometry.geom_type == "Polygon"
    assert layer.polygonization_report["polygonization_status"] in {"Complete", "Partial"}


def test_open_linework_fails_polygonization_without_bounding_error():
    layer = GeometryLayer(
        "Panels",
        "Panel / Compartment",
        [GeometryFeature(LineString([(0, 0), (1, 0), (1, 1), (2, 1)]), "Panel / Compartment", "Open")],
    )
    try:
        _polygonize_closed_linework(layer)
        raise AssertionError("Expected ValueError")
    except ValueError as exc:
        assert "closed polygon boundaries" in str(exc)


def test_zipped_shapefile_extraction_directory_persists_and_cleanup_is_safe(tmp_path):
    base = tmp_path / "persisted"
    writer = shapefile.Writer(str(base))
    writer.shapeType = shapefile.POLYGON
    writer.field("Name", "C")
    writer.poly([[(0, 0), (1, 0), (1, 1), (0, 1), (0, 0)]])
    writer.record("P")
    writer.close()
    (tmp_path / "persisted.prj").write_text("EPSG:4326", encoding="utf-8")

    payload = _zip_files({
        "persisted.shp": (tmp_path / "persisted.shp").read_bytes(),
        "persisted.shx": (tmp_path / "persisted.shx").read_bytes(),
        "persisted.dbf": (tmp_path / "persisted.dbf").read_bytes(),
        "persisted.prj": (tmp_path / "persisted.prj").read_bytes(),
    })

    layer = load_zipped_shapefile_bytes(payload, "Panel / Compartment", "Panels")
    assert layer.feature_count == 1
    assert any((Path(layer.source_name).exists() if layer.source_name else False) for _ in [0]) is False

