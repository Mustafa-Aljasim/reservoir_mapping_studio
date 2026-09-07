from copy import deepcopy
from io import BytesIO
from datetime import date
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
import rasterio
from streamlit.testing.v1 import AppTest

from core.control_import import apply_pending_pick, preview_control_import, read_control_file, suggest_control_columns
from core.crs import CRS_LIBRARY_MODE, _crs_catalog, search_crs, validate_epsg, local_crs, map_crs_snapshot
from core.engineering_controls import create_control_point, assign_panel_from_point
from core.layer_mapping import generate_layer_map_collection, map_status, MAP_STATUS_STALE
from core.project_io import save_project_archive, load_project_archive
from core.scenarios import create_map_scenario, scenario_to_generated_map
from core.geostatistics.validation import leave_one_out_cross_validation
from core.geostatistics.variogram import compute_experimental_variogram
from utils.export import map_geotiff_export_files, map_zmap_export_files, GEOTIFF_NODATA
from test_engineering_controls import _layer_frame, _generate, _panel_layer, _reservoir_boundary, MAPPINGS


DEFAULTS = {"Property": "Pressure", "Layer": "Lower", "Pressure_Reference_Date": "2026-01-01", "Unit": "psi", "Active": True}


def preview(frame, **kwargs):
    return preview_control_import(frame, suggest_control_columns(frame.columns), defaults=DEFAULTS,
        valid_properties=["Pressure", "Phi"], valid_layers=["Lower", "Upper"], valid_panels=["A", "B"], **kwargs)


def generate(crs, controls=()):
    return generate_layer_map_collection(dataframe=_layer_frame(), mappings=MAPPINGS,
        property_column="Pressure", property_type="Pressure", property_unit="psi", pressure_reference_date=date(2026, 1, 1),
        is_pressure_map=True, filter_values={}, selected_panels=[], layer_scope="Selected Layer", selected_layer="Lower",
        panel_interpolation_mode="Combined", include_state={}, duplicate_method="Average", interpolation_method="IDW",
        interpolation_parameters={"power": 2, "neighbors": 10, "min_neighbors": 1},
        grid_parameters={"nx": 3, "ny": 3, "buffer_fraction": 0}, mask_parameters={"mode": "No Mask"},
        interpolation_domain="Well Data Extent", coordinate_unit="meters", crs=crs,
        panel_layer=None, reservoir_boundary_layer=None, control_points=controls, x_col="X", y_col="Y", well_col="Well").maps["Lower"]


@pytest.mark.parametrize("query", ["32638", "EPSG:32638", "UTM zone 38N", "WGS 84"])
def test_crs_library_search(query):
    results = search_crs(query)
    assert any(item["epsg"] == 32638 and item["name"] == "WGS 84 / UTM zone 38N" for item in results)
    assert len(results) <= 50
    assert search_crs("no-such-crs-xyz") == []
    assert validate_epsg(32638).coordinate_unit == "metre"
    assert validate_epsg(32638).crs_type == "Projected CRS"
    assert _crs_catalog.cache_info().misses == 1


@pytest.mark.parametrize("epsg", [32638, 32639, None])
def test_generated_map_geotiff_exact_crs_and_geometry(epsg):
    crs = validate_epsg(epsg, CRS_LIBRARY_MODE).to_dict() if epsg else local_crs().to_dict()
    result = generate(crs)
    result["grid_z"][0, 0] = np.nan
    payload = map_geotiff_export_files(result, project_metadata={"crs": validate_epsg(4326).to_dict()})[0][1]
    with rasterio.io.MemoryFile(payload) as memory:
        with memory.open() as src:
            assert (src.crs.to_epsg() if src.crs else None) == epsg
            assert src.transform.a == .75
            assert src.transform.e == -.5
            assert tuple(src.bounds) == pytest.approx((-.375, -.25, 1.875, 1.25))
            assert src.nodata == GEOTIFF_NODATA
            np.testing.assert_allclose(src.read(1), np.where(np.isnan(result["grid_z"]), GEOTIFF_NODATA, result["grid_z"])[::-1], rtol=1e-6)


def test_crs_change_stales_active_map_and_preserves_scenario():
    project = validate_epsg(32638).to_dict()
    old = generate(project)
    saved = create_map_scenario("A", old)
    project.update(validate_epsg(32639).to_dict())
    current = generate(project)
    assert map_status(current["model_signature"], old) == MAP_STATUS_STALE
    reopened = scenario_to_generated_map(saved)
    with rasterio.io.MemoryFile(map_geotiff_export_files(reopened)[0][1]) as memory:
        with memory.open() as src:
            assert src.crs.to_epsg() == 32638
    assert map_zmap_export_files(reopened)[0][-1]["CRS_EPSG"] == 32638
    assert saved["EPSG"] == 32638


def test_missing_snapshot_uses_map_metadata_never_workspace():
    generated = generate(validate_epsg(32638).to_dict())
    generated.pop("crs")
    scenario = create_map_scenario("legacy", generated, {"crs": validate_epsg(4326).to_dict()})
    assert scenario["crs"]["epsg"] == 32638
    generated.pop("export_metadata")
    assert create_map_scenario("unknown", generated, {"crs": validate_epsg(4326).to_dict()})["crs"]["epsg"] is None


def test_contradictory_snapshot_is_rejected():
    generated = generate(validate_epsg(32638).to_dict())
    scenario = create_map_scenario("other", generate(validate_epsg(32639).to_dict()))
    with pytest.raises(ValueError, match="Map and saved scenario"):
        map_geotiff_export_files(generated, scenario=scenario)
    generated["crs"] = validate_epsg(4326).to_dict()
    with pytest.raises(ValueError, match="disagrees"):
        map_geotiff_export_files(generated)
    with pytest.raises(ValueError, match="disagree"):
        map_crs_snapshot({"crs": {"mode": "EPSG:32638 - WGS 84 / UTM zone 38N", "epsg": 4326}})
    with pytest.raises(ValueError, match="valid Excel"):
        read_control_file(b"not a workbook", "bad.xlsx")


@pytest.mark.parametrize("extension", ["csv", "xlsx"])
def test_file_scope_defaults_and_round_trip(extension):
    frame = pd.DataFrame({"X": [.2, .3, .4], "Y": [.4, .5, .6], "Value": [3075, 3010, 2985]})
    buffer = BytesIO()
    if extension == "xlsx":
        frame.to_excel(buffer, index=False)
    else:
        buffer.write(frame.to_csv(index=False).encode())
    controls, errors, _ = preview(read_control_file(buffer.getvalue(), f"controls.{extension}"))
    assert not errors and len(controls) == 3
    for control in controls:
        assert control["Property"] == "Pressure"
        assert control["Reservoir_Layer"] == "Lower"
        assert control["Pressure_Map_Reference_Date"] == "2026-01-01"
        assert control["Source_Type"] == "File Import"
    controls.append(create_control_point(x=.6, y=.4, value=4000, property_name="Pressure", property_unit="psi",
        reservoir_layer="Lower", pressure_reference_date="2026-01-01", source_type="Map Pick", comment="picked", active=False))
    saved = create_map_scenario("controls", generate(validate_epsg(32638).to_dict(), controls))
    state = {"working_df": _layer_frame(), "column_mappings": MAPPINGS, "engineering_control_points": controls,
             "crs": validate_epsg(32638).to_dict(), "map_scenarios": [saved]}
    restored = load_project_archive(save_project_archive(state))
    assert restored["engineering_control_points"] == controls
    controls[0]["Value"] = -1
    assert restored["map_scenarios"][0]["engineering_control_points"][0]["Value"] == 3075


def test_explicit_file_scopes_and_active_false():
    frame = pd.DataFrame({"Easting": [.2, 1.4], "Northing": [.3, .6], "Value": [3100, .2],
        "Property": ["Pressure", "Phi"], "Layer": ["Upper", "Lower"], "Panel": ["A", "B"],
        "Pressure_Reference_Date": ["2025-06-01", ""], "Active": ["true", "false"], "Unit": ["psi", "frac"]})
    controls, errors, _ = preview(frame)
    assert not errors
    assert controls[0]["Reservoir_Layer"] == "Upper"
    assert controls[0]["Pressure_Map_Reference_Date"] == "2025-06-01"
    assert controls[1]["Property"] == "Phi" and controls[1]["Panel"] == "B"
    assert controls[1]["Active"] is False


@pytest.mark.parametrize("field,value", [("X", np.inf), ("Y", "bad"), ("Value", np.nan), ("Layer", "Missing"),
    ("Property", "Missing"), ("Panel", "Missing"), ("Pressure_Reference_Date", "not-a-date"), ("Active", "maybe")])
def test_import_rejects_invalid_rows(field, value):
    row = {"X": .3, "Y": .4, "Value": 3000, field: value}
    _, errors, _ = preview(pd.DataFrame([row]))
    assert errors


def test_import_duplicate_ids_qc_and_spatial_assignment():
    rows = pd.DataFrame({"X": [0, 0, 5], "Y": [0, 0, 5], "Value": [8000, 8000, 8000]})
    controls, errors, warnings = preview(rows, panel_layer=_panel_layer(), reservoir_boundary_layer=_reservoir_boundary(),
        measured_dataframe=_layer_frame(), mappings=MAPPINGS)
    assert not errors and controls[0]["Panel"] == "A"
    assert any("Duplicate engineering" in warning for warning in warnings)
    assert any("measured" in warning.lower() for warning in warnings)
    assert any("outside" in warning.lower() for warning in warnings)
    rows["Control_ID"] = ["same", "same", "other"]
    assert preview(rows)[1]
    panel, warning = assign_panel_from_point(1, .5, _panel_layer())
    assert panel is None and "ambiguous" in warning


def test_pick_preview_does_not_mutate_model_and_add_stales():
    state = {"engineering_control_points": [], "pending_control_point_pick": {"x": .75, "y": .5}}
    old = generate(local_crs().to_dict())
    assert apply_pending_pick(state)
    assert state["engineering_control_x"] == .75 and state["engineering_control_y"] == .5
    assert state["engineering_control_points"] == []
    control = create_control_point(x=state["engineering_control_x"], y=state["engineering_control_y"], value=4200,
        property_name="Pressure", property_unit="psi", reservoir_layer="Lower", pressure_reference_date="2026-01-01", source_type="Map Pick")
    state["engineering_control_points"].append(control)
    new = generate(local_crs().to_dict(), state["engineering_control_points"])
    assert new["grid_z"][1, 1] == 4200
    assert map_status(new["model_signature"], old) == MAP_STATUS_STALE


def test_project_crs_library_widgets():
    app = AppTest.from_file("../pages/project.py", default_timeout=30).run()
    assert not app.exception
    next(widget for widget in app.radio if widget.label == "CRS Mode").set_value(CRS_LIBRARY_MODE).run()
    next(widget for widget in app.text_input if widget.label == "Search CRS").set_value("32638").run()
    next(widget for widget in app.button if widget.label == "Update CRS").click().run()
    assert not app.exception
    assert app.session_state["crs"]["epsg"] == 32638


def test_mapping_pick_inputs_and_add_widgets():
    app = AppTest.from_file("../pages/mapping_studio.py", default_timeout=30)
    app.session_state["working_df"] = _layer_frame()
    app.session_state["column_mappings"] = MAPPINGS
    app.run()
    assert not app.exception
    next(widget for widget in app.radio if widget.label == "Control Location").set_value("Pick on Map").run()
    next(widget for widget in app.button if widget.label == "PICK CONTROL LOCATION").click().run()
    assert app.session_state["control_point_picking_active"]
    # AppTest cannot execute iframe JavaScript; exercise its returned event at the rerun boundary.
    app.session_state["pending_control_point_pick"] = {"x": 431688.1, "y": 3363675.0}
    app.session_state["control_point_picking_active"] = False
    app.run()
    assert not app.exception
    assert app.number_input(key="engineering_control_x").value == 431688.1
    assert app.number_input(key="engineering_control_y").value == 3363675.0
    assert app.session_state["engineering_control_points"] == []
    next(widget for widget in app.button if widget.label == "Add Control Point").click().run()
    assert not app.exception
    assert app.session_state["engineering_control_points"][0]["Source_Type"] == "Map Pick"


@pytest.mark.parametrize("source", ["File Import", "Map Pick"])
def test_new_control_sources_condition_but_never_become_scientific_truth(source):
    controls, errors, _ = preview(pd.DataFrame({"X": [.75], "Y": [.5], "Value": [9000]}))
    assert not errors
    controls[0]["Source_Type"] = source
    result = generate(local_crs().to_dict(), controls)
    assert result["grid_z"][1, 1] == 9000
    measured = result["included_observations"]
    base = generate(local_crs().to_dict())["included_observations"]
    pd.testing.assert_frame_equal(measured, base)
    actual_variogram = compute_experimental_variogram(measured.X, measured.Y, measured.Pressure)
    expected_variogram = compute_experimental_variogram(base.X, base.Y, base.Pressure)
    np.testing.assert_allclose(actual_variogram.semivariance, expected_variogram.semivariance)
    truth = measured[["X", "Y", "Pressure"]].rename(columns={"Pressure": "Z"})
    conditioning = result["engineering_controls"][["X", "Y", "Pressure"]].rename(columns={"Pressure": "Z"})
    validation, metrics = leave_one_out_cross_validation(truth, "IDW", conditioning_points=conditioning)
    assert len(validation) == len(truth) == metrics["Count"]
    assert 9000 not in validation.Observed.values


def test_import_preview_and_commit_widgets():
    uploaded = BytesIO(b"X,Y,Value\n0.2,0.3,3075\n0.3,0.4,3010\n0.4,0.6,2985\n")
    uploaded.name = "three_controls.csv"
    app = AppTest.from_file("../pages/mapping_studio.py", default_timeout=30)
    app.session_state["working_df"] = _layer_frame()
    app.session_state["column_mappings"] = MAPPINGS
    app.session_state["selected_reservoir_layer"] = "Lower"
    with patch("streamlit.file_uploader", return_value=uploaded):
        app.run()
        assert not app.exception
        assert app.session_state["engineering_control_points"] == []
        button = next(widget for widget in app.button if widget.label == "IMPORT CONTROL POINTS")
        assert not button.disabled
        button.click().run()
        assert not app.exception
        controls = app.session_state["engineering_control_points"]
        assert len(controls) == 3
        assert all(control["Source_Type"] == "File Import" for control in controls)
        assert all(control["Reservoir_Layer"] == "Lower" for control in controls)
