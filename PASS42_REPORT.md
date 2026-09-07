# PASS 4.2 completion report

## A. Branch

`geometry-update`. The working tree was clean at the start. The initial branch was `main`; it was switched to `geometry-update` before any edits. No commit or merge was made.

## B. Files modified / added

- `core/crs.py`: cached library search, CRS detail fields, strict map snapshot resolution.
- `core/control_import.py` (new): CSV/XLSX reading, semantic mappings, preview validation, scope defaults, existing control QC, pending pick handling.
- `core/engineering_controls.py`: standardized source labels and legacy source migration.
- `core/layer_mapping.py`: CRS identity in generated-map change signatures.
- `core/scenarios.py`: map-owned CRS snapshots, explicit scenario CRS fields, removal of workspace CRS fallback.
- `core/plotting/engineering_controls.py`: source labels and legacy region overlay compatibility.
- `components/control_region_drawer/__init__.py`: single-point mode and preview payload.
- `components/control_region_drawer/frontend/index.html`: Cartesian single-click events and preview marker.
- `pages/project.py`: CRS library UI, selected definition details, update notice, pending pick reset on project open.
- `pages/mapping_studio.py`: point-pick workflow, import preview/commit, panel suggestions, preview markers, CRS mismatch notice.
- `pages/shared.py`: clear transient pick state when replacing input data.
- `utils/export.py`: shared strict map/scenario CRS resolution for GeoTIFF and ZMAP.
- `tests/test_pass42.py` (new): focused core and Streamlit AppTest regressions.
- `tests/test_engineering_controls.py`: existing measured-only Auto Fit test extended to Map Pick and File Import sources.
- `tests/control_point_component.cjs` (new): execute actual component JavaScript with a minimal DOM harness.
- `PASS42_REPORT.md` (new): this report.

## C. CRS library implementation

The Project page offers Local / Unknown XY, CRS Library, and Custom EPSG. The existing EPSG:32638 preset remains readable and is a convenient first library result. Searches match EPSG codes, CRS names, and UTM zone terms. A process-level `lru_cache` loads nondeprecated EPSG definitions from `pyproj.database.query_crs_info()` once; the UI receives at most 50 results. Selected definitions display EPSG, name, CRS type, and coordinate unit. Definitions are resolved through pyproj, with no manually maintained global catalog.

Project state retains the compatible `crs` dictionary (`mode`, `epsg`, `name`, `authority`) and adds `coordinate_unit` and `crs_type`. The UI explicitly says: “Coordinates are interpreted in this CRS and are not transformed.”

API reference: [pyproj database documentation](https://pyproj4.github.io/pyproj/stable/api/database.html).

## D. Root cause of GeoTIFF CRS mismatch

The normal writer already used the generated map's stored CRS; there was no WGS84 fallback in `grid_to_geotiff_bytes`.

The reproducible stale-map path was upstream: `build_layer_model_signature()` omitted CRS. Updating Project CRS changed `session_state.crs`, but left an existing generated map classified as up to date. Export therefore retained that map's previous CRS, which differed from the newly configured project. Selecting settings also required the existing Update CRS button; pending widget choices were not committed project state.

A second defect was in `create_map_scenario()`: if the generated map lacked its `crs` dictionary, it fell back to current `project_context.crs`, even when map export metadata preserved the original EPSG. This could attach the current workspace CRS to an older map.

Additional integrity gaps: preset normalization silently overrode a contradictory EPSG; map CRS and export metadata were not cross-checked. These now fail explicitly.

The historical temporary metadata file shown in the IDE was no longer readable at its supplied path, so the original manual-export artifact itself could not be inspected. The report identifies code paths and regression evidence, rather than claiming a forensic diagnosis of that unavailable file.

## E. GeoTIFF CRS correction

Generation still takes Project CRS and snapshots it. CRS identity now participates in the map change signature. Editing Project CRS marks an old map stale without rewriting its snapshot. Scenario creation and exports resolve map-owned metadata only; they never borrow workspace or geometry CRS. Contradictory map/scenario snapshots are rejected.

Mandatory generation-to-export-to-Rasterio test: `src.crs.to_epsg() == 32638` **PASS**. EPSG:32639 also passes. Local XY exports retain `src.crs is None`. Tests verify affine transform, bounds, north-up orientation, and NoData. ZMAP uses the same snapshot source and preserves X/Y.

## F. Map-pick control workflow

Engineering Controls → Point → Pick on Map → PICK CONTROL LOCATION activates the existing Cartesian component. One click returns unrounded Cartesian X/Y; the next rerun populates Control X/Y and displays a preview marker. Selected panel geometry supplies an assignment or an ambiguity warning. A manual panel selection is available when assignment is unresolved.

Picking alone does not mutate engineering controls or the model signature. Add Control Point commits the control with `Source_Type = Map Pick`; model status becomes stale when the new active control applies to the map. Manual entry remains available with `Source_Type = Manual Entry`.

## G. File-import workflow

Engineering Controls → Point → Import Control Points accepts CSV/XLSX. Required semantics are X, Y, Value. Column selectors support custom headings, with suggestions for standard names and aliases. Optional IDs, property, unit, layer, panel, pressure date, active state, and comments are preserved.

Missing-column defaults and spatial panel assignment are disclosed before the normalized preview. Explicit scopes are validated against the whole project, not only current filters. Invalid numeric values, unknown scopes, duplicate IDs, invalid pressure dates, and invalid Active flags block commit. Existing PASS 3 QC supplies duplicate-coordinate, measured-conflict, and reservoir-domain warnings. Pressure controls conflicting with measured data remain subject to the existing exclusion rules.

IMPORT CONTROL POINTS appends to the existing engineering-control collection. The existing editor, Active flag, and delete controls apply. Imports carry `Source_Type = File Import`. Region-generated controls use `Region Generated`; legacy source labels remain understood.

## H. Scientific treatment

Imported and picked controls condition interpolation. They remain separate from measured observations, experimental variograms, Auto Fit inputs, and LOOCV targets/metrics. Tests cover both new source types, including the existing measured-only kriging Auto Fit test. No interpolation mathematics or methods changed.

## I. Persistence / scenarios

The existing project archive format preserves new controls, source types, scopes, comments, and active states. Scenario control copies remain independent of workspace edits/removal. Scenarios retain map-owned CRS plus explicit `CRS_Mode`, `EPSG`, `CRS_Name`, and `Coordinate_Unit` fields. Transient map-pick state is cleared when opening a project or replacing source data.

## J. Tests added

28 new pytest cases plus two additional parameterizations of the existing Auto Fit test: library search/caching, exact CRS exports and geometry, stale status, immutable scenarios, missing/contradictory snapshots, CSV/XLSX defaults and explicit scopes, invalid imports, QC, spatial ambiguity, persistence, scientific exclusion, and Streamlit widget flows. The standalone JavaScript harness checks actual single-click coordinate emission and polygon drawing regression.

## K. Final results

- `python -B -m compileall app.py core pages tests utils components`: PASS.
- `python -B -m pytest -q`: **158 passed**; 13 existing Rasterio/Affine pending-deprecation warnings.
- `python -B -c "import app; print('OK')"`: **OK**.
- Streamlit: started at `http://localhost:8502`; **HTTP 200**. Port 8501 was occupied.
- `node tests/control_point_component.cjs`: PASS.
- `git diff --check`: PASS. Git emits line-ending conversion notices, with no whitespace errors.

## L. Remaining known issues / verification limits

No connected browser was available. Real browser canvas clicking, visual acceptance, and browser upload/download interaction remain unverified. Streamlit AppTest verifies the CRS widgets, pick-result-to-input/Add workflow, and three-row import preview/commit; the component harness verifies its JavaScript coordinate/event behavior. These are automated checks, not a claim of manual browser acceptance.

Malformed or contradictory saved CRS metadata now fails explicitly and requires correcting the snapshot or regenerating the map. No automatic reprojection or relabeling is performed.

## M. Git confirmation

`main` was not modified or merged. Changes remain uncommitted on `geometry-update`.

Suggested commit: `fix: align CRS exports and improve control point input`
