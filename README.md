# Reservoir Mapping Studio

Reservoir Mapping Studio is a production-oriented Streamlit application for building and comparing 2D reservoir property maps from well observations. It supports deterministic, surface, and geostatistical interpolation methods; engineering control points; soft control regions; panel-aware masking; Reservoir Layer selection; all-layer map batches; Ordinary and Universal Kriging; geometry layers; project persistence; saved map scenarios; side-by-side map comparison; delta maps; and pressure-change workflows while keeping the app focused on engineering map generation.

## Installation

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

## Project Navigation

The main workspace includes:

- Project: create a project, save/open `.rmsproj` archives, manage the dirty state, and maintain CRS metadata.
- Data Manager: import CSV/Excel data, map columns, review QC, filters, and geometry.
- Mapping Studio: generate selected-layer maps or all-layer map batches, apply masks, configure units, run interpolation, configure kriging, style surfaces, switch displayed map snapshots, export maps, and save scenarios from a persistent controls/map workspace.
- Map Comparison: select two saved maps, align grids, compare them visually, and generate delta / pressure-change surfaces.

## Core V1 Workflow

The tested workflow is:

```text
Create Project
  -> Load well data
  -> Load geometry
  -> Map columns
  -> Choose Reservoir Layer scope
  -> Generate selected layer or all layers
  -> Validate
  -> Save map scenario
  -> Generate second map
  -> Compare maps
  -> Calculate delta / pressure change
  -> Export result
  -> Save project
  -> Reopen project
  -> Continue working
```

## Project Persistence

Projects are saved as portable ZIP archives with the `.rmsproj` extension and include a schema version plus JSON/CSV/GeoJSON/NPZ contents as needed.

The archive stores:

- project metadata
- data mapping and units
- filters and included/excluded states
- geometry layers
- interpolation settings
- engineering control points and soft control regions
- selected Reservoir Layer scope, selected panels, panel interpolation mode, interpolation domain, and generated layer maps
- variogram settings
- saved map scenarios
- CRS metadata

This gives a reproducible engineering workspace without relying on Python pickles.

## Saved Maps and Map Library

Use the Map Library to save, rename, duplicate, delete, and load map scenarios. Each saved map keeps the computational definition of the generated surface, including the property, reference date, grid definition, interpolation method, method family, method parameters, engineering controls used for conditioning, mask details, geometry metadata, CRS metadata, uncertainty grids where available, and style settings.

The Mapping Studio map pane includes a `Displayed Map` selector. `Current Workspace` shows the active generated map or context map. Saved scenarios are selected by their internal scenario ID and shown with readable labels such as `Pressure | 01-Jan-2025 | Lower | Kriging`; duplicate labels are disambiguated without changing the saved ID. Selecting a saved scenario is display-only: it renders the stored grid, uncertainty grid, property/unit, layer, reference date, mask/domain metadata, and saved style without rerunning interpolation, changing filters, or mutating the scenario. Use `Load Scenario Into Workspace` only when a saved scenario should become the editable/generated workspace map.

## Reservoir Layers

Reservoir Layer is an explicit vertical mapping dimension, separate from metadata filters and separate from lateral panel geometry. A source column named `Zone` can still be mapped as `Layer`, but `Zone` is not a first-class semantic field in V1.

The V1 model keeps three concepts separate:

- Reservoir Layer: vertical/correlative reservoir subdivision that controls selected-layer or all-layer map outputs.
- Panel: lateral grouping or compartment context, usually from a mapped data `Panel` column and/or imported panel polygons.
- Geometry Type: either Reservoir Boundary or Custom Geometry during new uploads.

Mapping Studio supports:

- Selected Layer: generate one map for the selected reservoir layer.
- All Layers: generate one independent map per active reservoir layer.
- Layer map switching: switch among generated layer maps without recomputing interpolation.
- Map status: generated maps show whether computational inputs are up to date or stale.

Reservoir-layer availability is derived from the active tabular data after ordinary filters, pressure reference date selection, and mapped data-panel selection when a `Panel` column exists. It is not inferred from polygon geometry alone, so loading or selecting panel polygons does not hide valid Reservoir Layer values. In combined panel mode, each layer map pools observations from the selected panels. In independent panel mode, each Reservoir Layer still produces one map, but interpolation is performed separately inside each selected panel/compartment.

## Engineering Controls

Engineering controls are explicit conditioning inputs supplied by the engineer. They are not measured observations and are stored/exported separately from the well observation tables.

Mapping Studio supports:

- a raw measured-point context map before interpolation, including active reservoir boundary, panels, faults, custom geometry, manual controls, and saved control regions
- manual engineering control points with active/inactive status, panel/date/layer/property scope, and comments
- automatic panel assignment from panel polygons, or manual panel selection when only a data panel column exists
- soft control regions drawn directly on the Cartesian Mapping Studio canvas, from selected-well convex hulls with optional buffer, or from loaded polygon geometry
- drawn region validation for finite X/Y vertices, polygon area, safe topology repair, reservoir-domain clipping, selected-panel-domain clipping, layer scope, pressure reference date, target value, and point spacing
- generated region control points on a regular internal spacing constrained to the region and selected panels
- distinct map overlays for Engineering Controls, Control Regions, and optional Generated Region Points
- QC warnings for duplicate controls, measured-well conflicts, overlapping regions with different targets, outside-boundary controls, wrong layer/panel/date scope, missing pressure dates, and non-finite values

All interpolation methods use active matching controls as conditioning points only after Generate/Update is pressed. Drawing a polygon is transient; saving, editing, toggling, duplicating, or deleting controls changes the model signature and marks existing maps stale.

Drawn control regions use the uploaded engineering X/Y coordinate system directly. No lat/lon conversion, basemap, Folium, or Leaflet workflow is used.

## Map Comparison and Delta Maps

The Map Comparison page supports:

- side-by-side map selection
- common-range or independent color scaling
- compatibility checks before arithmetic
- grid alignment when needed
- delta surface calculation
- pressure-change calculation using the pressure map reference date

Delta output follows the rule:

```text
Map B - Map A
```

Pressure change is available only when both saved maps are pressure maps with valid, different pressure map reference dates. The app orders the two maps by date regardless of Map A / Map B selection and labels the result explicitly:

```text
Pressure Change 01-Jan-2026 minus 01-Jan-2025
```

Negative values indicate pressure decline.

## CRS Awareness

Project-level CRS support is optional and non-blocking. Supported modes are:

- Local / Unknown XY
- EPSG:32638 - WGS 84 / UTM zone 38N
- Custom EPSG

The application validates Custom EPSG values with `pyproj`, stores the resolved CRS metadata, and uses it for project/map compatibility checks and export metadata.

V1 does not transform or reproject coordinates. The configured CRS means: the supplied X/Y coordinates are interpreted in this CRS. EPSG:32638 is a projected CRS defined in meters; if the project coordinate unit is not meters, the app warns the user but does not convert any values.

## Units and Pressure Semantics

The app preserves:

- coordinate units such as meters and feet
- property units such as psi, bar, kPa, %, and field-specific units
- original measurement dates versus pressure reference dates

The pressure workflow remains date-aware but does not perform temporal extrapolation. The `Pressure_Map_Reference_Date` is the authoritative date for map comparisons, not the original well test date.

## Geometry and Masking

The application supports:

- reservoir boundary polygon masking
- legacy panel/compartment masking and assignment for existing projects
- legacy fault display and validation for existing projects
- custom geometry layers containing supported polygons or lines
- maximum-distance and convex-hull mask combinations

New geometry uploads are classified only as `Reservoir Boundary` or `Custom Geometry`. Reservoir Boundary remains special because it can define the interpolation domain, spatial mask, or outer map boundary and must be Polygon/MultiPolygon. Custom Geometry can store Polygon, MultiPolygon, LineString, or MultiLineString features as overlays/reference geometry while preserving source attributes. Files that older projects stored as Panel / Compartment or Fault continue to load and display.

Advanced panel behavior is chosen contextually in Mapping Studio. Combined Selected Panels can operate from the dataframe `Panel` field alone. Independent by Panel / Compartment requires compatible polygon boundary geometry; a legacy panel layer or a custom polygon layer can be selected as `Panel Boundary Geometry`.

Spatial Domain and Spatial Mask are separate controls:

- Interpolation Domain controls the rectangular grid extent used for interpolation: Well Data Extent, Reservoir Boundary Extent, or Selected Panel Union Extent.
- Spatial Mask controls which interpolated grid cells remain valid after interpolation: no mask, reservoir boundary, selected panel union, maximum distance, convex hull, or supported combinations.
- Selected Panel Union Extent uses the union bounds of selected panel polygons. In combined mode, there is no artificial internal panel gap across shared panel boundaries.
- Independent by Panel / Compartment uses the same Reservoir Layer output model, but computes each selected panel from its own assigned observations and stores the compartment grid for traceability.

## Interpolation and Kriging

Supported methods include:

- Basic: IDW, Linear, Cubic, RBF
- Geological / Surface: Natural Neighbor, Minimum Curvature, Convergent Interpolation
- Geostatistical: Ordinary Kriging, Universal Kriging

Ordinary Kriging and Universal Kriging are configured inside Mapping Studio. Ordinary Kriging uses the constant-mean assumption. Universal Kriging supports a deterministic XY trend, starting with Linear XY and optionally Quadratic XY. Variogram model, auto-fit/manual mode, practical range, sill/variance, nugget, anisotropy, local-neighborhood settings, uncertainty display, and compact diagnostics remain available without exposing a standalone geostatistics page in normal navigation.

Natural Neighbor uses bounded Voronoi-cell area weights. It honors conditioning points exactly and does not extrapolate outside the conditioning-point support. Because it computes true bounded cell intersections, it is conservative and can be slower on large grids.

Minimum Curvature is implemented as a thin-plate spline biharmonic minimum-bending-energy surface with optional smoothing. It generates a global smooth surface over the selected computational domain before masks are applied.

Convergent Interpolation is Reservoir Mapping Studio's iterative convergent interpolation implementation inspired by standard iterative surface-conditioning concepts. It is not an exact Petrel algorithm reproduction. The implementation builds an initial smooth surface, samples that surface at conditioning points, computes residuals as `Observed - Estimated`, interpolates residual corrections, updates the surface with a relaxation factor, and repeats until RMSE reaches the target tolerance, RMSE improvement falls below tolerance, or maximum iterations is reached. Generated maps store iterations, initial RMSE, final RMSE, tolerance, convergence status, and RMSE history.

Engineering controls condition interpolation estimates, but they are not measured truth. Validation metrics preserve the measured/synthetic distinction and do not let synthetic controls artificially improve measured LOOCV statistics.

## Exports

The app supports:

- CSV and Excel exports
- separate engineering-control CSV and Excel exports
- map image exports in PNG, SVG, and PDF
- XYZ ASCII export
- ZMAP Grid ASCII export
- metadata JSON export
- GeoTIFF export with embedded EPSG CRS when known, or local XY geometry with no CRS when unknown

## Sample Data

Bundled example inputs include:

- `data/sample_reservoir_data.csv`
- `data/sample_reservoir_boundary.geojson`
- `data/sample_panels.geojson`
- `data/sample_faults.geojson`

## Testing

Run:

```bash
python -B -m compileall .
python -B -m pytest -q
python -B -c "import app"
```

## Optional Features Policy

Secondary features such as ZMAP, GeoTIFF, and PDF export are wired through the shared V1 export helpers so grid orientation, CRS metadata, NoData, and scenario snapshots remain consistent. Core engineering flow remains the priority: project persistence, saved maps, comparison, delta maps, and export quality.

- Observed vs Predicted scatter with a 1:1 reference line
- Residual histogram
- Residual map with signed residual color and absolute-error marker sizing

## Algorithm Comparison

The Method Comparison tab compares selected methods using the same active observations and LOOCV framework. It reports the validation metrics side by side and highlights the lowest validation RMSE without automatically changing the engineer's selected map.

## Exports

Filtered observations can be downloaded as CSV or Excel.

Interpolated grid exports include:

- X
- Y
- Value for deterministic methods
- Estimated_Value for kriging outputs
- Kriging_Variance where available
- Kriging_StdDev where available
- Panel where panel-constrained interpolation is active

Excel grid export includes `Interpolated_Grid` and `Map_Metadata` sheets. When applicable, it also includes `Engineering_Controls`, `Control_Regions`, and `Validation` sheets. Engineering controls are exported separately from measured observations with control ID, source type, region ID, X/Y, property, value, unit, Reservoir Layer, panel, pressure reference date, active status, and comment.

Export metadata records the domain/mask split with `Interpolation_Domain_Type`, `Domain_Bounds`, `Domain_Geometry_Source`, `Mask`, selected panels, selected Reservoir Layer, panel interpolation mode, CRS, grid dimensions, and NoData value. Saved scenarios snapshot this metadata so reopening a project or map does not change the export definition.

### GeoTIFF

GeoTIFF export writes the generated MapResult grid exactly; it does not regenerate interpolation. Grid X/Y arrays are treated as cell centers. The affine transform uses west and north edge coordinates calculated one half grid spacing outside the center limits. Raster rows are written north-to-south and columns west-to-east, matching north-up GIS raster convention while preserving the app's south-to-north NumPy meshgrid order.

Known EPSG CRS settings are embedded in the raster. Local / Unknown XY exports are allowed; those rasters preserve local Cartesian coordinates, pixel size, bounds, NoData, and masks, but contain no EPSG spatial reference. Masked and invalid cells are written as GeoTIFF NoData (`-9999.0`). Reservoir boundary masks, panel masks, maximum-distance masks, Linear/Cubic NaNs, internal NaNs, and delta-map non-overlap are preserved.

All-layer export writes one GeoTIFF per generated Reservoir Layer, not one multi-band raster. Ordinary Kriging exports include estimate, kriging variance, and kriging standard deviation rasters with identical transform, CRS, bounds, and dimensions.

### ZMAP Grid ASCII

Reservoir Mapping Studio V1 writes a documented ZMAP-style gridded ASCII dialect:

- comment lines start with `!` and include Reservoir Mapping Studio metadata
- core header starts with `@GRID FILE, GRID, 4`
- format line is `values_per_line, null_value, , decimal_places, 1`
- geometry line is `NX, NY, X_MIN, X_MAX, Y_MIN, Y_MAX`
- rotation/origin line is `0.0, 0.0, 0.0`
- a single `@` line precedes grid values
- X/Y header min/max are grid cell-center limits
- grid spacing is `(X_MAX - X_MIN) / (NX - 1)` and `(Y_MAX - Y_MIN) / (NY - 1)`
- null value is `-999.25`
- numeric values use fixed decimal formatting with 6 decimal places
- grid values are written north-to-south by row and west-to-east by column

ZMAP exports preserve projected or local Cartesian X/Y values exactly. EPSG:32638 and Custom EPSG settings are recorded in comments and companion metadata JSON; Local / Unknown XY records the coordinate unit and unknown CRS state. All-layer export writes one ZMAP file per generated Reservoir Layer. This is described as ZMAP Grid ASCII / RMS V1 ZMAP-style gridded ASCII; no Petrel compatibility claim is made without external import validation.

Kriging diagnostics and validation exports remain available through the Mapping Studio workflow and supporting geostatistics modules. The standalone Geostatistics Lab page is hidden from normal V1 navigation, but the backend modules are still used for Ordinary Kriging, Universal Kriging, variogram models, validation, and uncertainty.

## Sample Data

Bundled sample files:

- `data/sample_reservoir_data.csv`
- `data/sample_reservoir_data.xlsx`
- `data/sample_reservoir_boundary.geojson`
- `data/sample_panels.geojson`
- `data/sample_faults.geojson`

The sample has 42 synthetic wells in North, Central, and South panels. Pressure values use one common `Pressure_Map_Reference_Date` of `2026-01-01`; original measurement dates vary by well. North pressure is higher, Central is intermediate, and South is lower so unrestricted interpolation can be compared against panel-constrained interpolation. The sample also includes a duplicate coordinate pair, one missing pressure value, missing coordinate examples, multiple numeric properties, and one intentional pressure outlier for QC testing.

## Testing

Run:

```bash
python -m compileall .
python -m pytest -q
```

The automated suite covers Level 1.1 regression behavior plus Level 2 geometry masks, panel assignment, overlap detection, compartment interpolation, variogram calculation, candidate fitting, kriging estimate/uncertainty output, validation metrics, explicit Reservoir Layer generation, all-layer batches, PASS 4.1 spatial domain decoupling, stale map signatures, engineering control scoping/conditioning/persistence, pressure-change guards, CRS round trips, GeoTIFF and ZMAP round trips, saved-scenario display switching, contour-line toggling, advanced interpolation methods, and project persistence.

## Current Limitations

This level does not implement CRS reprojection/transformation, barrier/fault Kriging, Regression Kriging, Co-Kriging, Sequential Gaussian Simulation, 3D mapping, corner-point grids, interactive GIS vertex editing, database backend/authentication, pressure datum-depth correction, or temporal pressure extrapolation.

Static PNG, SVG, and PDF image export uses Plotly/Kaleido. If Kaleido is missing or unavailable, the app shows an export failure message instead of crashing.
