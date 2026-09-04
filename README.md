# Reservoir Mapping Studio

Reservoir Mapping Studio is a production-oriented Streamlit application for building and comparing 2D reservoir property maps from well observations. It supports deterministic interpolation, engineering control points, soft control regions, panel-aware masking, Reservoir Layer selection, all-layer map batches, Ordinary Kriging, geometry layers, project persistence, saved map scenarios, side-by-side map comparison, delta maps, and pressure-change workflows while keeping the app focused on engineering use rather than geoscience research tooling.

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
- Mapping Studio: generate selected-layer maps or all-layer map batches, apply masks, configure units, run interpolation, style surfaces, and save map scenarios from a persistent controls/map workspace.
- Geostatistics Lab: fit variograms, run cross validation, inspect residuals, and compare methods for the active Reservoir Layer.
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
- selected Reservoir Layer scope and generated layer maps
- variogram settings
- saved map scenarios
- CRS metadata

This gives a reproducible engineering workspace without relying on Python pickles.

## Saved Maps and Map Library

Use the Map Library to save, rename, duplicate, delete, and reopen map scenarios. Each saved map keeps the computational definition of the generated surface, including the property, reference date, grid definition, interpolation method, engineering controls used for conditioning, mask details, geometry metadata, and style settings.

## Reservoir Layers

Reservoir Layer is an explicit mapping dimension, separate from metadata filters. A source column named `Zone` can still be mapped as `Layer`, but `Zone` is not a first-class semantic field in V1.

Mapping Studio supports:

- Selected Layer: generate one map for the selected reservoir layer.
- All Layers: generate one independent map per active reservoir layer.
- Layer map switching: switch among generated layer maps without recomputing interpolation.
- Map status: generated maps show whether computational inputs are up to date or stale.

Panel selection is applied before layer splitting. In combined panel mode, each layer map pools observations from the selected panels. In independent panel mode, each layer map respects panel/compartment boundaries.

## Engineering Controls

Engineering controls are explicit conditioning inputs supplied by the engineer. They are not measured observations and are stored/exported separately from the well observation tables.

Mapping Studio supports:

- manual engineering control points with active/inactive status, panel/date/layer/property scope, and comments
- automatic panel assignment from panel polygons, or manual panel selection when only a data panel column exists
- soft control regions from selected-well convex hulls with optional buffer, or from loaded polygon geometry
- generated region control points on a regular internal spacing constrained to the region and selected panels
- distinct map overlays for Engineering Controls, Control Regions, and optional Generated Region Points
- QC warnings for duplicate controls, measured-well conflicts, overlapping regions with different targets, outside-boundary controls, wrong layer/panel/date scope, missing pressure dates, and non-finite values

All interpolation methods use active matching controls as conditioning points only after Generate/Update is pressed. Adding, editing, toggling, or deleting controls changes the model signature and marks existing maps stale.

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
- EPSG Code

The application validates EPSG values with `pyproj`, stores the resolved CRS metadata, and uses it for project/map compatibility checks and export metadata.

## Units and Pressure Semantics

The app preserves:

- coordinate units such as meters and feet
- property units such as psi, bar, kPa, %, and field-specific units
- original measurement dates versus pressure reference dates

The pressure workflow remains date-aware but does not perform temporal extrapolation. The `Pressure_Map_Reference_Date` is the authoritative date for map comparisons, not the original well test date.

## Geometry and Masking

The application supports:

- reservoir boundary polygon masking
- panel/compartment masking and assignment
- fault display and validation
- custom geometry layers
- maximum-distance and convex-hull mask combinations

## Interpolation and Kriging

Supported methods include:

- IDW
- Linear
- Cubic
- RBF
- Ordinary Kriging

The geostatistics workflow includes experimental variogram analysis, model fitting, anisotropy, kriging uncertainty, validation metrics, and residual diagnostics. Engineering controls condition kriging estimates and validation predictions, but they are excluded from experimental variograms, variogram auto-fit, LOOCV target sets, and validation metrics.

## Exports

The app supports:

- CSV and Excel exports
- separate engineering-control CSV and Excel exports
- map image exports in PNG, SVG, and PDF
- XYZ ASCII export
- metadata JSON export
- GeoTIFF export when a valid EPSG CRS is present

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

## Known Limitations

This V1 is intentionally focused and not a full GIS or reservoir modeling platform. It does not implement:

- Petrel replacement
- full GIS system
- 3D geological grids
- dynamic simulation
- material balance
- co-kriging
- sequential Gaussian simulation
- universal kriging beyond the existing scope
- database backend / authentication

## Optional Features Policy

Secondary features such as ZMAP, GeoTIFF, and PDF export remain supported only when they are stable and reliable. Core engineering flow remains the priority: project persistence, saved maps, comparison, delta maps, and export quality.

- Observed vs Predicted scatter with a 1:1 reference line
- Residual histogram
- Residual map with signed residual color and absolute-error marker sizing

These diagnostics live in the Geostatistics Lab so the main map remains focused on map generation.

## Algorithm Comparison

The Method Comparison tab compares selected methods using the same active observations and LOOCV framework. It reports the validation metrics side by side and highlights the lowest validation RMSE without automatically changing the engineer's selected map.

## Exports

Filtered observations can be downloaded as CSV or Excel.

Interpolated grid exports include:

- X
- Y
- Property value for deterministic methods
- Estimated_Value for kriging outputs
- Kriging_Variance where available
- Kriging_StdDev where available
- Panel where panel-constrained interpolation is active

Excel grid export includes a `Map_Metadata` sheet with property, units, grid setup, duplicate handling, mask method, Reservoir Layer scope, geometry context, custom layer count, kriging parameters, anisotropy settings, and distance/range units.

Cross-validation results can be downloaded as CSV or Excel from the Geostatistics Lab.

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

The automated suite covers Level 1.1 regression behavior plus Level 2 geometry masks, panel assignment, overlap detection, compartment interpolation, variogram calculation, candidate fitting, kriging estimate/uncertainty output, validation metrics, explicit Reservoir Layer generation, all-layer batches, stale map signatures, engineering control scoping/conditioning/persistence, pressure-change guards, and project persistence.

## Current Limitations

This level does not implement Universal Kriging, Regression Kriging, Co-Kriging, Sequential Gaussian Simulation, mathematical barrier Kriging, CRS transformations, 3D geostatistics, GeoTIFF expansion beyond the existing EPSG-gated grid export, Petrel/ZMAP export, database connections, pressure datum-depth correction, or temporal pressure extrapolation.

Static map image export is still left to the Plotly mode bar/browser workflow rather than a dedicated server-side image export dependency.
