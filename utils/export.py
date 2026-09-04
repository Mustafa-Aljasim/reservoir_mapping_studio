"""Export helpers for tabular data and interpolated grids."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from io import BytesIO
import json
import re
import zipfile

import numpy as np
import pandas as pd

from core.crs import LOCAL_CRS_MODE, normalize_crs_config
from core.engineering_controls import serialize_control_regions
from core.scenarios import json_safe
from utils.constants import APP_NAME
from utils.constants import INTERNAL_ROW_ID
from utils.units import coordinate_unit_symbol


GEOTIFF_NODATA = -9999.0
ZMAP_NULL_VALUE = -999.25
ZMAP_VALUES_PER_LINE = 5
ZMAP_DECIMAL_PLACES = 6
ZMAP_DIALECT = "Reservoir Mapping Studio V1 ZMAP-style Grid ASCII"


@dataclass(frozen=True)
class RegularGridDefinition:
    """Regular cell-center grid definition used by all gridded exports."""

    x_axis: np.ndarray
    y_axis: np.ndarray
    x_order_west_to_east: np.ndarray
    y_order_north_to_south: np.ndarray
    dx: float
    dy: float
    x_min: float
    x_max: float
    y_min: float
    y_max: float
    west: float
    east: float
    south: float
    north: float
    width: int
    height: int

    def metadata(self) -> dict[str, object]:
        return {
            "NX": self.width,
            "NY": self.height,
            "Grid_NX": self.width,
            "Grid_NY": self.height,
            "X_Min": self.x_min,
            "X_Max": self.x_max,
            "Y_Min": self.y_min,
            "Y_Max": self.y_max,
            "X_Spacing": self.dx,
            "Y_Spacing": self.dy,
            "Bounds": {
                "west_edge": self.west,
                "east_edge": self.east,
                "south_edge": self.south,
                "north_edge": self.north,
                "x_min_center": self.x_min,
                "x_max_center": self.x_max,
                "y_min_center": self.y_min,
                "y_max_center": self.y_max,
            },
            "Grid_Node_Convention": (
                "Grid X/Y arrays are cell centers; raster edge bounds are one half spacing outside the center limits."
            ),
        }


@dataclass(frozen=True)
class MapValueExport:
    key: str
    label: str
    filename_suffix: str
    value_column: str
    unit: str
    grid: np.ndarray


def dataframe_to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


def dataframe_to_excel_bytes(df: pd.DataFrame, sheet_name: str = "Data") -> bytes:
    buffer = BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name=sheet_name[:31] or "Data")
    return buffer.getvalue()


def metadata_to_dataframe(metadata: dict[str, object]) -> pd.DataFrame:
    def cell_value(value):
        safe = json_safe(value)
        if isinstance(safe, (dict, list, tuple)):
            return json.dumps(safe, default=str)
        return safe

    return pd.DataFrame(
        [{"Parameter": key, "Value": cell_value(value)} for key, value in metadata.items()]
    )


def grid_to_excel_bytes(
    grid_df: pd.DataFrame,
    metadata: dict[str, object] | None = None,
    engineering_controls: pd.DataFrame | None = None,
    control_regions: pd.DataFrame | None = None,
    validation_df: pd.DataFrame | None = None,
) -> bytes:
    buffer = BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        grid_df.to_excel(writer, index=False, sheet_name="Interpolated_Grid")
        if metadata:
            metadata_to_dataframe(metadata).to_excel(writer, index=False, sheet_name="Map_Metadata")
        if engineering_controls is not None and not engineering_controls.empty:
            engineering_controls.to_excel(writer, index=False, sheet_name="Engineering_Controls")
        if control_regions is not None and not control_regions.empty:
            control_regions.to_excel(writer, index=False, sheet_name="Control_Regions")
        if validation_df is not None and not validation_df.empty:
            validation_df.to_excel(writer, index=False, sheet_name="Validation")
    return buffer.getvalue()


def metadata_to_json_bytes(metadata: dict[str, object]) -> bytes:
    return json.dumps(json_safe(metadata), indent=2, default=str).encode("utf-8")


def figure_to_image_bytes(
    figure,
    image_format: str = "png",
    width: int = 1600,
    height: int = 1000,
    scale: float = 2.0,
) -> bytes:
    try:
        import plotly.io as pio
    except ImportError as exc:  # pragma: no cover - plotly is a core dependency
        raise RuntimeError("Plotly is required for static image export.") from exc
    try:
        return pio.to_image(
            figure,
            format=image_format.lower(),
            width=int(width),
            height=int(height),
            scale=float(scale),
        )
    except Exception as exc:
        raise RuntimeError(f"Static {image_format.upper()} export failed: {exc}") from exc


def grid_to_dataframe(
    grid_x: np.ndarray,
    grid_y: np.ndarray,
    grid_z: np.ndarray,
    property_name: str,
    include_nan: bool = False,
    grid_variance: np.ndarray | None = None,
    panel_grid: np.ndarray | None = None,
) -> pd.DataFrame:
    value_column = "Estimated_Value" if grid_variance is not None else "Value"
    data = {
        "X": np.asarray(grid_x).ravel(),
        "Y": np.asarray(grid_y).ravel(),
        value_column: np.asarray(grid_z).ravel(),
    }
    if grid_variance is not None:
        variance = np.asarray(grid_variance).ravel()
        data["Kriging_Variance"] = variance
        data["Kriging_StdDev"] = np.sqrt(np.maximum(variance, 0.0))
    if panel_grid is not None:
        data["Panel"] = np.asarray(panel_grid).ravel()
    output = pd.DataFrame(data)
    if not include_nan:
        output = output.dropna(subset=[value_column])
    return output.reset_index(drop=True)


def grid_to_xyz_ascii_bytes(grid_df: pd.DataFrame, value_column: str | None = None) -> bytes:
    if grid_df.empty:
        return b""
    if value_column is None:
        candidates = [column for column in grid_df.columns if column not in {"X", "Y", "Panel", "Kriging_Variance", "Kriging_StdDev"}]
        value_column = candidates[0] if candidates else grid_df.columns[-1]
    lines = [f"# Generic XYZ ASCII export", f"# Columns: X Y {value_column}"]
    for _, row in grid_df.dropna(subset=["X", "Y", value_column]).iterrows():
        lines.append(f"{float(row['X']):.10g} {float(row['Y']):.10g} {float(row[value_column]):.10g}")
    return ("\n".join(lines) + "\n").encode("utf-8")


def regular_grid_definition(grid_x: np.ndarray, grid_y: np.ndarray) -> RegularGridDefinition:
    """Analyze RMS cell-center meshgrid arrays and return an explicit export grid."""

    x = np.asarray(grid_x, dtype=float)
    y = np.asarray(grid_y, dtype=float)
    if x.ndim != 2 or y.ndim != 2 or x.shape != y.shape:
        raise ValueError("Gridded export requires matching 2D grid X/Y arrays.")
    if x.shape[0] < 2 or x.shape[1] < 2:
        raise ValueError("Gridded export requires at least a 2 by 2 grid.")
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("Grid X/Y coordinates must be finite for gridded export.")

    x_axis = np.asarray(x[0, :], dtype=float)
    y_axis = np.asarray(y[:, 0], dtype=float)
    if not np.allclose(x, x_axis[None, :], rtol=1e-9, atol=1e-9):
        raise ValueError("Grid X coordinates must be regular by column.")
    if not np.allclose(y, y_axis[:, None], rtol=1e-9, atol=1e-9):
        raise ValueError("Grid Y coordinates must be regular by row.")

    x_order = np.argsort(x_axis)
    y_south_order = np.argsort(y_axis)
    x_sorted = x_axis[x_order]
    y_sorted = y_axis[y_south_order]
    x_diff = np.diff(x_sorted)
    y_diff = np.diff(y_sorted)
    if np.any(x_diff <= 0) or np.any(y_diff <= 0):
        raise ValueError("Grid coordinates must be unique along each axis.")
    dx = float(np.nanmedian(x_diff))
    dy = float(np.nanmedian(y_diff))
    if dx <= 0 or dy <= 0:
        raise ValueError("Gridded export requires positive grid spacing.")
    if not np.allclose(x_diff, dx, rtol=1e-6, atol=max(abs(dx), 1.0) * 1e-9):
        raise ValueError("Gridded export requires regular X spacing.")
    if not np.allclose(y_diff, dy, rtol=1e-6, atol=max(abs(dy), 1.0) * 1e-9):
        raise ValueError("Gridded export requires regular Y spacing.")

    x_min = float(x_sorted[0])
    x_max = float(x_sorted[-1])
    y_min = float(y_sorted[0])
    y_max = float(y_sorted[-1])
    return RegularGridDefinition(
        x_axis=x_axis,
        y_axis=y_axis,
        x_order_west_to_east=x_order,
        y_order_north_to_south=y_south_order[::-1],
        dx=dx,
        dy=dy,
        x_min=x_min,
        x_max=x_max,
        y_min=y_min,
        y_max=y_max,
        west=x_min - dx / 2.0,
        east=x_max + dx / 2.0,
        south=y_min - dy / 2.0,
        north=y_max + dy / 2.0,
        width=int(x.shape[1]),
        height=int(x.shape[0]),
    )


def orient_grid_north_to_south(
    grid_x: np.ndarray,
    grid_y: np.ndarray,
    grid_z: np.ndarray,
) -> tuple[RegularGridDefinition, np.ndarray]:
    definition = regular_grid_definition(grid_x, grid_y)
    z = np.asarray(grid_z, dtype=float)
    if z.shape != (definition.height, definition.width):
        raise ValueError("Gridded export requires grid Z to match grid X/Y dimensions.")
    data = z[np.ix_(definition.y_order_north_to_south, definition.x_order_west_to_east)]
    return definition, np.asarray(data, dtype=float)


def _coerce_crs(config_or_epsg) -> dict[str, object]:
    if isinstance(config_or_epsg, dict):
        return normalize_crs_config(config_or_epsg)
    if config_or_epsg in (None, ""):
        return normalize_crs_config(None)
    return normalize_crs_config({"mode": "Custom EPSG", "epsg": config_or_epsg})


def grid_to_geotiff_bytes(
    grid_x: np.ndarray,
    grid_y: np.ndarray,
    grid_z: np.ndarray,
    crs: dict[str, object] | int | str | None = None,
    nodata: float = GEOTIFF_NODATA,
    metadata: dict[str, object] | None = None,
) -> bytes:
    try:
        from rasterio.io import MemoryFile
        from rasterio.transform import from_origin
    except ImportError as exc:  # pragma: no cover - dependency is installed in V1 requirements
        raise RuntimeError("Rasterio is required for GeoTIFF export.") from exc

    definition, data = orient_grid_north_to_south(grid_x, grid_y, grid_z)
    data = np.where(np.isfinite(data), data, nodata).astype("float32")
    crs_config = _coerce_crs(crs)
    raster_crs = f"EPSG:{int(crs_config['epsg'])}" if crs_config.get("epsg") else None
    transform = from_origin(definition.west, definition.north, definition.dx, definition.dy)
    with MemoryFile() as memory_file:
        with memory_file.open(
            driver="GTiff",
            height=definition.height,
            width=definition.width,
            count=1,
            dtype="float32",
            crs=raster_crs,
            transform=transform,
            nodata=nodata,
        ) as dataset:
            dataset.write(data, 1)
            if metadata:
                tag_values = {
                    str(key): str(value)
                    for key, value in metadata.items()
                    if isinstance(value, (str, int, float, bool)) and value not in (None, "")
                }
                if tag_values:
                    dataset.update_tags(**tag_values)
        return memory_file.read()


def _metadata_seed(map_result: dict[str, object]) -> dict[str, object]:
    metadata = map_result.get("export_metadata")
    if isinstance(metadata, dict) and metadata:
        return dict(metadata)
    metadata = map_result.get("metadata")
    return dict(metadata) if isinstance(metadata, dict) else {}


def _crs_from_map_result(map_result: dict[str, object], seed: dict[str, object] | None = None) -> dict[str, object]:
    if map_result.get("crs"):
        return _coerce_crs(map_result.get("crs"))
    seed = seed or _metadata_seed(map_result)
    epsg = seed.get("CRS_EPSG") or seed.get("EPSG")
    if epsg not in (None, ""):
        return _coerce_crs({"mode": seed.get("CRS_Mode") or "Custom EPSG", "epsg": epsg})
    return normalize_crs_config(None)


def _iso_text(value) -> str:
    if value in (None, ""):
        return ""
    if hasattr(value, "isoformat"):
        return str(value.isoformat())[:10]
    return str(value)[:10]


def _project_schema_version() -> str:
    try:
        from core.project_io import PROJECT_SCHEMA_VERSION
    except Exception:
        return ""
    return PROJECT_SCHEMA_VERSION


def _scenario_value(scenario: dict[str, object] | None, key: str, default=""):
    if not scenario:
        return default
    return scenario.get(key, default)


def value_exports_for_map(
    map_result: dict[str, object],
    include_uncertainty: bool = True,
) -> list[MapValueExport]:
    property_name = str(map_result.get("property_col") or map_result.get("property") or "Value")
    unit = str(map_result.get("unit") or map_result.get("property_unit") or "")
    grid_z = np.asarray(map_result["grid_z"], dtype=float)
    variance = map_result.get("grid_variance")
    has_variance = variance is not None
    estimate_label = property_name if str(map_result.get("method") or "").lower() == "delta" else "Estimated Property"
    exports = [
        MapValueExport(
            key="estimate",
            label=estimate_label,
            filename_suffix="Estimate" if has_variance else "",
            value_column="Estimated_Value" if has_variance else "Value",
            unit=unit,
            grid=grid_z,
        )
    ]
    if include_uncertainty and has_variance:
        variance_grid = np.asarray(variance, dtype=float)
        exports.append(
            MapValueExport(
                key="kriging_variance",
                label="Kriging Variance",
                filename_suffix="KrigingVariance",
                value_column="Kriging_Variance",
                unit=f"{unit}^2" if unit else "property unit^2",
                grid=variance_grid,
            )
        )
        exports.append(
            MapValueExport(
                key="kriging_stddev",
                label="Kriging Standard Deviation",
                filename_suffix="KrigingStdDev",
                value_column="Kriging_StdDev",
                unit=unit,
                grid=np.sqrt(np.maximum(variance_grid, 0.0)),
            )
        )
    return exports


def engineering_controls_export_dataframe(map_result: dict[str, object]) -> pd.DataFrame:
    property_name = str(map_result.get("property_col") or map_result.get("property") or "")
    property_unit = str(map_result.get("unit") or map_result.get("property_unit") or "")
    reservoir_layer = str(map_result.get("reservoir_layer") or "")
    pressure_reference_date = _iso_text(map_result.get("map_reference_date") or map_result.get("pressure_reference_date"))
    controls = map_result.get("engineering_controls", pd.DataFrame())
    if not isinstance(controls, pd.DataFrame):
        controls = pd.DataFrame(controls)
    rows: list[dict[str, object]] = []
    for _, row in controls.iterrows():
        value = row.get("Value")
        if value in (None, "") and property_name in controls.columns:
            value = row.get(property_name)
        rows.append(
            {
                "Control_ID": row.get("Control_ID", ""),
                "Source_Type": row.get("Source_Type", ""),
                "Region_ID": row.get("Region_ID", ""),
                "X": row.get("X", row.get(map_result.get("x_col") or "X", "")),
                "Y": row.get("Y", row.get(map_result.get("y_col") or "Y", "")),
                "Property": row.get("Property", property_name),
                "Value": value,
                "Unit": row.get("Property_Unit", property_unit),
                "Reservoir_Layer": row.get("Reservoir_Layer", row.get("Layer", reservoir_layer)),
                "Panel": row.get("Panel", ""),
                "Pressure_Reference_Date": row.get("Pressure_Map_Reference_Date", pressure_reference_date),
                "Active": row.get("Active", True),
                "Comment": row.get("Comment", ""),
            }
        )
    return pd.DataFrame(
        rows,
        columns=[
            "Control_ID",
            "Source_Type",
            "Region_ID",
            "X",
            "Y",
            "Property",
            "Value",
            "Unit",
            "Reservoir_Layer",
            "Panel",
            "Pressure_Reference_Date",
            "Active",
            "Comment",
        ],
    )


def control_regions_export_dataframe(map_result: dict[str, object]) -> pd.DataFrame:
    regions = map_result.get("engineering_control_regions", []) or []
    rows: list[dict[str, object]] = []
    for region in regions:
        rows.append(
            {
                "Region_ID": region.get("Region_ID", ""),
                "Region_Name": region.get("Region_Name", ""),
                "Region_Source": region.get("Region_Source", ""),
                "Property": region.get("Property", map_result.get("property_col", "")),
                "Target_Value": region.get("Target_Value", ""),
                "Unit": region.get("Property_Unit", map_result.get("unit", "")),
                "Reservoir_Layer": region.get("Reservoir_Layer", map_result.get("reservoir_layer", "")),
                "Panel_Context": region.get("Panel", ""),
                "Pressure_Reference_Date": region.get("Pressure_Map_Reference_Date", ""),
                "Control_Point_Spacing": region.get("Control_Point_Spacing", ""),
                "Generated_Control_Count": region.get("Generated_Control_Count", ""),
                "Active": region.get("Active", True),
                "Comment": region.get("Comment", ""),
            }
        )
    return pd.DataFrame(rows)


def build_map_export_metadata(
    map_result: dict[str, object],
    *,
    project_metadata: dict[str, object] | None = None,
    scenario: dict[str, object] | None = None,
    value_export: MapValueExport | None = None,
    nodata: float = GEOTIFF_NODATA,
    extra: dict[str, object] | None = None,
) -> dict[str, object]:
    seed = _metadata_seed(map_result)
    crs = _crs_from_map_result(map_result, seed)
    grid_definition = regular_grid_definition(map_result["grid_x"], map_result["grid_y"])
    metadata = dict(seed)
    property_name = str(map_result.get("property_col") or map_result.get("property") or metadata.get("Property") or "Value")
    property_unit = str(map_result.get("unit") or map_result.get("property_unit") or metadata.get("Property_Unit") or "")
    selected_panels = map_result.get("selected_panels", metadata.get("Selected_Panels", []))
    selected_layers = map_result.get("selected_layers", metadata.get("Selected_Layers", []))
    value_unit = value_export.unit if value_export else property_unit
    value_label = value_export.label if value_export else "Estimated Property"
    interpolation_domain = (
        map_result.get("interpolation_domain")
        or metadata.get("Interpolation_Domain_Type")
        or metadata.get("Interpolation_Domain")
        or ""
    )
    domain_bounds = map_result.get("domain_bounds") or metadata.get("Domain_Bounds")
    if isinstance(domain_bounds, (list, tuple)) and len(domain_bounds) >= 4:
        domain_bounds = {
            "min_x": float(domain_bounds[0]),
            "min_y": float(domain_bounds[1]),
            "max_x": float(domain_bounds[2]),
            "max_y": float(domain_bounds[3]),
        }
    if interpolation_domain == "Reservoir Boundary Extent":
        domain_source = "Reservoir Boundary"
    elif interpolation_domain in {"Selected Panel Union Extent", "Selected Panel Extent"}:
        domain_source = "Selected Panel Union"
        interpolation_domain = "Selected Panel Union Extent"
    elif interpolation_domain:
        domain_source = "Well Data"
    else:
        domain_source = metadata.get("Domain_Geometry_Source", "")
    metadata.update(
        {
            "Application": APP_NAME,
            "Project_Schema_Version": _project_schema_version(),
            "Project_Name": (project_metadata or {}).get("name", ""),
            "Scenario_ID": _scenario_value(scenario, "id", map_result.get("scenario_id", metadata.get("Scenario_ID", ""))),
            "Scenario_Name": _scenario_value(
                scenario,
                "name",
                map_result.get("name") or map_result.get("title") or metadata.get("Scenario_Name", ""),
            ),
            "Property": property_name,
            "Property_Type": map_result.get("property_type")
            or metadata.get("Property_Type")
            or ("Pressure" if map_result.get("is_pressure_map") else "Generic"),
            "Property_Unit": property_unit,
            "Export_Value": value_label,
            "Export_Unit": value_unit,
            "Pressure_Map_Reference_Date": _iso_text(
                map_result.get("map_reference_date")
                or map_result.get("pressure_reference_date")
                or metadata.get("Pressure_Map_Reference_Date")
            ),
            "Reservoir_Layer": map_result.get("reservoir_layer") or metadata.get("Reservoir_Layer", ""),
            "Layer_Mapping_Scope": map_result.get("layer_mapping_scope") or metadata.get("Layer_Mapping_Scope", ""),
            "Selected_Panels": selected_panels,
            "Selected_Layers": selected_layers,
            "Panel_Mode": map_result.get("panel_interpolation_mode") or metadata.get("Panel_Interpolation_Mode", ""),
            "Panel_Interpolation_Mode": map_result.get("panel_interpolation_mode")
            or metadata.get("Panel_Interpolation_Mode", ""),
            "X_Field": map_result.get("x_col") or metadata.get("X_Field") or metadata.get("X_Column", "X"),
            "Y_Field": map_result.get("y_col") or metadata.get("Y_Field") or metadata.get("Y_Column", "Y"),
            "Coordinate_Unit": coordinate_unit_symbol(map_result.get("coordinate_unit") or metadata.get("Coordinate_Unit")),
            "CRS_Mode": crs.get("mode", LOCAL_CRS_MODE),
            "CRS_EPSG": crs.get("epsg", ""),
            "EPSG": crs.get("epsg", ""),
            "CRS_Name": crs.get("name", ""),
            "CRS_Authority": crs.get("authority", ""),
            "NoData": float(nodata),
            "Interpolation_Method": map_result.get("method") or map_result.get("interpolation_method") or metadata.get("Interpolation_Method", ""),
            "Interpolation_Parameters": json_safe(
                map_result.get("method_parameters") or map_result.get("interpolation_parameters") or {}
            ),
            "Interpolation_Domain": interpolation_domain,
            "Interpolation_Domain_Type": interpolation_domain,
            "Domain_Bounds": domain_bounds or "",
            "Domain_Geometry_Source": domain_source,
            "Mask": (map_result.get("mask_parameters") or {}).get("mode", metadata.get("Mask", "")),
            "Mask_Info": json_safe(map_result.get("mask_info", {})),
            "Variogram_Convention": (
                (map_result.get("method_parameters") or {}).get("variogram_range_convention")
                or metadata.get("Variogram_Range_Convention", "")
            ),
            "Variogram_Model": (
                (map_result.get("method_parameters") or {}).get("variogram_model")
                or metadata.get("Variogram_Model", "")
            ),
            "Range": (map_result.get("method_parameters") or {}).get("range", metadata.get("Range_Value", "")),
            "Variance_or_Sill": (map_result.get("method_parameters") or {}).get(
                "variance",
                metadata.get("Variance_or_Sill", ""),
            ),
            "Nugget": (map_result.get("method_parameters") or {}).get("nugget", metadata.get("Nugget", "")),
            "Anisotropy": {
                "enabled": (map_result.get("method_parameters") or {}).get(
                    "anisotropy_enabled",
                    metadata.get("Anisotropy_Enabled", False),
                ),
                "angle": (map_result.get("method_parameters") or {}).get(
                    "anisotropy_angle",
                    metadata.get("Anisotropy_Direction", 0.0),
                ),
                "ratio": (map_result.get("method_parameters") or {}).get(
                    "anisotropy_ratio",
                    metadata.get("Anisotropy_Ratio", 1.0),
                ),
            },
            "Kriging_Variance_Available": map_result.get("grid_variance") is not None,
            "Kriging_StdDev_Available": map_result.get("grid_variance") is not None,
            "Measured_Observation_Count": int(map_result.get("measured_observation_count") or 0),
            "Engineering_Control_Count": int(map_result.get("engineering_control_count") or 0),
            "Control_Region_Count": int(map_result.get("control_region_count") or 0),
            "Geometry_References": json_safe(map_result.get("geometry_references", {})),
            "Validation_Metrics": json_safe(map_result.get("validation_metrics", {})),
            "Control_Regions": json_safe(serialize_control_regions(map_result.get("engineering_control_regions", []))),
            "Export_Timestamp": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        }
    )
    metadata.update(grid_definition.metadata())
    if extra:
        metadata.update(json_safe(extra))
    return metadata


def _comment_value(value) -> str:
    if isinstance(value, (list, tuple, dict)):
        return json.dumps(json_safe(value), default=str)
    return str(value)


def _zmap_comment_lines(metadata: dict[str, object] | None) -> list[str]:
    metadata = metadata or {}
    comment_keys = [
        ("Property", "Property"),
        ("Unit", "Export_Unit"),
        ("Reservoir Layer", "Reservoir_Layer"),
        ("CRS", "CRS_Authority"),
        ("CRS_NAME", "CRS_Name"),
        ("XY_UNIT", "Coordinate_Unit"),
        ("Reference_Date", "Pressure_Map_Reference_Date"),
        ("Interpolation_Method", "Interpolation_Method"),
    ]
    lines = [
        "! Reservoir Mapping Studio",
        f"! ZMAP_DIALECT: {ZMAP_DIALECT}",
        "! GRID_NODE_CONVENTION: X/Y coordinates are cell centers.",
        "! VALUE_ORDER: north-to-south rows, west-to-east columns.",
    ]
    for label, key in comment_keys:
        value = metadata.get(key)
        if value not in (None, ""):
            lines.append(f"! {label}: {_comment_value(value)}")
    return lines


def grid_to_zmap_ascii_bytes(
    grid_x: np.ndarray,
    grid_y: np.ndarray,
    grid_z: np.ndarray,
    metadata: dict[str, object] | None = None,
    nodata: float = ZMAP_NULL_VALUE,
    values_per_line: int = ZMAP_VALUES_PER_LINE,
    decimal_places: int = ZMAP_DECIMAL_PLACES,
) -> bytes:
    definition, data = orient_grid_north_to_south(grid_x, grid_y, grid_z)
    output = np.where(np.isfinite(data), data, nodata)
    lines = _zmap_comment_lines(metadata)
    lines.extend(
        [
            "@GRID FILE, GRID, 4",
            f"{int(values_per_line)}, {float(nodata):.10g}, , {int(decimal_places)}, 1",
            (
                f"{definition.width}, {definition.height}, "
                f"{definition.x_min:.10g}, {definition.x_max:.10g}, "
                f"{definition.y_min:.10g}, {definition.y_max:.10g}"
            ),
            "0.0, 0.0, 0.0",
            "@",
        ]
    )
    number_format = f"{{:.{int(decimal_places)}f}}"
    values = [number_format.format(float(value)) for value in output.ravel(order="C")]
    for start in range(0, len(values), int(values_per_line)):
        lines.append(" ".join(values[start : start + int(values_per_line)]))
    return ("\n".join(lines) + "\n").encode("utf-8")


def parse_zmap_grid_ascii_bytes(data: bytes | str) -> dict[str, object]:
    """Minimal parser for the RMS V1 ZMAP-style dialect used in tests."""

    text = data.decode("utf-8") if isinstance(data, bytes) else str(data)
    raw_lines = [line.strip() for line in text.splitlines() if line.strip()]
    lines = [line for line in raw_lines if not line.startswith("!")]
    try:
        header_index = next(index for index, line in enumerate(lines) if line.upper().startswith("@GRID"))
    except StopIteration as exc:
        raise ValueError("ZMAP grid header was not found.") from exc
    if len(lines) <= header_index + 4:
        raise ValueError("ZMAP grid header is incomplete.")

    def numbers(line: str) -> list[float]:
        return [float(item) for item in re.split(r"[,\s]+", line.strip()) if item]

    format_line = numbers(lines[header_index + 1])
    bounds_line = numbers(lines[header_index + 2])
    if len(format_line) < 2 or len(bounds_line) < 6:
        raise ValueError("ZMAP grid header values are incomplete.")
    values_per_line = int(format_line[0])
    nodata = float(format_line[1])
    nx = int(bounds_line[0])
    ny = int(bounds_line[1])
    x_min, x_max, y_min, y_max = (float(value) for value in bounds_line[2:6])
    value_start = header_index + 5 if lines[header_index + 4] == "@" else header_index + 4
    values: list[float] = []
    for line in lines[value_start:]:
        values.extend(numbers(line))
    if len(values) < nx * ny:
        raise ValueError("ZMAP grid does not contain enough values.")
    data_north_south = np.asarray(values[: nx * ny], dtype=float).reshape((ny, nx))
    data_north_south[np.isclose(data_north_south, nodata)] = np.nan
    x_axis = np.linspace(x_min, x_max, nx)
    y_axis = np.linspace(y_min, y_max, ny)
    grid_x, grid_y = np.meshgrid(x_axis, y_axis)
    return {
        "nx": nx,
        "ny": ny,
        "x_min": x_min,
        "x_max": x_max,
        "y_min": y_min,
        "y_max": y_max,
        "dx": float((x_max - x_min) / (nx - 1)) if nx > 1 else None,
        "dy": float((y_max - y_min) / (ny - 1)) if ny > 1 else None,
        "null_value": nodata,
        "values_per_line": values_per_line,
        "grid_x": grid_x,
        "grid_y": grid_y,
        "grid_z": np.flipud(data_north_south),
    }


def sanitize_filename_part(value: object, fallback: str = "Map") -> str:
    text = str(value or "").strip()
    text = re.sub(r'[<>:"/\\|?*\s]+', "_", text)
    text = re.sub(r"_+", "_", text).strip("._")
    return text or fallback


def map_base_filename(
    map_result: dict[str, object],
    *,
    project_metadata: dict[str, object] | None = None,
    value_export: MapValueExport | None = None,
) -> str:
    seed = _metadata_seed(map_result)
    earlier = seed.get("Date_Earlier") or seed.get("Earlier_Reference_Date")
    later = seed.get("Date_Later") or seed.get("Later_Reference_Date")
    if earlier and later:
        parts = ["PressureChange", _iso_text(earlier), "to", _iso_text(later)]
    else:
        parts = [map_result.get("property_col") or map_result.get("property") or seed.get("Property") or "Map"]
        project_name = (project_metadata or {}).get("field") or (project_metadata or {}).get("name")
        if project_name and str(project_name) != "Untitled Project":
            parts.append(project_name)
        layer = map_result.get("reservoir_layer") or seed.get("Reservoir_Layer")
        if layer:
            parts.append(layer)
        reference_date = map_result.get("map_reference_date") or map_result.get("pressure_reference_date") or seed.get("Pressure_Map_Reference_Date")
        if reference_date:
            parts.append(_iso_text(reference_date))
        method = map_result.get("method") or map_result.get("interpolation_method") or seed.get("Interpolation_Method")
        if method and method != "Delta":
            parts.append(method)
    if value_export and value_export.filename_suffix:
        parts.append(value_export.filename_suffix)
    return "_".join(sanitize_filename_part(part, "") for part in parts if sanitize_filename_part(part, ""))


def map_geotiff_export_files(
    map_result: dict[str, object],
    *,
    project_metadata: dict[str, object] | None = None,
    scenario: dict[str, object] | None = None,
    include_uncertainty: bool = True,
    nodata: float = GEOTIFF_NODATA,
) -> list[tuple[str, bytes, dict[str, object]]]:
    files: list[tuple[str, bytes, dict[str, object]]] = []
    crs = _crs_from_map_result(map_result)
    for value_export in value_exports_for_map(map_result, include_uncertainty=include_uncertainty):
        metadata = build_map_export_metadata(
            map_result,
            project_metadata=project_metadata,
            scenario=scenario,
            value_export=value_export,
            nodata=nodata,
            extra={"Export_Format": "GeoTIFF"},
        )
        filename = f"{map_base_filename(map_result, project_metadata=project_metadata, value_export=value_export)}.tif"
        files.append(
            (
                filename,
                grid_to_geotiff_bytes(
                    map_result["grid_x"],
                    map_result["grid_y"],
                    value_export.grid,
                    crs,
                    nodata=nodata,
                    metadata=metadata,
                ),
                metadata,
            )
        )
    return files


def map_zmap_export_files(
    map_result: dict[str, object],
    *,
    project_metadata: dict[str, object] | None = None,
    scenario: dict[str, object] | None = None,
    include_uncertainty: bool = True,
    nodata: float = ZMAP_NULL_VALUE,
) -> list[tuple[str, bytes, str, bytes, dict[str, object]]]:
    files: list[tuple[str, bytes, str, bytes, dict[str, object]]] = []
    for value_export in value_exports_for_map(map_result, include_uncertainty=include_uncertainty):
        metadata = build_map_export_metadata(
            map_result,
            project_metadata=project_metadata,
            scenario=scenario,
            value_export=value_export,
            nodata=nodata,
            extra={"Export_Format": "ZMAP Grid ASCII", "ZMAP_Dialect": ZMAP_DIALECT},
        )
        base = map_base_filename(map_result, project_metadata=project_metadata, value_export=value_export)
        zmap_name = f"{base}.zmap"
        metadata_name = f"{base}_metadata.json"
        files.append(
            (
                zmap_name,
                grid_to_zmap_ascii_bytes(
                    map_result["grid_x"],
                    map_result["grid_y"],
                    value_export.grid,
                    metadata=metadata,
                    nodata=nodata,
                ),
                metadata_name,
                metadata_to_json_bytes(metadata),
                metadata,
            )
        )
    return files


def batch_map_package_zip_bytes(
    layer_maps: dict[str, dict[str, object]],
    *,
    project_metadata: dict[str, object] | None = None,
    include_formats: set[str] | list[str] | tuple[str, ...] | None = None,
    include_uncertainty: bool = True,
) -> bytes:
    formats = set(include_formats or {"geotiff", "zmap", "xyz", "excel", "metadata"})
    buffer = BytesIO()
    file_records: list[dict[str, object]] = []
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for layer_name, map_result in sorted(layer_maps.items(), key=lambda item: str(item[0])):
            base = map_base_filename(map_result, project_metadata=project_metadata)
            metadata = build_map_export_metadata(
                map_result,
                project_metadata=project_metadata,
                nodata=GEOTIFF_NODATA,
                extra={"Export_Format": "Batch Package"},
            )
            controls_df = engineering_controls_export_dataframe(map_result)
            regions_df = control_regions_export_dataframe(map_result)
            layer_record = {"Reservoir_Layer": layer_name, "Files": []}
            if "metadata" in formats:
                name = f"{base}_metadata.json"
                archive.writestr(name, metadata_to_json_bytes(metadata))
                layer_record["Files"].append(name)
            if "xyz" in formats or "csv" in formats:
                grid_df = grid_to_dataframe(
                    map_result["grid_x"],
                    map_result["grid_y"],
                    map_result["grid_z"],
                    str(map_result.get("property_col") or "Value"),
                    include_nan=False,
                    grid_variance=map_result.get("grid_variance"),
                    panel_grid=map_result.get("panel_grid"),
                )
                if "csv" in formats:
                    name = f"{base}.csv"
                    archive.writestr(name, dataframe_to_csv_bytes(grid_df))
                    layer_record["Files"].append(name)
                name = f"{base}.xyz"
                archive.writestr(name, grid_to_xyz_ascii_bytes(grid_df))
                layer_record["Files"].append(name)
            if "excel" in formats:
                grid_df = grid_to_dataframe(
                    map_result["grid_x"],
                    map_result["grid_y"],
                    map_result["grid_z"],
                    str(map_result.get("property_col") or "Value"),
                    include_nan=False,
                    grid_variance=map_result.get("grid_variance"),
                    panel_grid=map_result.get("panel_grid"),
                )
                name = f"{base}.xlsx"
                archive.writestr(
                    name,
                    grid_to_excel_bytes(
                        grid_df,
                        metadata,
                        engineering_controls=controls_df,
                        control_regions=regions_df,
                    ),
                )
                layer_record["Files"].append(name)
            if "geotiff" in formats:
                for filename, data, _ in map_geotiff_export_files(
                    map_result,
                    project_metadata=project_metadata,
                    include_uncertainty=include_uncertainty,
                ):
                    archive.writestr(filename, data)
                    layer_record["Files"].append(filename)
            if "zmap" in formats:
                for zmap_name, zmap_data, metadata_name, metadata_data, _ in map_zmap_export_files(
                    map_result,
                    project_metadata=project_metadata,
                    include_uncertainty=include_uncertainty,
                ):
                    archive.writestr(zmap_name, zmap_data)
                    layer_record["Files"].append(zmap_name)
                    if metadata_name not in layer_record["Files"]:
                        archive.writestr(metadata_name, metadata_data)
                        layer_record["Files"].append(metadata_name)
            file_records.append(layer_record)

        batch_metadata = {
            "Application": APP_NAME,
            "Project_Schema_Version": _project_schema_version(),
            "Project_Name": (project_metadata or {}).get("name", ""),
            "Property": next(iter(layer_maps.values())).get("property_col", "") if layer_maps else "",
            "Pressure_Map_Reference_Date": _iso_text(next(iter(layer_maps.values())).get("map_reference_date", "")) if layer_maps else "",
            "Generated_Reservoir_Layers": list(sorted(layer_maps)),
            "Selected_Panels": next(iter(layer_maps.values())).get("selected_panels", []) if layer_maps else [],
            "Panel_Mode": next(iter(layer_maps.values())).get("panel_interpolation_mode", "") if layer_maps else "",
            "CRS": _crs_from_map_result(next(iter(layer_maps.values()))) if layer_maps else normalize_crs_config(None),
            "Interpolation_Method": next(iter(layer_maps.values())).get("method", "") if layer_maps else "",
            "Files_By_Reservoir_Layer": file_records,
            "Export_Timestamp": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        }
        archive.writestr("metadata.json", metadata_to_json_bytes(batch_metadata))
    return buffer.getvalue()


def map_package_zip_bytes(
    grid_df: pd.DataFrame,
    metadata: dict[str, object],
    figure=None,
    validation_df: pd.DataFrame | None = None,
    engineering_controls: pd.DataFrame | None = None,
    control_regions: pd.DataFrame | None = None,
) -> bytes:
    buffer = BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("grid.csv", dataframe_to_csv_bytes(grid_df))
        archive.writestr("metadata.json", metadata_to_json_bytes(metadata))
        if validation_df is not None and not validation_df.empty:
            archive.writestr("validation.csv", dataframe_to_csv_bytes(validation_df))
        if engineering_controls is not None and not engineering_controls.empty:
            archive.writestr("engineering_controls.csv", dataframe_to_csv_bytes(engineering_controls))
        if control_regions is not None and not control_regions.empty:
            archive.writestr("control_regions.csv", dataframe_to_csv_bytes(control_regions))
        if figure is not None:
            try:
                archive.writestr("map.png", figure_to_image_bytes(figure, "png"))
            except RuntimeError:
                archive.writestr("map_export_warning.txt", b"PNG export was unavailable in this runtime.")
    return buffer.getvalue()


def drop_internal_columns(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop(columns=[INTERNAL_ROW_ID], errors="ignore")
