"""Export helpers for tabular data and interpolated grids."""

from __future__ import annotations

from io import BytesIO
import json
import zipfile

import numpy as np
import pandas as pd

from utils.constants import INTERNAL_ROW_ID


def dataframe_to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


def dataframe_to_excel_bytes(df: pd.DataFrame, sheet_name: str = "Data") -> bytes:
    buffer = BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name=sheet_name[:31] or "Data")
    return buffer.getvalue()


def metadata_to_dataframe(metadata: dict[str, object]) -> pd.DataFrame:
    return pd.DataFrame(
        [{"Parameter": key, "Value": value} for key, value in metadata.items()]
    )


def grid_to_excel_bytes(
    grid_df: pd.DataFrame,
    metadata: dict[str, object] | None = None,
) -> bytes:
    buffer = BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        grid_df.to_excel(writer, index=False, sheet_name="Interpolated_Grid")
        if metadata:
            metadata_to_dataframe(metadata).to_excel(writer, index=False, sheet_name="Map_Metadata")
    return buffer.getvalue()


def metadata_to_json_bytes(metadata: dict[str, object]) -> bytes:
    return json.dumps(metadata, indent=2, default=str).encode("utf-8")


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
    value_column = "Estimated_Value" if grid_variance is not None else property_name
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


def grid_to_geotiff_bytes(
    grid_x: np.ndarray,
    grid_y: np.ndarray,
    grid_z: np.ndarray,
    epsg: int | str | None,
    nodata: float = -9999.0,
) -> bytes:
    if not epsg:
        raise ValueError("GeoTIFF export requires a known EPSG CRS.")
    try:
        import rasterio
        from rasterio.io import MemoryFile
        from rasterio.transform import from_origin
    except ImportError as exc:  # pragma: no cover - dependency is installed in V1 requirements
        raise RuntimeError("Rasterio is required for GeoTIFF export.") from exc

    x = np.asarray(grid_x, dtype=float)
    y = np.asarray(grid_y, dtype=float)
    z = np.asarray(grid_z, dtype=float)
    if x.ndim != 2 or y.ndim != 2 or z.shape != x.shape or y.shape != x.shape:
        raise ValueError("GeoTIFF export requires matching 2D grid arrays.")
    if x.shape[0] < 2 or x.shape[1] < 2:
        raise ValueError("GeoTIFF export requires at least a 2 by 2 grid.")

    dx = abs(float(np.nanmedian(np.diff(x[0, :]))))
    dy = abs(float(np.nanmedian(np.diff(y[:, 0]))))
    if dx <= 0 or dy <= 0:
        raise ValueError("GeoTIFF export requires regular grid spacing.")
    west = float(np.nanmin(x)) - dx / 2.0
    north = float(np.nanmax(y)) + dy / 2.0
    data = np.flipud(z).astype("float32")
    data = np.where(np.isfinite(data), data, nodata).astype("float32")
    transform = from_origin(west, north, dx, dy)
    with MemoryFile() as memory_file:
        with memory_file.open(
            driver="GTiff",
            height=data.shape[0],
            width=data.shape[1],
            count=1,
            dtype="float32",
            crs=f"EPSG:{int(epsg)}",
            transform=transform,
            nodata=nodata,
        ) as dataset:
            dataset.write(data, 1)
        return memory_file.read()


def map_package_zip_bytes(
    grid_df: pd.DataFrame,
    metadata: dict[str, object],
    figure=None,
    validation_df: pd.DataFrame | None = None,
) -> bytes:
    buffer = BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("grid.csv", dataframe_to_csv_bytes(grid_df))
        archive.writestr("metadata.json", metadata_to_json_bytes(metadata))
        if validation_df is not None and not validation_df.empty:
            archive.writestr("validation.csv", dataframe_to_csv_bytes(validation_df))
        if figure is not None:
            try:
                archive.writestr("map.png", figure_to_image_bytes(figure, "png"))
            except RuntimeError:
                archive.writestr("map_export_warning.txt", b"PNG export was unavailable in this runtime.")
    return buffer.getvalue()


def drop_internal_columns(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop(columns=[INTERNAL_ROW_ID], errors="ignore")
