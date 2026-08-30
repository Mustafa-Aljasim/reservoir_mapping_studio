"""Point-to-compartment assignment utilities."""

from __future__ import annotations

import pandas as pd
from shapely.geometry import Point

from core.geometry.models import GeometryLayer


def assign_points_to_polygons(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    panel_layer: GeometryLayer | None,
    dataset_panel_col: str | None = None,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    if panel_layer is None or not panel_layer.polygon_features:
        return pd.DataFrame(columns=["Row_Index", "Spatial_Panel", "Assignment_Status"])

    for row_index, row in df.iterrows():
        x = pd.to_numeric(pd.Series([row.get(x_col)]), errors="coerce").iloc[0]
        y = pd.to_numeric(pd.Series([row.get(y_col)]), errors="coerce").iloc[0]
        assigned: list[str] = []
        if pd.notna(x) and pd.notna(y):
            point = Point(float(x), float(y))
            assigned = [feature.name for feature in panel_layer.polygon_features if feature.geometry.covers(point)]

        spatial_panel = None if not assigned else "|".join(assigned)
        status = "Outside"
        if len(assigned) == 1:
            status = "Assigned"
        elif len(assigned) > 1:
            status = "Ambiguous"

        dataset_panel = row.get(dataset_panel_col) if dataset_panel_col and dataset_panel_col in df.columns else None
        if dataset_panel is not None and pd.notna(dataset_panel) and spatial_panel:
            status = "Match" if str(dataset_panel) in assigned else "Mismatch"

        rows.append(
            {
                "Row_Index": row_index,
                "Spatial_Panel": spatial_panel,
                "Dataset_Panel": dataset_panel,
                "Assignment_Status": status,
            }
        )
    return pd.DataFrame(rows)


def outside_panel_count(assignments: pd.DataFrame) -> int:
    if assignments.empty or "Assignment_Status" not in assignments.columns:
        return 0
    return int((assignments["Assignment_Status"] == "Outside").sum())
