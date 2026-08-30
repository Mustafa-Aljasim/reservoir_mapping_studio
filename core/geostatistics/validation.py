"""Cross-validation and residual metrics."""

from __future__ import annotations

import numpy as np
import pandas as pd

from core.geostatistics.kriging import kriging_predict_point
from core.interpolation.idw import idw_interpolate


def validation_metrics(results: pd.DataFrame) -> dict[str, float | int | None]:
    valid = results.dropna(subset=["Observed", "Predicted"]).copy()
    if valid.empty:
        return {"Count": 0, "MAE": None, "RMSE": None, "Bias": None, "R2": None, "Median_Absolute_Error": None, "Max_Absolute_Error": None}
    residual = valid["Residual"].astype(float)
    observed = valid["Observed"].astype(float)
    predicted = valid["Predicted"].astype(float)
    sst = float(((observed - observed.mean()) ** 2).sum())
    sse = float(((observed - predicted) ** 2).sum())
    return {
        "Count": int(len(valid)),
        "MAE": float(valid["Absolute_Error"].mean()),
        "RMSE": float(np.sqrt(valid["Squared_Error"].mean())),
        "Bias": float(residual.mean()),
        "R2": None if sst == 0 else float(1.0 - sse / sst),
        "Median_Absolute_Error": float(valid["Absolute_Error"].median()),
        "Max_Absolute_Error": float(valid["Absolute_Error"].max()),
    }


def _predict_idw(train: pd.DataFrame, row: pd.Series, parameters: dict) -> float:
    prediction = idw_interpolate(
        train["X"],
        train["Y"],
        train["Z"],
        np.asarray([[row["X"]]], dtype=float),
        np.asarray([[row["Y"]]], dtype=float),
        power=float(parameters.get("power", 2.0)),
        neighbors=int(parameters.get("neighbors", 12)),
        search_radius=parameters.get("search_radius"),
        min_neighbors=int(parameters.get("min_neighbors", 1)),
    )
    return float(prediction[0, 0])


def _predict_kriging(train: pd.DataFrame, row: pd.Series, parameters: dict) -> float:
    estimate, _ = kriging_predict_point(train["X"], train["Y"], train["Z"], float(row["X"]), float(row["Y"]), parameters)
    return estimate


def leave_one_out_cross_validation(
    prepared_df: pd.DataFrame,
    method: str,
    method_parameters: dict | None = None,
    well_names: list[object] | None = None,
    panels: list[object] | None = None,
    respect_compartments: bool = False,
) -> tuple[pd.DataFrame, dict[str, float | int | None]]:
    """Run fixed-parameter LOOCV.

    Residual sign convention is Observed - Predicted.
    """

    parameters = method_parameters or {}
    method_key = method.strip().lower().replace(" ", "_")
    rows: list[dict[str, object]] = []
    working = prepared_df.reset_index(drop=True).copy()
    if well_names is None:
        well_names = [f"Obs-{index + 1}" for index in range(len(working))]
    else:
        well_names = list(well_names[: len(working)]) + [
            f"Obs-{index + 1}" for index in range(len(well_names), len(working))
        ]
    if panels is None:
        panels = [None] * len(working)
    else:
        panels = list(panels[: len(working)]) + [None] * max(0, len(working) - len(panels))

    for index, row in working.iterrows():
        train = working.drop(index).copy()
        target_panel = panels[index]
        if respect_compartments:
            if target_panel is None:
                train = working.iloc[0:0].copy()
            else:
                train_mask = [panel == target_panel and panel_index != index for panel_index, panel in enumerate(panels)]
                train = working.loc[train_mask].copy()
        predicted = np.nan
        try:
            if method_key == "idw":
                predicted = _predict_idw(train, row, parameters)
            elif method_key in {"ordinary_kriging", "kriging"}:
                predicted = _predict_kriging(train, row, parameters)
        except Exception:
            predicted = np.nan
        observed = float(row["Z"])
        residual = observed - predicted if np.isfinite(predicted) else np.nan
        rows.append(
            {
                "Well": well_names[index],
                "X": float(row["X"]),
                "Y": float(row["Y"]),
                "Observed": observed,
                "Predicted": predicted,
                "Residual": residual,
                "Absolute_Error": abs(residual) if np.isfinite(residual) else np.nan,
                "Squared_Error": residual**2 if np.isfinite(residual) else np.nan,
                "Panel": target_panel,
            }
        )
    results = pd.DataFrame(rows)
    return results, validation_metrics(results)
