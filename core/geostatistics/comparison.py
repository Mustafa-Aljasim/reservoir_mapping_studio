"""Algorithm comparison helpers."""

from __future__ import annotations

import pandas as pd

from core.geostatistics.validation import leave_one_out_cross_validation


def compare_methods(
    prepared_df: pd.DataFrame,
    method_parameters: dict[str, dict],
    methods: list[str],
    well_names: list[object] | None = None,
    panels: list[object] | None = None,
    respect_compartments: bool = False,
    conditioning_points: pd.DataFrame | None = None,
    conditioning_panels: list[object] | None = None,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for method in methods:
        results, metrics = leave_one_out_cross_validation(
            prepared_df,
            method,
            method_parameters.get(method, {}),
            well_names=well_names,
            panels=panels,
            respect_compartments=respect_compartments,
            conditioning_points=conditioning_points,
            conditioning_panels=conditioning_panels,
        )
        rows.append({"Method": method, **metrics})
    return pd.DataFrame(rows)
