"""Shared map style normalization for interactive and static map figures."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable

import numpy as np
import pandas as pd

from core.plotting.styling import format_numeric


WELL_LABEL_CONTENT_OPTIONS = ("Well Name", "Property Value", "Well Name + Value")
LABEL_DENSITY_OPTIONS = ("All", "Smart / Declutter", "None")
LABEL_POSITION_OPTIONS = (
    "top center",
    "top left",
    "top right",
    "middle right",
    "middle left",
    "bottom center",
)


def _bool(value, default: bool = False) -> bool:
    if value is None:
        return default
    return bool(value)


def _int(value, default: int, *, minimum: int | None = None, maximum: int | None = None) -> int:
    try:
        result = int(value)
    except (TypeError, ValueError):
        result = default
    if minimum is not None:
        result = max(minimum, result)
    if maximum is not None:
        result = min(maximum, result)
    return result


def _float(value, default: float, *, minimum: float | None = None, maximum: float | None = None) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        result = default
    if minimum is not None:
        result = max(minimum, result)
    if maximum is not None:
        result = min(maximum, result)
    return result


def normalize_label_content(value: object, *, well_col: str | None = None) -> str:
    text = str(value or "Property Value")
    if text in {"None", "Off"}:
        return "Property Value" if not well_col else "Well Name"
    if text == "Value":
        return "Property Value"
    if text == "Well Name + Property Value":
        return "Well Name + Value"
    if text not in WELL_LABEL_CONTENT_OPTIONS:
        return "Property Value" if not well_col else "Well Name"
    if text == "Well Name" and not well_col:
        return "Property Value"
    if text == "Well Name + Value" and not well_col:
        return "Property Value"
    return text


def normalize_label_density(value: object) -> str:
    text = str(value or "All")
    return text if text in LABEL_DENSITY_OPTIONS else "All"


def normalize_label_position(value: object) -> str:
    text = str(value or "top center")
    return text if text in LABEL_POSITION_OPTIONS else "top center"


@dataclass(frozen=True)
class MapStyle:
    color_scale: str = "Turbo"
    reverse_colors: bool = False
    z_range_mode: str = "Auto"
    zmin: float | None = None
    zmax: float | None = None
    contour_mode: str = "Auto interval"
    contour_interval: float | None = None
    show_surface: bool = True
    show_contour_lines: bool = True
    show_contour_labels: bool = False
    contour_line_width: float = 0.75
    contour_label_font_size: int = 10
    show_wells: bool = True
    marker_size: int = 9
    marker_opacity: float = 0.9
    marker_outline: bool = True
    engineering_control_marker_size: int = 13
    show_well_labels: bool = False
    well_label_mode: str = "Property Value"
    well_label_font_size: int = 10
    well_label_position: str = "top center"
    label_density: str = "All"
    label_min_separation_mode: str = "Auto"
    label_min_separation: float = 0.06
    label_decimal_places: int = 0
    label_include_unit: bool = False
    show_excluded: bool = True
    title_override: str = ""
    height: int = 720
    axis_font_size: int = 12
    title_font_size: int = 16
    colorbar_font_size: int = 11

    @classmethod
    def from_settings(cls, settings: dict | None, *, well_col: str | None = None, generated_surface: bool = True) -> "MapStyle":
        settings = settings or {}
        legacy_label_mode = str(settings.get("well_label_mode", "None"))
        show_well_labels = _bool(settings.get("show_well_labels"), legacy_label_mode != "None")
        label_density = normalize_label_density(settings.get("label_density", "All"))
        if label_density == "None":
            show_well_labels = False
        show_contour_lines = _bool(settings.get("show_contour_lines"), True) and generated_surface
        show_contour_labels = _bool(settings.get("show_contour_labels"), False) and show_contour_lines
        return cls(
            color_scale=str(settings.get("color_scale", "Turbo")),
            reverse_colors=_bool(settings.get("reverse_colors"), False),
            z_range_mode=str(settings.get("z_range_mode", "Auto")),
            zmin=settings.get("zmin"),
            zmax=settings.get("zmax"),
            contour_mode=str(settings.get("contour_mode", "Auto interval")),
            contour_interval=settings.get("contour_interval"),
            show_surface=_bool(settings.get("show_surface"), True) and generated_surface,
            show_contour_lines=show_contour_lines,
            show_contour_labels=show_contour_labels,
            contour_line_width=_float(settings.get("contour_line_width"), 0.75, minimum=0.0, maximum=10.0),
            contour_label_font_size=_int(settings.get("contour_label_font_size"), 10, minimum=6, maximum=24),
            show_wells=_bool(settings.get("show_wells"), True),
            marker_size=_int(settings.get("marker_size"), 9, minimum=1, maximum=60),
            marker_opacity=_float(settings.get("marker_opacity"), 0.9, minimum=0.0, maximum=1.0),
            marker_outline=_bool(settings.get("marker_outline"), True),
            engineering_control_marker_size=_int(
                settings.get("engineering_control_marker_size"),
                13,
                minimum=4,
                maximum=36,
            ),
            show_well_labels=show_well_labels,
            well_label_mode=normalize_label_content(settings.get("well_label_mode"), well_col=well_col),
            well_label_font_size=_int(
                settings.get("well_label_font_size", settings.get("label_text_size")),
                10,
                minimum=6,
                maximum=36,
            ),
            well_label_position=normalize_label_position(settings.get("well_label_position")),
            label_density=label_density,
            label_min_separation_mode=str(settings.get("label_min_separation_mode", "Auto")),
            label_min_separation=_float(settings.get("label_min_separation"), 0.06, minimum=0.0, maximum=1.0),
            label_decimal_places=_int(settings.get("label_decimal_places"), 0, minimum=0, maximum=6),
            label_include_unit=_bool(settings.get("label_include_unit"), False),
            show_excluded=_bool(settings.get("show_excluded"), True),
            title_override=str(settings.get("title_override", "")),
            height=_int(settings.get("height"), 720, minimum=240, maximum=4000),
            axis_font_size=_int(settings.get("axis_font_size"), 12, minimum=6, maximum=28),
            title_font_size=_int(settings.get("title_font_size"), 16, minimum=8, maximum=36),
            colorbar_font_size=_int(settings.get("colorbar_font_size"), 11, minimum=6, maximum=28),
        )

    def visible_well_labels(
        self,
        frame: pd.DataFrame,
        *,
        x_col: str,
        y_col: str,
        property_col: str,
        well_col: str | None,
        unit: str | None,
    ) -> list[str] | None:
        if not self.show_well_labels or self.label_density == "None" or frame.empty:
            return None
        labels = [
            self._label_for_row(row, property_col=property_col, well_col=well_col, unit=unit)
            for _, row in frame.iterrows()
        ]
        if self.label_density == "Smart / Declutter":
            return declutter_labels(
                pd.to_numeric(frame[x_col], errors="coerce"),
                pd.to_numeric(frame[y_col], errors="coerce"),
                labels,
                minimum_separation=self.effective_label_separation(len(labels)),
            )
        return labels

    def effective_label_separation(self, label_count: int) -> float:
        if str(self.label_min_separation_mode) == "Manual":
            return self.label_min_separation
        if label_count <= 1:
            return 0.0
        return max(0.04, min(0.10, 0.75 / math.sqrt(float(label_count))))

    def _label_for_row(self, row: pd.Series, *, property_col: str, well_col: str | None, unit: str | None) -> str:
        well = str(row.get(well_col, "")) if well_col and pd.notna(row.get(well_col)) else ""
        value = format_label_value(row.get(property_col), self.label_decimal_places)
        if self.label_include_unit and unit:
            value = f"{value} {unit}"
        if self.well_label_mode == "Well Name":
            return well
        if self.well_label_mode == "Well Name + Value":
            return f"{well}<br>{value}" if well else value
        return value


def format_label_value(value: object, decimals: int) -> str:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return format_numeric(value)
    if not np.isfinite(numeric):
        return format_numeric(value)
    decimals = max(0, int(decimals))
    return f"{numeric:.{decimals}f}"


def declutter_labels(
    x_values: Iterable[object],
    y_values: Iterable[object],
    labels: list[str],
    *,
    minimum_separation: float,
) -> list[str]:
    x = np.asarray(list(x_values), dtype=float)
    y = np.asarray(list(y_values), dtype=float)
    if len(labels) == 0:
        return labels
    finite = np.isfinite(x) & np.isfinite(y)
    if finite.sum() <= 1 or minimum_separation <= 0:
        return labels
    x_span = float(np.nanmax(x[finite]) - np.nanmin(x[finite])) or 1.0
    y_span = float(np.nanmax(y[finite]) - np.nanmin(y[finite])) or 1.0
    x_norm = (x - float(np.nanmin(x[finite]))) / x_span
    y_norm = (y - float(np.nanmin(y[finite]))) / y_span
    retained: list[tuple[float, float]] = []
    output: list[str] = []
    min_distance_sq = minimum_separation * minimum_separation
    for index, label in enumerate(labels):
        if not label or not finite[index]:
            output.append("")
            continue
        candidate = (float(x_norm[index]), float(y_norm[index]))
        if all((candidate[0] - kept[0]) ** 2 + (candidate[1] - kept[1]) ** 2 >= min_distance_sq for kept in retained):
            retained.append(candidate)
            output.append(label)
        else:
            output.append("")
    return output
