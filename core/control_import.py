"""Preview-only ingestion of engineering controls; never creates observations."""
from __future__ import annotations

from io import BytesIO
import math
import re
from zipfile import BadZipFile

import pandas as pd

from core.active_data import parse_reference_date
from core.engineering_controls import create_control_point, assign_panel_from_point, engineering_controls_for_context


FIELDS = ("X", "Y", "Value", "Control_ID", "Property", "Unit", "Layer", "Panel", "Pressure_Reference_Date", "Active", "Comment")


def read_control_file(data: bytes, filename: str) -> pd.DataFrame:
    if filename.lower().endswith(".csv"):
        return pd.read_csv(BytesIO(data))
    if filename.lower().endswith(".xlsx"):
        try:
            return pd.read_excel(BytesIO(data), engine="openpyxl")
        except BadZipFile as exc:
            raise ValueError("The XLSX file is not a valid Excel workbook.") from exc
    raise ValueError("Control files must be CSV or XLSX.")


def suggest_control_columns(columns) -> dict[str, str | None]:
    def key(value):
        return re.sub(r"[^a-z0-9]", "", str(value).lower())
    aliases = {"Layer": ("Reservoir_Layer",), "Unit": ("Property_Unit",),
               "Pressure_Reference_Date": ("Pressure_Map_Reference_Date", "Reference_Date"),
               "X": ("Easting",), "Y": ("Northing",), "Value": ("Control_Value",)}
    lookup = {key(col): col for col in columns}
    return {field: next((lookup[key(name)] for name in (field,) + aliases.get(field, ()) if key(name) in lookup), None)
            for field in FIELDS}


def apply_pending_pick(state) -> bool:
    """Apply before number_input widgets are instantiated on the next rerun."""
    picked = state.pop("pending_control_point_pick", None)
    if picked is None:
        return False
    x, y = float(picked["x"]), float(picked["y"])
    if not math.isfinite(x) or not math.isfinite(y):
        raise ValueError("Picked X/Y must be finite.")
    state["engineering_control_x"] = x
    state["engineering_control_y"] = y
    state["control_point_preview"] = {"x": x, "y": y}
    return True


def preview_control_import(frame, columns, *, defaults, valid_properties, valid_layers,
                           valid_panels=(), existing_controls=(), panel_layer=None,
                           reservoir_boundary_layer=None, measured_dataframe=None,
                           mappings=None, pressure_properties=("Pressure",)):
    """Return normalized controls, blocking errors and existing scientific QC warnings."""
    errors, warnings, controls = [], [], []
    if any(not columns.get(field) for field in ("X", "Y", "Value")):
        return [], ["Map X, Y and Value columns before importing."], []
    mapped = [value for value in columns.values() if value]
    if len(set(mapped)) != len(mapped):
        return [], ["Each source column can map to only one field."], []
    used_ids = {str(row.get("Control_ID")) for row in existing_controls}
    mappings = mappings or {"x": "X", "y": "Y"}
    for index, (_, row) in enumerate(frame.iterrows(), 1):
        def value(field):
            column = columns.get(field)
            raw = row[column] if column else defaults.get(field, "")
            return "" if pd.isna(raw) else raw
        try:
            xyz = [float(value(field)) for field in ("X", "Y", "Value")]
            if not all(math.isfinite(number) for number in xyz):
                raise ValueError("X, Y and Value must be finite")
            prop, layer, panel = (str(value(field)).strip() for field in ("Property", "Layer", "Panel"))
            if prop not in valid_properties:
                raise ValueError(f"Unknown Property: {prop}")
            if valid_layers and layer not in valid_layers:
                raise ValueError(f"Unknown Layer: {layer}")
            if not valid_layers and layer:
                raise ValueError(f"Unknown Layer: {layer}")
            if panel and panel not in valid_panels:
                raise ValueError(f"Unknown Panel: {panel}")
            reference = value("Pressure_Reference_Date")
            if prop in pressure_properties and parse_reference_date(reference) is None:
                raise ValueError("Pressure requires a valid reference date")
            if not columns.get("Panel"):
                panel, warning = assign_panel_from_point(xyz[0], xyz[1], panel_layer)
                if warning:
                    warnings.append(f"Row {index}: {warning}")
            active = str(value("Active")).strip().lower()
            if active not in {"true", "false", "1", "0", "yes", "no", "1.0", "0.0"}:
                raise ValueError("Active must be true/false, yes/no or 1/0")
            identifier = str(value("Control_ID")).strip()
            if identifier and identifier in used_ids:
                raise ValueError(f"Duplicate Control ID: {identifier}")
            unit = str(value("Unit")) if columns.get("Unit") or prop == defaults.get("Property") else ""
            control = create_control_point(x=xyz[0], y=xyz[1], value=xyz[2], property_name=prop,
                property_unit=unit, reservoir_layer=layer, panel=panel,
                pressure_reference_date=reference, active=active in {"true", "1", "yes", "1.0"},
                comment=str(value("Comment")), source_type="File Import", control_id=identifier or None,
                existing_controls=list(existing_controls) + controls)
            used_ids.add(control["Control_ID"])
            controls.append(control)
        except (ValueError, TypeError, OverflowError) as exc:
            errors.append(f"Row {index}: {exc}")
    # Reuse PASS 3 conflicts/domain QC separately for every explicit file scope.
    scopes = {(c["Property"], c["Property_Unit"], c["Reservoir_Layer"], c["Pressure_Map_Reference_Date"]) for c in controls}
    for prop, unit, layer, reference in sorted(scopes):
        measured = measured_dataframe.copy() if measured_dataframe is not None else pd.DataFrame()
        for semantic, selected in (("layer", layer), ("map_reference_date", reference)):
            column = mappings.get(semantic)
            if column in measured and selected:
                if semantic == "map_reference_date":
                    measured = measured[measured[column].map(lambda x: str(parse_reference_date(x) or "")) == selected]
                else:
                    measured = measured[measured[column].astype(str) == selected]
        selection = engineering_controls_for_context(control_points=list(existing_controls) + controls,
            control_regions=[], measured_dataframe=measured, mappings=mappings, property_col=prop,
            property_type="Pressure" if prop in pressure_properties else "Generic", property_unit=unit,
            pressure_reference_date=reference, reservoir_layer=layer or None, selected_panels=[],
            panel_interpolation_mode="Combined", x_col=mappings.get("x") or "X", y_col=mappings.get("y") or "Y",
            panel_layer=panel_layer, reservoir_boundary_layer=reservoir_boundary_layer)
        warnings.extend(selection.warnings)
    return controls, errors, list(dict.fromkeys(warnings))
