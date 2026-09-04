"""Data Manager page."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from core.column_mapper import normalize_column_mappings, numeric_property_candidates, suggest_mappings
from core.data_loader import get_excel_sheet_names, load_uploaded_dataframe, summarize_dataframe
from core.data_qc import build_qc_summary, duplicate_coordinate_rows, flag_outliers
from core.geometry.loader import (
    geojson_property_names,
    get_zipped_shapefile_candidates,
    load_geojson_bytes,
    load_geometry_csv,
    load_zipped_shapefile_bytes,
    read_zipped_shapefile_fields,
    read_zipped_shapefile_geometry_types,
    suggest_geometry_name_attribute,
)
from core.geometry.validation import layer_total_area, validate_geometry_layer
from core.pressure_dates import format_map_date, summarize_measurement_dates, validate_date_column
from pages.shared import (
    coordinate_unit_input,
    ensure_session_state,
    filter_candidate_options,
    load_sample_dataset,
    render_filter_controls,
    selectbox_with_none,
    set_active_dataframe,
)
from utils.constants import INTERNAL_ROW_ID, OUTLIER_COLUMN, SEMANTIC_FIELDS
from utils.units import coordinate_unit_symbol, format_area


ensure_session_state()

st.title("Reservoir Mapping Studio")
st.subheader("Data Manager")

left, right = st.columns([2, 1])
with left:
    uploaded_file = st.file_uploader("Upload CSV or Excel data", type=["csv", "xlsx", "xls"])
with right:
    st.write("")
    st.write("")
    if st.button("Load Sample Dataset", width="stretch"):
        load_sample_dataset()
        st.success("Sample dataset loaded.")

if uploaded_file is not None:
    extension = uploaded_file.name.lower().rsplit(".", 1)[-1]
    selected_sheet = None
    if extension in {"xlsx", "xls"}:
        try:
            sheets = get_excel_sheet_names(uploaded_file)
            selected_sheet = st.selectbox("Worksheet", sheets, key="excel_sheet_selector")
        except Exception as exc:
            st.error(f"Could not read workbook sheets: {exc}")
            sheets = []
    file_key = f"{uploaded_file.name}:{uploaded_file.size}:{selected_sheet}"
    if st.session_state.get("source_key") != file_key and (extension == "csv" or selected_sheet):
        try:
            loaded = load_uploaded_dataframe(uploaded_file, uploaded_file.name, selected_sheet)
            set_active_dataframe(loaded, uploaded_file.name, file_key)
            st.success(f"Loaded {uploaded_file.name}.")
        except Exception as exc:
            st.error(f"Could not load file: {exc}")

df = st.session_state.get("working_df")
if df is None:
    st.info("Upload a CSV or Excel workbook, or load the bundled sample dataset to begin.")
    st.stop()

summary = summarize_dataframe(df.drop(columns=[INTERNAL_ROW_ID], errors="ignore"))
metric_cols = st.columns(4)
metric_cols[0].metric("Rows", f"{summary.rows:,}")
metric_cols[1].metric("Columns", f"{summary.columns:,}")
metric_cols[2].metric("Numeric Columns", f"{summary.numeric_columns:,}")
metric_cols[3].metric("Text Columns", f"{summary.text_columns:,}")

st.caption(f"Active source: {st.session_state.get('source_name')}")

preview_tab, mapping_tab, qc_tab, filters_tab, geometry_tab = st.tabs(
    ["Preview", "Column Mapping", "QC", "Filters", "Spatial Geometry"]
)

with preview_tab:
    st.dataframe(df.drop(columns=[INTERNAL_ROW_ID], errors="ignore").head(200), width="stretch")

with mapping_tab:
    st.markdown("#### Coordinate and Metadata Mapping")
    st.caption("Suggestions are based on common reservoir data column names and can be overridden.")
    mappings = normalize_column_mappings(st.session_state.get("column_mappings", {}))
    suggestions = suggest_mappings(df)
    columns = [column for column in df.columns if column != INTERNAL_ROW_ID]

    first_row = st.columns(3)
    with first_row[0]:
        mappings["x"] = selectbox_with_none(
            SEMANTIC_FIELDS["x"],
            columns,
            mappings.get("x"),
            suggestions.get("x"),
            "mapping_x",
            required=True,
        )
    with first_row[1]:
        mappings["y"] = selectbox_with_none(
            SEMANTIC_FIELDS["y"],
            columns,
            mappings.get("y"),
            suggestions.get("y"),
            "mapping_y",
            required=True,
        )
    with first_row[2]:
        coordinate_unit_input("data_manager_coordinate_unit")

    identity_row = st.columns(3)
    with identity_row[0]:
        mappings["well"] = selectbox_with_none(
            SEMANTIC_FIELDS["well"],
            columns,
            mappings.get("well"),
            suggestions.get("well"),
            "mapping_well",
        )
    with identity_row[1]:
        mappings["field"] = selectbox_with_none(
            SEMANTIC_FIELDS["field"],
            columns,
            mappings.get("field"),
            suggestions.get("field"),
            "mapping_field",
        )
    with identity_row[2]:
        mappings["reservoir"] = selectbox_with_none(
            SEMANTIC_FIELDS["reservoir"],
            columns,
            mappings.get("reservoir"),
            suggestions.get("reservoir"),
            "mapping_reservoir",
        )

    optional_keys = [
        "panel",
        "layer",
        "formation",
        "measurement_date",
        "map_reference_date",
        "well_type",
        "well_status",
    ]
    for row_start in range(0, len(optional_keys), 3):
        cols = st.columns(3)
        for index, key in enumerate(optional_keys[row_start : row_start + 3]):
            with cols[index]:
                mappings[key] = selectbox_with_none(
                    SEMANTIC_FIELDS[key],
                    columns,
                    mappings.get(key),
                    suggestions.get(key),
                    f"mapping_{key}",
                )

    st.session_state.column_mappings = mappings

    property_options = numeric_property_candidates(df, mappings)
    if property_options:
        current_property = st.session_state.get("current_property")
        if current_property not in property_options:
            current_property = property_options[0]
        st.session_state.current_property = st.selectbox(
            "Default Property for QC",
            property_options,
            index=property_options.index(current_property),
            key="default_property_qc",
        )
    else:
        st.warning("No numeric property columns were detected beyond the mapped coordinates.")

    additional_options = filter_candidate_options(df, mappings)
    selected_additional = [
        column for column in st.session_state.get("additional_filter_columns", []) if column in additional_options
    ]
    st.session_state.additional_filter_columns = st.multiselect(
        "Additional Filters",
        additional_options,
        default=selected_additional,
        help="Low-cardinality non-coordinate columns can be used as extra filters.",
    )

with qc_tab:
    mappings = st.session_state.get("column_mappings", {})
    x_col = mappings.get("x")
    y_col = mappings.get("y")
    property_col = st.session_state.get("current_property")
    well_col = mappings.get("well")
    if not x_col or not y_col:
        st.warning("Map X and Y coordinate columns before QC can run.")
    else:
        qc = build_qc_summary(df, x_col, y_col, property_col, well_col)
        qc_cols = st.columns(4)
        qc_cols[0].metric("Total Rows", f"{qc.total_rows:,}")
        qc_cols[1].metric("Valid X/Y Rows", f"{qc.valid_xy_rows:,}")
        qc_cols[2].metric("Missing X", f"{qc.missing_x:,}")
        qc_cols[3].metric("Missing Y", f"{qc.missing_y:,}")

        if property_col:
            stat_cols = st.columns(5)
            stat_cols[0].metric("Missing Property", f"{qc.missing_property or 0:,}")
            stat_cols[1].metric("Minimum", "" if qc.property_min is None else f"{qc.property_min:.4g}")
            stat_cols[2].metric("Maximum", "" if qc.property_max is None else f"{qc.property_max:.4g}")
            stat_cols[3].metric("Mean", "" if qc.property_mean is None else f"{qc.property_mean:.4g}")
            stat_cols[4].metric("Std Dev", "" if qc.property_std is None else f"{qc.property_std:.4g}")

        warnings: list[str] = []
        if qc.missing_x:
            warnings.append(f"{qc.missing_x} record(s) contain missing or non-numeric X coordinates.")
        if qc.missing_y:
            warnings.append(f"{qc.missing_y} record(s) contain missing or non-numeric Y coordinates.")
        if qc.duplicate_xy_rows:
            warnings.append(f"{qc.duplicate_xy_rows} record(s) share duplicate XY coordinates.")
        if qc.outlier_count:
            warnings.append(f"{qc.outlier_count} potential property outlier(s) were detected.")
        for warning in warnings:
            st.warning(warning)

        duplicate_rows = duplicate_coordinate_rows(df, x_col, y_col)
        if not duplicate_rows.empty:
            with st.expander("Duplicate Coordinate Rows", expanded=False):
                st.dataframe(duplicate_rows.drop(columns=[INTERNAL_ROW_ID], errors="ignore"), width="stretch")

        if property_col:
            flagged = flag_outliers(df, property_col)
            outliers = flagged[flagged[OUTLIER_COLUMN]]
            if not outliers.empty:
                with st.expander("Potential Outliers", expanded=False):
                    st.dataframe(outliers.drop(columns=[INTERNAL_ROW_ID], errors="ignore"), width="stretch")

        measurement_col = mappings.get("measurement_date")
        if measurement_col and measurement_col in df.columns:
            measurement_summary = summarize_measurement_dates(df[measurement_col])
            info_cols = st.columns(3)
            info_cols[0].metric(
                "Earliest Measurement",
                format_map_date(measurement_summary.earliest) if measurement_summary.earliest else "",
            )
            info_cols[1].metric(
                "Latest Measurement",
                format_map_date(measurement_summary.latest) if measurement_summary.latest else "",
            )
            info_cols[2].metric(
                "Measurement Span",
                "" if measurement_summary.span_days is None else f"{measurement_summary.span_days:,} days",
            )
            if measurement_summary.failed_count:
                st.warning(f"{measurement_summary.failed_count} measurement date value(s) could not be parsed.")

        reference_col = mappings.get("map_reference_date")
        if reference_col and reference_col in df.columns:
            reference_validation = validate_date_column(df[reference_col])
            if reference_validation.common_date:
                st.info(
                    "Detected common Pressure Map Reference Date: "
                    f"{format_map_date(reference_validation.common_date)}"
                )
            elif reference_validation.has_multiple_dates:
                st.warning(
                    "Multiple pressure map reference dates were detected. A single pressure map should normally "
                    "contain pressure values representing one common reference date."
                )
            if reference_validation.failed_count:
                st.warning(f"{reference_validation.failed_count} map reference date value(s) could not be parsed.")

with filters_tab:
    st.markdown("#### Metadata Filters")
    filtered = render_filter_controls(df, st.session_state.get("column_mappings", {}), "data_manager")
    st.caption(f"{len(filtered):,} of {len(df):,} rows match the current filters.")
    st.dataframe(filtered.drop(columns=[INTERNAL_ROW_ID], errors="ignore").head(300), width="stretch")

with geometry_tab:
    st.markdown("#### Spatial Geometry")
    st.caption("Imported geometry must use the same coordinate system and units as the well X/Y data.")
    mappings = st.session_state.get("column_mappings", {})
    x_col = mappings.get("x")
    y_col = mappings.get("y")
    coordinate_unit = coordinate_unit_input("data_manager_geometry_coordinate_unit")
    geometry_layers = st.session_state.geometry_layers

    summary_cols = st.columns(5)
    boundary_layer = geometry_layers.get("reservoir_boundary")
    panel_layer = geometry_layers.get("panels")
    fault_layer = geometry_layers.get("faults")
    custom_layers = geometry_layers.get("custom", [])
    summary_cols[0].metric("Reservoir Boundary", "Loaded" if boundary_layer else "Not loaded")
    summary_cols[1].metric("Panels", 0 if panel_layer is None else panel_layer.feature_count)
    summary_cols[2].metric("Faults", 0 if fault_layer is None else fault_layer.feature_count)
    summary_cols[3].metric("Reference Geometry", len(custom_layers))
    summary_cols[4].metric("Coordinate Unit", coordinate_unit_symbol(coordinate_unit))

    for label, key in [
        ("Reservoir Outer Boundary", "reservoir_boundary"),
        ("Panel / Compartment Polygons", "panels"),
        ("Fault Lines", "faults"),
    ]:
        layer = geometry_layers.get(key)
        if layer:
            st.write(f"{label}: {layer.feature_count} feature(s), total polygon area {format_area(layer_total_area(layer), coordinate_unit)}")
            if st.button(f"Remove {label}", key=f"remove_{key}"):
                geometry_layers[key] = None
                st.session_state.generated_map = None
                st.rerun()

    for index, layer in enumerate(list(custom_layers)):
        st.write(
            f"Reference / Custom Geometry - {layer.name}: {layer.feature_count} feature(s), "
            f"total polygon area {format_area(layer_total_area(layer), coordinate_unit)}"
        )
        if st.button(f"Remove Reference Geometry {layer.name}", key=f"remove_custom_{index}"):
            custom_layers.pop(index)
            geometry_layers["custom"] = custom_layers
            st.session_state.generated_map = None
            st.rerun()

    uploaded_geometry = st.file_uploader(
        "Upload Geometry",
        type=["geojson", "json", "csv", "zip"],
        key="geometry_uploader",
    )
    geometry_role_to_type = {
        "Reservoir Outer Boundary": "Reservoir Boundary",
        "Panel / Compartment Polygons": "Panel / Compartment",
        "Fault Lines": "Fault",
        "Reference / Custom Geometry": "Custom",
    }
    geometry_role = st.selectbox(
        "Geometry Role",
        list(geometry_role_to_type),
        key="geometry_layer_type",
    )
    layer_type = geometry_role_to_type[geometry_role]
    layer_name = st.text_input("Geometry Name", value=geometry_role, key="geometry_layer_name")

    if uploaded_geometry is not None:
        extension = uploaded_geometry.name.lower().rsplit(".", 1)[-1]
        name_attribute = None
        csv_df = None
        shapefile_choice = None
        convert_closed_linework = False
        try:
            raw_bytes = uploaded_geometry.getvalue()
            if extension in {"geojson", "json"}:
                property_names = geojson_property_names(raw_bytes)
                suggested = suggest_geometry_name_attribute(property_names)
                options = ["None"] + property_names
                default_index = options.index(suggested) if suggested in options else 0
                attr_label = "Panel Name Attribute" if layer_type == "Panel / Compartment" else "Name / ID Attribute"
                selected_name_attr = st.selectbox(attr_label, options, index=default_index)
                name_attribute = None if selected_name_attr == "None" else selected_name_attr
            elif extension == "csv":
                csv_df = pd.read_csv(uploaded_geometry)
                csv_columns = list(csv_df.columns)
                x_guess = "X" if "X" in csv_columns else csv_columns[0]
                y_guess = "Y" if "Y" in csv_columns else csv_columns[min(1, len(csv_columns) - 1)]
                geom_x_col = st.selectbox("Geometry X Column", csv_columns, index=csv_columns.index(x_guess))
                geom_y_col = st.selectbox("Geometry Y Column", csv_columns, index=csv_columns.index(y_guess))
                group_options = ["None"] + csv_columns
                default_group = "Panel" if "Panel" in csv_columns else ("Polygon_ID" if "Polygon_ID" in csv_columns else "None")
                group_label = "Panel Name Attribute" if layer_type == "Panel / Compartment" else "Polygon / Line ID Column"
                group_col = st.selectbox(group_label, group_options, index=group_options.index(default_group))
                name_attribute = None if group_col == "None" else group_col
            elif extension == "zip":
                zip_candidates = get_zipped_shapefile_candidates(raw_bytes)
                shapefile_choice = st.selectbox(
                    "Shapefile in ZIP",
                    zip_candidates,
                    index=0,
                    key="zip_selected_shapefile",
                    help="Choose the specific .shp to load when multiple shapefiles are present in the archive.",
                )
                zip_fields = read_zipped_shapefile_fields(raw_bytes, shapefile_choice)
                zip_field_options = ["None"] + zip_fields
                suggested_field = suggest_geometry_name_attribute(zip_fields)
                default_field_index = zip_field_options.index(suggested_field) if suggested_field in zip_field_options else 0
                attr_label = "Panel Name Attribute" if layer_type == "Panel / Compartment" else "Name / ID Attribute"
                selected_field = st.selectbox(
                    attr_label,
                    zip_field_options,
                    index=default_field_index,
                    key="zip_name_attribute",
                )
                name_attribute = None if selected_field == "None" else selected_field
                if layer_type in {"Reservoir Boundary", "Panel / Compartment"}:
                    detected_linework = read_zipped_shapefile_geometry_types(raw_bytes, shapefile_choice)
                    if detected_linework["has_line_geometry"]:
                        st.info("Boundary line geometry detected.")
                        convert_closed_linework = st.checkbox(
                            "Convert closed linework to polygons",
                            value=False,
                            key="convert_closed_linework",
                        )
                    else:
                        convert_closed_linework = False
                else:
                    convert_closed_linework = False

            if st.button("Validate And Add Geometry", type="primary"):
                if extension in {"geojson", "json"}:
                    layer = load_geojson_bytes(
                        raw_bytes,
                        layer_type,
                        layer_name,
                        name_attribute,
                        uploaded_geometry.name,
                        convert_closed_linework=convert_closed_linework if layer_type in {"Reservoir Boundary", "Panel / Compartment"} else False,
                    )
                elif extension == "csv" and csv_df is not None:
                    layer = load_geometry_csv(csv_df, layer_type, layer_name, geom_x_col, geom_y_col, name_attribute, uploaded_geometry.name)
                elif extension == "zip":
                    layer = load_zipped_shapefile_bytes(
                        raw_bytes,
                        layer_type,
                        layer_name,
                        name_attribute,
                        uploaded_geometry.name,
                        shapefile_choice,
                        convert_closed_linework=convert_closed_linework,
                    )
                else:
                    raise ValueError("Unsupported geometry file type.")

                extent = None
                if x_col and y_col:
                    numeric_x = pd.to_numeric(df[x_col], errors="coerce")
                    numeric_y = pd.to_numeric(df[y_col], errors="coerce")
                    if numeric_x.notna().any() and numeric_y.notna().any():
                        extent = (float(numeric_x.min()), float(numeric_y.min()), float(numeric_x.max()), float(numeric_y.max()))
                validated_layer, report = validate_geometry_layer(layer, extent)
                if layer_type == "Reservoir Boundary":
                    geometry_layers["reservoir_boundary"] = validated_layer
                elif layer_type == "Panel / Compartment":
                    geometry_layers["panels"] = validated_layer
                elif layer_type == "Fault":
                    geometry_layers["faults"] = validated_layer
                else:
                    geometry_layers.setdefault("custom", []).append(validated_layer)
                st.session_state.geometry_layers = geometry_layers
                st.session_state.generated_map = None
                st.success(
                    f"Added {validated_layer.name}: {report.valid_features} valid, "
                    f"{report.repaired_features} repaired, {report.invalid_features} invalid."
                )
                if report.overlaps:
                    for overlap in report.overlaps:
                        st.warning(
                            f"{overlap.first} and {overlap.second} overlap by "
                            f"{format_area(overlap.area, coordinate_unit)}."
                        )
                for issue in report.issues:
                    if issue.severity == "error":
                        st.error(f"{issue.feature_name}: {issue.message}")
                    else:
                        st.warning(f"{issue.feature_name}: {issue.message}")
        except Exception as exc:
            st.error(f"Could not load geometry: {exc}")
