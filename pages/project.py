"""Project management and saved-map library page."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from core.crs import EPSG_CRS_MODE, LOCAL_CRS_MODE, crs_display_name, local_crs, validate_epsg
from core.project_io import ProjectArchiveError, load_project_archive, save_project_archive
from core.scenarios import duplicate_scenario, rename_scenario, scenario_summary_table, scenario_to_generated_map
from pages.shared import (
    ensure_session_state,
    mark_project_dirty,
    mark_project_saved,
    project_display_name,
    reset_workspace_for_new_project,
    workflow_status,
)


ensure_session_state()


def _session_snapshot() -> dict[str, object]:
    keys = [
        "project_metadata",
        "original_df",
        "working_df",
        "source_name",
        "column_mappings",
        "additional_filter_columns",
        "filter_values",
        "coordinate_unit",
        "property_unit",
        "pressure_reference_date",
        "layer_mapping_scope",
        "selected_reservoir_layer",
        "active_generated_layer",
        "generated_layer_maps",
        "generated_layer_statuses",
        "generated_layer_batch_signature",
        "engineering_control_points",
        "engineering_control_regions",
        "crs",
        "include_state",
        "geometry_layers",
        "layer_settings",
        "style_settings",
        "current_property",
        "map_scenarios",
        "current_scenario_id",
    ]
    return {key: st.session_state.get(key) for key in keys}


def _safe_filename(name: str) -> str:
    safe = "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in name.strip())
    return safe or "Reservoir_Mapping_Project"


def _apply_loaded_project(state: dict[str, object]) -> None:
    for key, value in state.items():
        st.session_state[key] = value
    st.session_state.generated_map = None
    generated_layers = st.session_state.get("generated_layer_maps", {}) or {}
    active_layer = st.session_state.get("active_generated_layer")
    if generated_layers:
        if active_layer not in generated_layers:
            active_layer = next(iter(generated_layers))
            st.session_state.active_generated_layer = active_layer
        st.session_state.generated_map = generated_layers[active_layer]
        mark_project_saved()
        return
    current_id = st.session_state.get("current_scenario_id")
    for scenario in st.session_state.get("map_scenarios", []):
        if scenario.get("id") == current_id:
            st.session_state.generated_map = scenario_to_generated_map(scenario)
            break
    mark_project_saved()


st.title("Reservoir Mapping Studio")
st.subheader("Project")
st.caption(f"Project: {project_display_name()}")

status = workflow_status()
status_cols = st.columns(5)
for column, (label, ready) in zip(status_cols, status.items()):
    column.metric(label, "Ready" if ready else "Pending")

metadata = dict(st.session_state.get("project_metadata", {}))
data_rows = 0 if st.session_state.get("working_df") is None else len(st.session_state.working_df)
geometry = st.session_state.get("geometry_layers", {})
scenario_count = len(st.session_state.get("map_scenarios", []))
overview_cols = st.columns(4)
overview_cols[0].metric("Data", f"{data_rows:,} rows")
overview_cols[1].metric("Geometry", "Loaded" if any(geometry.get(key) for key in ("reservoir_boundary", "panels", "faults")) else "None")
overview_cols[2].metric("Saved Maps", scenario_count)
overview_cols[3].metric("CRS", crs_display_name(st.session_state.get("crs")))

project_tab, crs_tab, library_tab = st.tabs(["Project File", "CRS", "Map Library"])

with project_tab:
    st.markdown("#### New Project")
    new_cols = st.columns(2)
    with new_cols[0]:
        new_name = st.text_input("Project Name", value=metadata.get("name", "Untitled Project"))
        new_description = st.text_area("Description", value=metadata.get("description", ""), height=80)
    with new_cols[1]:
        new_engineer = st.text_input("Engineer / Author", value=metadata.get("engineer", ""))
        new_field = st.text_input("Field", value=metadata.get("field", ""))
        new_notes = st.text_area("Notes", value=metadata.get("notes", ""), height=80)
    if st.button("New Project", type="primary"):
        reset_workspace_for_new_project(
            {
                "name": new_name,
                "description": new_description,
                "engineer": new_engineer,
                "field": new_field,
                "notes": new_notes,
            }
        )
        st.success("New project created.")
        st.rerun()

    st.markdown("#### Open Project")
    uploaded_project = st.file_uploader("Open .rmsproj", type=["rmsproj", "zip"])
    if uploaded_project is not None and st.button("Open Project"):
        try:
            _apply_loaded_project(load_project_archive(uploaded_project.getvalue()))
            st.success("Project opened.")
            st.rerun()
        except (ProjectArchiveError, ValueError) as exc:
            st.error(str(exc))

    st.markdown("#### Save Project")
    try:
        project_bytes = save_project_archive(_session_snapshot())
        st.download_button(
            "Save Project",
            project_bytes,
            file_name=f"{_safe_filename(metadata.get('name', 'Reservoir_Mapping_Project'))}.rmsproj",
            mime="application/zip",
            on_click=mark_project_saved,
            width="stretch",
        )
    except Exception as exc:
        st.error(f"Project could not be packaged: {exc}")

with crs_tab:
    st.markdown("#### Coordinate Reference System")
    current_crs = dict(st.session_state.get("crs", local_crs().to_dict()))
    mode = st.radio(
        "CRS Mode",
        [LOCAL_CRS_MODE, EPSG_CRS_MODE],
        index=1 if current_crs.get("mode") == EPSG_CRS_MODE else 0,
        horizontal=True,
    )
    if mode == LOCAL_CRS_MODE:
        st.session_state.crs = local_crs().to_dict()
        st.caption("Local / Unknown XY remains fully supported. CRS is stored as project metadata only.")
    else:
        default_epsg = int(current_crs.get("epsg") or 32638)
        epsg = st.number_input("EPSG Code", min_value=1, max_value=999999, value=default_epsg, step=1)
        try:
            resolved = validate_epsg(epsg)
            st.session_state.crs = resolved.to_dict()
            st.success(crs_display_name(st.session_state.crs))
        except ValueError as exc:
            st.error(str(exc))
    if st.button("Update CRS"):
        mark_project_dirty()
        st.success("CRS settings updated.")

with library_tab:
    st.markdown("#### Saved Maps")
    scenarios = list(st.session_state.get("map_scenarios", []))
    if scenarios:
        st.dataframe(scenario_summary_table(scenarios), width="stretch", hide_index=True)
        labels = [str(scenario.get("name") or scenario.get("id")) for scenario in scenarios]
        selected_label = st.selectbox("Selected Map Scenario", labels)
        selected_index = labels.index(selected_label)
        selected = scenarios[selected_index]
        action_cols = st.columns(5)
        if action_cols[0].button("Open"):
            st.session_state.generated_map = scenario_to_generated_map(selected)
            st.session_state.current_scenario_id = selected.get("id")
            st.success("Scenario opened in Mapping Studio.")
        if action_cols[1].button("Duplicate"):
            scenarios.append(duplicate_scenario(selected))
            st.session_state.map_scenarios = scenarios
            mark_project_dirty()
            st.rerun()
        rename_value = st.text_input("Rename Selected Scenario", value=str(selected.get("name") or "Saved Map"))
        if action_cols[2].button("Rename"):
            scenarios[selected_index] = rename_scenario(selected, rename_value)
            st.session_state.map_scenarios = scenarios
            mark_project_dirty()
            st.rerun()
        if action_cols[3].button("Delete"):
            deleted_id = selected.get("id")
            st.session_state.map_scenarios = [scenario for scenario in scenarios if scenario.get("id") != deleted_id]
            if st.session_state.get("current_scenario_id") == deleted_id:
                st.session_state.current_scenario_id = None
                st.session_state.generated_map = None
            mark_project_dirty()
            st.rerun()
        if action_cols[4].button("Use In Comparison"):
            st.session_state.compare_map_a_id = selected.get("id")
            st.success("Selected as Map A in Map Comparison.")
    else:
        st.info("Generate a map in Mapping Studio, then save it as a scenario.")

    if st.session_state.get("generated_map"):
        st.markdown("#### Save Current Generated Map")
        default_name = str(st.session_state.generated_map.get("title") or "Saved Map")
        scenario_name = st.text_input("Scenario Name", value=default_name, key="project_save_current_scenario_name")
        if st.button("Save Current Map Scenario", type="primary"):
            from core.scenarios import create_map_scenario

            validation = st.session_state.get("geostatistics", {}).get("cross_validation")
            scenario = create_map_scenario(
                scenario_name,
                st.session_state.generated_map,
                {
                    "style_settings": st.session_state.get("style_settings", {}),
                    "layer_settings": st.session_state.get("layer_settings", {}),
                    "filter_values": st.session_state.get("filter_values", {}),
                    "include_state": st.session_state.get("include_state", {}),
                    "crs": st.session_state.get("crs", {}),
                },
                validation,
            )
            st.session_state.map_scenarios = scenarios + [scenario]
            st.session_state.current_scenario_id = scenario["id"]
            mark_project_dirty()
            st.success("Map scenario saved.")
