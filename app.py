"""Reservoir Mapping Studio application entrypoint."""

from __future__ import annotations

import streamlit as st

from pages.shared import ensure_session_state
from utils.constants import APP_NAME


def main() -> None:
    st.set_page_config(
        page_title=APP_NAME,
        page_icon="RMS",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    ensure_session_state()

    st.markdown(
        """
        <style>
        .block-container { padding-top: 1.6rem; padding-bottom: 2rem; }
        div[data-testid="stMetric"] {
            background: #f8fafc;
            border: 1px solid #e5e7eb;
            border-radius: 6px;
            padding: 0.7rem 0.8rem;
        }
        div[data-testid="stAlert"] { border-radius: 6px; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    pages = [
        st.Page("pages/project.py", title="Project", icon=":material/folder_open:"),
        st.Page("pages/data_manager.py", title="Data Manager", icon=":material/table:"),
        st.Page("pages/mapping_studio.py", title="Mapping Studio", icon=":material/map:"),
        st.Page("pages/geostatistics_lab.py", title="Geostatistics Lab", icon=":material/scatter_plot:"),
        st.Page("pages/map_comparison.py", title="Map Comparison", icon=":material/compare:"),
    ]
    navigation = st.navigation(pages)
    navigation.run()


if __name__ == "__main__":
    main()
