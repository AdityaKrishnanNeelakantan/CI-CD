"""Effective user defaults and read-only deployment settings."""

from __future__ import annotations

import streamlit as st

from synth_platform.application.dto.tool_commands import (
    ReadPreferencesCommand,
    ReadRuntimeSettingsCommand,
)
from synth_platform.application.dto.workspace import UserPreferences
from synth_platform.interfaces.streamlit.workspace import (
    get_workspace_tools,
    save_preferences_from_form,
)

st.title("Settings")
st.caption(
    "Effective generation defaults are stored locally. Security and output "
    "boundaries are deployment-controlled."
)
tools = get_workspace_tools()
preferences = tools.get_preferences(ReadPreferencesCommand())
locales = ["en_US", "en_GB", "en_IN", "de_DE", "fr_FR"]

with st.form("user_preferences", border=True):
    st.markdown("### Generation defaults")
    c1, c2 = st.columns(2)
    locale = c1.selectbox(
        "Default locale",
        locales,
        index=locales.index(preferences.locale) if preferences.locale in locales else 0,
    )
    default_rows = c2.number_input(
        "Default Schema rows",
        min_value=1,
        max_value=1_000_000,
        value=preferences.default_row_count,
    )
    export_format = c1.selectbox(
        "Default Schema table format",
        ["csv", "parquet"],
        index=["csv", "parquet"].index(preferences.default_export_format),
    )
    st.caption("These defaults are consumed by the Schema Twin configuration form.")
    if st.form_submit_button("Save defaults", type="primary"):
        save_preferences_from_form(
            UserPreferences(
                locale=locale,
                default_row_count=int(default_rows),
                default_export_format=export_format,
            )
        )
        st.success("Generation defaults saved locally.")

runtime = tools.runtime_settings(ReadRuntimeSettingsCommand())
with st.container(border=True):
    st.markdown("### Guardrails (read only)")
    c1, c2, c3 = st.columns(3)
    c1.metric("Policy", runtime.guardrail_policy_version)
    c2.metric("Model network", runtime.local_model_network_scope)
    c3.metric("Tool deletes", "default deny")
    st.write(
        "Model input/output checks and workspace tool schema/permission checks are "
        "deployment-enforced. Browser preferences cannot disable them."
    )
    st.caption(
        "Safety detection is deterministic and conservative; it is not presented as "
        "a comprehensive moderation classifier."
    )

with st.container(border=True):
    st.markdown("### Deployment controls (read only)")
    st.caption(
        "Change these through environment/deployment configuration, not browser "
        "preferences."
    )
    st.json(runtime.model_dump(mode="json"))
    st.info(
        "Credentials, connection URLs, and raw transcript text are never stored in "
        "workspace settings. This UI is supported in single-user localhost mode only."
    )
