from __future__ import annotations

from typing import Mapping, MutableMapping


def apply_secrets_to_env(secrets: Mapping, env: MutableMapping[str, str]) -> None:
    """Copy string secret values into ``env`` for keys that are not already set.

    Streamlit Cloud injects configuration via ``st.secrets``, but this app reads
    everything through ``os.getenv`` (see ``AppConfig.from_env``). This bridges
    the two: each top-level string secret is written into the environment unless
    a real environment variable already provides it (real env wins, so local
    ``.env`` / shell overrides are never clobbered). Non-string values (numbers,
    nested TOML sections) are skipped — only flat string settings map to env.
    """
    for key, value in secrets.items():
        if not isinstance(value, str):
            continue
        if key in env:
            continue
        env[key] = value


def load_streamlit_secrets_into_env() -> None:
    """Apply ``st.secrets`` to ``os.environ`` before config is read.

    A no-op when Streamlit is unavailable or no secrets file is configured, so
    it is safe to call unconditionally at startup (including in local/CLI runs).
    """
    import os

    try:
        import streamlit as st

        secrets = dict(st.secrets)
    except Exception:
        # No Streamlit context, or no secrets.toml configured: nothing to bridge.
        return

    apply_secrets_to_env(secrets, os.environ)
