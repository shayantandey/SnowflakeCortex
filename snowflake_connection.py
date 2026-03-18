from __future__ import annotations

import streamlit as st
import snowflake.connector
from snowflake.connector import SnowflakeConnection


def get_connection(params: dict) -> SnowflakeConnection:
    """Create and cache a Snowflake connection in session state."""
    conn_kwargs = {
        "account": params["account"],
        "user": params["user"],
        "role": params.get("role", ""),
        "warehouse": params.get("warehouse", ""),
        "database": params.get("database", ""),
        "schema": params.get("schema", "PUBLIC"),
        "session_parameters": {
            "QUERY_TAG": "CortexAnalystApp",
            "STATEMENT_TIMEOUT_IN_SECONDS": 120,
        },
    }

    if params.get("private_key_path"):
        from cryptography.hazmat.backends import default_backend
        from cryptography.hazmat.primitives import serialization

        with open(params["private_key_path"], "rb") as key_file:
            passphrase = params.get("private_key_passphrase", "")
            p_key = serialization.load_pem_private_key(
                key_file.read(),
                password=passphrase.encode() if passphrase else None,
                backend=default_backend(),
            )
        pkb = p_key.private_bytes(
            encoding=serialization.Encoding.DER,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption(),
        )
        conn_kwargs["private_key"] = pkb
    else:
        conn_kwargs["password"] = params["password"]

    conn = snowflake.connector.connect(**conn_kwargs)
    # Smoke test
    cur = conn.cursor()
    try:
        cur.execute("SELECT 1").fetchone()
    finally:
        cur.close()
    return conn


def get_auth_token(conn: SnowflakeConnection) -> str:
    """Extract the session token for REST API calls."""
    return conn.rest.token


def close_connection() -> None:
    conn: SnowflakeConnection | None = st.session_state.get("sf_connection")
    if conn:
        try:
            conn.close()
        except Exception:
            pass
        del st.session_state["sf_connection"]
