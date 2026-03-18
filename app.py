from __future__ import annotations

import re
import time

import pandas as pd
import streamlit as st

from audit_log import log_query
from cost_utils import get_query_cost, CREDIT_PRICE_USD, format_bytes
from config import settings
from snowflake_connection import get_connection, close_connection
from cortex_analyst import send_message, build_api_messages, CortexAnalystError
from semantic_model import validate_stage_path, validate_yaml, upload_to_stage
from chart_utils import render_result
from forecast_utils import get_date_columns, get_numeric_columns, run_forecast, render_forecast_chart

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Chat with Data – Snowflake Cortex",
    page_icon="❄️",
    layout="wide",
)

# ── Session state defaults ─────────────────────────────────────────────────────
if "conversation" not in st.session_state:
    st.session_state.conversation = []          # display messages
if "semantic_model_path" not in st.session_state:
    st.session_state.semantic_model_path = settings.CORTEX_SEMANTIC_MODEL_STAGE_PATH
if "row_limit" not in st.session_state:
    st.session_state.row_limit = settings.CORTEX_RESULT_ROW_LIMIT
if "last_activity" not in st.session_state:
    st.session_state.last_activity = time.time()
if "session_start" not in st.session_state:
    st.session_state.session_start = time.time()
if "session_queries" not in st.session_state:
    st.session_state.session_queries = 0
if "session_credits" not in st.session_state:
    st.session_state.session_credits = 0.0
if "session_cost_usd" not in st.session_state:
    st.session_state.session_cost_usd = 0.0


# ── Guardrail constants ────────────────────────────────────────────────────────
_MAX_QUESTION_LENGTH = 2000
_MAX_YAML_UPLOAD_BYTES = 512 * 1024          # 512 KB
_SESSION_TIMEOUT_MINUTES = 60
_WRITE_SQL_RE = re.compile(
    r'\b(INSERT|UPDATE|DELETE|DROP|CREATE|ALTER|TRUNCATE|MERGE|GRANT|REVOKE|EXECUTE|PUT|COPY)\b',
    re.IGNORECASE,
)
_PII_KEYWORDS = frozenset([
    "email", "phone", "mobile", "ssn", "social_security", "dob", "birthdate",
    "birth_date", "address", "street", "zipcode", "zip_code", "postcode",
    "firstname", "first_name", "lastname", "last_name", "fullname", "full_name",
    "passport", "license", "credit_card", "card_number", "iban", "ip_address",
])
_TOKEN_RE = re.compile(
    r'(?:[A-Za-z0-9+/]{20,}={0,2}\.){2}[A-Za-z0-9+/\-_]{20,}'  # JWT
    r'|[0-9a-f]{64,}'                                             # long hex
    r'|[A-Za-z0-9+/]{80,}={0,2}',                                # long base64
    re.IGNORECASE,
)


def _is_safe_sql(sql: str) -> tuple[bool, str]:
    """Reject SQL containing write or DDL operations; only SELECT/WITH allowed."""
    clean = re.sub(r'--[^\n]*', ' ', sql)
    clean = re.sub(r'/\*.*?\*/', ' ', clean, flags=re.DOTALL)
    m = _WRITE_SQL_RE.search(clean)
    if m:
        return False, f"Operation {m.group().upper()} is not permitted."
    first = clean.strip().split()[0].upper() if clean.strip().split() else ""
    if first not in ("SELECT", "WITH"):
        return False, f"Only SELECT queries are permitted (got: {first!r})."
    return True, ""


def _sanitize_error(msg: str) -> str:
    """Redact token-like strings from error messages before displaying."""
    return _TOKEN_RE.sub("[REDACTED]", str(msg))


def _warn_pii_columns(df: pd.DataFrame) -> None:
    """Show a warning if result columns match common PII field names."""
    pii = [
        c for c in df.columns
        if any(kw in c.lower().replace(" ", "_") for kw in _PII_KEYWORDS)
    ]
    if pii:
        st.warning(
            f"⚠️ Potential PII detected in column(s): **{', '.join(pii)}**. "
            "Review before sharing or exporting."
        )


def _render_query_cost(cost: dict) -> None:
    """Render per-query execution stats and estimated cost in an expander."""
    with st.expander("📊 Query Stats & Cost", expanded=False):
        c1, c2, c3 = st.columns(3)
        c1.metric("Execution Time", f"{cost['elapsed_ms'] / 1000:.2f}s")
        c2.metric("Credits Used", f"{cost['credits']:.6f}")
        c3.metric("Est. Cost (USD)", f"${cost['cost_usd']:.6f}")
        st.caption(
            f"Bytes scanned: {format_bytes(cost['bytes_scanned'])} · "
            f"Rows produced: {cost['rows_produced']:,} · "
            f"Warehouse: {cost['warehouse_size']}"
        )


def _check_session_expiry() -> None:
    """Disconnect if the Snowflake session has been idle beyond the timeout."""
    if "sf_connection" not in st.session_state:
        return
    elapsed_min = (time.time() - st.session_state.get("last_activity", time.time())) / 60
    if elapsed_min > _SESSION_TIMEOUT_MINUTES:
        st.warning(
            f"Session expired after {_SESSION_TIMEOUT_MINUTES} minutes of inactivity. "
            "You have been disconnected."
        )
        close_connection()
        st.session_state.last_activity = time.time()
        st.rerun()


_check_session_expiry()


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("❄️ Snowflake Connection")

    connected = "sf_connection" in st.session_state

    if connected:
        conn = st.session_state["sf_connection"]
        st.success(f"Connected as **{conn.user}**")
        if st.button("Disconnect", use_container_width=True):
            close_connection()
            st.rerun()
    else:
        with st.form("connection_form"):
            account = st.text_input(
                "Account", value=settings.SNOWFLAKE_ACCOUNT,
                placeholder="myorg-myaccount",
            )
            user = st.text_input("User", value=settings.SNOWFLAKE_USER)
            password = st.text_input(
                "Password", type="password",
                value=settings.SNOWFLAKE_PASSWORD.get_secret_value()
                if settings.SNOWFLAKE_PASSWORD else "",
            )
            col1, col2 = st.columns(2)
            with col1:
                role = st.text_input("Role", value=settings.SNOWFLAKE_ROLE)
                database = st.text_input("Database", value=settings.SNOWFLAKE_DATABASE)
            with col2:
                warehouse = st.text_input("Warehouse", value=settings.SNOWFLAKE_WAREHOUSE)
                schema = st.text_input("Schema", value=settings.SNOWFLAKE_SCHEMA)

            submitted = st.form_submit_button("Connect", use_container_width=True)

        if submitted:
            with st.spinner("Connecting…"):
                try:
                    conn = get_connection({
                        "account": account,
                        "user": user,
                        "password": password,
                        "role": role,
                        "warehouse": warehouse,
                        "database": database,
                        "schema": schema,
                    })
                    st.session_state["sf_connection"] = conn
                    st.rerun()
                except Exception as e:
                    st.error(f"Connection failed: {e}")

    st.divider()

    # ── Semantic model ────────────────────────────────────────────────────────
    st.subheader("Semantic Model")

    stage_path_input = st.text_input(
        "Stage path",
        value=st.session_state.semantic_model_path,
        placeholder="@DB.SCHEMA.STAGE/model.yaml",
        help="Full stage path to your Cortex Analyst semantic model YAML.",
    )

    if stage_path_input != st.session_state.semantic_model_path:
        valid, err = validate_stage_path(stage_path_input)
        if valid:
            st.session_state.semantic_model_path = stage_path_input
        else:
            st.warning(err)

    with st.expander("Upload YAML to Stage", expanded=not bool(st.session_state.semantic_model_path)):
        if st.session_state.semantic_model_path:
            st.caption("Model already configured. Only upload if you want to replace it.")
        uploaded_file = st.file_uploader("Choose a .yaml file", type=["yaml", "yml"])
        target_path = st.text_input(
            "Target stage path",
            placeholder="@DB.SCHEMA.STAGE/my_model.yaml",
        )
        if st.button("Upload", disabled=not (uploaded_file and target_path)):
            valid_path, err = validate_stage_path(target_path)
            if not valid_path:
                st.error(err)
            else:
                content = uploaded_file.read()
                if len(content) > _MAX_YAML_UPLOAD_BYTES:
                    st.error(
                        f"File too large ({len(content):,} bytes). "
                        f"Maximum allowed size is {_MAX_YAML_UPLOAD_BYTES // 1024} KB."
                    )
                elif b'\x00' in content[:512]:
                    st.error("File appears to be binary, not a valid YAML text file.")
                else:
                    valid_yaml, err = validate_yaml(content.decode("utf-8", errors="replace"))
                    if not valid_yaml:
                        st.error(f"Invalid semantic model: {err}")
                    elif not connected:
                        st.error("Connect to Snowflake first.")
                    else:
                        with st.spinner("Uploading…"):
                            try:
                                result_path = upload_to_stage(
                                    st.session_state["sf_connection"],
                                    content,
                                    target_path,
                                )
                                st.session_state.semantic_model_path = result_path
                                st.success(f"Uploaded to {result_path}")
                            except Exception as e:
                                st.error(f"Upload failed: {e}")

    st.divider()
    if st.button("Clear conversation", use_container_width=True):
        st.session_state.conversation = []
        st.rerun()

    with st.expander("Settings"):
        st.session_state.row_limit = st.number_input(
            "Max result rows", min_value=100, max_value=100000,
            value=st.session_state.row_limit, step=1000,
        )

    st.divider()
    st.subheader("💰 Session Cost")
    elapsed = time.time() - st.session_state.session_start
    h, rem = divmod(int(elapsed), 3600)
    m, s = divmod(rem, 60)
    st.caption(f"Session duration: {h:02d}:{m:02d}:{s:02d}")
    col1, col2 = st.columns(2)
    col1.metric("Queries Run", st.session_state.session_queries)
    col2.metric("Est. Cost (USD)", f"${st.session_state.session_cost_usd:.4f}")
    if st.session_state.session_credits > 0:
        st.caption(
            f"Total credits: {st.session_state.session_credits:.6f} "
            f"× ${CREDIT_PRICE_USD:.2f}/credit"
        )


_GUARDRAIL_KEYWORDS = frozenset([
    "guardrail", "guardrails", "restriction", "restrictions",
    "rule", "rules", "limit", "limits", "policy", "policies",
    "what can", "what cannot", "what can't", "not allowed", "blocked",
    "security", "safety", "protection",
])

_GUARDRAILS_RESPONSE = """\
The following guardrails are enforced in this application:

**1. SQL Write-Operation Blocking**
Only `SELECT` / `WITH` queries are permitted. Any SQL generated by Cortex Analyst that contains `INSERT`, `UPDATE`, `DELETE`, `DROP`, `CREATE`, `ALTER`, `TRUNCATE`, `MERGE`, `GRANT`, `REVOKE`, `EXECUTE`, `PUT`, or `COPY` is blocked before execution.

**2. SQL Query Timeout**
All Snowflake queries are subject to a 120-second statement timeout. Queries exceeding this limit are automatically cancelled.

**3. Conversation History Cap**
Only the last 20 question-answer turns are sent to the Cortex Analyst API per request, preventing unbounded payload growth.

**4. Question Length Limit**
User questions are capped at 2,000 characters.

**5. File Upload Restrictions**
Semantic model YAML uploads are limited to 512 KB. Binary files are rejected before parsing.

**6. Forecast Period Cap**
Forecast requests are capped at 365 periods server-side, regardless of the UI input.

**7. Stage Path Injection Prevention**
Snowflake stage identifiers are validated against a strict `@DB.SCHEMA.STAGE` pattern before any `LIST` command is executed.

**8. PII Column Detection**
Query results are scanned for column names matching common PII patterns (email, phone, SSN, address, passport, etc.) and a warning is shown if any are detected.

**9. Session Expiry**
Snowflake sessions are automatically disconnected after 60 minutes of inactivity.

**10. Error Token Redaction**
Error messages are scanned for JWT tokens, long hex strings, and base64 blobs before being displayed, replacing them with `[REDACTED]`.

**11. Audit Logging**
Every question, generated SQL, row count, and error is logged to `audit.log` with a UTC timestamp and the connected username.
"""


def _is_guardrail_intent(question: str) -> bool:
    q = question.lower()
    return any(kw in q for kw in _GUARDRAIL_KEYWORDS)


_FORECAST_KEYWORDS = frozenset([
    "forecast", "predict", "prediction", "projection",
    "next month", "next quarter", "next year", "next week",
])

_TIME_TERMS = frozenset([
    "month", "year", "quarter", "week", "day", "daily", "monthly", "yearly",
    "quarterly", "over time", "trend", "historical", "date", "period",
])
_RANK_TERMS = frozenset([
    "top", "bottom", "best", "worst", "highest", "lowest", "most", "least",
    "ranking", "ranked",
])
_COMPARE_TERMS = frozenset([
    "vs ", "versus", "compared", "previous", "last year", "last month",
    "growth", "change", "difference", "yoy", "mom",
])
_BREAKDOWN_TERMS = frozenset([
    "by region", "by country", "by city", "by product", "by category",
    "by customer", "by store", "by channel", "by segment", "breakdown",
])


def _generate_improved_suggestions(question: str) -> list[str]:
    """Generate enhanced versions of the user's question as follow-up suggestions."""
    q = question.lower()
    base = question.strip().rstrip("?")
    suggestions = []

    has_time = any(t in q for t in _TIME_TERMS)
    has_rank = any(t in q for t in _RANK_TERMS)
    has_compare = any(t in q for t in _COMPARE_TERMS)
    has_breakdown = any(t in q for t in _BREAKDOWN_TERMS)

    if not has_time:
        suggestions.append(f"{base} broken down by month?")
    if not has_rank:
        suggestions.append(f"What are the top 10 for {base.lower()}?")
    if not has_compare:
        suggestions.append(f"{base} compared to the same period last year?")
    if not has_breakdown:
        suggestions.append(f"{base} broken down by category?")
    if not has_time and not has_compare:
        suggestions.append(f"Show the trend for {base.lower()} over the last 12 months?")

    return suggestions[:3]


def _is_forecast_intent(question: str) -> bool:
    q = question.lower()
    return any(kw in q for kw in _FORECAST_KEYWORDS)


def _extract_periods(question: str) -> int:
    """Try to extract a number of periods from phrases like 'next 3 months'."""
    import re
    match = re.search(r"next\s+(\d+)\s*(day|week|month|quarter|year)", question.lower())
    if match:
        n = int(match.group(1))
        unit = match.group(2)
        return {"day": n, "week": n * 7, "month": n * 30, "quarter": n * 90, "year": n * 365}.get(unit, n)
    for phrase, val in [("next year", 365), ("next quarter", 90), ("next month", 30), ("next week", 7)]:
        if phrase in question.lower():
            return val
    return 30


def _to_historical_query(question: str) -> str:
    """Rewrite a forecast question into a historical time-series data request."""
    q = question.lower()
    for kw in ["forecast", "predict", "projection", "project"]:
        q = q.replace(kw, "show historical")
    q = q.replace("next month", "over time").replace("next quarter", "over time") \
         .replace("next year", "over time").replace("next week", "over time")
    import re
    q = re.sub(r"next\s+\d+\s*(day|week|month|quarter|year)s?", "over time", q)
    return f"{q.strip()} with a date breakdown by month"


def _render_forecast_ui(conn, df: pd.DataFrame, key: str) -> None:
    """Show a forecast expander below any query result that has date + numeric columns."""
    date_cols = get_date_columns(df)
    num_cols = get_numeric_columns(df)
    if not date_cols or not num_cols:
        return

    with st.expander("📈 Forecast with Snowflake Cortex ML", expanded=False):
        c1, c2, c3 = st.columns(3)
        ts_col = c1.selectbox("Timestamp column", date_cols, key=f"fc_ts_{key}")
        tgt_col = c2.selectbox("Target column", num_cols, key=f"fc_tgt_{key}")
        periods = c3.number_input("Periods to forecast", min_value=1, max_value=365, value=30, key=f"fc_periods_{key}")

        result_key = f"fc_result_{key}"

        if st.button("Run Forecast", key=f"fc_btn_{key}", use_container_width=True):
            with st.spinner("Training forecast model on Snowflake…"):
                try:
                    st.session_state[result_key] = run_forecast(conn, df, ts_col, tgt_col, int(periods))
                except Exception as e:
                    st.session_state[result_key] = None
                    st.error(f"Forecast failed: {e}")

        if st.session_state.get(result_key) is not None:
            render_forecast_chart(df, st.session_state[result_key], ts_col, tgt_col)
            with st.expander("Forecast data table", expanded=False):
                st.dataframe(st.session_state[result_key], use_container_width=True)


# ── Main area ──────────────────────────────────────────────────────────────────
st.title("Chat with Your Data")
st.caption("Powered by Snowflake Cortex Analyst — ask questions in plain English.")

if not connected:
    st.info("Connect to Snowflake using the sidebar to get started.")
elif not st.session_state.semantic_model_path:
    st.warning("Set a semantic model stage path in the sidebar.")

# Render conversation history
for msg_idx, msg in enumerate(st.session_state.conversation):
    role = msg["role"]

    if role == "user":
        with st.chat_message("user"):
            st.write(msg["text"])

    elif role == "analyst":
        with st.chat_message("assistant", avatar="❄️"):
            if msg.get("error"):
                st.error(msg["error"])
            else:
                if msg.get("text"):
                    st.write(msg["text"])

                if msg.get("sql"):
                    with st.expander("Generated SQL", expanded=False):
                        st.code(msg["sql"], language="sql")

                if msg.get("df") is not None:
                    render_result(msg["df"], row_limit=st.session_state.row_limit)
                    if "sf_connection" in st.session_state:
                        _render_forecast_ui(st.session_state["sf_connection"], msg["df"], key=f"hist_{msg_idx}")

                if msg.get("suggestions"):
                    st.markdown("**Suggested follow-up questions:**")
                    for sug_idx, suggestion in enumerate(msg["suggestions"]):
                        if st.button(suggestion, key=f"sug_{msg_idx}_{sug_idx}"):
                            st.session_state["_pending_question"] = suggestion
                            st.rerun()

                if msg.get("query_cost"):
                    _render_query_cost(msg["query_cost"])

                if msg.get("request_id"):
                    with st.expander("Debug info", expanded=False):
                        st.caption(f"Request ID: `{msg['request_id']}`")


# ── Chat input ────────────────────────────────────────────────────────────────
pending = st.session_state.pop("_pending_question", None)

question = st.chat_input(
    "Ask a question about your data…",
    disabled=not connected or not st.session_state.semantic_model_path,
)

if pending:
    question = pending

if question:
    if len(question) > _MAX_QUESTION_LENGTH:
        st.warning(
            f"Question too long ({len(question):,} chars). "
            f"Please keep it under {_MAX_QUESTION_LENGTH:,} characters."
        )
        st.stop()
    st.session_state.last_activity = time.time()

    if _is_guardrail_intent(question):
        st.session_state.conversation.append({"role": "user", "text": question})
        with st.chat_message("user"):
            st.write(question)
        with st.chat_message("assistant", avatar="❄️"):
            st.markdown(_GUARDRAILS_RESPONSE)
        st.session_state.conversation.append({
            "role": "analyst",
            "text": _GUARDRAILS_RESPONSE,
            "sql": "",
            "suggestions": [],
            "error": "",
            "df": None,
            "request_id": "",
        })
        st.stop()

    forecast_mode = _is_forecast_intent(question)
    forecast_periods = _extract_periods(question) if forecast_mode else 30
    api_question = _to_historical_query(question) if forecast_mode else question

    # Append user message
    st.session_state.conversation.append({"role": "user", "text": question})

    with st.chat_message("user"):
        st.write(question)
        if forecast_mode:
            st.caption(f"Fetching historical data to forecast {forecast_periods} periods…")

    with st.chat_message("assistant", avatar="❄️"):
        with st.spinner("Thinking…"):
            try:
                # Build messages but substitute the rephrased question for forecast intent
                conversation_for_api = st.session_state.conversation[:-1] + [
                    {"role": "user", "text": api_question}
                ] if forecast_mode else st.session_state.conversation
                api_messages = build_api_messages(conversation_for_api)
                response = send_message(
                    conn=st.session_state["sf_connection"],
                    messages=api_messages,
                    semantic_model_file=st.session_state.semantic_model_path,
                )

                improved = _generate_improved_suggestions(question)
                merged_suggestions = response.suggestions + [
                    s for s in improved if s not in response.suggestions
                ]
                analyst_msg: dict = {
                    "role": "analyst",
                    "text": response.text,
                    "sql": response.sql,
                    "suggestions": merged_suggestions[:5],
                    "error": response.error,
                    "df": None,
                    "request_id": response.request_id,
                    "query_id": "",
                    "query_cost": None,
                }

                if response.error:
                    st.error(response.error)
                else:
                    if response.text:
                        st.write(response.text)

                    if response.sql:
                        with st.expander("Generated SQL", expanded=False):
                            st.code(response.sql, language="sql")

                        safe_sql, sql_reason = _is_safe_sql(response.sql)
                        if not safe_sql:
                            st.error(f"Query blocked — {sql_reason}")
                            log_query(
                                user=st.session_state["sf_connection"].user,
                                question=question,
                                sql=response.sql,
                                error=f"Blocked: {sql_reason}",
                            )
                        else:
                            with st.spinner("Running query…"):
                                try:
                                    cur = st.session_state["sf_connection"].cursor()
                                    cur.execute(response.sql)
                                    df = cur.fetch_pandas_all()
                                    analyst_msg["df"] = df
                                    analyst_msg["query_id"] = cur.sfqid or ""
                                    log_query(
                                        user=st.session_state["sf_connection"].user,
                                        question=question,
                                        sql=response.sql,
                                        row_count=len(df),
                                    )
                                    render_result(df, row_limit=st.session_state.row_limit)
                                    _warn_pii_columns(df)

                                    if forecast_mode:
                                        date_cols = get_date_columns(df)
                                        num_cols = get_numeric_columns(df)
                                        if date_cols and num_cols:
                                            with st.spinner("Running Snowflake Cortex ML forecast…"):
                                                try:
                                                    forecast_df = run_forecast(
                                                        st.session_state["sf_connection"],
                                                        df,
                                                        date_cols[0],
                                                        num_cols[0],
                                                        forecast_periods,
                                                    )
                                                    render_forecast_chart(df, forecast_df, date_cols[0], num_cols[0])
                                                    with st.expander("Forecast data table", expanded=False):
                                                        st.dataframe(forecast_df, use_container_width=True)
                                                except Exception as e:
                                                    st.error(f"Forecast failed: {e}")
                                        else:
                                            st.warning("Could not detect date and numeric columns for forecasting.")
                                    else:
                                        _render_forecast_ui(st.session_state["sf_connection"], df, key="current")
                                except Exception as e:
                                    st.error(f"Query execution failed: {e}")

                    with st.expander("Debug info", expanded=False):
                        st.caption(f"Request ID: `{response.request_id}`")

            except CortexAnalystError as e:
                err_text = _sanitize_error(f"Cortex Analyst error ({e.status_code}): {e.detail}")
                st.error(err_text)
                log_query(
                    user=st.session_state["sf_connection"].user,
                    question=question,
                    error=err_text,
                )
                analyst_msg = {
                    "role": "analyst",
                    "text": "",
                    "sql": "",
                    "suggestions": [],
                    "error": err_text,
                    "df": None,
                    "request_id": "",
                }
            except Exception as e:
                err_text = _sanitize_error(f"Unexpected error: {e}")
                st.error(err_text)
                log_query(
                    user=st.session_state["sf_connection"].user,
                    question=question,
                    error=err_text,
                )
                analyst_msg = {
                    "role": "analyst",
                    "text": "",
                    "sql": "",
                    "suggestions": [],
                    "error": err_text,
                    "df": None,
                    "request_id": "",
                }

        if analyst_msg.get("query_id") and not analyst_msg.get("error"):
            cost = get_query_cost(
                st.session_state["sf_connection"], analyst_msg["query_id"]
            )
            if cost:
                analyst_msg["query_cost"] = cost
                st.session_state.session_queries += 1
                st.session_state.session_credits += cost["credits"]
                st.session_state.session_cost_usd += cost["cost_usd"]
                _render_query_cost(cost)

        if analyst_msg.get("suggestions") and not analyst_msg.get("error"):
            st.markdown("**Suggested follow-up questions:**")
            for sug_idx, suggestion in enumerate(analyst_msg["suggestions"]):
                if st.button(suggestion, key=f"new_sug_{sug_idx}"):
                    st.session_state["_pending_question"] = suggestion
                    st.rerun()

        st.session_state.conversation.append(analyst_msg)
