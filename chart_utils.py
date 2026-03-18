from __future__ import annotations

from enum import Enum

import pandas as pd
import plotly.express as px
import streamlit as st

# Whole-word suffixes/prefixes to identify time-dimension columns.
# Checked against the column name split by underscores, not substring, to avoid
# false positives like "category" matching "at" or "data" matching "at".
_TIME_KEYWORDS = frozenset(
    ["date", "time", "year", "month", "week", "day", "ts", "created_at", "updated_at"]
)


class ChartType(str, Enum):
    TABLE = "table"
    LINE = "line"
    BAR = "bar"


def _is_time_column(col: str, series: pd.Series) -> bool:
    # Split on underscores so "category" doesn't match "date", etc.
    col_parts = set(col.lower().split("_"))
    if col_parts & _TIME_KEYWORDS:
        return True
    return pd.api.types.is_datetime64_any_dtype(series)


def detect_chart_type(df: pd.DataFrame) -> tuple[ChartType, str | None, str | None]:
    """
    Returns (chart_type, x_col, y_col).
    Uses heuristics: temporal x + numeric y → LINE, categorical x + numeric y → BAR.
    """
    if df.empty or len(df.columns) < 2:
        return ChartType.TABLE, None, None

    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    non_numeric_cols = [c for c in df.columns if c not in numeric_cols]

    if not numeric_cols:
        return ChartType.TABLE, None, None

    y_col = numeric_cols[0]

    for x_col in non_numeric_cols:
        if _is_time_column(x_col, df[x_col]):
            return ChartType.LINE, x_col, y_col

    # Check numeric columns that look like time dimensions
    for x_col in numeric_cols:
        if _is_time_column(x_col, df[x_col]) and x_col != y_col:
            return ChartType.LINE, x_col, y_col

    if non_numeric_cols:
        x_col = non_numeric_cols[0]
        # Only use BAR for reasonably low cardinality
        if df[x_col].nunique() <= 50:
            return ChartType.BAR, x_col, y_col

    return ChartType.TABLE, None, None


def render_result(df: pd.DataFrame, row_limit: int = 10000) -> None:
    """Render a DataFrame as a table with optional charts."""
    if df is None or df.empty:
        st.info("Query returned no results.")
        return

    was_truncated = len(df) > row_limit
    if was_truncated:
        df = df.head(row_limit)
        st.warning(f"Results truncated to {row_limit:,} rows.")

    chart_type, x_col, y_col = detect_chart_type(df)

    tab_labels = ["Table"]
    if chart_type == ChartType.LINE:
        tab_labels.append("Line Chart")
    elif chart_type == ChartType.BAR:
        tab_labels.append("Bar Chart")

    tabs = st.tabs(tab_labels)

    with tabs[0]:
        st.dataframe(df, use_container_width=True)
        st.caption(f"{len(df):,} rows × {len(df.columns)} columns")

    if len(tabs) > 1:
        with tabs[1]:
            if chart_type == ChartType.LINE:
                fig = px.line(df, x=x_col, y=y_col)
            else:
                fig = px.bar(df, x=x_col, y=y_col)
            st.plotly_chart(fig, use_container_width=True)
