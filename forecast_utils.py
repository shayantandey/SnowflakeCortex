from __future__ import annotations

import re
import uuid

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from snowflake.connector import SnowflakeConnection
from snowflake.connector.pandas_tools import write_pandas


_DATE_PATTERN = re.compile(r"\d{4}[-/]\d{2}[-/]\d{2}|\d{2}[-/]\d{2}[-/]\d{4}")


def get_date_columns(df: pd.DataFrame) -> list[str]:
    result = []
    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            result.append(col)
        elif pd.api.types.is_object_dtype(df[col]) or pd.api.types.is_string_dtype(df[col]):
            sample = df[col].dropna().head(5).astype(str)
            # Require values to look like actual dates (contain separators + year)
            if sample.str.match(_DATE_PATTERN).all():
                try:
                    pd.to_datetime(sample, format="mixed")
                    result.append(col)
                except Exception:
                    pass
    return result


def get_numeric_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]


MAX_FORECAST_PERIODS = 365


def run_forecast(
    conn: SnowflakeConnection,
    df: pd.DataFrame,
    timestamp_col: str,
    target_col: str,
    periods: int,
) -> pd.DataFrame:
    if periods > MAX_FORECAST_PERIODS:
        raise ValueError(
            f"Forecast periods ({periods}) exceeds the maximum allowed ({MAX_FORECAST_PERIODS})."
        )
    uid = uuid.uuid4().hex[:8].upper()
    temp_table = f"TEMP_FC_INPUT_{uid}"
    model_name = f"TEMP_FC_MODEL_{uid}"

    df_upload = df[[timestamp_col, target_col]].copy()
    df_upload.columns = ["TS", "TARGET_VAL"]
    df_upload["TS"] = pd.to_datetime(df_upload["TS"], format="mixed", dayfirst=False)
    df_upload["TARGET_VAL"] = pd.to_numeric(df_upload["TARGET_VAL"], errors="coerce")
    df_upload = df_upload.dropna()

    # Aggregate duplicate timestamps (sum) and sort — required by Snowflake ML FORECAST
    df_upload = (
        df_upload.groupby("TS", as_index=False)["TARGET_VAL"]
        .sum()
        .sort_values("TS")
        .reset_index(drop=True)
    )

    if len(df_upload) < 10:
        raise ValueError(
            f"Not enough data points for forecasting ({len(df_upload)} rows). "
            "Need at least 10 rows. Try a query that returns more historical data."
        )

    cur = conn.cursor()
    db = schema = None
    try:
        cur.execute("SELECT CURRENT_DATABASE(), CURRENT_SCHEMA()")
        row = cur.fetchone()
        db, schema = row[0], row[1]

        # Upload raw data (write_pandas may create TS as VARCHAR)
        raw_table = f"{temp_table}_RAW"
        write_pandas(
            conn, df_upload, raw_table,
            database=db, schema=schema,
            auto_create_table=True, overwrite=True,
            quote_identifiers=False,
        )

        # Create a properly typed table with TS as TIMESTAMP_NTZ
        cur.execute(f"""
            CREATE OR REPLACE TEMPORARY TABLE {db}.{schema}.{temp_table} AS
            SELECT TS::TIMESTAMP_NTZ AS TS, TARGET_VAL::FLOAT AS TARGET_VAL
            FROM {db}.{schema}.{raw_table}
        """)
        cur.execute(f"DROP TABLE IF EXISTS {db}.{schema}.{raw_table}")

        cur.execute(f"""
            CREATE OR REPLACE SNOWFLAKE.ML.FORECAST {db}.{schema}.{model_name}(
                INPUT_DATA => SYSTEM$REFERENCE('TABLE', '{db}.{schema}.{temp_table}'),
                TIMESTAMP_COLNAME => 'TS',
                TARGET_COLNAME => 'TARGET_VAL'
            )
        """)

        cur.execute(f"CALL {db}.{schema}.{model_name}!FORECAST(FORECASTING_PERIODS => {periods})")
        rows = cur.fetchall()
        cols = [d[0] for d in cur.description]
        return pd.DataFrame(rows, columns=cols)

    finally:
        for obj in [f"TABLE IF EXISTS {db}.{schema}.{temp_table}",
                    f"TABLE IF EXISTS {db}.{schema}.{temp_table}_RAW"]:
            try:
                cur.execute(f"DROP {obj}")
            except Exception:
                pass
        try:
            cur.execute(f"DROP SNOWFLAKE.ML.FORECAST IF EXISTS {db}.{schema}.{model_name}")
        except Exception:
            pass
        cur.close()


def render_forecast_chart(
    historical_df: pd.DataFrame,
    forecast_df: pd.DataFrame,
    timestamp_col: str,
    target_col: str,
) -> None:
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=pd.to_datetime(historical_df[timestamp_col]),
        y=historical_df[target_col],
        name="Historical",
        line=dict(color="#29b5e8", width=2),
    ))

    ts_col = next((c for c in forecast_df.columns if "TS" in c.upper()), forecast_df.columns[0])
    fc_col = next((c for c in forecast_df.columns if "FORECAST" in c.upper()), forecast_df.columns[1])
    lower_col = next((c for c in forecast_df.columns if "LOWER" in c.upper()), None)
    upper_col = next((c for c in forecast_df.columns if "UPPER" in c.upper()), None)

    if lower_col and upper_col:
        x_band = (
            list(pd.to_datetime(forecast_df[ts_col]))
            + list(pd.to_datetime(forecast_df[ts_col]))[::-1]
        )
        y_band = list(forecast_df[upper_col]) + list(forecast_df[lower_col])[::-1]
        fig.add_trace(go.Scatter(
            x=x_band, y=y_band,
            fill="toself",
            fillcolor="rgba(255,127,14,0.15)",
            line=dict(color="rgba(0,0,0,0)"),
            name="95% Confidence Interval",
            hoverinfo="skip",
        ))

    fig.add_trace(go.Scatter(
        x=pd.to_datetime(forecast_df[ts_col]),
        y=forecast_df[fc_col],
        name="Forecast",
        line=dict(color="#ff7f0e", width=2, dash="dash"),
    ))

    fig.update_layout(
        title="Historical vs Forecast",
        xaxis_title="Date",
        yaxis_title=target_col,
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    st.plotly_chart(fig, use_container_width=True)
