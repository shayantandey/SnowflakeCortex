from __future__ import annotations

from snowflake.connector import SnowflakeConnection

# Snowflake warehouse credits consumed per hour by size
_CREDITS_PER_HOUR: dict[str, float] = {
    "X-SMALL":  1.0,
    "SMALL":    2.0,
    "MEDIUM":   4.0,
    "LARGE":    8.0,
    "X-LARGE":  16.0,
    "2X-LARGE": 32.0,
    "3X-LARGE": 64.0,
    "4X-LARGE": 128.0,
}

CREDIT_PRICE_USD: float = 2.00  # Standard Snowflake on-demand price per credit


def get_query_cost(conn: SnowflakeConnection, query_id: str) -> dict | None:
    """
    Return execution stats and estimated credit cost for a completed query.
    Uses INFORMATION_SCHEMA.QUERY_HISTORY_BY_SESSION which is near-real-time
    for the current session.
    """
    if not query_id:
        return None
    try:
        cur = conn.cursor()
        cur.execute(
            "SELECT TOTAL_ELAPSED_TIME, BYTES_SCANNED, ROWS_PRODUCED, WAREHOUSE_SIZE "
            "FROM TABLE(INFORMATION_SCHEMA.QUERY_HISTORY_BY_SESSION(RESULT_LIMIT => 100)) "
            "WHERE QUERY_ID = %s",
            (query_id,),
        )
        row = cur.fetchone()
        cur.close()
        if not row:
            return None
        elapsed_ms, bytes_scanned, rows_produced, warehouse_size = row
        credits_per_hour = _CREDITS_PER_HOUR.get((warehouse_size or "").upper(), 1.0)
        credits_used = (int(elapsed_ms or 0) / 3_600_000) * credits_per_hour
        return {
            "elapsed_ms":    int(elapsed_ms or 0),
            "bytes_scanned": int(bytes_scanned or 0),
            "rows_produced": int(rows_produced or 0),
            "warehouse_size": warehouse_size or "Unknown",
            "credits":  credits_used,
            "cost_usd": credits_used * CREDIT_PRICE_USD,
        }
    except Exception:
        return None


def format_bytes(n: int) -> str:
    """Human-readable byte size."""
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024:
            return f"{n:.1f} {unit}"
        n //= 1024
    return f"{n:.1f} PB"
