from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone

_LOG_FILE = os.path.join(os.path.dirname(__file__), "audit.log")

_handler = logging.FileHandler(_LOG_FILE, encoding="utf-8")
_handler.setFormatter(logging.Formatter("%(message)s"))

_logger = logging.getLogger("audit")
_logger.setLevel(logging.INFO)
if not _logger.handlers:
    _logger.addHandler(_handler)


def log_query(
    user: str,
    question: str,
    sql: str = "",
    row_count: int | None = None,
    error: str = "",
) -> None:
    """Append a structured JSON record to audit.log."""
    record = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "user": user,
        "question": question[:500],
        "sql": sql[:2000] if sql else "",
        "row_count": row_count,
        "error": error[:500] if error else "",
    }
    _logger.info(json.dumps(record))
