from __future__ import annotations

from dataclasses import dataclass, field

import requests
from snowflake.connector import SnowflakeConnection

from snowflake_connection import get_auth_token

ANALYST_ENDPOINT = "/api/v2/cortex/analyst/message"
MAX_HISTORY_TURNS = 20  # max user+analyst pairs sent to the API per request


class CortexAnalystError(Exception):
    def __init__(self, status_code: int, detail: str):
        self.status_code = status_code
        self.detail = detail
        super().__init__(f"HTTP {status_code}: {detail}")


@dataclass
class AnalystResponse:
    request_id: str
    text: str = ""
    sql: str = ""
    suggestions: list[str] = field(default_factory=list)
    error: str = ""


def send_message(
    conn: SnowflakeConnection,
    messages: list[dict],
    semantic_model_file: str,
    timeout: int = 60,
) -> AnalystResponse:
    host = f"https://{conn.host}"
    token = get_auth_token(conn)

    payload = {
        "messages": messages,
        "semantic_model_file": semantic_model_file,
    }

    response = requests.post(
        f"{host}{ANALYST_ENDPOINT}",
        json=payload,
        headers={
            "Authorization": f'Snowflake Token="{token}"',
            "Content-Type": "application/json",
        },
        timeout=timeout,
    )

    if response.status_code >= 400:
        try:
            detail = response.json().get("message", response.text)
        except Exception:
            detail = response.text
        raise CortexAnalystError(status_code=response.status_code, detail=detail)

    return _parse_response(response.json())


def _parse_response(raw: dict) -> AnalystResponse:
    request_id = raw.get("request_id", "")
    message = raw.get("message", {})
    content_items = message.get("content", [])

    result = AnalystResponse(request_id=request_id)

    for item in content_items:
        item_type = item.get("type", "")
        if item_type == "text":
            result.text = item.get("text", "")
        elif item_type == "sql":
            result.sql = item.get("statement", "")
        elif item_type == "suggestions":
            result.suggestions = item.get("suggestions", [])
        elif item_type == "error":
            result.error = item.get("message", "Unknown error from Cortex Analyst")

    return result


def build_api_messages(conversation: list[dict]) -> list[dict]:
    """
    Convert the display conversation to the Cortex Analyst API message format.

    Processes the conversation as strict user+analyst pairs. Pairs where the
    analyst response has no text/SQL (e.g. errors) are skipped entirely.
    The final unpaired user message (the current question) is always appended last.
    History is capped at MAX_HISTORY_TURNS pairs to bound API payload size.
    """
    # Apply sliding window: keep at most MAX_HISTORY_TURNS pairs + current user msg
    # Each pair = 2 items (user + analyst); +1 for the current unpaired user msg
    max_items = MAX_HISTORY_TURNS * 2 + 1
    if len(conversation) > max_items:
        trimmed = conversation[-max_items:]
        # Ensure the window starts with a user message
        if trimmed and trimmed[0]["role"] != "user":
            trimmed = trimmed[1:]
        conversation = trimmed

    api_messages = []
    i = 0

    # Process complete user+analyst pairs
    while i + 1 < len(conversation):
        user_msg = conversation[i]
        analyst_msg = conversation[i + 1]
        if user_msg["role"] == "user" and analyst_msg["role"] == "analyst":
            analyst_content = []
            if analyst_msg.get("text"):
                analyst_content.append({"type": "text", "text": analyst_msg["text"]})
            if analyst_msg.get("sql"):
                analyst_content.append({"type": "sql", "statement": analyst_msg["sql"]})
            if analyst_content:
                api_messages.append({
                    "role": "user",
                    "content": [{"type": "text", "text": user_msg["text"]}],
                })
                api_messages.append({"role": "analyst", "content": analyst_content})
        i += 2

    # Append the current (unpaired) user message
    if i < len(conversation) and conversation[i]["role"] == "user":
        api_messages.append({
            "role": "user",
            "content": [{"type": "text", "text": conversation[i]["text"]}],
        })

    return api_messages
