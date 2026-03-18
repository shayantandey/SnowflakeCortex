from __future__ import annotations

import re
import tempfile
import os
import shutil

import yaml
from snowflake.connector import SnowflakeConnection


STAGE_PATH_PATTERN = re.compile(r"^@[\w$]+\.[\w$]+\.[\w$]+/\S+\.yaml$", re.IGNORECASE)


def validate_stage_path(path: str) -> tuple[bool, str]:
    """Validate the stage path format: @DB.SCHEMA.STAGE/file.yaml"""
    path = path.strip()
    if not path:
        return False, "Stage path is empty."
    if not STAGE_PATH_PATTERN.match(path):
        return False, (
            "Stage path must match @DATABASE.SCHEMA.STAGE/filename.yaml  "
            f"Got: {path!r}"
        )
    return True, ""


def validate_yaml(content: str) -> tuple[bool, str]:
    """Parse YAML and check for required Cortex Analyst semantic model keys."""
    try:
        model = yaml.safe_load(content)
    except yaml.YAMLError as e:
        return False, f"YAML parse error: {e}"

    if not isinstance(model, dict):
        return False, "Semantic model must be a YAML mapping (dict)."

    if "name" not in model:
        return False, "Missing required field: 'name'"

    tables = model.get("tables")
    if not tables or not isinstance(tables, list):
        return False, "Missing or empty 'tables' list."

    for i, table in enumerate(tables):
        base = table.get("base_table", {})
        for key in ("database", "schema", "table"):
            if not base.get(key):
                return False, f"tables[{i}].base_table.{key} is required."
        has_columns = any(
            table.get(k) for k in ("dimensions", "measures", "time_dimensions")
        )
        if not has_columns:
            return False, (
                f"tables[{i}] must have at least one of: "
                "dimensions, measures, time_dimensions."
            )

    return True, ""


def upload_to_stage(
    conn: SnowflakeConnection,
    yaml_content: bytes,
    stage_path: str,
) -> str:
    """
    Upload a local YAML file to a Snowflake stage.
    stage_path format: @DB.SCHEMA.STAGE/filename.yaml
    Returns the stage path on success.
    """
    # Extract the stage directory and the intended filename from the stage path.
    # PUT uploads the source filename to the stage, so we must write the temp file
    # with the exact target filename, otherwise the staged object won't match.
    last_slash = stage_path.rfind("/")
    stage_dir = stage_path[: last_slash] if last_slash != -1 else stage_path
    filename = stage_path[last_slash + 1 :] if last_slash != -1 else "model.yaml"

    tmp_dir = tempfile.mkdtemp()
    tmp_path = os.path.join(tmp_dir, filename)
    try:
        with open(tmp_path, "wb") as f:
            f.write(yaml_content)
        put_sql = (
            f"PUT 'file://{tmp_path}' '{stage_dir}' "
            f"AUTO_COMPRESS=FALSE OVERWRITE=TRUE"
        )
        conn.cursor().execute(put_sql)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    return stage_path


def list_stage_files(conn: SnowflakeConnection, stage: str) -> list[str]:
    """List .yaml files on a stage, e.g. @DB.SCHEMA.MY_STAGE"""
    if not re.match(r'^@[\w$]+\.[\w$]+\.[\w$]+$', stage.strip(), re.IGNORECASE):
        raise ValueError(f"Invalid stage identifier: {stage!r}")
    cur = conn.cursor()
    cur.execute(f"LIST {stage}")
    rows = cur.fetchall()
    return [row[0] for row in rows if row[0].endswith(".yaml")]
