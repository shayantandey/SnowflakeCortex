from __future__ import annotations

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # Snowflake connection
    SNOWFLAKE_ACCOUNT: str = ""
    SNOWFLAKE_USER: str = ""
    SNOWFLAKE_PASSWORD: SecretStr | None = None
    SNOWFLAKE_PRIVATE_KEY_PATH: str = ""
    SNOWFLAKE_ROLE: str = "SYSADMIN"
    SNOWFLAKE_WAREHOUSE: str = ""
    SNOWFLAKE_DATABASE: str = ""
    SNOWFLAKE_SCHEMA: str = "PUBLIC"

    # Cortex Analyst
    CORTEX_SEMANTIC_MODEL_STAGE_PATH: str = ""
    CORTEX_RESULT_ROW_LIMIT: int = 10000

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()
