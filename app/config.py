"""Central configuration loaded from environment / .env file.

Every other module imports `settings` from here instead of reading
os.environ directly. One source of truth for all configuration.
"""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # --- Google Gemini ---
    google_api_key: str  # required: app fails fast if missing

    # --- Models ---
    llm_model: str = "gemini-2.5-flash"
    embedding_model: str = "models/gemini-embedding-001"
    embedding_dim: int = 3072  # gemini-embedding-001 -> 3072 dims

    # --- Qdrant ---
    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = "documents"
    # Empty locally; required by Qdrant Cloud (managed) in production.
    qdrant_api_key: str = ""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",  # ignore unrelated env vars instead of erroring
    )


# Single shared instance used across the whole app.
settings = Settings()
