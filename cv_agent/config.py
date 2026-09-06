"""
Configuration module using Pydantic Settings.
All values can be overridden via environment variables or a .env file.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Project root = two levels up from this file (cv_agent/config.py → cv_agent/ → project root)
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_ENV_FILE = _PROJECT_ROOT / ".env"


class Settings(BaseSettings):
    """Central configuration for the CV QA Agent."""

    model_config = SettingsConfigDict(
        env_file=str(_ENV_FILE),
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── LLM Provider ──────────────────────────────────────────────────────────
    llm_provider: Literal["gemini", "openai", "ollama"] = Field(
        default="gemini",
        description="Which LLM backend to use.",
    )
    google_api_key: str = Field(
        default="",
        description="Google Gemini API key (required when llm_provider=gemini).",
    )
    openai_api_key: str = Field(
        default="",
        description="OpenAI API key (required when llm_provider=openai).",
    )
    ollama_base_url: str = Field(
        default="http://localhost:11434",
        description="Ollama server base URL.",
    )

    # ── Model names ───────────────────────────────────────────────────────────
    llm_model: str = Field(
        default="gemini-3.5-flash-lite",
        description="LLM model name.",
    )
    embedding_model: str = Field(
        default="gemini-embedding-001",
        description="Embedding model name.",
    )

    # ── Chunking ──────────────────────────────────────────────────────────────
    chunk_size: int = Field(default=800, ge=100, le=4000)
    chunk_overlap: int = Field(default=150, ge=0, le=1000)

    # ── Retrieval ─────────────────────────────────────────────────────────────
    retrieval_top_k: int = Field(default=6, ge=1, le=20)
    retrieval_score_threshold: float = Field(
        default=1.0,
        ge=0.0,
        description=(
            "Maximum FAISS L2 distance score to consider a chunk relevant. "
            "Lower scores mean closer match. Calibrate this against your actual CV "
            "by inspecting real retrieval scores for known-answer and unrelated queries. "
            "Set via RETRIEVAL_SCORE_THRESHOLD in .env."
        ),
    )

    # ── Paths ─────────────────────────────────────────────────────────────────
    cv_file_path: Path = Field(
        default=Path("data/cv.pdf"),
        description="Path to the CV file (PDF or DOCX).",
    )
    faiss_index_path: Path = Field(
        default=Path("data/faiss_index"),
        description="Directory where the FAISS index is persisted.",
    )
    output_dir: Path = Field(
        default=Path("output"),
        description="Directory where session output files are saved.",
    )

    # ── Logging ───────────────────────────────────────────────────────────────
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(
        default="INFO",
        description="Root log level.",
    )
    log_file: Path | None = Field(
        default=Path("output/agent.log"),
        description="Optional log file path. Set to empty string to disable.",
    )

    # ── Agent behaviour ───────────────────────────────────────────────────────
    agent_max_iterations: int = Field(
        default=5,
        description="Maximum tool-call iterations per turn.",
    )
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)

    @field_validator("cv_file_path", "faiss_index_path", "output_dir", mode="before")
    @classmethod
    def _coerce_path(cls, v: object) -> Path:
        return Path(v) if not isinstance(v, Path) else v

    @field_validator("log_file", mode="before")
    @classmethod
    def _coerce_log_file(cls, v: object) -> Path | None:
        if v == "" or v is None:
            return None
        return Path(v)


# Module-level singleton — import this everywhere
settings = Settings()
