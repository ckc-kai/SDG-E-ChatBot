"""Configuration-driven factories shared by ingest, query, and evaluation.

The checked-in YAML contains non-secret behavior and deployment settings.
Passwords and API credentials are resolved from environment variables so the
same code works with a local ``.env`` file, RDS, or Aurora PostgreSQL.
"""

from __future__ import annotations

import functools
import logging
import math
import os
from pathlib import Path
from typing import Any

import anthropic
import psycopg2
import yaml
from dotenv import load_dotenv
from pgvector.psycopg2 import register_vector
from sentence_transformers import SentenceTransformer
from sentence_transformers.cross_encoder import CrossEncoder

load_dotenv()

logger = logging.getLogger(__name__)

DEFAULT_CONFIG_PATH = "config/config.yaml"
CONFIG_PATH_ENV = "SDGE_CONFIG_PATH"
SUPPORTED_DATABASE_DEPLOYMENTS = {"local", "rds", "aurora"}
SUPPORTED_MODEL_PROVIDERS = {"sentence_transformers"}
SUPPORTED_OUTPUT_MODES = {"grouped", "legacy_flat"}
SUPPORTED_REWRITE_MODES = {"off", "auto", "always"}
SUPPORTED_EMBEDDING_MODES = {"raw", "contextual", "hybrid"}
SUPPORTED_HYBRID_POOL_MODES = {"rrf", "union"}


def _mapping(parent: dict[str, Any], key: str) -> dict[str, Any]:
    value = parent.get(key, {})
    if not isinstance(value, dict):
        raise ValueError(f"{key} must be a YAML mapping")
    return value


def _integer(value: Any, field: str, *, minimum: int = 1) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{field} must be an integer, not a boolean")
    if isinstance(value, float) and (
        not math.isfinite(value) or not value.is_integer()
    ):
        raise ValueError(f"{field} must be a whole integer")
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be an integer") from exc
    if parsed < minimum:
        raise ValueError(f"{field} must be at least {minimum}")
    return parsed


def _number(value: Any, field: str, *, minimum: float = 0.0) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{field} must be numeric, not a boolean")
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be numeric") from exc
    if not math.isfinite(parsed) or parsed < minimum:
        raise ValueError(f"{field} must be finite and at least {minimum}")
    return parsed


def _validate_config(config: dict[str, Any]) -> None:
    database = _mapping(config, "database")
    if database.get("provider", "postgresql") != "postgresql":
        raise ValueError("Only the 'postgresql' database provider is implemented")
    deployment = database.get("deployment", "local")
    if deployment not in SUPPORTED_DATABASE_DEPLOYMENTS:
        raise ValueError(
            f"Unknown database deployment {deployment!r}; expected one of "
            f"{sorted(SUPPORTED_DATABASE_DEPLOYMENTS)}"
        )
    for field in ("host", "user"):
        if not isinstance(database.get(field), str) or not database[field].strip():
            raise ValueError(f"database.{field} must be a non-empty string")
    port = _integer(database.get("port", 5432), "database.port")
    if port > 65535:
        raise ValueError("database.port must be at most 65535")
    _integer(
        database.get("connect_timeout_seconds", 10),
        "database.connect_timeout_seconds",
    )
    models = _mapping(config, "models")
    for role in ("embedding", "reranker"):
        model = _mapping(models, role)
        provider = model.get("provider", "sentence_transformers")
        if provider not in SUPPORTED_MODEL_PROVIDERS:
            raise ValueError(
                f"Model provider {provider!r} for {role} is not implemented; "
                f"expected one of {sorted(SUPPORTED_MODEL_PROVIDERS)}"
            )
        if not isinstance(model.get("name"), str) or not model["name"].strip():
            raise ValueError(f"models.{role}.name must be a non-empty string")
    embedding = _mapping(models, "embedding")
    _integer(embedding.get("dimensions"), "models.embedding.dimensions")
    max_tokens = _integer(
        embedding.get("max_tokens_per_chunk"),
        "models.embedding.max_tokens_per_chunk",
    )
    _integer(
        embedding.get("minimum_chunk_chars"),
        "models.embedding.minimum_chunk_chars",
    )
    overlap = _integer(
        embedding.get("token_overlap", 0),
        "models.embedding.token_overlap",
        minimum=0,
    )
    if overlap >= max_tokens:
        raise ValueError(
            "models.embedding.token_overlap must be smaller than max_tokens_per_chunk"
        )
    retrieval = _mapping(config, "retrieval")
    output_mode = retrieval.get("output_mode", "legacy_flat")
    if output_mode not in SUPPORTED_OUTPUT_MODES:
        raise ValueError(
            f"Unknown retrieval.output_mode {output_mode!r}; expected one of "
            f"{sorted(SUPPORTED_OUTPUT_MODES)}"
        )
    embedding_mode = retrieval.get("embedding_mode", "hybrid")
    if embedding_mode not in SUPPORTED_EMBEDDING_MODES:
        raise ValueError(
            f"Unknown retrieval.embedding_mode {embedding_mode!r}; expected one of "
            f"{sorted(SUPPORTED_EMBEDDING_MODES)}"
        )
    pool_mode = retrieval.get("hybrid_pool_mode", "union")
    if pool_mode not in SUPPORTED_HYBRID_POOL_MODES:
        raise ValueError(
            f"Unknown retrieval.hybrid_pool_mode {pool_mode!r}; expected one of "
            f"{sorted(SUPPORTED_HYBRID_POOL_MODES)}"
        )
    for key in ("retrieval_top_k", "rerank_top_k", "rrf_k"):
        _integer(retrieval.get(key), f"retrieval.{key}")
    for key in (
        "caption_rerank_weight",
        "lexical_caption_weight",
        "lexical_hint_weight",
        "lane_confidence_floor",
    ):
        _number(retrieval.get(key, 0.0), f"retrieval.{key}")
    query_rewrite = _mapping(config, "query_rewrite")
    rewrite_mode = query_rewrite.get("mode", "off")
    if rewrite_mode not in SUPPORTED_REWRITE_MODES:
        raise ValueError(
            f"Unknown query_rewrite.mode {rewrite_mode!r}; expected one of "
            f"{sorted(SUPPORTED_REWRITE_MODES)}"
        )
    _integer(
        query_rewrite.get("max_subquestions", 2),
        "query_rewrite.max_subquestions",
    )
    extraction = _mapping(config, "extraction")
    _mapping(extraction, "structured")
    object_storage = _mapping(config, "object_storage")
    storage_provider = object_storage.get("provider", "filesystem")
    if storage_provider not in {"filesystem", "s3"}:
        raise ValueError("object_storage.provider must be either 'filesystem' or 's3'")


@functools.lru_cache(maxsize=1)
def load_config() -> dict[str, Any]:
    config_path = Path(os.environ.get(CONFIG_PATH_ENV, DEFAULT_CONFIG_PATH))
    try:
        with config_path.open() as file:
            raw = yaml.safe_load(file) or {}
    except (OSError, yaml.YAMLError) as exc:
        logger.error("Cannot load configuration file: %s", config_path)
        raise FileNotFoundError(f"configuration file not found: {config_path}") from exc

    if not isinstance(raw, dict):
        raise ValueError("Configuration root must be a YAML mapping")
    if "local" in raw:
        raise ValueError(
            "Legacy config.local format is no longer accepted because it drops "
            "retrieval and extraction controls. Replace config/config.yaml with "
            "config/config.example.yaml and apply the local database settings."
        )
    config = raw
    _validate_config(config)
    return config


def database_config() -> dict[str, Any]:
    config = load_config()["database"]
    password_env = config.get("password_env")
    password = os.environ.get(password_env) if password_env else config.get("password")
    if password_env and password is None:
        raise RuntimeError(
            f"Database password environment variable {password_env!r} is not set"
        )
    connection = {
        "host": config["host"],
        "port": int(config.get("port", 5432)),
        "dbname": config.get("name", config.get("dbname", "postgres")),
        "user": config["user"],
        "password": password,
        "sslmode": config.get("sslmode", "prefer"),
        "connect_timeout": int(config.get("connect_timeout_seconds", 10)),
    }
    return {key: value for key, value in connection.items() if value is not None}


def connect_db() -> psycopg2.extensions.connection:
    conn = psycopg2.connect(**database_config())
    register_vector(conn)
    return conn


def embedding_config() -> dict[str, Any]:
    return load_config()["models"]["embedding"]


def reranker_config() -> dict[str, Any]:
    return load_config()["models"]["reranker"]


@functools.lru_cache(maxsize=1)
def get_embedding_model() -> SentenceTransformer:
    config = embedding_config()
    if config.get("provider", "sentence_transformers") != "sentence_transformers":
        raise ValueError("Configured embedding provider is not implemented")
    return SentenceTransformer(config["name"])


@functools.lru_cache(maxsize=1)
def get_reranker_model() -> CrossEncoder:
    config = reranker_config()
    if config.get("provider", "sentence_transformers") != "sentence_transformers":
        raise ValueError("Configured reranker provider is not implemented")
    return CrossEncoder(config["name"])


@functools.lru_cache(maxsize=1)
def get_anthropic_client() -> anthropic.Anthropic:
    return anthropic.Anthropic()
