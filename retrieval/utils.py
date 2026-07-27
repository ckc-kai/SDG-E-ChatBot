"""Configuration-driven factories shared by ingest, query, and evaluation.

The checked-in YAML contains non-secret behavior and deployment settings.
Passwords and API credentials are resolved from environment variables so the
same code works with a local ``.env`` file, RDS, or Aurora PostgreSQL.
"""

from __future__ import annotations

import functools
import logging
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


def _legacy_config(local: dict[str, Any]) -> dict[str, Any]:
    """Normalize the pre-unification ``local:`` config for a smooth checkout.

    ``config/config.yaml`` is intentionally ignored because it may contain a
    password. Existing developers can run the refactor before replacing their
    local file with the new example; README documents the preferred format.
    """
    db = local.get("db", {})
    embedding = local.get("embedding", {})
    rerank = local.get("rerank", {})
    structured = local.get("structured", {})
    return {
        "database": {
            "provider": "postgresql",
            "deployment": "local",
            "host": db.get("host", "localhost"),
            "port": db.get("port", 5432),
            "name": db.get("dbname", "postgres"),
            "user": db.get("user", "postgres"),
            "password": db.get("password"),
            "sslmode": db.get("sslmode", "prefer"),
        },
        "models": {
            "embedding": {
                "provider": "sentence_transformers",
                "name": embedding.get("model", "BAAI/bge-base-en-v1.5"),
                "dimensions": embedding.get("dimensions", 768),
                "max_tokens_per_chunk": embedding.get("max_tokens_per_chunk", 400),
                "minimum_chunk_chars": embedding.get("minimum_chunk_chars", 20),
                "token_overlap": embedding.get("token_overlap", 50),
            },
            "reranker": {
                "provider": "sentence_transformers",
                "name": rerank.get("model", "BAAI/bge-reranker-base"),
            },
        },
        "query_rewrite": local.get("query_rewrite", {}),
        "retrieval": local.get("retrieval", {}),
        "extraction": {
            "structured": {
                **structured,
                "figure_description": {
                    "generate": bool(structured.get("picture_description", True)),
                    "candidate_retrieval": True,
                    "reranking": False,
                    "answer_context": False,
                },
            }
        },
        "object_storage": {
            "provider": "filesystem",
            "filesystem": {
                "root": structured.get(
                    "figure_images_dir", "resources/wmp/figures"
                )
            },
        },
    }


def _validate_config(config: dict[str, Any]) -> None:
    database = config.get("database", {})
    if database.get("provider", "postgresql") != "postgresql":
        raise ValueError("Only the 'postgresql' database provider is implemented")
    deployment = database.get("deployment", "local")
    if deployment not in SUPPORTED_DATABASE_DEPLOYMENTS:
        raise ValueError(
            f"Unknown database deployment {deployment!r}; expected one of "
            f"{sorted(SUPPORTED_DATABASE_DEPLOYMENTS)}"
        )
    for role in ("embedding", "reranker"):
        provider = config.get("models", {}).get(role, {}).get(
            "provider", "sentence_transformers"
        )
        if provider not in SUPPORTED_MODEL_PROVIDERS:
            raise ValueError(
                f"Model provider {provider!r} for {role} is not implemented; "
                f"expected one of {sorted(SUPPORTED_MODEL_PROVIDERS)}"
            )


@functools.lru_cache(maxsize=1)
def load_config() -> dict[str, Any]:
    config_path = Path(os.environ.get(CONFIG_PATH_ENV, DEFAULT_CONFIG_PATH))
    try:
        with config_path.open() as file:
            raw = yaml.safe_load(file) or {}
    except (OSError, yaml.YAMLError) as exc:
        logger.error("Cannot load configuration file: %s", config_path)
        raise FileNotFoundError(f"configuration file not found: {config_path}") from exc

    config = _legacy_config(raw["local"]) if "local" in raw else raw
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
