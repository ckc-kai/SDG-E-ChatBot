import functools
import logging

import anthropic
import psycopg2
import yaml
from dotenv import load_dotenv
from pgvector.psycopg2 import register_vector
from sentence_transformers import SentenceTransformer
from sentence_transformers.cross_encoder import CrossEncoder

load_dotenv()

logger = logging.getLogger(__name__)

CONFIG_PATH = "config/config.yaml"


@functools.lru_cache(maxsize=1)
def load_config() -> dict:
    try:
        with open(CONFIG_PATH, "r") as file:
            return yaml.safe_load(file)
    except Exception:
        logger.error("Cannot find/load the configuration file: %s", CONFIG_PATH)
        raise FileNotFoundError(f"configuration file not found: {CONFIG_PATH}")


_db = load_config()["local"]["db"]
DB_CONFIG = {
    "host": _db["host"],
    "port": _db["port"],
    "dbname": _db["dbname"],
    "user": _db["user"],
    "password": _db["password"],
}


def connect_db() -> psycopg2.extensions.connection:
    conn = psycopg2.connect(**DB_CONFIG)
    register_vector(conn)
    return conn


@functools.lru_cache(maxsize=1)
def get_embedding_model() -> SentenceTransformer:
    model_name = load_config()["local"]["embedding"]["model"]
    return SentenceTransformer(model_name)


@functools.lru_cache(maxsize=1)
def get_reranker_model() -> CrossEncoder:
    model_name = load_config()["local"]["rerank"]["model"]
    return CrossEncoder(model_name)


@functools.lru_cache(maxsize=1)
def get_anthropic_client() -> anthropic.Anthropic:
    return anthropic.Anthropic()
