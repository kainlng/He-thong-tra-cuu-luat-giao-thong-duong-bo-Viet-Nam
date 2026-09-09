import os
from pathlib import Path
from dotenv import load_dotenv

# Base paths (project root is 3 levels up: src/core/config.py -> core -> src -> root)
BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
VECTOR_DB_DIR = BASE_DIR / "vector_db"

RAW_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
VECTOR_DB_DIR.mkdir(parents=True, exist_ok=True)

# Load environment variables
load_dotenv(BASE_DIR / ".env")

import re
from typing import List

# Hugging Face & API Keys
HF_TOKEN = os.getenv("HF_TOKEN", "")
if HF_TOKEN and "HUGGING_FACE_HUB_TOKEN" not in os.environ:
    os.environ["HUGGING_FACE_HUB_TOKEN"] = HF_TOKEN


def get_llama_cloud_api_keys() -> List[str]:
    """
    Collect all configured LlamaCloud API keys from environment variables.
    Supports:
    - LLAMA_CLOUD_API_KEY_1, LLAMA_CLOUD_API_KEY_2, LLAMA_CLOUD_API_KEY_3, ...
    - LLAMA_CLOUD_API_KEYS (comma or semicolon separated string)
    - LLAMA_CLOUD_API_KEY or LLAMA_PARSE_API_KEY
    Returns a deduplicated list of non-empty API keys in order.
    """
    keys: List[str] = []

    # 1. Numbered env vars: LLAMA_CLOUD_API_KEY_1, LLAMA_CLOUD_API_KEY_2, ...
    idx = 1
    while True:
        k = os.getenv(f"LLAMA_CLOUD_API_KEY_{idx}")
        if not k:
            break
        k_clean = k.strip()
        if k_clean and k_clean not in keys:
            keys.append(k_clean)
        idx += 1

    # 2. Plural or CSV env var: LLAMA_CLOUD_API_KEYS
    raw_plural = os.getenv("LLAMA_CLOUD_API_KEYS", "")
    if raw_plural:
        for part in re.split(r"[,;]", raw_plural):
            part_clean = part.strip()
            if part_clean and part_clean not in keys:
                keys.append(part_clean)

    # 3. Standard env vars: LLAMA_CLOUD_API_KEY, LLAMA_PARSE_API_KEY
    for env_name in ["LLAMA_CLOUD_API_KEY", "LLAMA_PARSE_API_KEY"]:
        val = os.getenv(env_name, "")
        if val:
            for part in re.split(r"[,;]", val):
                part_clean = part.strip()
                if part_clean and part_clean not in keys:
                    keys.append(part_clean)

    return keys


LLAMA_CLOUD_API_KEYS = get_llama_cloud_api_keys()
LLAMA_CLOUD_API_KEY = LLAMA_CLOUD_API_KEYS[0] if LLAMA_CLOUD_API_KEYS else ""
LLAMA_PARSE_API_KEY = LLAMA_CLOUD_API_KEY
QWEN_API_KEY = os.getenv("QWEN_API_KEY", "")
OPEN_AI_API_KEY = os.getenv("OPEN_AI_API_KEY", "")

# PostgreSQL Settings
POSTGRES_HOST = os.getenv("POSTGRES_HOST", "localhost")
POSTGRES_PORT = int(os.getenv("POSTGRES_PORT", 5432))
POSTGRES_DB = os.getenv("POSTGRES_DB", "legal_traffic_db")
POSTGRES_USER = os.getenv("POSTGRES_USER", "postgres")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD", "111206")

POSTGRES_URL = os.getenv(
    "POSTGRES_URL",
    f"postgresql://{POSTGRES_USER}:{POSTGRES_PASSWORD}@{POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DB}"
)

# Qdrant Settings
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "")
QDRANT_COLLECTION_NAME = os.getenv("QDRANT_COLLECTION_NAME", "legal_chunks")
QDRANT_LOCAL_PATH = os.getenv("QDRANT_LOCAL_PATH", str(VECTOR_DB_DIR / "qdrant_storage"))

# Embedding Model Configuration
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "intfloat/multilingual-e5-base")
EMBEDDING_DIMENSION = int(os.getenv("EMBEDDING_DIMENSION", 768))

# LLM Model Configuration
LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")

# Reranker Model Configuration
RERANKER_MODEL_NAME = os.getenv("RERANKER_MODEL_NAME", "ms-marco-TinyBERT-L-2-v2")
RERANKER_TOP_N = int(os.getenv("RERANKER_TOP_N", 4))
RETRIEVAL_TOP_K = int(os.getenv("RETRIEVAL_TOP_K", 20))
