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

TESSDATA_DIR = DATA_DIR / "tessdata"
TESSDATA_DIR.mkdir(parents=True, exist_ok=True)
SYSTEM_TESSDATA_DIR = Path("C:/Program Files/Tesseract-OCR/tessdata")


def resolve_tessdata_dir() -> Path:
    """Resolve directory containing Tesseract traineddata models (e.g. vie.traineddata).
    Priority:
    1. TESSDATA_PREFIX environment variable
    2. Local project directory: data/tessdata
    3. Windows system installation directory: C:/Program Files/Tesseract-OCR/tessdata
    """
    env_tess = os.getenv("TESSDATA_PREFIX", "")
    if env_tess and Path(env_tess).exists():
        return Path(env_tess)
    if (TESSDATA_DIR / "vie.traineddata").exists():
        return TESSDATA_DIR
    if (SYSTEM_TESSDATA_DIR / "vie.traineddata").exists():
        return SYSTEM_TESSDATA_DIR
    return TESSDATA_DIR


# Xberg Extraction Configuration
XBERG_OCR_STRATEGY = os.getenv("XBERG_OCR_STRATEGY", "scanned_pages")
XBERG_OCR_MIN_CONFIDENCE = float(os.getenv("XBERG_OCR_MIN_CONFIDENCE", 0.5))
XBERG_OCR_LANGUAGE = ["vie"]
XBERG_MAX_CONTENT_SIZE = int(os.getenv("XBERG_MAX_CONTENT_SIZE", 1 * 1024 * 1024 * 1024))  # 1 GB limit for uncompressed OCR buffer

# Hugging Face & API Keys
HF_TOKEN = os.getenv("HF_TOKEN", "")
if HF_TOKEN and "HUGGING_FACE_HUB_TOKEN" not in os.environ:
    os.environ["HUGGING_FACE_HUB_TOKEN"] = HF_TOKEN


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
QDRANT_COLLECTION_NAME = os.getenv("QDRANT_COLLECTION_NAME", "legal_chunks_v2")
QDRANT_LOCAL_PATH = os.getenv("QDRANT_LOCAL_PATH", str(VECTOR_DB_DIR / "qdrant_storage"))

# Embedding Model Configuration & Token Control
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "intfloat/multilingual-e5-base")
EMBEDDING_DIMENSION = int(os.getenv("EMBEDDING_DIMENSION", 768))
EMBED_TARGET_TOKENS = int(os.getenv("EMBED_TARGET_TOKENS", 350))
EMBED_HARD_MAX_TOKENS = int(os.getenv("EMBED_HARD_MAX_TOKENS", 450))

# Pipeline Versioning
PIPELINE_VERSION = "2.0.0"
CHUNKING_VERSION = "legal-hierarchical-v2"

# LLM Model Configuration
LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")

# Reranker Model Configuration
RERANKER_MODEL_NAME = os.getenv("RERANKER_MODEL_NAME", "ms-marco-TinyBERT-L-2-v2")
RERANKER_TOP_N = int(os.getenv("RERANKER_TOP_N", 4))
RETRIEVAL_TOP_K = int(os.getenv("RETRIEVAL_TOP_K", 20))
