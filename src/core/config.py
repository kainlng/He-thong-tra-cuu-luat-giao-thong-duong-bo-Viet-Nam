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

# Hugging Face & API Keys
HF_TOKEN = os.getenv("HF_TOKEN", "")
if HF_TOKEN and "HUGGING_FACE_HUB_TOKEN" not in os.environ:
    os.environ["HUGGING_FACE_HUB_TOKEN"] = HF_TOKEN

LLAMA_PARSE_API_KEY = os.getenv("LLAMA_PARSE_API_KEY", "")
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
