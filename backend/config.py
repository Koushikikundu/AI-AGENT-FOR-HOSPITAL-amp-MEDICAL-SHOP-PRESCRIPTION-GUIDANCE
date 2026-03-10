from pathlib import Path
import os
from dotenv import load_dotenv

# Project paths
ROOT_DIR = Path(__file__).resolve().parents[1]
BACKEND_DIR = ROOT_DIR / "backend"
DATA_DIR = BACKEND_DIR / "data"
VECTOR_STORE_DIR = BACKEND_DIR / "vector_store" / "chroma_db"

# Load environment variables from project .env if present.
load_dotenv(ROOT_DIR / ".env")

# Data
CSV_PATH = DATA_DIR / "medicine_dataset_50k_unique.csv"

# Embeddings / vector DB
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")
CHROMA_COLLECTION_NAME = "hospital_medicines"
TOP_K = 4

# Stock backend
STOCK_BACKEND = os.getenv("STOCK_BACKEND", "csv").lower()  # mongodb | postgres | csv
MONGO_URI = os.getenv("MONGO_URI", "")
MONGO_DB_NAME = os.getenv("MONGO_DB_NAME", "hospital_ai")
MONGO_STOCK_COLLECTION = os.getenv("MONGO_STOCK_COLLECTION", "medicine_stock")
DATABASE_URL = os.getenv("DATABASE_URL", "")
POSTGRES_STOCK_TABLE = os.getenv("POSTGRES_STOCK_TABLE", "medicine_stock")

# LLMs
GENERATION_MODEL_NAME = os.getenv("GENERATION_MODEL_NAME", "mistralai/Mistral-7B-Instruct-v0.2")
EXTRACTION_MODEL_NAME = os.getenv("EXTRACTION_MODEL_NAME", "google/flan-t5-base")
MAX_NEW_TOKENS = 256
TEMPERATURE = 0.2

# Query routing
STOCK_INTENT_KEYWORDS = {
    "stock",
    "available",
    "availability",
    "have",
    "inventory",
    "in stock",
    "left",
    "qty",
    "quantity",
}

# Fuzzy matching
FUZZY_SCORE_CUTOFF = 65
