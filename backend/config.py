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
CHROMA_RELEVANCE_MAX_DISTANCE = float(os.getenv("CHROMA_RELEVANCE_MAX_DISTANCE", "1.2"))

# Stock backend
STOCK_BACKEND = os.getenv("STOCK_BACKEND", "csv").lower()  # mongodb | postgres | csv
MONGO_URI = os.getenv("MONGO_URI", "")
MONGO_DB_NAME = os.getenv("MONGO_DB_NAME", "hospital_ai")
MONGO_STOCK_COLLECTION = os.getenv("MONGO_STOCK_COLLECTION", "medicine_stock")

# LLMs
GENERATION_MODEL_NAME = os.getenv("GENERATION_MODEL_NAME", "mistralai/Mistral-7B-Instruct-v0.2")
EXTRACTION_MODEL_NAME = os.getenv("EXTRACTION_MODEL_NAME", "google/flan-t5-base")
MAX_NEW_TOKENS = 256
TEMPERATURE = 0.2
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY", "")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1:latest")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
HUGGINGFACE_INFERENCE_MODEL = os.getenv("HUGGINGFACE_INFERENCE_MODEL", "google/flan-t5-base")

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
