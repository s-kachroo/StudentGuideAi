import os

# OpenAI
OPENAI_API_KEY    = os.getenv("OPENAI_API_KEY")

# Data paths
DATA_DIR          = "../data"

# Chunking parameters
CHUNK_SIZE        = 1500
CHUNK_OVERLAP     = 100

# Vector DB paths
DB_PATH_DEFAULT   = "./chroma_db"
DB_PATH_QINDEX    = "./chroma_qdb"

# RAG parameters
TOP_K             = 3
NUM_QUESTIONS     = 5      # how many questions per chunk

# Cache paths
CHUNKS_CACHE_PATH = "./chunks_cache.pkl"
