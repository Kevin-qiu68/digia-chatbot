import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# API Configuration
COHERE_API_KEY = os.getenv("COHERE_API_KEY")

# Vector Database Configuration
VECTORDB_PATH = "./vectordb"
COLLECTION_NAME = "digia_knowledge"

# Document Configuration
DATA_PATH = "./data/company_docs"

# Text Chunking Configuration
CHUNK_SIZE = 500  # Characters per chunk
CHUNK_OVERLAP = 50  # Overlapping characters between chunks

# Retrieval Configuration
TOP_K_RETRIEVAL = 20  # Initial retrieval count
TOP_K_RERANK = 3  # Number of results after reranking

# LLM Configuration
COHERE_MODEL = "command-a-03-2025" #"command-r-plus-08-2024"
TEMPERATURE = 0.3
MAX_TOKENS = 1000