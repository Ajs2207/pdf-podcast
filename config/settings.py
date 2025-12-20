import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from the .env file
load_dotenv()

# Define the root directory of the project
BASE_DIR = Path(__file__).resolve().parent.parent

# ---------- Core Paths ----------
# Where uploaded PDFs will be saved
UPLOAD_FOLDER = BASE_DIR / os.getenv("UPLOAD_FOLDER", "data/uploads")
# Where ChromaDB will persist its data
CHROMA_DB_PATH = BASE_DIR / os.getenv("CHROMA_DB_PATH", "storage/chroma_db")

# Ensure these directories exist
UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)
CHROMA_DB_PATH.mkdir(parents=True, exist_ok=True)

# ---------- API Keys ----------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
LANGCHAIN_API_KEY = os.getenv("LANGCHAIN_API_KEY")

# LangChain / LangSmith settings
LANGCHAIN_PROJECT = os.getenv("LANGCHAIN_PROJECT", "pdf-podcast")
LANGCHAIN_TRACING_V2 = os.getenv("LANGCHAIN_TRACING_V2", "false").lower() == "true"

# ---------- Runtime Settings ----------
MAX_UPLOAD_SIZE_MB = int(os.getenv("MAX_UPLOAD_SIZE_MB", 50))
MAX_FILES_PER_USER = int(os.getenv("MAX_FILES_PER_USER", 10))

# ---------- Redis ----------
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")