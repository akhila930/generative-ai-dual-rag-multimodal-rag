from pathlib import Path
import os

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
INDEX_DIR = DATA_DIR / "faiss_index"
PAGE_IMAGE_DIR = DATA_DIR / "page_images"

HANDBOOK_PATH = DATA_DIR / "university_handbook.pdf"
PAGES_JSON = DATA_DIR / "processed_pages.json"
QUESTIONS_JSON = DATA_DIR / "evaluation_questions.json"

EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL",
    "sentence-transformers/all-MiniLM-L6-v2",
)

SMOLLM_MODEL = os.getenv(
    "SMOLLM_MODEL",
    "HuggingFaceTB/SmolLM2-135M-Instruct",
)

LLAMA_MODEL = os.getenv(
    "LLAMA_MODEL",
    "meta-llama/Llama-3.2-1B-Instruct",
)

VLM_MODEL = os.getenv(
    "VLM_MODEL",
    "HuggingFaceTB/SmolVLM-256M-Instruct",
)

TOP_K = int(os.getenv("TOP_K", "3"))
MAX_NEW_TOKENS = int(os.getenv("MAX_NEW_TOKENS", "180"))
