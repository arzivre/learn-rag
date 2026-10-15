import json
from pathlib import Path

BM25_K1 = 1.5

PROJECT_ROOT = Path(__file__).resolve().parents[3]
# Directory
DATA_DIR = PROJECT_ROOT / "src" / "data"
CACHE_DIR = PROJECT_ROOT / "src" / "cache"
# File
MOVIES_PATH = DATA_DIR / "movies.json"
STOPWORDS_PATH = DATA_DIR / "stopwords.txt"


def load_movies() -> list[dict]:
    with open(MOVIES_PATH, "r") as f:
        data = json.load(f)
    return data["movies"]


def load_stopwords():
    with open(STOPWORDS_PATH, "r") as f:
        data = f.read().splitlines()
    return data
