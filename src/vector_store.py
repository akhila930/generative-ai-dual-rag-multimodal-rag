import json
from pathlib import Path
import numpy as np
import faiss

from .config import PAGES_JSON, INDEX_DIR, TOP_K
from .embeddings import embed_texts, embed_query

INDEX_FILE = INDEX_DIR / "handbook.faiss"
META_FILE = INDEX_DIR / "metadata.json"

def load_pages():
    if not PAGES_JSON.exists():
        raise FileNotFoundError(
            "processed_pages.json not found. Run: python -m src.document"
        )
    return json.loads(PAGES_JSON.read_text(encoding="utf-8"))

def build_index():
    pages = load_pages()
    texts = [p["text"] for p in pages]
    vectors = np.asarray(embed_texts(texts), dtype="float32")

    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)

    faiss.write_index(index, str(INDEX_FILE))
    META_FILE.write_text(json.dumps(pages, indent=2, ensure_ascii=False), encoding="utf-8")
    return index, pages

def load_index():
    if not INDEX_FILE.exists() or not META_FILE.exists():
        build_index()
    index = faiss.read_index(str(INDEX_FILE))
    pages = json.loads(META_FILE.read_text(encoding="utf-8"))
    return index, pages

def retrieve(query, top_k=TOP_K):
    index, pages = load_index()
    q = np.asarray(embed_query(query), dtype="float32")
    scores, ids = index.search(q, top_k)

    results = []
    for score, idx in zip(scores[0], ids[0]):
        if idx < 0:
            continue
        item = dict(pages[int(idx)])
        item["score"] = float(score)
        results.append(item)
    return results

if __name__ == "__main__":
    index, pages = build_index()
    print(f"Indexed {len(pages)} pages.")
