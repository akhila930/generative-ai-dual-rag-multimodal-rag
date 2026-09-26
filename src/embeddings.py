from functools import lru_cache
from sentence_transformers import SentenceTransformer
from .config import EMBEDDING_MODEL

@lru_cache(maxsize=1)
def get_embedding_model():
    return SentenceTransformer(EMBEDDING_MODEL)

def embed_texts(texts):
    model = get_embedding_model()
    return model.encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=True,
    )

def embed_query(query):
    model = get_embedding_model()
    return model.encode(
        [query],
        normalize_embeddings=True,
        show_progress_bar=False,
    )
