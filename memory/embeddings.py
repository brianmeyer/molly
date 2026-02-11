import logging
import threading
from typing import Optional

import numpy as np

log = logging.getLogger(__name__)

_model = None
_model_lock = threading.Lock()


def _get_model():
    """Lazy-load EmbeddingGemma-300M. ~200MB RAM, stays resident."""
    global _model
    if _model is not None:
        return _model
    with _model_lock:
        if _model is None:
            from sentence_transformers import SentenceTransformer

            log.info("Loading embedding model (google/embeddinggemma-300m)...")
            _model = SentenceTransformer("google/embeddinggemma-300m")
            log.info("Embedding model loaded.")
    return _model


def embed(text: str) -> list[float]:
    """Embed a single text string. Returns 768-dim float vector."""
    model = _get_model()
    vec = model.encode(text, normalize_embeddings=True)
    return vec.tolist()


def embed_batch(texts: list[str]) -> list[list[float]]:
    """Embed multiple texts. Returns list of 768-dim float vectors."""
    if not texts:
        return []
    model = _get_model()
    vecs = model.encode(texts, normalize_embeddings=True, batch_size=32)
    return vecs.tolist()


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Cosine similarity between two normalized vectors."""
    a_arr = np.array(a)
    b_arr = np.array(b)
    return float(np.dot(a_arr, b_arr))
