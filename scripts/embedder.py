"""embedder.py — cached sentence embeddings, the only semantic judgement the
environment offers.

Used by WDGraphEnv.rank() to order entities the agent already holds, and by
scout() to measure whether a frontier is getting closer to a target. Both are
deterministic and cost no environment reads: judgement is free, reading is not,
and that asymmetry is what forces the agent to think before spending a read.

Prerequisites: sentence-transformers, and the model weights (downloaded on first
    use). Override the model with WD_RANK_MODEL; default BAAI/bge-m3.
Outputs: none directly — vectors are cached in data/graph/embcache_<model>.npz
    by whoever passes cache_path, so repeated runs skip re-encoding.

Contract: embeddings are L2-normalised, so a dot product IS the cosine. Cache
    keys are the raw text; the E5 prefix is applied at encode time only, so a
    cache stays valid across models of the same family.

Why: this lives beside the environment rather than inside an experiment
    script, because rank() and scout() depend on it and the environment must
    not depend on experiment code.
"""

from __future__ import annotations

import pathlib

import numpy as np


class Embedder:
    """Lazy, cached sentence embeddings (CPU).

    E5-family models require the 'query: ' prefix (symmetric use per the E5
    paper); it is applied transparently and cache keys stay unprefixed.
    """

    def __init__(self, model_name="BAAI/bge-m3", cache_path=None):
        from sentence_transformers import SentenceTransformer
        self.model_name = model_name
        self.prefix = "query: " if "e5" in model_name.lower() else ""
        self.model = SentenceTransformer(model_name)
        self.cache: dict[str, np.ndarray] = {}
        self.cache_path = cache_path
        if cache_path and pathlib.Path(cache_path).exists():
            stored = np.load(cache_path, allow_pickle=True)
            for text, vector in zip(stored["keys"].tolist(), stored["vecs"]):
                self.cache[text] = vector

    def save(self):
        if not self.cache_path or not self.cache:
            return
        keys = list(self.cache)
        np.savez(self.cache_path, keys=np.array(keys, dtype=object),
                 vecs=np.stack([self.cache[k] for k in keys]))

    def get(self, texts):
        missing = [t for t in dict.fromkeys(texts) if t not in self.cache]
        if missing:
            vectors = self.model.encode([self.prefix + t for t in missing],
                                        batch_size=64,
                                        normalize_embeddings=True,
                                        show_progress_bar=False)
            for text, vector in zip(missing, vectors):
                self.cache[text] = vector
        return np.stack([self.cache[t] for t in texts])

    def centroid(self, texts):
        if not texts:
            return None
        vector = self.get(texts).mean(axis=0)
        norm = np.linalg.norm(vector)
        return vector / norm if norm else vector
