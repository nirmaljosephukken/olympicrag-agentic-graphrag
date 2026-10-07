"""
Local, free, deterministic embeddings: model2vec static retrieval model (minishlab/potion-retrieval-32M, 512-d).

Why static embeddings: the corpus is 22.5k chunks (~5.5M tokens). A transformer embedder runs at ~1 chunk/s
on a 2-core CPU (hours), and API embedding quotas are tight. potion-retrieval embeds ~2,000 chunks/s on CPU,
needs no GPU or API key, and gives every pipeline the same reproducible vectors.
"""
from __future__ import annotations

import json
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np

MODEL = "minishlab/potion-retrieval-32M"
DIM = 512


@lru_cache(maxsize=1)
def _model():
    from model2vec import StaticModel
    return StaticModel.from_pretrained(MODEL)


def _unit(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.where(n == 0, 1, n)


def embed_query(q: str) -> list[float]:
    return [round(float(x), 6) for x in _unit(_model().encode([q]))[0]]


def embed_texts(texts: list[str]) -> np.ndarray:
    return _unit(_model().encode(texts)).astype(np.float32)


if __name__ == "__main__":
    build = Path(sys.argv[1] if len(sys.argv) > 1 else "build")
    chunks = [json.loads(l) for l in open(build / "chunks.jsonl", encoding="utf-8")]
    vecs = embed_texts([c["text"] for c in chunks])
    np.save(build / "chunk_emb.npy", vecs)
    print(f"embedded {len(chunks)} chunks -> {build / 'chunk_emb.npy'} {vecs.shape}")
