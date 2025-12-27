"""Retrieval utilities."""
from __future__ import annotations

import logging
from typing import Dict, List, Tuple

from rank_bm25 import BM25Okapi

from .utils import tokenize

LOGGER = logging.getLogger(__name__)


def build_bm25(corpus: List[Dict[str, str]], k1: float = 1.2, b: float = 0.75) -> Tuple[BM25Okapi, List[List[str]]]:
    tokenized = [tokenize(doc.get("text") or "") for doc in corpus]
    bm25 = BM25Okapi(tokenized, k1=k1, b=b)
    return bm25, tokenized


def retrieve_top_k(
    query: str,
    corpus: List[Dict[str, str]],
    bm25: BM25Okapi,
    top_k: int = 10,
) -> List[Dict[str, str]]:
    scores = bm25.get_scores(tokenize(query))
    ranked = sorted(range(len(corpus)), key=lambda idx: scores[idx], reverse=True)
    results = []
    for idx in ranked[:top_k]:
        doc = dict(corpus[idx])
        doc["score"] = float(scores[idx])
        results.append(doc)
    return results
