"""Stress test generators."""
from __future__ import annotations

import logging
import random
from typing import Dict, Iterable, List, Tuple

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

LOGGER = logging.getLogger(__name__)


def inject_noise(
    retrieved: List[Dict[str, str]],
    pool: List[Dict[str, str]],
    distractor_k: int = 5,
    seed: int = 13,
) -> List[Dict[str, str]]:
    random.seed(seed)
    pmids = {doc["pmid"] for doc in retrieved}
    candidates = [doc for doc in pool if doc.get("pmid") not in pmids]
    random.shuffle(candidates)
    noise = candidates[:distractor_k]
    LOGGER.info("Injected %s distractor docs", len(noise))
    return retrieved + noise


def detect_conflicts(snippets: List[Dict[str, str]], threshold: float = 0.3) -> List[Tuple[Dict[str, str], Dict[str, str]]]:
    if len(snippets) < 2:
        return []
    texts = [s["sentence"] for s in snippets]
    vectorizer = TfidfVectorizer(stop_words="english")
    vecs = vectorizer.fit_transform(texts)
    sim = cosine_similarity(vecs)
    conflicts: List[Tuple[Dict[str, str], Dict[str, str]]] = []
    for i in range(len(snippets)):
        for j in range(i + 1, len(snippets)):
            if sim[i, j] > threshold and snippets[i]["pmid"] != snippets[j]["pmid"]:
                conflicts.append((snippets[i], snippets[j]))
    return conflicts


def remove_supporting_snippets(snippets: List[Dict[str, str]], remove_top_n: int = 2) -> List[Dict[str, str]]:
    ranked = sorted(snippets, key=lambda s: s.get("score", 0.0), reverse=True)
    return ranked[remove_top_n:]
