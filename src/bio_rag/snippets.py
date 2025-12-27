"""Snippet selection utilities."""
from __future__ import annotations

import logging
from typing import Dict, List, Tuple

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .utils import normalize_whitespace, simple_sentence_split

LOGGER = logging.getLogger(__name__)


def build_candidate_snippets(
    docs: List[Dict[str, str]],
    max_sentences_per_doc: int = 50,
) -> List[Dict[str, str]]:
    snippets: List[Dict[str, str]] = []
    for doc in docs:
        sentences = simple_sentence_split(doc.get("text") or "")[:max_sentences_per_doc]
        for sent in sentences:
            snippets.append(
                {
                    "pmid": doc.get("pmid"),
                    "sentence": normalize_whitespace(sent),
                    "doc_score": doc.get("score", 0.0),
                }
            )
    return snippets


def score_snippets(query: str, snippets: List[Dict[str, str]]) -> List[Dict[str, str]]:
    if not snippets:
        return []
    texts = [query] + [s["sentence"] for s in snippets]
    vectorizer = TfidfVectorizer(stop_words="english")
    vectors = vectorizer.fit_transform(texts)
    query_vec = vectors[0]
    snippet_vecs = vectors[1:]
    scores = cosine_similarity(query_vec, snippet_vecs).flatten()
    for snippet, score in zip(snippets, scores):
        snippet["score"] = float(score)
    return snippets


def select_top_snippets(
    snippets: List[Dict[str, str]],
    snippet_k: int = 10,
    mmr_lambda: float = 0.7,
) -> List[Dict[str, str]]:
    if not snippets:
        return []
    ranked = sorted(snippets, key=lambda s: (s.get("score", 0.0), s.get("doc_score", 0.0)), reverse=True)
    selected: List[Dict[str, str]] = []
    for snippet in ranked:
        if len(selected) >= snippet_k:
            break
        selected.append(snippet)
    if mmr_lambda >= 1.0 or len(selected) <= 1:
        return selected

    # Basic diversity: down-weight near-duplicate sentences
    sentences = [s["sentence"] for s in selected]
    vectorizer = TfidfVectorizer(stop_words="english")
    vectors = vectorizer.fit_transform(sentences)
    similarity = cosine_similarity(vectors)
    final_selection = []
    for idx, snippet in enumerate(selected):
        if len(final_selection) >= snippet_k:
            break
        if all(similarity[idx][jdx] < 0.8 for jdx in range(len(final_selection))):
            final_selection.append(snippet)
    return final_selection or selected[:snippet_k]
