"""Evaluation metrics for BioRAG stress tests."""
from __future__ import annotations

import logging
from typing import Dict, Iterable, List, Tuple

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .utils import tokenize

LOGGER = logging.getLogger(__name__)


def recall_at_k(gold: List[str], retrieved: List[str], k: int = 10) -> float:
    gold_set = set(gold)
    if not gold_set:
        return 0.0
    retrieved_set = set(retrieved[:k])
    return len(gold_set & retrieved_set) / float(len(gold_set))


def token_overlap_f1(gold: str, pred: str) -> float:
    gold_tokens = set(tokenize(gold))
    pred_tokens = set(tokenize(pred))
    if not gold_tokens or not pred_tokens:
        return 0.0
    precision = len(gold_tokens & pred_tokens) / len(pred_tokens)
    recall = len(gold_tokens & pred_tokens) / len(gold_tokens)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def snippets_overlap_f1(gold_snippets: List[str], pred_snippets: List[str]) -> float:
    if not gold_snippets or not pred_snippets:
        return 0.0
    scores = []
    for gold in gold_snippets:
        best = max(token_overlap_f1(gold, pred) for pred in pred_snippets)
        scores.append(best)
    return float(sum(scores) / len(scores)) if scores else 0.0


def groundedness_score(pred_texts: List[str], snippets: List[str]) -> float:
    if not snippets:
        return 0.0
    if not pred_texts:
        pred_texts = []
    texts = pred_texts + snippets
    vectorizer = TfidfVectorizer(stop_words="english")
    vecs = vectorizer.fit_transform(texts)
    pred_vecs = vecs[: len(pred_texts)] if pred_texts else vecs[:1]
    snippet_vecs = vecs[len(pred_texts) :]
    if pred_texts:
        sims = cosine_similarity(pred_vecs, snippet_vecs)
        return float(sims.max(axis=1).mean())
    sims = cosine_similarity(vecs[0], snippet_vecs).flatten()
    return float(sims.mean())


def abstention_accuracy(abstain_flags: List[bool], predictions: List[str]) -> float:
    if not abstain_flags:
        return 0.0
    correct = 0
    for flag, pred in zip(abstain_flags, predictions):
        if flag and pred:
            if pred.lower().strip() == "insufficient evidence":
                correct += 1
        elif not flag:
            if pred.lower().strip() != "insufficient evidence":
                correct += 1
    return correct / len(abstain_flags)


def evaluate_run(
    dataset: List[Dict[str, object]],
    predictions: List[Dict[str, object]],
) -> Tuple[Dict[str, float], pd.DataFrame]:
    pred_map = {p["question_id"]: p for p in predictions}
    metrics = {
        "recall@10": [],
        "snippet_f1": [],
        "groundedness": [],
        "abstain_accuracy": [],
    }
    rows = []
    for question in dataset:
        qid = question["id"]
        pred = pred_map.get(qid, {})
        gold_docs = []
        for doc in question.get("documents") or []:
            if isinstance(doc, str):
                gold_docs.append(doc.split("/")[-1])
        retrieved = pred.get("retrieved_pmids") or []
        metrics["recall@10"].append(recall_at_k(gold_docs, retrieved))

        gold_snippets = [s.get("text") for s in question.get("snippets") or [] if isinstance(s, dict)]
        pred_snippets = [s["sentence"] for s in pred.get("snippets") or []]
        metrics["snippet_f1"].append(snippets_overlap_f1(gold_snippets, pred_snippets))

        pred_texts = []
        if pred.get("predicted_exact"):
            pred_texts.append(str(pred["predicted_exact"]))
        if pred.get("predicted_ideal"):
            pred_texts.append(str(pred["predicted_ideal"]))
        metrics["groundedness"].append(groundedness_score(pred_texts, pred_snippets))

        abstain_flag = pred.get("is_unanswerable", False)
        abstain_pred = pred.get("predicted_exact") or ""
        metrics["abstain_accuracy"].append(1.0 if abstain_flag and abstain_pred.lower() == "insufficient evidence" else 0.0)

        rows.append(
            {
                "question_id": qid,
                "recall@10": metrics["recall@10"][-1],
                "snippet_f1": metrics["snippet_f1"][-1],
                "groundedness": metrics["groundedness"][-1],
                "abstain_accuracy": metrics["abstain_accuracy"][-1],
            }
        )
    summary = {k: float(sum(v) / len(v)) if v else 0.0 for k, v in metrics.items()}
    return summary, pd.DataFrame(rows)
