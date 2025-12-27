"""Run stress tests on the baseline pipeline."""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional

import requests

from bio_rag.config import load_config
from bio_rag.corpus import load_corpus
from bio_rag.dataset import parse_dataset
from bio_rag.pico import extract_pico, pico_mismatch_score
from bio_rag.retrieval import build_bm25, retrieve_top_k
from bio_rag.snippets import build_candidate_snippets, score_snippets, select_top_snippets
from bio_rag.stressors import detect_conflicts, inject_noise, remove_supporting_snippets
from bio_rag.utils import ensure_dir, load_env, read_json, safe_get_env, setup_logging, timestamp_run_id, write_json

LOGGER = logging.getLogger(__name__)


def llm_conflict_judge(snippet_a: str, snippet_b: str, api_key: str) -> bool:
    payload = {
        "model": "gpt-4o-mini",
        "messages": [
            {"role": "system", "content": "Decide if the two snippets are contradictory. Reply JSON {conflict: true|false}."},
            {"role": "user", "content": f"Snippet A: {snippet_a}\nSnippet B: {snippet_b}"},
        ],
        "response_format": {"type": "json_object"},
        "temperature": 0,
    }
    response = requests.post(
        "https://api.openai.com/v1/chat/completions",
        headers={"Authorization": f"Bearer {api_key}"},
        json=payload,
        timeout=30,
    )
    response.raise_for_status()
    content = response.json()["choices"][0]["message"]["content"]
    return json.loads(content).get("conflict", False)


def run_pipeline(
    question: Dict[str, object],
    corpus: List[Dict[str, str]],
    bm25,
    config: Dict[str, object],
    noise: bool = False,
    unanswerable: bool = False,
) -> Dict[str, object]:
    retrieved = retrieve_top_k(question["body"], corpus, bm25, config["retrieval"]["top_k"])
    if noise:
        retrieved = inject_noise(retrieved, corpus, config["stressors"]["noise"]["distractor_k"])
    candidates = build_candidate_snippets(retrieved, config["snippets"]["max_sentences_per_doc"])
    scored = score_snippets(question["body"], candidates)
    selected = select_top_snippets(scored, config["snippets"]["snippet_k"], config["snippets"]["mmr_lambda"])
    if unanswerable:
        selected = remove_supporting_snippets(selected, config["stressors"]["unanswerable"]["remove_top_n"])
    return {
        "question_id": question["id"],
        "retrieved_pmids": [d["pmid"] for d in retrieved],
        "snippets": selected,
        "predicted_exact": "insufficient evidence" if unanswerable else None,
        "predicted_ideal": None,
        "is_unanswerable": unanswerable,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--config", default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    setup_logging(config.get("logging", {}).get("level", "INFO"))
    load_env()

    api_key = safe_get_env("OPENAI_API_KEY")

    questions = parse_dataset(read_json(args.dataset))
    corpus = load_corpus(args.corpus)
    bm25, _ = build_bm25(corpus, config["retrieval"]["bm25_k1"], config["retrieval"]["bm25_b"])

    runs_dir = Path(config["paths"]["runs_dir"])

    if config["stressors"]["noise"]["enabled"]:
        run_id = timestamp_run_id("noise")
        run_dir = runs_dir / run_id
        ensure_dir(run_dir)
        predictions = [run_pipeline(q, corpus, bm25, config, noise=True) for q in questions]
        write_json(run_dir / "predictions.json", predictions)
        LOGGER.info("Noise run saved to %s", run_dir)

    if config["stressors"]["conflict"]["enabled"]:
        run_id = timestamp_run_id("conflict")
        run_dir = runs_dir / run_id
        ensure_dir(run_dir)
        predictions = []
        for question in questions:
            pred = run_pipeline(question, corpus, bm25, config)
            conflicts = detect_conflicts(pred["snippets"], config["stressors"]["conflict"]["similarity_threshold"])
            conflict_pairs = []
            for a, b in conflicts:
                is_conflict = True
                if config["stressors"]["conflict"]["llm_judge"] and api_key:
                    try:
                        is_conflict = llm_conflict_judge(a["sentence"], b["sentence"], api_key)
                    except Exception as exc:  # pylint: disable=broad-except
                        LOGGER.warning("LLM conflict judge failed: %s", exc)
                if is_conflict:
                    conflict_pairs.append({"a": a, "b": b})
            pred["conflict_pairs"] = conflict_pairs
            pred["is_conflict"] = len(conflict_pairs) > 0
            predictions.append(pred)
        write_json(run_dir / "predictions.json", predictions)
        LOGGER.info("Conflict run saved to %s", run_dir)

    if config["stressors"]["unanswerable"]["enabled"]:
        run_id = timestamp_run_id("unanswerable")
        run_dir = runs_dir / run_id
        ensure_dir(run_dir)
        predictions = [run_pipeline(q, corpus, bm25, config, unanswerable=True) for q in questions]
        write_json(run_dir / "predictions.json", predictions)
        LOGGER.info("Unanswerable run saved to %s", run_dir)

    if config["stressors"]["pico_mismatch"]["enabled"]:
        run_id = timestamp_run_id("pico_mismatch")
        run_dir = runs_dir / run_id
        ensure_dir(run_dir)
        predictions = []
        for question in questions:
            pred = run_pipeline(question, corpus, bm25, config)
            question_pico = extract_pico(question["body"], api_key if config["pico"]["llm_enabled"] else None)
            mismatch_scores = []
            for snippet in pred["snippets"]:
                snippet_pico = extract_pico(snippet["sentence"], api_key if config["pico"]["llm_enabled"] else None)
                mismatch_scores.append(pico_mismatch_score(question_pico, snippet_pico))
            avg_mismatch = float(sum(mismatch_scores) / len(mismatch_scores)) if mismatch_scores else 0.0
            pred["pico_mismatch_score"] = avg_mismatch
            pred["is_pico_mismatch"] = avg_mismatch >= config["stressors"]["pico_mismatch"]["mismatch_threshold"]
            predictions.append(pred)
        write_json(run_dir / "predictions.json", predictions)
        LOGGER.info("PICO mismatch run saved to %s", run_dir)

    return 0


if __name__ == "__main__":
    sys.exit(main())
