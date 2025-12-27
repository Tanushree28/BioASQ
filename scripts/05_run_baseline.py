"""Run baseline BM25 + snippet selection pipeline."""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from bio_rag.config import load_config
from bio_rag.corpus import load_corpus
from bio_rag.dataset import parse_dataset
from bio_rag.retrieval import build_bm25, retrieve_top_k
from bio_rag.snippets import build_candidate_snippets, score_snippets, select_top_snippets
from bio_rag.utils import ensure_dir, read_json, setup_logging, timestamp_run_id, write_json

LOGGER = logging.getLogger(__name__)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--config", default=None)
    parser.add_argument("--run_id", default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    setup_logging(config.get("logging", {}).get("level", "INFO"))

    questions = parse_dataset(read_json(args.dataset))
    corpus = load_corpus(args.corpus)

    bm25, _ = build_bm25(corpus, config["retrieval"]["bm25_k1"], config["retrieval"]["bm25_b"])

    run_id = args.run_id or timestamp_run_id("baseline")
    run_dir = Path(config["paths"]["runs_dir"]) / run_id
    ensure_dir(run_dir)

    predictions = []
    for question in questions:
        retrieved = retrieve_top_k(question["body"], corpus, bm25, config["retrieval"]["top_k"])
        candidates = build_candidate_snippets(retrieved, config["snippets"]["max_sentences_per_doc"])
        scored = score_snippets(question["body"], candidates)
        selected = select_top_snippets(scored, config["snippets"]["snippet_k"], config["snippets"]["mmr_lambda"])
        predictions.append(
            {
                "question_id": question["id"],
                "retrieved_pmids": [d["pmid"] for d in retrieved],
                "snippets": selected,
                "predicted_exact": None,
                "predicted_ideal": None,
            }
        )

    write_json(run_dir / "predictions.json", predictions)
    LOGGER.info("Saved predictions to %s", run_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
