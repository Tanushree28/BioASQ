"""Validate BioASQ-style dataset structure."""
from __future__ import annotations

import argparse
import logging
import sys

from bio_rag.config import load_config
from bio_rag.dataset import parse_dataset
from bio_rag.utils import read_json, setup_logging

LOGGER = logging.getLogger(__name__)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, help="Path to BioASQ-style JSON dataset")
    parser.add_argument("--config", default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    setup_logging(config.get("logging", {}).get("level", "INFO"))

    raw = read_json(args.dataset)
    questions = parse_dataset(raw)
    if not questions:
        LOGGER.error("No questions found in dataset")
        return 1
    missing = [q for q in questions if not q.get("id") or not q.get("body")]
    if missing:
        LOGGER.warning("%s questions missing id/body", len(missing))
    LOGGER.info("Dataset looks valid with %s questions", len(questions))
    return 0


if __name__ == "__main__":
    sys.exit(main())
