"""Extract gold PMIDs from dataset."""
from __future__ import annotations

import argparse
import logging
import sys

from bio_rag.config import load_config
from bio_rag.dataset import extract_gold_pmids, parse_dataset
from bio_rag.utils import read_json, setup_logging, write_json

LOGGER = logging.getLogger(__name__)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--config", default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    setup_logging(config.get("logging", {}).get("level", "INFO"))

    questions = parse_dataset(read_json(args.dataset))
    pmids = extract_gold_pmids(questions)
    write_json(args.out, pmids)
    LOGGER.info("Saved PMIDs to %s", args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
