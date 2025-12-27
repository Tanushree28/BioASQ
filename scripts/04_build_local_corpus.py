"""Build local corpus JSONL from cached PubMed records."""
from __future__ import annotations

import argparse
import logging
import sys

from bio_rag.config import load_config
from bio_rag.corpus import build_corpus_from_cache
from bio_rag.utils import setup_logging

LOGGER = logging.getLogger(__name__)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--config", default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    setup_logging(config.get("logging", {}).get("level", "INFO"))

    build_corpus_from_cache(args.db, args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
