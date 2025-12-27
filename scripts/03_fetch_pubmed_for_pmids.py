"""Fetch PubMed records for a list of PMIDs and cache them."""
from __future__ import annotations

import argparse
import logging
import sys

from tqdm import tqdm

from bio_rag.config import load_config
from bio_rag.pubmed import cache_records, fetch_pubmed_batch, get_cached_pmids, init_cache
from bio_rag.utils import load_env, read_json, safe_get_env, setup_logging

LOGGER = logging.getLogger(__name__)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pmids", required=True, help="JSON list of PMIDs")
    parser.add_argument("--db", required=True)
    parser.add_argument("--config", default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    setup_logging(config.get("logging", {}).get("level", "INFO"))
    load_env()

    email = safe_get_env("NCBI_EMAIL")
    api_key = safe_get_env("NCBI_API_KEY")
    if not email:
        LOGGER.error("NCBI_EMAIL must be set in environment")
        return 1

    pmids = read_json(args.pmids)
    init_cache(args.db)
    cached = get_cached_pmids(args.db, pmids)
    missing = [pmid for pmid in pmids if pmid not in cached]
    LOGGER.info("%s PMIDs cached, %s missing", len(cached), len(missing))

    records = []
    for chunk_start in tqdm(range(0, len(missing), 50), desc="Fetching PMIDs"):
        chunk = missing[chunk_start : chunk_start + 50]
        records.extend(fetch_pubmed_batch(chunk, email=email, api_key=api_key, rate_limit=3.0))
        if records:
            cache_records(args.db, records)
            records = []
    LOGGER.info("Fetch complete")
    return 0


if __name__ == "__main__":
    sys.exit(main())
