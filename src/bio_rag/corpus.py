"""Local corpus utilities."""
from __future__ import annotations

import logging
import sqlite3
from typing import Dict, Iterable, List

from .utils import read_jsonl, write_jsonl

LOGGER = logging.getLogger(__name__)


def build_corpus_from_cache(db_path: str, out_path: str) -> List[Dict[str, str]]:
    with sqlite3.connect(db_path) as conn:
        rows = conn.execute("SELECT pmid, title, abstract, text FROM pubmed").fetchall()
    docs = [
        {
            "pmid": row[0],
            "title": row[1],
            "abstract": row[2],
            "text": row[3],
        }
        for row in rows
    ]
    write_jsonl(out_path, docs)
    LOGGER.info("Wrote %s docs to %s", len(docs), out_path)
    return docs


def load_corpus(path: str) -> List[Dict[str, str]]:
    return read_jsonl(path)
