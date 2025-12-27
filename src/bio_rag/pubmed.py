"""PubMed fetching and caching via NCBI E-utilities."""
from __future__ import annotations

import logging
import sqlite3
import time
from typing import Dict, Iterable, List, Optional
from xml.etree import ElementTree

import requests

from .utils import normalize_whitespace

LOGGER = logging.getLogger(__name__)


def init_cache(db_path: str) -> None:
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS pubmed (
                pmid TEXT PRIMARY KEY,
                title TEXT,
                abstract TEXT,
                text TEXT
            )
            """
        )
        conn.commit()


def get_cached_pmids(db_path: str, pmids: Iterable[str]) -> Dict[str, Dict[str, str]]:
    pmid_list = list(pmids)
    if not pmid_list:
        return {}
    placeholders = ",".join("?" for _ in pmid_list)
    query = f"SELECT pmid, title, abstract, text FROM pubmed WHERE pmid IN ({placeholders})"
    with sqlite3.connect(db_path) as conn:
        rows = conn.execute(query, pmid_list).fetchall()
    return {row[0]: {"pmid": row[0], "title": row[1], "abstract": row[2], "text": row[3]} for row in rows}


def cache_records(db_path: str, records: Iterable[Dict[str, str]]) -> None:
    with sqlite3.connect(db_path) as conn:
        conn.executemany(
            "INSERT OR REPLACE INTO pubmed (pmid, title, abstract, text) VALUES (?, ?, ?, ?)",
            [(r["pmid"], r.get("title"), r.get("abstract"), r.get("text")) for r in records],
        )
        conn.commit()


def fetch_pubmed_record(pmid: str, email: str, api_key: Optional[str] = None) -> Optional[Dict[str, str]]:
    params = {
        "db": "pubmed",
        "id": pmid,
        "retmode": "xml",
        "email": email,
    }
    if api_key:
        params["api_key"] = api_key
    response = requests.get("https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi", params=params, timeout=30)
    response.raise_for_status()
    root = ElementTree.fromstring(response.text)
    title = None
    abstract = None
    for article in root.findall(".//Article"):
        title_el = article.find("ArticleTitle")
        title = title_el.text if title_el is not None else None
        abstract_texts = []
        for abs_el in article.findall(".//AbstractText"):
            if abs_el.text:
                abstract_texts.append(abs_el.text)
        abstract = " ".join(abstract_texts) if abstract_texts else None
    if not title and not abstract:
        return None
    text = normalize_whitespace(" ".join([t for t in [title, abstract] if t]))
    return {
        "pmid": pmid,
        "title": normalize_whitespace(title or ""),
        "abstract": normalize_whitespace(abstract or ""),
        "text": text,
    }


def fetch_pubmed_batch(
    pmids: List[str],
    email: str,
    api_key: Optional[str] = None,
    rate_limit: float = 3.0,
) -> List[Dict[str, str]]:
    records = []
    delay = 1.0 / rate_limit if rate_limit > 0 else 0
    for pmid in pmids:
        try:
            record = fetch_pubmed_record(pmid, email=email, api_key=api_key)
            if record:
                records.append(record)
            time.sleep(delay)
        except requests.HTTPError as exc:
            LOGGER.warning("Failed to fetch PMID %s: %s", pmid, exc)
            time.sleep(max(1.0, delay))
    return records
