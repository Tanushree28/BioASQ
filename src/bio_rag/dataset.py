"""Dataset parsing utilities for BioASQ-style JSON."""
from __future__ import annotations

import logging
import re
from typing import Any, Dict, Iterable, List

from .utils import normalize_whitespace

LOGGER = logging.getLogger(__name__)


PMID_RE = re.compile(r"(\d{6,})")


def parse_dataset(raw: Any) -> List[Dict[str, Any]]:
    if isinstance(raw, dict):
        if "questions" in raw:
            items = raw["questions"]
        else:
            items = list(raw.values())
    elif isinstance(raw, list):
        items = raw
    else:
        raise ValueError("Unsupported dataset format")

    questions: List[Dict[str, Any]] = []
    for entry in items:
        if not isinstance(entry, dict):
            continue
        qid = str(entry.get("id") or entry.get("qid") or entry.get("question_id") or "")
        body = normalize_whitespace(entry.get("body") or entry.get("question") or "")
        q_type = entry.get("type") or entry.get("question_type")
        documents = entry.get("documents") or []
        snippets = entry.get("snippets") or []
        exact_answer = entry.get("exact_answer")
        ideal_answer = entry.get("ideal_answer")
        questions.append(
            {
                "id": qid,
                "body": body,
                "type": q_type,
                "documents": documents,
                "snippets": snippets,
                "exact_answer": exact_answer,
                "ideal_answer": ideal_answer,
                "raw": entry,
            }
        )
    LOGGER.info("Loaded %s questions", len(questions))
    return questions


def extract_gold_pmids(questions: Iterable[Dict[str, Any]]) -> List[str]:
    pmids = []
    for question in questions:
        for doc in question.get("documents") or []:
            if isinstance(doc, str):
                match = PMID_RE.search(doc)
                if match:
                    pmids.append(match.group(1))
        for snippet in question.get("snippets") or []:
            doc = snippet.get("document") if isinstance(snippet, dict) else None
            if isinstance(doc, str):
                match = PMID_RE.search(doc)
                if match:
                    pmids.append(match.group(1))
    unique = sorted(set(pmids))
    LOGGER.info("Extracted %s unique PMIDs", len(unique))
    return unique
