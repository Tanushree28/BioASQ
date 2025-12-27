"""PICO extraction and mismatch scoring."""
from __future__ import annotations

import json
import logging
import re
from typing import Dict, Optional

import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .utils import normalize_whitespace

LOGGER = logging.getLogger(__name__)


PICO_SCHEMA = {
    "type": "object",
    "properties": {
        "population": {"type": "string"},
        "intervention": {"type": "string"},
        "outcome": {"type": "string"},
    },
    "required": ["population", "intervention", "outcome"],
}


def heuristic_pico(text: str) -> Dict[str, str]:
    text = normalize_whitespace(text)
    population = ""
    intervention = ""
    outcome = ""

    pop_match = re.search(r"(patients|adults|children|subjects|participants|women|men)[^,.]*", text, re.I)
    if pop_match:
        population = pop_match.group(0)
    int_match = re.search(r"(treat|therapy|drug|intervention|procedure)[^,.]*", text, re.I)
    if int_match:
        intervention = int_match.group(0)
    out_match = re.search(r"(outcome|effect|response|survival|mortality)[^,.]*", text, re.I)
    if out_match:
        outcome = out_match.group(0)

    return {
        "population": population,
        "intervention": intervention,
        "outcome": outcome,
    }


def llm_pico(text: str, api_key: str) -> Dict[str, str]:
    payload = {
        "model": "gpt-4o-mini",
        "messages": [
            {
                "role": "system",
                "content": "Extract PICO elements as JSON with keys population, intervention, outcome.",
            },
            {"role": "user", "content": text},
        ],
        "response_format": {"type": "json_schema", "json_schema": {"name": "pico", "schema": PICO_SCHEMA}},
        "temperature": 0,
    }
    response = requests.post(
        "https://api.openai.com/v1/chat/completions",
        headers={"Authorization": f"Bearer {api_key}"},
        json=payload,
        timeout=30,
    )
    response.raise_for_status()
    data = response.json()
    content = data["choices"][0]["message"]["content"]
    return json.loads(content)


def extract_pico(text: str, api_key: Optional[str] = None) -> Dict[str, str]:
    if api_key:
        try:
            return llm_pico(text, api_key)
        except Exception as exc:  # pylint: disable=broad-except
            LOGGER.warning("LLM PICO extraction failed: %s", exc)
    return heuristic_pico(text)


def pico_similarity(a: str, b: str) -> float:
    texts = [a or "", b or ""]
    vectorizer = TfidfVectorizer(stop_words="english")
    vecs = vectorizer.fit_transform(texts)
    return float(cosine_similarity(vecs[0], vecs[1])[0, 0])


def pico_mismatch_score(question_pico: Dict[str, str], snippet_pico: Dict[str, str]) -> float:
    p_sim = pico_similarity(question_pico.get("population", ""), snippet_pico.get("population", ""))
    i_sim = pico_similarity(question_pico.get("intervention", ""), snippet_pico.get("intervention", ""))
    o_sim = pico_similarity(question_pico.get("outcome", ""), snippet_pico.get("outcome", ""))
    return float(1.0 - (p_sim + i_sim + o_sim) / 3.0)
