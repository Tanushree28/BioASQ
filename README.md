# bio-rag-stress-tests

Diagnose what hurts Retrieval-Augmented Generation (RAG) in biomedical question answering using BioASQ-style datasets. This repo builds a lightweight local corpus from gold PMIDs, runs a BM25 baseline, and evaluates stressors like noise, conflict, unanswerability, and PICO mismatch.

## Quick start

### 1) Setup
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2) Configure environment
Copy `.env.example` to `.env` and fill in values.
```bash
cp .env.example .env
```

### 3) Prepare data
Put your BioASQ-style dataset in `data/dataset.json`.

### 4) Run pipeline end-to-end
```bash
python scripts/01_validate_dataset.py --dataset data/dataset.json
python scripts/02_extract_gold_pmids.py --dataset data/dataset.json --out data/gold_pmids.json
python scripts/03_fetch_pubmed_for_pmids.py --pmids data/gold_pmids.json --db data/pubmed_cache.sqlite
python scripts/04_build_local_corpus.py --db data/pubmed_cache.sqlite --out data/corpus.jsonl
python scripts/05_run_baseline.py --dataset data/dataset.json --corpus data/corpus.jsonl
python scripts/06_run_stress_tests.py --dataset data/dataset.json --corpus data/corpus.jsonl
python scripts/07_evaluate_runs.py --dataset data/dataset.json --runs_dir data/runs
```

## Repository layout
```
config/config.yaml          # default hyperparameters and toggles
src/bio_rag/                # pipeline modules
scripts/01..07_*.py         # entry points
```

## Outputs
- `data/runs/<run_id>/predictions.json`
- `data/runs/<run_id>/report.json`
- `data/runs/<run_id>/report.csv`

## Notes
- No GPU required.
- PubMed retrieval is performed via NCBI E-utilities and cached in SQLite.
- The system runs fully offline after caching.
