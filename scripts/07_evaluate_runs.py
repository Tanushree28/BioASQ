"""Evaluate runs and output report JSON/CSV."""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

from bio_rag.config import load_config
from bio_rag.dataset import parse_dataset
from bio_rag.evaluation import evaluate_run
from bio_rag.utils import read_json, setup_logging, write_json

LOGGER = logging.getLogger(__name__)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--runs_dir", required=True)
    parser.add_argument("--config", default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    setup_logging(config.get("logging", {}).get("level", "INFO"))

    dataset = parse_dataset(read_json(args.dataset))
    runs_dir = Path(args.runs_dir)

    reports = []
    for run_path in runs_dir.iterdir():
        if not run_path.is_dir():
            continue
        predictions_path = run_path / "predictions.json"
        if not predictions_path.exists():
            continue
        predictions = read_json(predictions_path)
        summary, detail_df = evaluate_run(dataset, predictions)
        report = {"run_id": run_path.name, **summary}
        reports.append(report)
        write_json(run_path / "report.json", report)
        detail_df.to_csv(run_path / "report.csv", index=False)
        LOGGER.info("Saved report for %s", run_path.name)

    if reports:
        summary_df = pd.DataFrame(reports)
        summary_df.to_csv(runs_dir / "report.csv", index=False)
        write_json(runs_dir / "report.json", reports)
        LOGGER.info("Saved aggregate report to %s", runs_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
