"""
Re-run evaluation only (no PDF processing).

Usage:
    python -m src.nlp.validation.evaluate_only ^
        --pred "D:\Mes projets\mof-co2-ml\data\output\dataset.json" ^
        --gold "D:\Mes projets\mof-co2-ml\data\output\gold.json" ^
        --out  "D:\Mes projets\mof-co2-ml\data\output"
"""
from __future__ import annotations
import argparse
import json
import logging
from pathlib import Path

from .metrics import evaluate, TARGET_FIELDS
from .error_report import write_error_report
from ..io.writer import write_json

logger = logging.getLogger(__name__)


def _print_table(report: dict) -> None:
    print()
    print(f"Gold rows: {report['_gold_records']} | "
          f"matched: {report['_matched_gold_rows']} | "
          f"coverage: {report['_coverage_gold_rows']:.1%}")
    print()
    header = f"{'Field':28s} {'n_ref':>5s} {'cov':>5s} {'P':>6s} {'R':>6s} {'F1':>6s}"
    print(header)
    print("-" * len(header))
    for label in TARGET_FIELDS:
        v = report.get(label, {})
        n_ref = v.get("n_ref", 0)
        cov = v.get("coverage")
        p, r, f1 = v.get("precision"), v.get("recall"), v.get("f1")
        cov_s = f"{cov:.0%}" if cov is not None else "n/a"
        p_s = f"{p:.2f}" if p is not None else "n/a"
        r_s = f"{r:.2f}" if r is not None else "n/a"
        f_s = f"{f1:.2f}" if f1 is not None else "n/a"
        print(f"{label:28s} {n_ref:5d} {cov_s:>5s} {p_s:>6s} {r_s:>6s} {f_s:>6s}")
    print()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True, type=Path)
    ap.add_argument("--gold", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    with open(args.pred, "r", encoding="utf-8") as f:
        predicted = json.load(f)
    with open(args.gold, "r", encoding="utf-8") as f:
        gold = json.load(f)

    logger.info("Loaded %d predicted, %d gold", len(predicted), len(gold))

    report = evaluate(predicted, gold)
    args.out.mkdir(parents=True, exist_ok=True)
    write_json(report, args.out / "eval_report.json")
    write_error_report(predicted, gold, args.out / "error_report.txt")

    _print_table(report)
    logger.info("Wrote: %s", args.out / "eval_report.json")
    logger.info("Wrote: %s", args.out / "error_report.txt")


if __name__ == "__main__":
    main()