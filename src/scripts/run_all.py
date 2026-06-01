"""
End-to-end orchestrator: generate -> preprocess -> train -> evaluate.

Runs the full reproducible pipeline and produces all data CSVs, model artifacts,
metric tables, and figures.

Run:
    python scripts/run_all.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config
from pipelines import evaluate, generate_data, preprocess, train


def main() -> None:
    """Execute every pipeline stage in order."""
    config.ensure_dirs()
    print("=" * 70, "\n[1/4] Generating synthetic data\n", "=" * 70)
    generate_data.main()
    print("=" * 70, "\n[2/4] Preprocessing\n", "=" * 70)
    preprocess.run()
    print("=" * 70, "\n[3/4] Training CLIQUE (grid search)\n", "=" * 70)
    train.run(do_grid_search=True)
    print("=" * 70, "\n[4/4] Evaluating + exporting results\n", "=" * 70)
    evaluate.run()
    print("\nDone. See results/metrics/ (CSV) and results/figures/ (PNG).")


if __name__ == "__main__":
    main()
