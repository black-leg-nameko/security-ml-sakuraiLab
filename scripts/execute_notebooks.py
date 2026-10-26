"""Execute each standalone notebook in a fresh kernel."""

import argparse
from pathlib import Path

import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=ROOT / "artifacts" / "executed")
    parser.add_argument(
        "--quick", action="store_true", help="Limit search time in execution copies only"
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for path in sorted((ROOT / "notebooks").glob("*.ipynb")):
        book = nbformat.read(path, as_version=4)
        if args.quick:
            book.metadata["validation_profile"] = {
                "name": "quick",
                "optuna_timeout_seconds": 2,
                "lightgbm_time_budget_seconds": 10,
            }
            for cell in book.cells:
                if cell.cell_type != "code":
                    continue
                cell.source = cell.source.replace(
                    "study.optimize(objective, timeout=60)", "study.optimize(objective, timeout=2)"
                )
                cell.source = cell.source.replace(
                    "olgb.LightGBMTunerCV(params, train, num_boost_round=100)",
                    "olgb.LightGBMTunerCV(params, train, num_boost_round=100, time_budget=10)",
                )
        client = NotebookClient(
            book,
            timeout=3600,
            kernel_name="python3",
            resources={"metadata": {"path": str(ROOT)}},
        )
        client.execute()
        target = args.output_dir / f"{path.stem}.executed.ipynb"
        nbformat.write(book, target)
        print(f"Executed: {path.name}", flush=True)


if __name__ == "__main__":
    main()
