"""Execute each standalone notebook in a fresh kernel."""

import argparse
from pathlib import Path

import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=ROOT / "artifacts" / "executed")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for path in sorted((ROOT / "notebooks").glob("*.ipynb")):
        book = nbformat.read(path, as_version=4)
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
