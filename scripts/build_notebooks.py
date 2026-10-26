"""Build the main notebooks from the book's published code cells."""

import argparse
import json
import re
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = "565def0834d063281ff0776918ad829020919752"


def notes(chapter, section):
    source = (ROOT / "notebooks" / "notes" / f"{chapter}.md").read_text(encoding="utf-8")
    parts = re.split(r"(?m)^<!-- section: ([a-z_]+) -->\n", source)
    text = dict(zip(parts[1::2], parts[2::2], strict=True))[section]
    text = text.replace(
        "](../figures/",
        "](https://raw.githubusercontent.com/black-leg-nameko/"
        "security-ml-sakuraiLab/main/notebooks/figures/",
    )
    return nbf.v4.new_markdown_cell(text.strip())


def code(source):
    return nbf.v4.new_code_cell(source.rstrip())


def notebook(cells):
    book = nbf.v4.new_notebook(cells=cells)
    book.metadata = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.12"},
        "colab": {"provenance": []},
        "book_source": {
            "repository": "oreilly-japan/ml-security-jp",
            "revision": UPSTREAM,
            "path": "ch02/Chapter2.ipynb",
        },
    }
    for i, cell in enumerate(book.cells):
        cell.id = f"cell-{i:03d}"
    return book


def chapter_one():
    return notebook(
        [
            notes("01_environment", "introduction"),
            notes("01_environment", "environment"),
            code("!ls"),
            code("!pip install optuna"),
            notes("01_environment", "runtime"),
            notes("01_environment", "workflow"),
            notes("01_environment", "discussion"),
        ]
    )


def chapter_two():
    original = json.loads((ROOT / "notebooks/book/02_code.json").read_text(encoding="utf-8"))
    replacements = {
        0: [("!wget ", "!wget -O dataset.csv ")],
        1: [
            (
                "!pip install optuna",
                "!pip install optuna==4.5.0 optuna-integration[lightgbm]==4.5.0 lightgbm==4.6.0",
            )
        ],
        8: [("['auto', 'sqrt', 'log2']", "['sqrt', 'log2']")],
        10: [("!wget ", "!wget -O enron1.zip ")],
        11: [("!unzip -q ", "!unzip -oq ")],
        17: [
            (
                'TfidfVectorizer(stop_words="english", lowercase=False)',
                'TfidfVectorizer(stop_words="english")',
            ),
            ("tfidf.get_feature_names()", "tfidf.get_feature_names_out()"),
        ],
        21: [("verbose_eval=0,", "callbacks=[lgb.log_evaluation(0)],")],
    }
    introductions = {
        0: "data",
        2: "load",
        3: "baseline",
        4: "cv",
        5: "tuning",
        6: "evaluation",
        7: "metrics",
        8: "tree",
        10: "email",
        14: "read_email",
        17: "tfidf",
        19: "lightgbm",
        20: "tuner_score",
        21: "predict_email",
        22: "importance",
        23: "subject",
    }
    cells = [notes("02_phishing_and_spam", "introduction")]
    for index, source in enumerate(original):
        if index in introductions:
            cells.append(notes("02_phishing_and_spam", introductions[index]))
        for old, new in replacements.get(index, []):
            if source.count(old) != 1:
                raise ValueError(f"Book code changed: cell {index}, {old}")
            source = source.replace(old, new)
        cell = code(source)
        cell.metadata["book_code_cell"] = index
        cells.append(cell)
    cells.append(notes("02_phishing_and_spam", "discussion"))
    return notebook(cells)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    for name, book in (("01_environment", chapter_one()), ("02_phishing_and_spam", chapter_two())):
        path = ROOT / "notebooks" / f"{name}.ipynb"
        content = nbf.writes(book) + "\n"
        if args.check:
            if not path.exists() or path.read_text(encoding="utf-8") != content:
                raise SystemExit(f"Regenerate {path.relative_to(ROOT)}")
        else:
            path.write_text(content, encoding="utf-8")


if __name__ == "__main__":
    main()
