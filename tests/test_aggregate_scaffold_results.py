import csv
import importlib.util
from pathlib import Path


SCRIPT = (
    Path(__file__).parents[1]
    / "scripts"
    / "data-val"
    / "aggregate_scaffold_results.py"
)
SPEC = importlib.util.spec_from_file_location("aggregate_scaffold_results", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _write_tags(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    all_tags = ",".join(tag for tag, _ in MODULE.TAG_CATEGORIES)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["SMILES", "Tags"])
        for smiles in rows:
            writer.writerow([smiles, all_tags])


def test_category_analysis_uses_unique_outer_fold_predictions(tmp_path):
    tag_dir = tmp_path / "data" / "MMF-3" / "tags"
    _write_tags(tag_dir / "pos_compound_tags.csv", ["CCO", "CCN"])
    _write_tags(tag_dir / "neg_compound_tags.csv", ["CCC", "CCCl", "missing"])
    predictions = [
        {"kind": "cv", "mode": "pos", "fold": 1, "smiles": "CCO", "correct": 1.0, "predict": 1.0},
        {"kind": "cv", "mode": "pos", "fold": 2, "smiles": "CCN", "correct": 2.0, "predict": 2.0},
        {"kind": "cv", "mode": "neg", "fold": 1, "smiles": "CCC", "correct": 3.0, "predict": 3.0},
        {"kind": "cv", "mode": "neg", "fold": 2, "smiles": "CCCl", "correct": 4.0, "predict": 4.0},
    ]

    metrics, details, coverage = MODULE._category_analysis(tmp_path, predictions)

    assert len(metrics) == 20
    assert all(row["n"] == 2 for row in metrics)
    assert all(row["r2"] == 1.0 for row in metrics)
    assert len(details) == 40
    assert coverage[0]["tags_without_prediction_count"] == 0
    assert coverage[1]["tags_without_prediction_count"] == 1
    assert coverage[1]["tags_without_prediction_smiles"] == "missing"

