#!/usr/bin/env python3
"""Validate and aggregate the eight G/GP/GPT/GPTC random-split runs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np


FEATURE_LEVELS: Tuple[Tuple[str, str], ...] = (
    ("G", "graph"),
    ("GP", "graph_physchem"),
    ("GPT", "graph_physchem_toxicity"),
    ("GPTC", "all"),
)
METRICS = ("mae", "rmse", "r2", "pcc")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    residual = y_pred - y_true
    denominator = float(np.sum((y_true - np.mean(y_true)) ** 2))
    return {
        "mae": float(np.mean(np.abs(residual))),
        "rmse": float(np.sqrt(np.mean(residual ** 2))),
        "r2": float(1.0 - np.sum(residual ** 2) / denominator),
        "pcc": float(np.corrcoef(y_true, y_pred)[0, 1]),
    }


def _read_predictions(path: Path) -> List[Dict[str, object]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError("No predictions in {}".format(path))
    return rows


def _write_csv(path: Path, rows: Iterable[Dict[str, object]], fields: Sequence[str]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _bootstrap(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    seed: int,
    iterations: int,
) -> Tuple[List[Dict[str, object]], Dict[str, float], int]:
    sample_size = min(100, max(10, int(len(y_true) * 0.10)))
    rng = np.random.RandomState(seed)
    rows: List[Dict[str, object]] = []
    attempts = 0
    while len(rows) < iterations:
        attempts += 1
        if attempts > iterations * 20:
            raise ValueError("Could not draw enough non-degenerate bootstrap samples")
        selected = rng.randint(0, len(y_true), size=sample_size)
        if np.ptp(y_true[selected]) == 0 or np.ptp(y_pred[selected]) == 0:
            continue
        values = _metrics(y_true[selected], y_pred[selected])
        if not all(np.isfinite(values[metric]) for metric in METRICS):
            continue
        rows.append({"bootstrap_iteration": len(rows) + 1, **values})
    summary = {
        metric + "_bootstrap_sd": float(np.std([row[metric] for row in rows], ddof=1))
        for metric in METRICS
    }
    return rows, summary, sample_size


def aggregate(run_root: Path, bootstrap_iterations: int) -> Dict[str, object]:
    manifest = json.loads((run_root / "run_manifest.json").read_text(encoding="utf-8-sig"))
    metric_rows: List[Dict[str, object]] = []
    bootstrap_rows: List[Dict[str, object]] = []
    prediction_rows: List[Dict[str, object]] = []
    inventory_rows: List[Dict[str, object]] = []
    required = (
        "experiment_config.json", "split_manifest.json", "partition_assignments.csv",
        "best.pt", "model.pt", "predictions_test.csv", "metrics.json",
    )

    completed = [task for task in manifest["tasks"] if task.get("status") == "completed"]
    if len(completed) != 8:
        raise ValueError("Expected 8 completed experiments, found {}".format(len(completed)))

    for task in completed:
        experiment_dir = Path(task["output_dir"])
        for name in required:
            if not (experiment_dir / name).is_file():
                raise FileNotFoundError("{} is missing {}".format(task["id"], name))
        predictions = _read_predictions(experiment_dir / "predictions_test.csv")
        with (experiment_dir / "partition_assignments.csv").open(
                "r", encoding="utf-8", newline="") as handle:
            assignments = list(csv.DictReader(handle))
        test_assignment_count = sum(row["subset"] == "test" for row in assignments)
        if len(predictions) != test_assignment_count:
            raise ValueError(
                "{} prediction count {} != model-used test count {}".format(
                    task["id"], len(predictions), test_assignment_count
                )
            )
        y_true = np.asarray([float(row["source_rti"]) for row in predictions], dtype=float)
        y_pred = np.asarray([float(row["predicted_rti"]) for row in predictions], dtype=float)
        recalculated = _metrics(y_true, y_pred)
        stored = json.loads((experiment_dir / "metrics.json").read_text(encoding="utf-8"))
        max_delta = max(abs(recalculated[key] - float(stored[key])) for key in METRICS)
        # The training log stores full-precision metrics, while the prediction
        # text is rounded before CSV normalization. Allow only that small,
        # expected serialization difference.
        if max_delta > 1e-4:
            raise ValueError("Metric mismatch in {}: {}".format(task["id"], max_delta))

        boot, boot_summary, sample_size = _bootstrap(
            y_true, y_pred, int(manifest["split_seed"]), bootstrap_iterations
        )
        record = {
            "mode": task["mode"],
            "feature_code": task["feature_code"],
            "model": "ViSRTI-" + task["feature_code"],
            "feature_level": task["feature_level"],
            "n": len(predictions),
            **recalculated,
            **boot_summary,
            "bootstrap_iterations": bootstrap_iterations,
            "bootstrap_sample_size": sample_size,
            "experiment_dir": str(experiment_dir),
            "max_metric_recalculation_delta": max_delta,
        }
        metric_rows.append(record)
        for row in boot:
            bootstrap_rows.append({
                "mode": task["mode"],
                "feature_code": task["feature_code"],
                **row,
            })
        for row in predictions:
            prediction_rows.append({
                "mode": task["mode"],
                "feature_code": task["feature_code"],
                **row,
            })
        for artifact in sorted(experiment_dir.iterdir()):
            if artifact.is_file():
                inventory_rows.append({
                    "mode": task["mode"],
                    "feature_code": task["feature_code"],
                    "artifact": artifact.name,
                    "size_bytes": artifact.stat().st_size,
                    "sha256": _sha256(artifact),
                    "path": str(artifact),
                })

    order = {code: index for index, (code, _) in enumerate(FEATURE_LEVELS)}
    metric_rows.sort(key=lambda row: ((0 if row["mode"] == "pos" else 1), order[row["feature_code"]]))
    metric_fields = [
        "mode", "feature_code", "model", "feature_level", "n", *METRICS,
        *(metric + "_bootstrap_sd" for metric in METRICS),
        "bootstrap_iterations", "bootstrap_sample_size", "experiment_dir",
        "max_metric_recalculation_delta",
    ]
    _write_csv(run_root / "feature_fusion_metrics.csv", metric_rows, metric_fields)
    _write_csv(
        run_root / "feature_fusion_bootstrap_metrics.csv",
        bootstrap_rows,
        ["mode", "feature_code", "bootstrap_iteration", *METRICS],
    )
    _write_csv(
        run_root / "feature_fusion_predictions.csv",
        prediction_rows,
        [
            "mode", "feature_code", "smiles", "source_rti", "predicted_rti",
            "residual_source_minus_prediction",
        ],
    )
    _write_csv(
        run_root / "artifact_inventory.csv",
        inventory_rows,
        ["mode", "feature_code", "artifact", "size_bytes", "sha256", "path"],
    )
    plot_script = Path(manifest["repo_root"]) / "scripts" / "plt" / "plot_test_metrics_bar_line.py"
    subprocess.run(
        [
            sys.executable,
            str(plot_script),
            "--input",
            str(run_root / "feature_fusion_metrics.csv"),
            "--output-dir",
            str(run_root),
        ],
        check=True,
    )
    report = {
        "run_count": len(metric_rows),
        "prediction_count": len(prediction_rows),
        "bootstrap_iterations_per_experiment": bootstrap_iterations,
        "all_required_artifacts_present": True,
        "maximum_metric_recalculation_delta": max(
            float(row["max_metric_recalculation_delta"]) for row in metric_rows
        ),
        "figures": [
            str(run_root / ("Figure2ef_feature_fusion." + suffix))
            for suffix in ("svg", "pdf", "png", "tiff")
        ],
    }
    (run_root / "aggregation_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--bootstrap-iterations", type=int, default=1000)
    args = parser.parse_args()
    report = aggregate(args.run_root.resolve(), args.bootstrap_iterations)
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
