#!/usr/bin/env python3
"""Validate baseline predictions, calculate four metrics, and reconcile K/M."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
from docx import Document
from openpyxl import load_workbook


MODEL_ORDER = ("GNN-RT", "GIN", "SMILES-BiLSTM", "ViSRTI-G")
METRICS = ("r2", "mae", "rmse", "pcc")


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError("No rows in {}".format(path))
    return rows


def write_csv(path: Path, rows: Iterable[Mapping[str, object]], fields: Sequence[str]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def calculate(actual: np.ndarray, predicted: np.ndarray) -> Dict[str, float]:
    residual = predicted - actual
    denominator = float(np.sum((actual - np.mean(actual)) ** 2))
    return {
        "r2": float(1.0 - np.sum(residual ** 2) / denominator),
        "mae": float(np.mean(np.abs(residual))),
        "rmse": float(np.sqrt(np.mean(residual ** 2))),
        "pcc": float(np.corrcoef(actual, predicted)[0, 1]),
    }


def bootstrap(actual: np.ndarray, predicted: np.ndarray, seed: int, iterations: int) -> List[Dict[str, float]]:
    sample_size = min(100, max(10, len(actual) // 10))
    rng = np.random.RandomState(seed)
    rows = []
    while len(rows) < iterations:
        selected = rng.randint(0, len(actual), size=sample_size)
        if np.ptp(actual[selected]) == 0 or np.ptp(predicted[selected]) == 0:
            continue
        result = calculate(actual[selected], predicted[selected])
        if all(np.isfinite(result[metric]) for metric in METRICS):
            rows.append(result)
    return rows


def workbook_means(path: Path, source: str) -> Dict[Tuple[str, str], Tuple[float, float]]:
    workbook = load_workbook(path, read_only=True, data_only=True)
    sheet = workbook[workbook.sheetnames[2]]
    values = list(sheet.values)[3:]
    columns = list(zip(*values))
    if source == "K":
        mappings = {("pos", "GIN"): (1, 3), ("neg", "GIN"): (5, 7)}
    else:
        mappings = {
            ("pos", "GNN-RT"): (0, 2),
            ("neg", "GNN-RT"): (4, 6),
            ("pos", "ChemBERTa-LSTM"): (1, 3),
            ("neg", "ChemBERTa-LSTM"): (5, 7),
        }
    result = {}
    for key, (r2_column, mae_column) in mappings.items():
        r2_values = [float(value) for value in columns[r2_column] if value is not None]
        mae_values = [float(value) for value in columns[mae_column] if value is not None]
        result[key] = (float(np.mean(r2_values)), float(np.mean(mae_values)))
    return result


def supplementary_values(path: Path) -> Dict[Tuple[str, str], Tuple[float, float]]:
    table = Document(path).tables[0]
    result = {}
    for row in table.rows[1:]:
        cells = [cell.text.strip() for cell in row.cells]
        if len(cells) >= 4 and cells[0] in ("pos", "neg") and cells[1] and cells[2] and cells[3]:
            result[(cells[0], cells[1])] = (float(cells[2]), float(cells[3]))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--feature-root", type=Path, required=True)
    parser.add_argument("--k-workbook", type=Path, required=True)
    parser.add_argument("--m-workbook", type=Path, required=True)
    parser.add_argument("--supplementary", type=Path, required=True)
    parser.add_argument("--bootstrap-iterations", type=int, default=1000)
    args = parser.parse_args()

    run_root, feature_root = args.run_root.resolve(), args.feature_root.resolve()
    manifest = json.loads((run_root / "run_manifest.json").read_text(encoding="utf-8-sig"))
    completed = [task for task in manifest["tasks"] if task["status"] == "completed"]
    if len(completed) != 6:
        raise ValueError("Expected six completed baseline tasks; found {}".format(len(completed)))

    prediction_sets: Dict[Tuple[str, str], List[Dict[str, str]]] = {}
    qualifications: Dict[str, str] = {}
    for task in completed:
        experiment = Path(task["output_dir"])
        for required in ("best.pt", "model.pt", "experiment_config.json", "split_manifest.json", "metrics.json", "predictions_test.csv"):
            if not (experiment / required).is_file():
                raise FileNotFoundError("{} is missing {}".format(task["id"], required))
        config = json.loads((experiment / "experiment_config.json").read_text(encoding="utf-8"))
        label = config["model_label"]
        qualifications[label] = config["model_qualification"]
        prediction_sets[(task["mode"], label)] = read_csv(experiment / "predictions_test.csv")
    qualifications["ViSRTI-G"] = "completed feature-fusion G run on the identical random partition"
    for mode in ("pos", "neg"):
        prediction_sets[(mode, "ViSRTI-G")] = read_csv(
            feature_root / "experiments" / ("G_" + mode) / "predictions_test.csv"
        )

    metric_rows, bootstrap_rows, combined_predictions = [], [], []
    maximum_actual_delta = 0.0
    for mode_index, mode in enumerate(("pos", "neg")):
        reference = prediction_sets[(mode, "ViSRTI-G")]
        if manifest.get("smoke"):
            smoke_count = len(prediction_sets[(mode, "GNN-RT")])
            reference = reference[:smoke_count]
            prediction_sets[(mode, "ViSRTI-G")] = reference
        reference_smiles = [row["smiles"] for row in reference]
        reference_actual = np.asarray([float(row["source_rti"]) for row in reference])
        for model_index, model in enumerate(MODEL_ORDER):
            rows = prediction_sets[(mode, model)]
            smiles = [row["smiles"] for row in rows]
            if smiles != reference_smiles:
                raise ValueError("Test-row identity/order mismatch for {} {}".format(mode, model))
            actual = np.asarray([float(row["source_rti"]) for row in rows])
            predicted = np.asarray([float(row["predicted_rti"]) for row in rows])
            maximum_actual_delta = max(maximum_actual_delta, float(np.max(np.abs(actual - reference_actual))))
            if maximum_actual_delta > 1e-3:
                raise ValueError("Test targets differ across models")
            calculated = calculate(actual, predicted)
            boot = bootstrap(actual, predicted, 1234 + 100 * mode_index + model_index, args.bootstrap_iterations)
            metric_rows.append({
                "mode": mode,
                "model": model,
                "n": len(rows),
                **calculated,
                **{metric + "_bootstrap_sd": float(np.std([row[metric] for row in boot], ddof=1)) for metric in METRICS},
                "qualification": qualifications[model],
            })
            for iteration, values in enumerate(boot, 1):
                bootstrap_rows.append({"mode": mode, "model": model, "bootstrap_iteration": iteration, **values})
            for row_id, row in enumerate(rows):
                combined_predictions.append({
                    "mode": mode,
                    "model": model,
                    "row_id": row.get("row_id", row_id),
                    "smiles": row["smiles"],
                    "source_rti": row["source_rti"],
                    "predicted_rti": row["predicted_rti"],
                })

    write_csv(run_root / "baseline_metrics.csv", metric_rows, ("mode", "model", "n", *METRICS, *(metric + "_bootstrap_sd" for metric in METRICS), "qualification"))
    write_csv(run_root / "baseline_bootstrap_metrics.csv", bootstrap_rows, ("mode", "model", "bootstrap_iteration", *METRICS))
    write_csv(run_root / "baseline_predictions.csv", combined_predictions, ("mode", "model", "row_id", "smiles", "source_rti", "predicted_rti"))

    historical = {}
    historical.update(workbook_means(args.k_workbook.resolve(), "K"))
    historical.update(workbook_means(args.m_workbook.resolve(), "M"))
    supplementary = supplementary_values(args.supplementary.resolve())
    reconciliation = []
    source_labels = {"GNN-RT": "GNN-RT", "GIN": "GIN", "SMILES-BiLSTM": "ChemBERTa-LSTM", "ViSRTI-G": "ViSRTI-G"}
    fresh_lookup = {(row["mode"], row["model"]): row for row in metric_rows}
    for mode in ("pos", "neg"):
        for model in MODEL_ORDER:
            source_label = source_labels[model]
            old = historical.get((mode, source_label))
            supplement_label = "RoBERTa-BiLSTM" if model == "SMILES-BiLSTM" else model
            si = supplementary.get((mode, supplement_label))
            fresh = fresh_lookup[(mode, model)]
            if model == "SMILES-BiLSTM":
                determination = "Historical labels conflict (M: ChemBERTa-LSTM; SI: RoBERTa-BiLSTM); fresh local sequence baseline is not treated as the same pretrained model."
            elif old and si:
                determination = "Historical workbook and SI compared; fresh prediction-derived metrics are authoritative for the rebuilt table."
            else:
                determination = "No raw historical predictions; fresh prediction-derived metrics are authoritative."
            reconciliation.append({
                "mode": mode,
                "fresh_model": model,
                "historical_model_label": source_label if old else "",
                "historical_r2_mean": old[0] if old else "",
                "historical_mae_mean": old[1] if old else "",
                "supplementary_model_label": supplement_label if si else "",
                "supplementary_r2": si[0] if si else "",
                "supplementary_mae": si[1] if si else "",
                "historical_minus_supplementary_r2": old[0] - si[0] if old and si else "",
                "historical_minus_supplementary_mae": old[1] - si[1] if old and si else "",
                "fresh_r2": fresh["r2"],
                "fresh_mae": fresh["mae"],
                "fresh_rmse": fresh["rmse"],
                "fresh_pcc": fresh["pcc"],
                "determination": determination,
            })
    write_csv(
        run_root / "source_metric_reconciliation.csv",
        reconciliation,
        ("mode", "fresh_model", "historical_model_label", "historical_r2_mean", "historical_mae_mean", "supplementary_model_label", "supplementary_r2", "supplementary_mae", "historical_minus_supplementary_r2", "historical_minus_supplementary_mae", "fresh_r2", "fresh_mae", "fresh_rmse", "fresh_pcc", "determination"),
    )
    plot_script = Path(manifest["repo_root"]) / "scripts" / "plt" / "plot_baseline_violin.py"
    subprocess.run(
        [sys.executable, str(plot_script), "--input", str(run_root / "baseline_bootstrap_metrics.csv"), "--output-stem", str(run_root / "Figure2ad_baseline")],
        check=True,
    )
    report = {
        "status": "completed",
        "model_mode_runs": len(metric_rows),
        "fresh_prediction_rows": len(combined_predictions),
        "bootstrap_iterations_per_model_mode": args.bootstrap_iterations,
        "maximum_cross_model_target_delta": maximum_actual_delta,
        "same_test_rows_and_order": True,
        "historical_limit": "K/M contain bootstrap R2 and MAE only; RMSE and PCC cannot be recovered without predictions.",
        "sequence_identity_resolution": "M names ChemBERTa-LSTM, SI names RoBERTa-BiLSTM, and neither original weights nor raw predictions were supplied. The fresh model is therefore labelled SMILES-BiLSTM.",
    }
    (run_root / "aggregation_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
