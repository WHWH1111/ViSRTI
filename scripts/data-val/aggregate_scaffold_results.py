#!/usr/bin/env python3
"""Aggregate scaffold runs, out-of-fold category metrics, and Figure 5."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


METRICS = ("mae", "rmse", "r2", "pcc")
TAG_CATEGORIES: Tuple[Tuple[str, str], ...] = (
    ("Aliphatic", "Aliphatic"),
    ("Aromatic", "Aromatic"),
    ("Halogen_containing", "Halogenated"),
    ("Oxygen_containing", "Oxygen-containing"),
    ("Nitrogen_containing", "Nitrogen-containing"),
    ("Phosphorus_containing", "Organophosphorus"),
    ("Sulfur_containing", "Organosulfur"),
    ("Hydrophilic", "Hydrophilic"),
    ("Medium_hydrophobic", "Moderately hydrophobic"),
    ("High_hydrophobic", "Highly hydrophobic"),
)


def _read_key_values(path: Path) -> Dict[str, float]:
    values: Dict[str, float] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        try:
            values[key.strip()] = float(value.strip())
        except ValueError:
            continue
    return values


def _read_predictions(path: Path) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        required = {"Smiles", "Correct", "Predict"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError("Unexpected prediction columns in {}".format(path))
        for row in reader:
            correct = float(row["Correct"])
            predict = float(row["Predict"])
            if not (math.isfinite(correct) and math.isfinite(predict)):
                raise ValueError("Non-finite prediction in {}".format(path))
            rows.append({
                "smiles": row["Smiles"],
                "correct": correct,
                "predict": predict,
            })
    if not rows:
        raise ValueError("No predictions in {}".format(path))
    return rows


def _calculate_metrics(predictions: Iterable[Dict[str, object]]) -> Dict[str, float]:
    rows = list(predictions)
    y_true = np.asarray([float(row["correct"]) for row in rows], dtype=float)
    y_pred = np.asarray([float(row["predict"]) for row in rows], dtype=float)
    residual = y_pred - y_true
    denominator = float(np.sum((y_true - np.mean(y_true)) ** 2))
    return {
        "mae": float(np.mean(np.abs(residual))),
        "rmse": float(np.sqrt(np.mean(residual ** 2))),
        "r2": float(1.0 - np.sum(residual ** 2) / denominator),
        "pcc": float(np.corrcoef(y_true, y_pred)[0, 1]),
    }


def _read_training_progress(path: Path) -> Dict[str, object]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    if not rows:
        raise ValueError("No epoch records in {}".format(path))
    valid = [row for row in rows if math.isfinite(float(row["MAE_dev"]))]
    selected = min(valid, key=lambda row: float(row["MAE_dev"]))
    return {
        "epochs_completed": len(rows),
        "selected_epoch": int(selected["Epoch"]),
        "selected_dev_mae": float(selected["MAE_dev"]),
        "training_seconds": float(rows[-1]["Time(sec)"]),
    }


def _expected_counts(repo_root: Path) -> Dict[str, Dict[str, object]]:
    output: Dict[str, Dict[str, object]] = {}
    for mode in ("pos", "neg"):
        path = repo_root / "data" / "MMF-3" / ("MMF_GNN_" + mode) / "split_verification.json"
        output[mode] = json.loads(path.read_text(encoding="utf-8"))
    return output


def _write_csv(path: Path, rows: List[Dict[str, object]], fieldnames: List[str]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _read_tags(path: Path) -> Dict[str, Tuple[str, ...]]:
    tags_by_smiles: Dict[str, Tuple[str, ...]] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not {"SMILES", "Tags"}.issubset(reader.fieldnames or []):
            raise ValueError("Unexpected tag columns in {}".format(path))
        for row in reader:
            smiles = row["SMILES"]
            if smiles in tags_by_smiles:
                raise ValueError("Duplicate SMILES in {}: {}".format(path, smiles))
            tags_by_smiles[smiles] = tuple(
                tag.strip() for tag in row["Tags"].split(",") if tag.strip()
            )
    return tags_by_smiles


def _category_analysis(
    repo_root: Path,
    prediction_rows: Sequence[Dict[str, object]],
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]], List[Dict[str, object]]]:
    category_rows: List[Dict[str, object]] = []
    detail_rows: List[Dict[str, object]] = []
    coverage_rows: List[Dict[str, object]] = []
    for mode in ("pos", "neg"):
        predictions = [
            row for row in prediction_rows
            if row["kind"] == "cv" and row["mode"] == mode
        ]
        prediction_by_smiles: Dict[str, Dict[str, object]] = {}
        for row in predictions:
            smiles = str(row["smiles"])
            if smiles in prediction_by_smiles:
                raise ValueError(
                    "OOF prediction is not unique for {} mode: {}".format(mode, smiles)
                )
            prediction_by_smiles[smiles] = row
        tag_path = repo_root / "data" / "MMF-3" / "tags" / (mode + "_compound_tags.csv")
        tags_by_smiles = _read_tags(tag_path)
        prediction_smiles = set(prediction_by_smiles)
        tag_smiles = set(tags_by_smiles)
        missing_predictions = sorted(tag_smiles - prediction_smiles)
        missing_tags = sorted(prediction_smiles - tag_smiles)
        coverage_rows.append({
            "mode": mode,
            "tag_record_count": len(tags_by_smiles),
            "oof_prediction_count": len(prediction_by_smiles),
            "tags_without_prediction_count": len(missing_predictions),
            "predictions_without_tags_count": len(missing_tags),
            "tags_without_prediction_smiles": " | ".join(missing_predictions),
            "predictions_without_tags_smiles": " | ".join(missing_tags),
        })
        if missing_tags:
            raise ValueError(
                "OOF predictions without category tags for {} mode: {}".format(
                    mode, len(missing_tags)
                )
            )
        for tag, label in TAG_CATEGORIES:
            members = [
                prediction_by_smiles[smiles]
                for smiles, tags in tags_by_smiles.items()
                if tag in tags and smiles in prediction_by_smiles
            ]
            if len(members) < 2:
                raise ValueError("Too few records for {} / {}".format(mode, tag))
            metrics = _calculate_metrics(members)
            category_rows.append({
                "mode": mode,
                "category": tag,
                "category_label": label,
                "n": len(members),
                **metrics,
            })
            for row in members:
                detail_rows.append({
                    "mode": mode,
                    "fold": row["fold"],
                    "category": tag,
                    "category_label": label,
                    "smiles": row["smiles"],
                    "source_rti": row["correct"],
                    "predicted_rti": row["predict"],
                    "residual_source_minus_prediction": (
                        float(row["correct"]) - float(row["predict"])
                    ),
                })
    return category_rows, detail_rows, coverage_rows


def _set_figure_style() -> None:
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans", "sans-serif"],
        "font.size": 7,
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "legend.frameon": False,
    })


def _save_figure(
    fig: mpl.figure.Figure,
    base: Path,
    *,
    raster_dpi: int = 600,
    tight: bool = True,
) -> List[str]:
    outputs: List[str] = []
    for suffix, options in (
        ("svg", {}),
        ("pdf", {}),
        ("png", {"dpi": raster_dpi}),
        ("tiff", {"dpi": raster_dpi, "pil_kwargs": {"compression": "tiff_lzw"}}),
    ):
        path = base.with_suffix("." + suffix)
        save_options = dict(options)
        if tight:
            save_options["bbox_inches"] = "tight"
        fig.savefig(path, **save_options)
        outputs.append(str(path))
    plt.close(fig)
    return outputs


def _draw_fold_summary(rows: List[Dict[str, object]], output_dir: Path) -> List[str]:
    _set_figure_style()
    colors = {"mae": "#4C78A8", "rmse": "#9ECAE1", "r2": "#8C6BB1", "pcc": "#59A14F"}
    fig, axes = plt.subplots(1, 2, figsize=(183 / 25.4, 82 / 25.4), sharex=True)
    for panel, (mode, title) in enumerate((("pos", "Positive ESI"), ("neg", "Negative ESI"))):
        ax = axes[panel]
        ax2 = ax.twinx()
        subset = sorted((row for row in rows if row["mode"] == mode), key=lambda row: int(row["fold"]))
        folds = [int(row["fold"]) for row in subset]
        for metric, marker in (("mae", "o"), ("rmse", "s")):
            ax.plot(folds, [float(row[metric]) for row in subset], marker=marker,
                    color=colors[metric], linewidth=1.2, markersize=3.5, label=metric.upper())
        for metric, marker in (("r2", "^"), ("pcc", "D")):
            ax2.plot(folds, [float(row[metric]) for row in subset], marker=marker,
                     color=colors[metric], linewidth=1.2, markersize=3.5,
                     linestyle="--", label="R²" if metric == "r2" else "PCC")
        agreement = [float(row[metric]) for row in subset for metric in ("r2", "pcc")]
        ax2.set_ylim(min(-0.1, min(agreement) - 0.05), 1.02)
        ax.set_xticks(folds)
        ax.set_xlabel("Outer fold")
        ax.set_ylabel("Error (RTI units)")
        ax2.set_ylabel("Agreement")
        ax.set_title("{}  {}".format("ab"[panel], title), loc="left", fontweight="bold", pad=28)
        handles1, labels1 = ax.get_legend_handles_labels()
        handles2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(handles1 + handles2, labels1 + labels2, ncol=4, loc="lower center",
                  bbox_to_anchor=(0.5, 1.0), fontsize=6, handlelength=1.5, columnspacing=0.8)
    fig.tight_layout(w_pad=2.0)
    return _save_figure(fig, output_dir / "scaffold_cv_fold_metrics")


def _draw_category_panel(
    mode: str,
    all_predictions: Sequence[Dict[str, object]],
    category_rows: Sequence[Dict[str, object]],
    tags_by_smiles: Mapping[str, Tuple[str, ...]],
    output_dir: Path,
    *,
    figure_number: str = "6",
    large_typography: bool = False,
) -> List[str]:
    # Figure 5 follows the original author's visual grammar:
    # two separate 4 x 3 panels, Times-style serif text, boxed axes, grey
    # overall points, pink category points, and the same labels and legends.
    mpl.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "font.size": 10,
        "axes.linewidth": 0.8,
        "axes.spines.top": True,
        "axes.spines.right": True,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "legend.frameon": False,
    })
    mode_predictions = [row for row in all_predictions if row["mode"] == mode and row["kind"] == "cv"]
    metric_by_category = {
        str(row["category"]): row for row in category_rows if row["mode"] == mode
    }
    all_x = np.asarray([float(row["predict"]) for row in mode_predictions])
    all_y = np.asarray([float(row["correct"]) - float(row["predict"]) for row in mode_predictions])
    if large_typography:
        fig, axes = plt.subplots(5, 2, figsize=(10, 12.5), sharex=True, sharey=True)
        base_font_size = 16
        tick_font_size = 13
        annotation_font_size = 13
        legend_font_size = 11
        axis_font_size = 15
    else:
        fig, axes = plt.subplots(3, 4, figsize=(20, 13), sharex=True, sharey=True)
        base_font_size = 10
        tick_font_size = 9
        annotation_font_size = 10
        legend_font_size = 9
        axis_font_size = 10
    mpl.rcParams["font.size"] = base_font_size
    overall_style = {
        "s": 18,
        "facecolors": "#D9D9D9",
        "edgecolors": "#AFAFAF",
        "linewidths": 0.45,
        "alpha": 0.72,
    }
    category_style = {
        "s": 18,
        "facecolors": "#FFC0C0",
        "edgecolors": "#FF7777",
        "linewidths": 0.45,
        "alpha": 0.78,
    }
    for index, (tag, label) in enumerate(TAG_CATEGORIES):
        ax = axes.flat[index]
        members = [row for row in mode_predictions if tag in tags_by_smiles[str(row["smiles"])]]
        x = np.asarray([float(row["predict"]) for row in members])
        y = np.asarray([float(row["correct"]) - float(row["predict"]) for row in members])
        legend_label = "{} compounds".format(label)
        if large_typography and len(legend_label) > 24:
            legend_label = legend_label.replace(" compounds", "\ncompounds")
        ax.scatter(all_x, all_y, rasterized=True, label="Overall", **overall_style)
        ax.scatter(
            x,
            y,
            rasterized=True,
            label=legend_label,
            **category_style,
        )
        ax.axhline(0.0, color="#8E8E8E", linewidth=0.75, linestyle="--", zorder=0)
        ax.set_xlim(-100, 1050)
        ax.set_ylim(-550, 550)
        ax.set_xticks([0, 200, 400, 600, 800, 1000])
        ax.set_yticks([-400, -200, 0, 200, 400])
        ax.tick_params(
            direction="out",
            length=4,
            width=0.8,
            labelsize=tick_font_size,
            labelbottom=True,
            labelleft=True,
        )
        metrics = metric_by_category[tag]
        ax.text(
            0.065,
            0.975,
            "Sample size = {n}\nR² = {r2:.4f}\nMAE = {mae:.4f}".format(**metrics),
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=annotation_font_size,
            linespacing=1.25,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.9, "pad": 1.0},
        )
        ax.legend(
            loc="upper right",
            frameon=True,
            facecolor="white",
            edgecolor="none",
            framealpha=0.9,
            borderaxespad=0.35,
            handletextpad=0.5,
            labelspacing=0.35,
            fontsize=legend_font_size,
            markerscale=1.25,
        )
        ax.set_xlabel("Predicted Values", fontsize=axis_font_size)
        ax.set_ylabel("Residuals", fontsize=axis_font_size)
    if not large_typography:
        for blank_index in (10, 11):
            axes.flat[blank_index].set_visible(False)
    panel_label = "a" if mode == "pos" else "b"
    fig.text(
        0.034,
        0.985,
        "({}) {}-RTI".format(panel_label, mode),
        ha="left",
        va="top",
        fontsize=20 if not large_typography else 22,
    )
    if large_typography:
        fig.subplots_adjust(left=0.09, right=0.98, bottom=0.06, top=0.96, wspace=0.20, hspace=0.30)
    else:
        fig.subplots_adjust(left=0.064, right=0.947, bottom=0.065, top=0.948, wspace=0.19, hspace=0.24)
    return _save_figure(
        fig,
        output_dir / ("Figure{}{}_scaffold_oof_categories_{}".format(figure_number, panel_label, mode)),
        raster_dpi=600 if large_typography else 200,
        tight=False,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()

    run_root = args.run_root.resolve()
    manifest_path = run_root / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    repo_root = Path(manifest["repo_root"])
    expected = _expected_counts(repo_root)
    figure_number = str(manifest.get("figure_number", "6"))
    figure_prefix = "figure" + figure_number
    large_typography = bool(manifest.get("large_typography", False))
    model_labels = manifest.get("model_labels", {
        "pos": "ViSRTI-GPT with KOC masked",
        "neg": "ViSRTI-GPT with KOC masked",
    })

    metric_rows: List[Dict[str, object]] = []
    prediction_rows: List[Dict[str, object]] = []
    inventory_rows: List[Dict[str, object]] = []
    for task in manifest["tasks"]:
        if task.get("status") != "completed":
            continue
        log_dir = Path(task["log_dir"])
        prediction_file = log_dir / "train_prediction_test.txt"
        stored_file = log_dir / "test_prediction.txt"
        progress_file = log_dir / "MAEs.txt"
        predictions = _read_predictions(prediction_file)
        calculated = _calculate_metrics(predictions)
        stored = _read_key_values(stored_file)
        progress = _read_training_progress(progress_file)
        mode = task["mode"]
        fold = task.get("fold")
        if task["kind"] == "cv":
            split_record = expected[mode]["cross_validation"][int(fold) - 1]
            expected_count = int(split_record["test_count"])
            overlap_count = int(split_record["scaffold_overlap_count"])
        else:
            expected_count = int(expected[mode]["holdout"]["counts"]["test"])
            overlap_count = int(expected[mode]["holdout"]["overlap_counts"]["train_test"])
        row: Dict[str, object] = {
            "task_id": task["id"],
            "kind": task["kind"],
            "mode": mode,
            "fold": fold if fold is not None else "",
            "split_seed": 1234,
            "expected_test_count": expected_count,
            "predicted_test_count": len(predictions),
            "excluded_or_failed_count": expected_count - len(predictions),
            "scaffold_overlap_count": overlap_count,
            **calculated,
            **progress,
            "log_dir": str(log_dir),
        }
        stored_values = {
            "mae": stored.get("MAE_test"),
            "rmse": math.sqrt(stored["MSE_test"]) if "MSE_test" in stored else None,
            "r2": stored.get("R2_test"),
            "pcc": stored.get("PCC_test"),
        }
        for metric in METRICS:
            value = stored_values[metric]
            row["stored_" + metric] = value
            row["abs_delta_" + metric] = abs(calculated[metric] - value) if value is not None else ""
        metric_rows.append(row)
        for prediction in predictions:
            prediction_rows.append({
                "task_id": task["id"],
                "kind": task["kind"],
                "mode": mode,
                "fold": fold if fold is not None else "",
                **prediction,
                "residual": float(prediction["predict"]) - float(prediction["correct"]),
            })
        for artifact in sorted(log_dir.iterdir()):
            if artifact.is_file():
                inventory_rows.append({
                    "task_id": task["id"],
                    "artifact": artifact.name,
                    "size_bytes": artifact.stat().st_size,
                    "path": str(artifact),
                })

    expected_tasks = int(manifest.get("total_tasks", 12))
    if not args.allow_partial and len(metric_rows) != expected_tasks:
        raise SystemExit("Expected {} completed runs, found {}".format(expected_tasks, len(metric_rows)))
    metric_rows.sort(key=lambda row: (0 if row["kind"] == "cv" else 1, row["mode"], str(row["fold"])))
    metric_fields = [
        "task_id", "kind", "mode", "fold", "split_seed", "expected_test_count",
        "predicted_test_count", "excluded_or_failed_count", "scaffold_overlap_count",
        "mae", "rmse", "r2", "pcc", "stored_mae", "stored_rmse", "stored_r2",
        "stored_pcc", "abs_delta_mae", "abs_delta_rmse", "abs_delta_r2",
        "abs_delta_pcc", "epochs_completed", "selected_epoch", "selected_dev_mae",
        "training_seconds", "log_dir",
    ]
    _write_csv(run_root / "all_run_metrics.csv", metric_rows, metric_fields)
    _write_csv(run_root / "all_test_predictions.csv", prediction_rows,
               ["task_id", "kind", "mode", "fold", "smiles", "correct", "predict", "residual"])
    _write_csv(run_root / "artifact_inventory.csv", inventory_rows,
               ["task_id", "artifact", "size_bytes", "path"])

    cv_rows = [row for row in metric_rows if row["kind"] == "cv"]
    _write_csv(run_root / (figure_prefix + "_source_data.csv"), cv_rows, metric_fields)
    summary_rows: List[Dict[str, object]] = []
    for mode in ("pos", "neg"):
        subset = [row for row in cv_rows if row["mode"] == mode]
        if not subset:
            continue
        record: Dict[str, object] = {"mode": mode, "folds": len(subset)}
        for metric in METRICS:
            values = [float(row[metric]) for row in subset]
            record[metric + "_mean"] = statistics.mean(values)
            record[metric + "_sd"] = statistics.stdev(values) if len(values) > 1 else 0.0
        record["total_predictions"] = sum(int(row["predicted_test_count"]) for row in subset)
        record["total_expected"] = sum(int(row["expected_test_count"]) for row in subset)
        summary_rows.append(record)
    summary_fields = ["mode", "folds", "total_expected", "total_predictions"] + [
        name for metric in METRICS for name in (metric + "_mean", metric + "_sd")
    ]
    _write_csv(run_root / (figure_prefix + "_summary.csv"), summary_rows, summary_fields)

    figure_outputs: List[str] = []
    if len(cv_rows) == 10:
        figure_outputs.extend(_draw_fold_summary(cv_rows, run_root))
        category_rows, category_prediction_rows, coverage_rows = _category_analysis(
            repo_root, prediction_rows
        )
        _write_csv(
            run_root / (figure_prefix + "_category_metrics.csv"),
            category_rows,
            ["mode", "category", "category_label", "n", "mae", "rmse", "r2", "pcc"],
        )
        _write_csv(
            run_root / (figure_prefix + "_category_predictions.csv"),
            category_prediction_rows,
            [
                "mode", "fold", "category", "category_label", "smiles", "source_rti",
                "predicted_rti", "residual_source_minus_prediction",
            ],
        )
        _write_csv(
            run_root / (figure_prefix + "_category_coverage.csv"),
            coverage_rows,
            [
                "mode", "tag_record_count", "oof_prediction_count",
                "tags_without_prediction_count", "predictions_without_tags_count",
                "tags_without_prediction_smiles", "predictions_without_tags_smiles",
            ],
        )
        for mode in ("pos", "neg"):
            tags_by_smiles = _read_tags(
                repo_root / "data" / "MMF-3" / "tags" / (mode + "_compound_tags.csv")
            )
            figure_outputs.extend(
                _draw_category_panel(
                    mode,
                    prediction_rows,
                    category_rows,
                    tags_by_smiles,
                    run_root,
                    figure_number=figure_number,
                    large_typography=large_typography,
                )
            )
        pos_counts = [str(row["predicted_test_count"]) for row in cv_rows if row["mode"] == "pos"]
        neg_counts = [str(row["predicted_test_count"]) for row in cv_rows if row["mode"] == "neg"]
        pos_coverage = next(row for row in coverage_rows if row["mode"] == "pos")
        neg_coverage = next(row for row in coverage_rows if row["mode"] == "neg")
        omitted = []
        for label, coverage in (("positive", pos_coverage), ("negative", neg_coverage)):
            count = int(coverage["tags_without_prediction_count"])
            if count:
                omitted.append(
                    "{} {}-mode compound{} omitted after preprocessing failed".format(
                        count, label, " was" if count == 1 else "s were"
                    )
                )
        coverage_text = "; ".join(omitted) if omitted else "all tagged compounds yielded predictions"
        caption = (
            "Figure {} | Category-stratified agreement with NORMAN-predicted RTI labels "
            "from scaffold-disjoint five-fold cross-validation of fixed candidate "
            "models with development-set masking ({} for positive ESI and {} for negative ESI). (a) Positive ESI and "
            "(b) negative ESI. Each molecule was predicted "
            "only when assigned to the outer test fold, whose training data contained no "
            "shared Bemis-Murcko scaffold. Grey points show all out-of-fold predictions "
            "for the ionization mode and red points show the indicated, non-exclusive "
            "chemical or hydrophobicity category. Residuals are source RTI minus predicted "
            "RTI. The positive-mode fold sizes were {}; the negative-mode fold sizes were "
            "{}; {}."
        ).format(
            figure_number,
            model_labels["pos"],
            model_labels["neg"],
            ", ".join(pos_counts),
            ", ".join(neg_counts),
            coverage_text,
        )
        (run_root / ("Figure" + figure_number + "_caption.txt")).write_text(
            caption + "\n", encoding="utf-8"
        )

    max_delta = max(
        float(row["abs_delta_" + metric])
        for row in metric_rows for metric in METRICS
        if row["abs_delta_" + metric] != ""
    ) if metric_rows else None
    report = {
        "run_count": len(metric_rows),
        "cv_run_count": len(cv_rows),
        "prediction_count": len(prediction_rows),
        "max_metric_recalculation_delta": max_delta,
        "all_scaffold_overlaps_zero": all(int(row["scaffold_overlap_count"]) == 0 for row in metric_rows),
        "figure_outputs": figure_outputs,
        "source_files": [
            "all_run_metrics.csv", "all_test_predictions.csv", "artifact_inventory.csv",
            figure_prefix + "_source_data.csv", figure_prefix + "_summary.csv",
            figure_prefix + "_category_metrics.csv", figure_prefix + "_category_predictions.csv",
            figure_prefix + "_category_coverage.csv", "Figure" + figure_number + "_caption.txt",
        ],
    }
    (run_root / "aggregation_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
