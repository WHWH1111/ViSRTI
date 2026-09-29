#!/usr/bin/env python3
"""Build fixed-split evidence and figures for development-selected models."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


METRICS = ("mae", "rmse", "r2", "pcc")
CATEGORIES: Tuple[Tuple[str, str], ...] = (
    ("Sulfur_containing", "Organosulfur compounds"),
    ("Phosphorus_containing", "Organophosphorus compounds"),
    ("Oxygen_containing", "Oxygen-containing compounds"),
    ("Nitrogen_containing", "Nitrogen-containing compounds"),
    ("Medium_hydrophobic", "Moderately hydrophobic compounds"),
    ("Hydrophilic", "Hydrophilic compounds"),
    ("High_hydrophobic", "Highly hydrophobic compounds"),
    ("Halogen_containing", "Halogenated compounds"),
    ("Aromatic", "Aromatic compounds"),
    ("Aliphatic", "Aliphatic compounds"),
)


def _read_tsv_predictions(path: Path) -> List[Dict[str, object]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if not {"Smiles", "Correct", "Predict"}.issubset(reader.fieldnames or []):
            raise ValueError("Unexpected prediction columns in {}".format(path))
        rows = [
            {
                "smiles": row["Smiles"],
                "source_rti": float(row["Correct"]),
                "predicted_rti": float(row["Predict"]),
            }
            for row in reader
        ]
    if not rows:
        raise ValueError("No predictions in {}".format(path))
    for row in rows:
        row["residual_source_minus_prediction"] = (
            float(row["source_rti"]) - float(row["predicted_rti"])
        )
    return rows


def _metrics(rows: Iterable[Mapping[str, object]]) -> Dict[str, float]:
    values = list(rows)
    y_true = np.asarray([float(row["source_rti"]) for row in values], dtype=float)
    y_pred = np.asarray([float(row["predicted_rti"]) for row in values], dtype=float)
    residual = y_pred - y_true
    denominator = float(np.sum((y_true - np.mean(y_true)) ** 2))
    if not math.isfinite(denominator) or denominator == 0:
        raise ValueError("Cannot calculate R2 for a constant target")
    return {
        "mae": float(np.mean(np.abs(residual))),
        "rmse": float(np.sqrt(np.mean(residual ** 2))),
        "r2": float(1.0 - np.sum(residual ** 2) / denominator),
        "pcc": float(np.corrcoef(y_true, y_pred)[0, 1]),
    }


def _read_tags(path: Path) -> Dict[str, Tuple[str, ...]]:
    output: Dict[str, Tuple[str, ...]] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not {"SMILES", "Tags"}.issubset(reader.fieldnames or []):
            raise ValueError("Unexpected tag columns in {}".format(path))
        for row in reader:
            output[row["SMILES"]] = tuple(
                value.strip() for value in row["Tags"].split(",") if value.strip()
            )
    return output


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]], fields: Sequence[str]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _figure_style() -> None:
    mpl.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "font.size": 12,
        "axes.linewidth": 1.0,
        "axes.spines.top": True,
        "axes.spines.right": True,
        "legend.frameon": False,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
    })


def _save_figure(fig: mpl.figure.Figure, base: Path) -> None:
    for suffix, options in (
        (".png", {"dpi": 600}),
        (".tif", {"dpi": 600, "pil_kwargs": {"compression": "tiff_lzw"}}),
        (".svg", {}),
        (".pdf", {}),
    ):
        fig.savefig(str(base) + suffix, bbox_inches="tight", facecolor="white", **options)


def _draw_figure3(
    output_base: Path,
    test_rows: Mapping[str, Sequence[Mapping[str, object]]],
    labels: Mapping[str, str],
) -> None:
    _figure_style()
    fig, axes = plt.subplots(2, 2, figsize=(12.6, 10.2))
    colors = {"pos": "#91C8A7", "neg": "#E79C9C"}
    for column, mode in enumerate(("pos", "neg")):
        rows = list(test_rows[mode])
        y_true = np.asarray([float(row["source_rti"]) for row in rows])
        y_pred = np.asarray([float(row["predicted_rti"]) for row in rows])
        residual = y_true - y_pred
        values = _metrics(rows)

        ax = axes[0, column]
        ax.scatter(
            y_true, y_pred, s=22, color=colors[mode], edgecolors="white",
            linewidths=0.35, alpha=0.72,
        )
        lower = float(min(y_true.min(), y_pred.min()))
        upper = float(max(y_true.max(), y_pred.max()))
        padding = (upper - lower) * 0.04
        ax.plot(
            [lower - padding, upper + padding],
            [lower - padding, upper + padding],
            linestyle="--", color="#C44E52", linewidth=1.4,
        )
        ax.set_xlim(lower - padding, upper + padding)
        ax.set_ylim(lower - padding, upper + padding)
        ax.set_xlabel("Source RTI")
        ax.set_ylabel("Predicted RTI")
        ax.text(
            0.04, 0.96,
            "Sample size = {n}\nR² = {r2:.4f}\nMAE = {mae:.4f}\nRMSE = {rmse:.4f}\nPCC = {pcc:.4f}".format(
                n=len(rows), **values
            ),
            transform=ax.transAxes, ha="left", va="top", fontsize=12,
        )
        ax.text(
            0.97, 0.05, labels[mode], transform=ax.transAxes,
            ha="right", va="bottom", fontsize=13,
        )
        ax.text(-0.14, 1.04, "({})".format("ab"[column]), transform=ax.transAxes,
                fontsize=17, va="top")

        ax = axes[1, column]
        ax.scatter(
            y_pred, residual, s=22, color=colors[mode], edgecolors="white",
            linewidths=0.35, alpha=0.72,
        )
        ax.axhline(0, linestyle="--", color="#666666", linewidth=1.2)
        ax.set_xlabel("Predicted RTI")
        ax.set_ylabel("Residuals")
        ax.text(
            0.97, 0.95, "{}\nSource RTI - predicted RTI".format(labels[mode]),
            transform=ax.transAxes, ha="right", va="top", fontsize=12,
        )
        ax.text(-0.14, 1.04, "({})".format("cd"[column]), transform=ax.transAxes,
                fontsize=17, va="top")
        ax.tick_params(labelsize=11)

    fig.subplots_adjust(left=0.09, right=0.98, bottom=0.08, top=0.98, wspace=0.26, hspace=0.26)
    _save_figure(fig, output_base)
    plt.close(fig)


def _draw_figure4(
    output_base: Path,
    category_rows: Sequence[Mapping[str, object]],
    labels: Mapping[str, str],
) -> None:
    _figure_style()
    fig, axes = plt.subplots(1, 2, figsize=(15.2, 8.2), sharey=True)
    for panel, mode in enumerate(("pos", "neg")):
        rows = [row for row in category_rows if row["mode"] == mode]
        rows_by_tag = {str(row["category"]): row for row in rows}
        ordered = [rows_by_tag[tag] for tag, _ in CATEGORIES]
        labels_y = [label for _, label in CATEGORIES]
        y = np.arange(len(ordered))

        ax_r2 = axes[panel]
        ax_mae = ax_r2.twiny()
        ax_n = ax_r2.twiny()
        ax_n.spines["top"].set_position(("outward", 42))

        r2 = np.asarray([float(row["r2"]) for row in ordered])
        mae = np.asarray([float(row["mae"]) for row in ordered])
        n = np.asarray([int(row["n"]) for row in ordered])
        r2_min = max(0.0, math.floor((float(r2.min()) - 0.02) * 100) / 100)
        r2_max = min(1.0, math.ceil((float(r2.max()) + 0.02) * 100) / 100)
        if r2_max - r2_min < 0.10:
            r2_min = max(0.0, r2_max - 0.12)

        ax_r2.barh(y, r2 - r2_min, left=r2_min, height=0.36,
                   color="#DDF29C", edgecolor="none", zorder=1)
        ax_mae.barh(y, mae, height=0.36, color="#F3CBC7", edgecolor="none", zorder=0)
        ax_n.plot(n, y, color="#198FA3", marker="o", markersize=5.5,
                  markerfacecolor="white", markeredgewidth=1.5, linewidth=1.2, zorder=4)

        ax_r2.set_xlim(r2_min, r2_max)
        ax_mae.set_xlim(max(0.0, float(mae.min()) - 5.0), float(mae.max()) + 5.0)
        ax_n.set_xlim(0, max(100, int(math.ceil(float(n.max()) / 100.0) * 100)))
        ax_r2.set_yticks(y)
        ax_r2.set_yticklabels(labels_y, fontsize=12)
        ax_r2.set_xlabel("R²", color="#7D9B35", fontsize=14)
        ax_mae.set_xlabel("MAE", color="#E98A99", fontsize=14)
        ax_n.set_xlabel("number of compounds", color="#198FA3", fontsize=14)
        ax_r2.tick_params(axis="x", colors="#7D9B35", labelsize=11)
        ax_mae.tick_params(axis="x", colors="#B88E88", labelsize=11)
        ax_n.tick_params(axis="x", colors="#198FA3", labelsize=11)
        ax_r2.spines["bottom"].set_color("#7D9B35")
        ax_mae.spines["top"].set_color("#B88E88")
        ax_n.spines["top"].set_color("#198FA3")
        ax_r2.text(-0.12, 1.10, "({}) {}".format("ab"[panel], mode),
                   transform=ax_r2.transAxes, fontsize=18, va="top")
        ax_r2.text(0.98, 0.02, labels[mode], transform=ax_r2.transAxes,
                   ha="right", va="bottom", fontsize=11)

    axes[0].tick_params(axis="y", labelleft=True)
    axes[1].tick_params(axis="y", labelleft=True)
    axes[0].invert_yaxis()
    fig.subplots_adjust(left=0.18, right=0.98, bottom=0.11, top=0.80, wspace=0.70)
    _save_figure(fig, output_base)
    plt.close(fig)


def _draw_figure_s5(
    output_base: Path,
    history_by_mode: Mapping[str, Mapping[str, Sequence[object]]],
    test_rows: Mapping[str, Sequence[Mapping[str, object]]],
    labels: Mapping[str, str],
) -> None:
    _figure_style()
    fig, axes = plt.subplots(2, 3, figsize=(15.4, 9.6))
    panel_labels = (("a", "b", "c"), ("d", "e", "f"))
    scatter_colors = {"pos": "#BBDDA8", "neg": "#B9D8E9"}
    for row_index, mode in enumerate(("pos", "neg")):
        history = history_by_mode[mode]
        epoch = np.asarray(history["Epoch"], dtype=int)

        ax = axes[row_index, 0]
        ax.plot(epoch, history["MAE_train"], color="#222222", linewidth=1.5, label="Train")
        ax.plot(epoch, history["MAE_dev"], color="#D95F5F", linewidth=1.5, label="Development")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("MAE")
        ax.legend(fontsize=10)

        ax = axes[row_index, 1]
        ax.plot(epoch, history["MSE_train"], color="#222222", linewidth=1.5, label="Train")
        ax.plot(epoch, history["MSE_dev"], color="#D95F5F", linewidth=1.5, label="Development")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("MSE")
        ax.legend(fontsize=10)

        rows = list(test_rows[mode])
        y_true = np.asarray([float(item["source_rti"]) for item in rows])
        y_pred = np.asarray([float(item["predicted_rti"]) for item in rows])
        values = _metrics(rows)
        lower = float(min(y_true.min(), y_pred.min()))
        upper = float(max(y_true.max(), y_pred.max()))
        padding = (upper - lower) * 0.04
        ax = axes[row_index, 2]
        ax.scatter(
            y_true, y_pred, s=19, color=scatter_colors[mode], alpha=0.70,
            edgecolors="white", linewidths=0.30,
        )
        ax.plot(
            [lower - padding, upper + padding], [lower - padding, upper + padding],
            linestyle="--", color="#D95F5F", linewidth=1.25,
        )
        ax.set_xlim(lower - padding, upper + padding)
        ax.set_ylim(lower - padding, upper + padding)
        ax.set_xlabel("Source RTI")
        ax.set_ylabel("Predicted RTI")
        ax.text(
            0.04, 0.96,
            "R² = {r2:.4f}\nMAE = {mae:.4f}\nRMSE = {rmse:.4f}".format(**values),
            transform=ax.transAxes, ha="left", va="top", fontsize=10,
        )

        for column in range(3):
            axes[row_index, column].tick_params(labelsize=10)
            axes[row_index, column].text(
                -0.16, 1.04, "({})".format(panel_labels[row_index][column]),
                transform=axes[row_index, column].transAxes, fontsize=16, va="top",
            )
            if column == 2:
                axes[row_index, column].text(
                    0.98, 0.04, labels[mode], transform=axes[row_index, column].transAxes,
                    ha="right", va="bottom", fontsize=11,
                )

    fig.subplots_adjust(left=0.07, right=0.99, bottom=0.08, top=0.98, wspace=0.31, hspace=0.30)
    _save_figure(fig, output_base)
    plt.close(fig)


def build(repo_root: Path, output_dir: Path) -> Dict[str, object]:
    selection_root = repo_root / "outputs" / "model_selection_dev_seed1234"
    feature_root = repo_root / "outputs" / "feature_fusion_random_seed1234"
    selection = json.loads((selection_root / "selected_models.json").read_text(encoding="utf-8"))
    output_dir.mkdir(parents=True, exist_ok=True)

    metric_rows: List[Dict[str, object]] = []
    prediction_rows: List[Dict[str, object]] = []
    test_by_mode: Dict[str, List[Dict[str, object]]] = {}
    history_by_mode: Dict[str, Mapping[str, Sequence[object]]] = {}
    history_rows: List[Dict[str, object]] = []
    labels: Dict[str, str] = {}

    for mode in ("pos", "neg"):
        selected = selection["selected_models"][mode]
        code = selected["feature_code"]
        model = selected["model"]
        labels[mode] = model
        experiment = feature_root / "experiments" / (code + "_" + mode)
        history = json.loads((experiment / "training_metrics.json").read_text(encoding="utf-8"))
        history_by_mode[mode] = history
        for index, epoch in enumerate(history["Epoch"]):
            history_rows.append({
                "mode": mode,
                "feature_code": code,
                "model": model,
                "epoch": int(epoch),
                "mae_train": float(history["MAE_train"][index]),
                "mae_development": float(history["MAE_dev"][index]),
                "mse_train": float(history["MSE_train"][index]),
                "mse_development": float(history["MSE_dev"][index]),
                "r2_train": float(history["R2_train"][index]),
                "r2_development": float(history["R2_dev"][index]),
            })
        for subset, filename in (
            ("development", "train_prediction_dev.txt"),
            ("held_out_test", "train_prediction_test.txt"),
        ):
            rows = _read_tsv_predictions(experiment / filename)
            values = _metrics(rows)
            metric_rows.append({
                "mode": mode,
                "feature_code": code,
                "model": model,
                "subset": subset,
                "n": len(rows),
                **values,
                "selected_using": "development_mae",
            })
            for row in rows:
                prediction_rows.append({
                    "mode": mode,
                    "feature_code": code,
                    "model": model,
                    "subset": subset,
                    **row,
                })
            if subset == "held_out_test":
                test_by_mode[mode] = rows

    category_rows: List[Dict[str, object]] = []
    category_prediction_rows: List[Dict[str, object]] = []
    coverage_rows: List[Dict[str, object]] = []
    for mode in ("pos", "neg"):
        tags = _read_tags(repo_root / "data" / "MMF-3" / "tags" / (mode + "_compound_tags.csv"))
        predictions = {str(row["smiles"]): row for row in test_by_mode[mode]}
        missing_tags = sorted(set(predictions) - set(tags))
        coverage_rows.append({
            "mode": mode,
            "prediction_count": len(predictions),
            "prediction_without_tag_count": len(missing_tags),
            "prediction_without_tag_smiles": " | ".join(missing_tags),
        })
        if missing_tags:
            raise ValueError("Selected-model predictions lack tags: {} / {}".format(mode, len(missing_tags)))
        for tag, label in CATEGORIES:
            members = [row for smiles, row in predictions.items() if tag in tags[smiles]]
            if len(members) < 2:
                raise ValueError("Too few category members for {} / {}".format(mode, tag))
            values = _metrics(members)
            category_rows.append({
                "mode": mode,
                "feature_code": selection["selected_models"][mode]["feature_code"],
                "model": labels[mode],
                "category": tag,
                "category_label": label,
                "n": len(members),
                **values,
            })
            for row in members:
                category_prediction_rows.append({
                    "mode": mode,
                    "model": labels[mode],
                    "category": tag,
                    "category_label": label,
                    **row,
                })

    _write_csv(
        output_dir / "selected_model_metrics.csv", metric_rows,
        ["mode", "feature_code", "model", "subset", "n", *METRICS, "selected_using"],
    )
    _write_csv(
        output_dir / "selected_model_predictions.csv", prediction_rows,
        [
            "mode", "feature_code", "model", "subset", "smiles", "source_rti",
            "predicted_rti", "residual_source_minus_prediction",
        ],
    )
    _write_csv(
        output_dir / "figure3_selected_model_predictions.csv",
        [row for row in prediction_rows if row["subset"] == "held_out_test"],
        [
            "mode", "feature_code", "model", "subset", "smiles", "source_rti",
            "predicted_rti", "residual_source_minus_prediction",
        ],
    )
    _write_csv(
        output_dir / "figure4_category_metrics.csv", category_rows,
        ["mode", "feature_code", "model", "category", "category_label", "n", *METRICS],
    )
    _write_csv(
        output_dir / "figure4_category_predictions.csv", category_prediction_rows,
        [
            "mode", "model", "category", "category_label", "smiles", "source_rti",
            "predicted_rti", "residual_source_minus_prediction",
        ],
    )
    _write_csv(
        output_dir / "category_tag_coverage.csv", coverage_rows,
        ["mode", "prediction_count", "prediction_without_tag_count", "prediction_without_tag_smiles"],
    )
    _write_csv(
        output_dir / "figureS5_selected_model_training_history.csv", history_rows,
        [
            "mode", "feature_code", "model", "epoch", "mae_train", "mae_development",
            "mse_train", "mse_development", "r2_train", "r2_development",
        ],
    )

    _draw_figure3(output_dir / "Figure3_selected_model_fixed_split", test_by_mode, labels)
    _draw_figure4(output_dir / "Figure4_selected_model_categories", category_rows, labels)
    _draw_figure_s5(
        output_dir / "FigureS5_selected_model_training_diagnostics",
        history_by_mode, test_by_mode, labels,
    )

    provenance = {
        "selection_rule": selection["selection_rule"],
        "selected_models": selection["selected_models"],
        "fixed_split_source": str(feature_root),
        "figure3": "held-out predictions from the development-selected model in each mode",
        "figure4": "non-exclusive category metrics from the same held-out predictions",
    }
    (output_dir / "selected_model_result_authority.json").write_text(
        json.dumps(provenance, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / "Figure3_caption.txt").write_text(
        "Figure 3 Held-out performance of the development-selected ViSRTI models. "
        "Positive-mode results use ViSRTI-GP and negative-mode results use ViSRTI-GPT. "
        "Panels (a-b) compare ViSRTI predictions with the source RTI labels; panels (c-d) "
        "show residuals (source RTI minus predicted RTI) against predicted RTI.\n",
        encoding="utf-8",
    )
    (output_dir / "Figure4_caption.txt").write_text(
        "Figure 4 Class-wise agreement with source RTI labels for the development-selected "
        "models on the held-out test sets: (a) positive-mode ViSRTI-GP and (b) negative-mode "
        "ViSRTI-GPT. Categories are non-exclusive.\n",
        encoding="utf-8",
    )
    (output_dir / "FigureS5_caption.txt").write_text(
        "Figure S5 Training diagnostics for the development-selected models. Panels (a-c) "
        "show positive-mode ViSRTI-GP and panels (d-f) show negative-mode ViSRTI-GPT. "
        "Training and development MAE and MSE are shown with held-out predicted-versus-source "
        "RTI agreement. The held-out test sets were not used for model selection.\n",
        encoding="utf-8",
    )
    return provenance


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("outputs/selected_model_fixed_split_seed1234"),
    )
    args = parser.parse_args()
    repo_root = args.repo_root.resolve()
    output_dir = args.output_dir
    if not output_dir.is_absolute():
        output_dir = repo_root / output_dir
    report = build(repo_root, output_dir.resolve())
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
