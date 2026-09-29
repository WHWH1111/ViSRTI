#!/usr/bin/env python3
"""Render Figure 2a-d in the visual style of the author's original figure."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List, Sequence

import matplotlib.pyplot as plt
import numpy as np


COLORS = ("#fb9a99", "#a6cee3", "#b2df8a", "#cab2d6")
MODEL_ORDER = ("GNN-RT", "GIN", "SMILES-BiLSTM", "ViSRTI-G")


def read_values(path: Path) -> Dict[str, Dict[str, Dict[str, List[float]]]]:
    values: Dict[str, Dict[str, Dict[str, List[float]]]] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("analysis") and row["analysis"] != "fixed_split_baseline":
                continue
            values.setdefault(row["mode"], {}).setdefault(row["model"], {"r2": [], "mae": []})
            values[row["mode"]][row["model"]]["r2"].append(float(row["r2"]))
            values[row["mode"]][row["model"]]["mae"].append(float(row["mae"]))
    return values


def half_violin(ax, datasets: Sequence[Sequence[float]], ylabel: str) -> None:
    positions = np.arange(1, len(datasets) + 1)
    violins = ax.violinplot(
        datasets,
        positions=positions,
        widths=0.70,
        showmeans=False,
        showmedians=False,
        showextrema=False,
    )
    for index, body in enumerate(violins["bodies"]):
        vertices = body.get_paths()[0].vertices
        center = positions[index]
        vertices[:, 0] = np.maximum(vertices[:, 0], center)
        body.set_facecolor(COLORS[index])
        body.set_edgecolor("#808080")
        body.set_linewidth(0.8)
        body.set_alpha(0.88)
        data = np.asarray(datasets[index])
        quantiles = np.quantile(data, np.linspace(0.04, 0.96, 24))
        ax.hlines(quantiles, center - 0.17, center - 0.04, color=COLORS[index], lw=1.4, alpha=0.75)
        mean, standard_deviation = float(np.mean(data)), float(np.std(data, ddof=1))
        ax.errorbar(
            center,
            mean,
            yerr=standard_deviation,
            fmt="o",
            color="#777777",
            ecolor="#888888",
            markerfacecolor="white",
            markeredgecolor="#b0b0b0",
            markersize=4.8,
            capsize=2.5,
            lw=0.8,
            zorder=5,
        )
    ax.set_ylabel(ylabel)
    ax.set_xticks(positions)
    ax.set_xticklabels(MODEL_ORDER, rotation=0)
    ax.tick_params(direction="out", length=3, width=0.8)
    for spine in ax.spines.values():
        spine.set_linewidth(0.9)
        spine.set_color("#333333")


def plot(input_path: Path, output_stem: Path) -> None:
    values = read_values(input_path)
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "font.size": 9,
        "axes.linewidth": 0.9,
        "svg.fonttype": "none",
    })
    fig, axes = plt.subplots(2, 2, figsize=(10.2, 7.0))
    panels = (("pos", "r2", "R$^2$"), ("pos", "mae", "MAE"), ("neg", "r2", "R$^2$"), ("neg", "mae", "MAE"))
    for index, (axis, (mode, metric, ylabel)) in enumerate(zip(axes.flat, panels)):
        datasets = [values[mode][model][metric] for model in MODEL_ORDER]
        half_violin(axis, datasets, ylabel)
        axis.text(0.02, 0.97, "({})".format(chr(ord("a") + index)), transform=axis.transAxes, ha="left", va="top", fontsize=13)
        axis.text(0.98, 0.97, "{}-RTI".format(mode), transform=axis.transAxes, ha="right", va="top")
        if index in (0, 2):
            proxy = axis.errorbar([], [], yerr=[], fmt="none", ecolor="#888888", capsize=3, label="Mean±1SD")
            mean_proxy = axis.scatter([], [], s=18, facecolor="white", edgecolor="#b0b0b0", label="Mean")
            axis.legend(handles=(proxy, mean_proxy), loc="lower left", frameon=False, fontsize=8)
        all_values = np.concatenate([np.asarray(dataset) for dataset in datasets])
        lower, upper = float(np.min(all_values)), float(np.max(all_values))
        span = max(upper - lower, 1e-6)
        axis.set_ylim(lower - 0.08 * span, upper + 0.08 * span)
    fig.tight_layout(pad=1.1, w_pad=2.0, h_pad=1.8)
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    for suffix, dpi in (("png", 600), ("tiff", 600), ("pdf", 300), ("svg", 300)):
        fig.savefig(output_stem.with_suffix("." + suffix), dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-stem", type=Path, required=True)
    args = parser.parse_args()
    plot(args.input.resolve(), args.output_stem.resolve())


if __name__ == "__main__":
    main()
