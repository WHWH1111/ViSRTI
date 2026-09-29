#!/usr/bin/env python3
"""Render Figure 2e-f in the visual style of the author's original figure."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


FEATURE_ORDER = ("G", "GP", "GPT", "GPTC")
METRICS = (
    ("mae", "mae_bootstrap_sd", "MAE", "#F4A6A6", "left"),
    ("rmse", "rmse_bootstrap_sd", "RMSE", "#A8C8DC", "left"),
    ("r2", "r2_bootstrap_sd", "R²", "#A8D5AE", "right"),
    ("pcc", "pcc_bootstrap_sd", "PCC", "#CAB3D2", "right"),
)


def _save_figure(fig: mpl.figure.Figure, base: Path) -> List[str]:
    outputs: List[str] = []
    for suffix, options in (
        ("svg", {}),
        ("pdf", {}),
        ("png", {"dpi": 300}),
        ("tiff", {"dpi": 300, "pil_kwargs": {"compression": "tiff_lzw"}}),
    ):
        path = base.with_suffix("." + suffix)
        fig.savefig(path, bbox_inches="tight", **options)
        outputs.append(str(path))
    plt.close(fig)
    return outputs


def plot_feature_fusion(metrics_file: Path, output_dir: Path) -> List[str]:
    table = pd.read_csv(metrics_file)
    if "analysis" in table.columns:
        table = table[table["analysis"] == "fixed_split_feature_fusion"].copy()
    required = {
        "mode", "feature_code", "mae", "rmse", "r2", "pcc",
        "mae_bootstrap_sd", "rmse_bootstrap_sd", "r2_bootstrap_sd",
        "pcc_bootstrap_sd",
    }
    missing = required - set(table.columns)
    if missing:
        raise ValueError("Missing columns: {}".format(", ".join(sorted(missing))))

    mpl.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "font.size": 10,
        "axes.linewidth": 1.0,
        "axes.spines.top": True,
        "axes.spines.right": True,
        "legend.frameon": False,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
    })
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.7))
    width = 0.20
    offsets = (-1.5 * width, -0.5 * width, 0.5 * width, 1.5 * width)
    legend_handles: Dict[str, object] = {}

    for panel_index, mode in enumerate(("pos", "neg")):
        ax_left = axes[panel_index]
        ax_right = ax_left.twinx()
        subset = table[table["mode"] == mode].copy()
        subset["feature_code"] = pd.Categorical(
            subset["feature_code"], categories=FEATURE_ORDER, ordered=True
        )
        subset = subset.sort_values("feature_code")
        if subset["feature_code"].astype(str).tolist() != list(FEATURE_ORDER):
            raise ValueError("Expected G, GP, GPT, GPTC for {} mode".format(mode))

        x = np.arange(len(FEATURE_ORDER), dtype=float)
        for offset, (metric, sd_key, label, color, axis_name) in zip(offsets, METRICS):
            axis = ax_left if axis_name == "left" else ax_right
            values = subset[metric].to_numpy(dtype=float)
            errors = subset[sd_key].to_numpy(dtype=float)
            bars = axis.bar(
                x + offset,
                values,
                width,
                yerr=errors,
                color=color,
                edgecolor="none",
                alpha=0.90,
                error_kw={
                    "ecolor": "#777777",
                    "elinewidth": 0.8,
                    "capsize": 2.5,
                    "capthick": 0.8,
                },
                label=label,
                zorder=2,
            )
            legend_handles.setdefault(label, bars)
            for bar, value in zip(bars, values):
                y_position = max(4.0, value * 0.42) if axis_name == "left" else 0.625
                axis.text(
                    bar.get_x() + bar.get_width() / 2,
                    y_position,
                    "{:.4f}".format(value),
                    ha="center",
                    va="center",
                    rotation=90,
                    fontsize=8.5,
                    color="#222222",
                    zorder=3,
                )

        ax_left.set_xlim(-0.55, len(FEATURE_ORDER) - 0.45)
        ax_left.set_ylim(0, 100)
        ax_left.set_yticks([0, 20, 40, 60, 80, 100])
        ax_right.set_ylim(0.55, 1.10)
        ax_right.set_yticks([0.6, 0.7, 0.8, 0.9, 1.0])
        ax_left.set_xticks(x)
        ax_left.set_xticklabels(["ViSRTI-" + code for code in FEATURE_ORDER])
        ax_left.set_ylabel("MAE & RMSE")
        ax_right.set_ylabel("R² & PCC", rotation=270, labelpad=17)
        ax_left.set_xlabel("{}-RTI".format(mode))
        ax_left.tick_params(direction="out", length=4, width=1.0)
        ax_right.tick_params(direction="out", length=4, width=1.0)
        ax_left.text(
            0.02,
            0.98,
            "({})".format("ef"[panel_index]),
            transform=ax_left.transAxes,
            ha="left",
            va="top",
            fontsize=18,
        )
        ax_left.grid(False)
        ax_right.grid(False)

    handles = [legend_handles[label] for label in ("MAE", "RMSE", "R²", "PCC")]
    labels = ["MAE", "RMSE", "R²", "PCC"]
    for axis in axes:
        axis.legend(
            handles,
            labels,
            loc="upper center",
            bbox_to_anchor=(0.60, 0.97),
            ncol=4,
            handlelength=1.5,
            columnspacing=1.0,
        )
    fig.subplots_adjust(left=0.07, right=0.94, top=0.96, bottom=0.16, wspace=0.34)
    output_dir.mkdir(parents=True, exist_ok=True)
    return _save_figure(fig, output_dir / "Figure2ef_feature_fusion")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Aggregated feature-fusion metrics CSV")
    parser.add_argument("--output-dir", type=Path, required=True, help="Figure output directory")
    args = parser.parse_args()
    for path in plot_feature_fusion(args.input, args.output_dir):
        print(path)


if __name__ == "__main__":
    main()
