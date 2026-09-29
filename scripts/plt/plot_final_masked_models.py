#!/usr/bin/env python3
"""Plot held-out evidence for the frozen development-selected GP/GPT masks."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.patches import Patch
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "data-val"))
from build_selected_model_fixed_split_results import CATEGORIES, _draw_figure_s5, _figure_style

METRICS = ("mae", "rmse", "r2", "pcc")
COLORS = ("#F4A6A6", "#A8C8DC", "#A8D5AE", "#CAB3D2")
LABELS = {
    "Monoiso_Mass": "Monoiso_Mass", "LogKow": "LogKow", "LogP": "LogP",
    "Koc": "Koc", "IC50": "IC₅₀", "48h-LC50": "48h-LC₅₀",
    "72h-EC50": "72h-EC₅₀", "96h-LC50": "96h-LC₅₀",
}
FEATURES = tuple(LABELS)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def read_rows(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        delimiter = "\t" if "\t" in handle.readline() else ","
        handle.seek(0)
        return list(csv.DictReader(handle, delimiter=delimiter))


def write_rows(path, rows):
    rows = list(rows)
    if not rows:
        raise ValueError("Cannot export an empty evidence table: {}".format(path))
    with Path(path).open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def resolve_path(value, run_root):
    path = Path(value)
    if path.is_absolute():
        return path
    candidate = run_root / path
    return candidate if candidate.exists() else ROOT / path


def read_predictions(path):
    output = []
    for row in read_rows(path):
        smiles = row.get("smiles", row.get("Smiles"))
        target = float(row.get("source_rti", row.get("Correct")))
        predicted = float(row.get("predicted_rti", row.get("Predict")))
        if not smiles or not np.isfinite([target, predicted]).all():
            raise ValueError("Invalid prediction record")
        output.append({"smiles": smiles, "source_rti": target, "predicted_rti": predicted,
                       "residual_source_minus_prediction": target - predicted})
    if not output or len({row["smiles"] for row in output}) != len(output):
        raise ValueError("Prediction rows must be nonempty with unique SMILES")
    return output


def metric_values(rows):
    y = np.asarray([row["source_rti"] for row in rows], dtype=float)
    prediction = np.asarray([row["predicted_rti"] for row in rows], dtype=float)
    residual = y - prediction
    total = float(np.sum((y - y.mean()) ** 2))
    if len(y) < 2 or total <= 0:
        raise ValueError("Metrics require at least two nonconstant responses")
    return dict(zip(METRICS, (float(np.mean(np.abs(residual))),
                             float(np.sqrt(np.mean(residual ** 2))),
                             float(1 - np.sum(residual ** 2) / total),
                             float(np.corrcoef(y, prediction)[0, 1]))))


def validate_steps(steps, run_root):
    """Check reported test values without using them to select a step."""
    selected = [step for step in steps if step.get("selected") is True or step.get("status") == "selected"]
    if len(selected) != 1 or not selected[0]["accepted"]:
        raise ValueError("Exactly one development-selected accepted step is required")
    accepted = [step for step in steps if step["accepted"]]
    if selected[0]["step"] != max(step["step"] for step in accepted):
        raise ValueError("Selected step must be the last accepted development step")
    baseline_order, baseline_y = None, None
    all_predictions = []
    previous_mae = None
    stopped = False
    for step in steps:
        if stopped:
            raise ValueError("Masking continued after its first rejected development step")
        development_mae = float(step["development_metrics"]["mae"])
        if previous_mae is not None:
            should_accept = development_mae <= previous_mae + 1e-8
            if bool(step["accepted"]) != should_accept:
                raise ValueError("Acceptance disagrees with the development MAE stopping rule")
        previous_mae = development_mae
        stopped = not step["accepted"]
        rows = read_predictions(resolve_path(step["prediction_file"], run_root))
        order = [row["smiles"] for row in rows]
        target = [row["source_rti"] for row in rows]
        if baseline_order is None:
            baseline_order, baseline_y = order, target
        if order != baseline_order or target != baseline_y:
            raise ValueError("Masking steps use different held-out samples or responses")
        calculated = metric_values(rows)
        for metric in METRICS:
            if not math.isclose(calculated[metric], float(step["metrics"][metric]), abs_tol=1e-6, rel_tol=1e-7):
                raise ValueError("Masking test metric does not match its prediction file")
        all_predictions.append(rows)
    return selected[0]["step"], all_predictions


def retained_shap(payload, model):
    if payload.get("partition") != "test" or payload.get("smoke_only", False):
        raise ValueError("Only complete test SHAP can be plotted here")
    names = list(payload["feature_names"])
    masks = list(model["physchem_mask"]) + list(model.get("toxicity_mask") or [])
    enabled = {name for name, flag in zip(FEATURES, masks) if flag}
    if model["feature_code"] == "GP":
        enabled &= set(FEATURES[:4])
    indices = [index for index, name in enumerate(names) if name in enabled]
    if set(names[index] for index in indices) != enabled or not indices:
        raise ValueError("SHAP columns do not cover the retained feature configuration")
    values = np.asarray(payload["shap_values"], dtype=float)
    features = np.asarray(payload["feature_values"], dtype=float)
    if values.shape != features.shape or values.shape != (len(payload["smiles"]), len(names)):
        raise ValueError("SHAP arrays and sample identities differ")
    if not np.isfinite(values).all() or not np.isfinite(features).all():
        raise ValueError("Non-finite SHAP evidence")
    result = {**payload, "feature_names": [names[index] for index in indices],
              "shap_values": values[:, indices].tolist(), "feature_values": features[:, indices].tolist()}
    if "raw_feature_values" in payload:
        raw = np.asarray(payload["raw_feature_values"], dtype=float)
        if raw.shape != values.shape or not np.isfinite(raw).all():
            raise ValueError("Raw feature evidence does not match the SHAP columns")
        result["raw_feature_values"] = raw[:, indices].tolist()
    return result


def category_evidence(predictions, shap_payload, tags):
    """Use the same held-out rows for class metrics and class-conditional SHAP."""
    smiles = [row["smiles"] for row in predictions]
    shap_smiles = shap_payload["smiles"]
    if len(set(shap_smiles)) != len(shap_smiles) or set(smiles) != set(shap_smiles):
        raise ValueError("Category SHAP and predictions must cover the same held-out compounds")
    missing = set(smiles) - set(tags)
    if missing:
        raise ValueError("Held-out predictions lack frozen category tags")
    lookup = {smiles: i for i, smiles in enumerate(shap_smiles)}
    values = np.asarray(shap_payload["shap_values"], dtype=float)
    names = shap_payload["feature_names"]
    metric_rows, shap_rows, member_rows = [], [], []
    for category, label in CATEGORIES:
        members = [row for row in predictions if category in tags[row["smiles"]]]
        if len(members) < 2:
            raise ValueError("Too few held-out members in {}".format(category))
        metric_rows.append({"category": category, "category_label": label, "n": len(members),
                            **metric_values(members)})
        mean_abs = np.abs(values[[lookup[row["smiles"]] for row in members]]).mean(0)
        span = float(mean_abs.max() - mean_abs.min())
        normalized = (mean_abs - mean_abs.min()) / span if span else np.zeros_like(mean_abs)
        for feature, raw, norm in zip(names, mean_abs, normalized):
            shap_rows.append({"category": category, "category_label": label, "n": len(members),
                              "feature": feature, "mean_absolute_shap": float(raw),
                              "normalized_mean_absolute_shap": float(norm), "constant_row": span == 0})
        member_rows.extend({"category": category, "category_label": label, **row} for row in members)
    return metric_rows, shap_rows, member_rows


def save_figure(fig, base):
    for suffix, options in ((".png", {"dpi": 300}), (".pdf", {}), (".svg", {}),
                            (".tif", {"dpi": 600, "pil_kwargs": {"compression": "tiff_lzw"}})):
        fig.savefig(str(base) + suffix, facecolor="white", bbox_inches="tight", **options)
    fig.savefig(str(base) + "_预览.png", dpi=130, facecolor="white", bbox_inches="tight")
    plt.close(fig)


def draw_mask_bars(ax, steps, model_label, selected_step, panel, mode):
    right = ax.twinx()
    x, width = np.arange(len(steps)), 0.20
    for j, (metric, color) in enumerate(zip(METRICS, COLORS)):
        axis = ax if j < 2 else right
        values = [float(step["metrics"][metric]) for step in steps]
        bars = axis.bar(x + (j - 1.5) * width, values, width=width, color=color, edgecolor="none")
        for bar, value in zip(bars, values):
            axis.text(bar.get_x() + width / 2, value * 0.47, "{:.4f}".format(value),
                      rotation=90, va="center", ha="center", fontsize=10)
    ymax = max(float(step["metrics"]["rmse"]) for step in steps)
    upper = math.ceil(ymax * 1.18 / 10) * 10
    ax.set_ylim(0, upper)
    ax.set_yticks(np.arange(0, upper + 1, 20))
    right.set_ylim(min(0, min(float(step["metrics"][name]) for step in steps for name in ("r2", "pcc"))), 1.1)
    right.set_yticks(np.arange(0, 1.01, 0.2))
    ax.set_ylabel("MAE & RMSE", fontsize=12)
    right.set_ylabel("R² & PCC", rotation=270, labelpad=18, fontsize=12)
    labels = []
    for step in steps:
        removed = step["removed_features"]
        label = model_label if not removed else "mask " + "+".join(LABELS[name] for name in removed)
        if len(label) > 23:
            label = label.replace("+", "+\n", 1)
        if step["step"] == selected_step:
            label += "\n(selected)"
        elif not step["accepted"]:
            label += "\n(stop)"
        labels.append(label)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10 if len(steps) <= 3 else 8.5)
    ax.set_xlim(-0.55, len(steps) - 0.45)
    ax.set_xlabel(mode + "-RTI (test set)", fontsize=12)
    ax.text(0.02, 0.98, panel, transform=ax.transAxes, fontsize=17, va="top")
    ax.legend([Patch(facecolor=color) for color in COLORS], ["MAE", "RMSE", "R²", "PCC"],
              loc="upper right", ncol=4, fontsize=10, frameon=False, handlelength=1.2, columnspacing=0.7)


def draw_figure3(payloads, figures):
    sys.path.insert(0, str(Path(__file__).parent))
    from plot_gp_gpt_masking import draw_shap
    _figure_style()
    fig, axes = plt.subplots(2, 2, figsize=(13.8, 10.5))
    fig.subplots_adjust(left=0.12, right=0.94, top=0.91, bottom=0.13, wspace=0.48, hspace=0.62)
    for index, mode in enumerate(("pos", "neg")):
        data = payloads[mode]
        cmap = draw_shap(axes[index, 0], data["unmasked_shap"], "(a)" if index == 0 else "(c)", mode)
        draw_mask_bars(axes[index, 1], data["steps"], data["unmasked_shap"]["model"],
                       data["selected_step"], "(b)" if index == 0 else "(d)", mode)
    fig.subplots_adjust(left=0.12, right=0.94, top=0.91, bottom=0.13, wspace=0.48, hspace=0.62)
    for row in range(2):
        box = axes[row, 0].get_position()
        cax = fig.add_axes([box.x1 + 0.013, box.y0, 0.004, box.height])
        cb = fig.colorbar(matplotlib.cm.ScalarMappable(norm=Normalize(0, 1), cmap=cmap), cax=cax)
        cb.set_ticks([0, 1])
        cb.set_ticklabels(["Low", "High"])
        cb.ax.tick_params(length=0, labelsize=11)
        cb.outline.set_visible(False)
    save_figure(fig, figures / "图3_GP与GPT完整测试集SHAP及逐步掩码")


def draw_category_panel(ax, rows, mode, panel):
    ordered = {row["category"]: row for row in rows}
    ordered = [ordered[category] for category, _ in CATEGORIES]
    y = np.arange(len(ordered))
    r2 = np.asarray([row["r2"] for row in ordered])
    mae = np.asarray([row["mae"] for row in ordered])
    n = np.asarray([row["n"] for row in ordered])
    ax_mae, ax_n = ax.twiny(), ax.twiny()
    ax_n.spines["top"].set_position(("outward", 43))
    low = math.floor((float(r2.min()) - 0.02) * 100) / 100
    high = min(1, math.ceil((float(r2.max()) + 0.02) * 100) / 100)
    ax.barh(y + 0.16, r2 - low, left=low, height=0.30, color="#DDF29C", edgecolor="none")
    ax_mae.barh(y - 0.16, mae, height=0.30, color="#F3CBC7", edgecolor="none")
    ax_n.plot(n, y, color="#198FA3", marker="o", markersize=5,
              markerfacecolor="white", markeredgewidth=1.2, linewidth=1.0)
    ax.set_xlim(low, high)
    ax_mae.set_xlim(max(0, float(mae.min()) - 5), float(mae.max()) + 5)
    ax_n.set_xlim(0, max(100, int(math.ceil(float(n.max()) / 100) * 100)))
    ax.set_yticks(y)
    ax.set_yticklabels([label for _, label in CATEGORIES], fontsize=10)
    ax.set_ylim(len(ordered) - 0.5, -0.5)
    ax.set_xlabel("R²", color="#7D9B35", fontsize=12)
    ax_mae.set_xlabel("MAE", color="#B88E88", fontsize=12)
    ax_n.set_xlabel("number of compounds", color="#198FA3", fontsize=12)
    for axis, color in ((ax, "#7D9B35"), (ax_mae, "#B88E88"), (ax_n, "#198FA3")):
        axis.tick_params(axis="x", colors=color, labelsize=9)
    ax.text(-0.05, 1.07, "({}) {}".format(panel, mode), transform=ax.transAxes,
            fontsize=16, va="bottom", ha="right")


def draw_heatmap(ax, rows, names, mode, panel):
    lookup = {(row["category"], row["feature"]): row["normalized_mean_absolute_shap"] for row in rows}
    matrix = np.asarray([[lookup[category, name] for name in names] for category, _ in CATEGORIES])
    image = ax.imshow(matrix, cmap="viridis_r", vmin=0, vmax=1, aspect="auto")
    ax.set_yticks(np.arange(len(CATEGORIES)))
    ax.set_yticklabels([label for _, label in CATEGORIES], fontsize=10)
    ax.set_xticks(np.arange(len(names)))
    ax.set_xticklabels([LABELS[name] for name in names], rotation=90, fontsize=11)
    ax.tick_params(length=0)
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            value = matrix[row, column]
            ax.text(column, row, "{:.2f}".format(value), ha="center", va="center",
                    color="white" if value > 0.52 else "black", fontsize=10)
    ax.set_xticks(np.arange(-0.5, len(names), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(CATEGORIES), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=0.55)
    ax.tick_params(which="minor", length=0)
    ax.set_title("({}) {}".format(panel, mode), loc="left", fontsize=16, pad=12)
    cb = ax.figure.colorbar(image, ax=ax, fraction=0.036, pad=0.035)
    cb.set_label("Normalized SHAP value", rotation=270, labelpad=18, fontsize=11)
    cb.ax.tick_params(labelsize=9)


def draw_figure4(payloads, figures):
    _figure_style()
    fig, axes = plt.subplots(2, 2, figsize=(15.8, 11.8), gridspec_kw={"height_ratios": [1, 1.05]})
    for column, mode in enumerate(("pos", "neg")):
        data = payloads[mode]
        draw_category_panel(axes[0, column], data["category_metrics"], mode, "ab"[column])
        draw_heatmap(axes[1, column], data["category_shap"], data["final_shap"]["feature_names"],
                     mode, "cd"[column])
    fig.subplots_adjust(left=0.20, right=0.95, top=0.86, bottom=0.15, wspace=0.88, hspace=0.40)
    save_figure(fig, figures / "图4_最终模型十类化合物性能与SHAP热图")


def read_history(experiment):
    rows = read_rows(Path(experiment) / "MAEs.txt")
    needed = ("Epoch", "MAE_train", "MAE_dev", "MSE_train", "MSE_dev")
    if not rows or not set(needed).issubset(rows[0]):
        raise ValueError("Selected model has no usable training history")
    return {name: [float(row[name]) for row in rows] for name in needed}


def shap_long_rows(payload, mode, stage):
    if payload.get("shap_unit", "standardized_target") != "standardized_target":
        raise ValueError("RTI export requires standardized-target SHAP input")
    property_std = float(payload["property_std"])
    if not np.isfinite(property_std) or property_std <= 0:
        raise ValueError("A positive saved target standard deviation is required")
    for index, smiles in enumerate(payload["smiles"]):
        for column, name in enumerate(payload["feature_names"]):
            shap_value = float(payload["shap_values"][index][column])
            yield {"mode": mode, "stage": stage, "partition": "test", "smiles": smiles,
                   "raw_feature_value": payload["raw_feature_values"][index][column],
                   "feature": name, "standardized_feature_value": payload["feature_values"][index][column],
                   "shap_value": shap_value, "shap_rti": shap_value * property_std}


def shap_rank_comparison(development, test, mode):
    """Report test ranking descriptively while preserving the frozen removal order."""
    if development["partition"] != "development" or test["partition"] != "test":
        raise ValueError("Ranking comparison requires development and test evidence")
    names = development["feature_names"]
    if set(names) != set(test["feature_names"]):
        raise ValueError("Development and test SHAP explain different feature configurations")
    dev_mean = np.abs(np.asarray(development["shap_values"], dtype=float)).mean(0)
    test_mean = np.abs(np.asarray(test["shap_values"], dtype=float)).mean(0)
    test_index = {name: index for index, name in enumerate(test["feature_names"])}
    dev_rank = {int(index): rank + 1 for rank, index in enumerate(np.argsort(-dev_mean, kind="stable"))}
    test_rank = {int(index): rank + 1 for rank, index in enumerate(np.argsort(-test_mean, kind="stable"))}
    removal_order = {int(index): rank + 1 for rank, index in enumerate(development["removal_order"])}
    if set(removal_order) != set(range(len(names))):
        raise ValueError("Frozen development removal order must include every feature once")
    return [{"mode": mode, "feature": name,
             "development_mean_absolute_shap": float(dev_mean[index]),
             "development_rank_high_to_low": dev_rank[index],
             "test_mean_absolute_shap": float(test_mean[test_index[name]]),
             "test_rank_high_to_low": test_rank[test_index[name]],
             "frozen_development_removal_order_low_to_high": removal_order[index]}
            for index, name in enumerate(names)]


def build(run_root, output_dir):
    run_root, output_dir = Path(run_root).resolve(), Path(output_dir).resolve()
    manifest = read_json(run_root / "selection_manifest.json")
    payloads, histories, labels = {}, {}, {}
    data_rows = {"steps": [], "predictions": [], "unmasked_shap": [], "final_shap": [],
                 "category_metrics": [], "category_shap": [], "memberships": [], "history": [],
                 "shap_ranks": []}
    for mode in ("pos", "neg"):
        model = manifest["models"][mode]
        mode_dir = run_root / mode
        predictions = read_predictions(mode_dir / "test_predictions.csv")
        steps = read_json(mode_dir / "masking_steps_test.json")
        selected_step, step_predictions = validate_steps(steps, run_root)
        selected_index = next(i for i, step in enumerate(steps) if step["step"] == selected_step)
        if step_predictions[selected_index] != predictions:
            raise ValueError("Final predictions differ from the development-selected masking step")
        selected_record = steps[selected_index]
        if selected_record["removed_features"] != model["removed_features"]:
            raise ValueError("Manifest and selected step use different removed features")
        unmasked = read_json(mode_dir / "unmasked_test_shap.json")
        baseline_model = {**model, "physchem_mask": [1] * 4,
                          "toxicity_mask": [1] * 4 if model["feature_code"] == "GPT" else None}
        unmasked = retained_shap(unmasked, baseline_model)
        development = read_json(resolve_path(manifest["mask_run"], run_root) /
                                (mode + "_" + model["feature_code"]) / "shap_result.json")
        data_rows["shap_ranks"].extend(shap_rank_comparison(development, unmasked, mode))
        final_shap = retained_shap(read_json(mode_dir / "final_test_shap.json"), model)
        if set(unmasked["smiles"]) != {row["smiles"] for row in predictions}:
            raise ValueError("Unmasked SHAP differs from the held-out test sample set")
        tag_rows = read_rows(ROOT / "data" / "MMF-3" / "tags" / (mode + "_compound_tags.csv"))
        tags = {row["SMILES"]: tuple(item.strip() for item in row["Tags"].split(",")) for row in tag_rows}
        if len(tags) != len(tag_rows):
            raise ValueError("Frozen category file contains duplicate SMILES")
        category_metrics, category_shap, memberships = category_evidence(predictions, final_shap, tags)
        histories[mode] = read_history(resolve_path(model["experiment"], run_root))
        labels[mode] = model["model_label"]
        payloads[mode] = {"steps": steps, "selected_step": selected_step, "predictions": predictions,
                          "unmasked_shap": unmasked, "final_shap": final_shap,
                          "category_metrics": category_metrics, "category_shap": category_shap}
        for step, rows in zip(steps, step_predictions):
            data_rows["steps"].append({"mode": mode, "model": model["feature_code"], "partition": "test",
                                       "step": step["step"], "removed_features": "+".join(step["removed_features"]),
                                       "n": len(rows), "selected_by_development": step["step"] == selected_step,
                                       "accepted_by_development": step["accepted"], **metric_values(rows),
                                       **{"development_" + key: step["development_metrics"][key] for key in METRICS}})
            data_rows["predictions"].extend({"mode": mode, "step": step["step"], **row} for row in rows)
        data_rows["unmasked_shap"].extend(shap_long_rows(unmasked, mode, "unmasked"))
        data_rows["final_shap"].extend(shap_long_rows(final_shap, mode, "selected_mask"))
        for name, rows in (("category_metrics", category_metrics), ("category_shap", category_shap),
                           ("memberships", memberships)):
            data_rows[name].extend({"mode": mode, "model": labels[mode], **row} for row in rows)
        for index, epoch in enumerate(histories[mode]["Epoch"]):
            data_rows["history"].append({"mode": mode, "model": labels[mode], "epoch": int(epoch),
                                        **{name: histories[mode][name][index] for name in histories[mode] if name != "Epoch"}})

    figures, data_dir = output_dir / "01_图片", output_dir / "02_绘图数据"
    for directory in (figures, data_dir):
        directory.mkdir(parents=True, exist_ok=True)
    filenames = {
        "steps": "图3绘制所需的逐步掩码完整测试集四指标及开发集选择数据.csv",
        "predictions": "图3绘制所需的逐步掩码各步测试集逐样本预测数据.csv",
        "unmasked_shap": "图3绘制所需的未掩码模型测试集逐样本SHAP数据.csv",
        "shap_ranks": "图3掩码顺序核查所需的开发集SHAP重要性与测试集SHAP重要性对照数据.csv",
        "final_shap": "图4绘制所需的最终模型测试集逐样本SHAP数据.csv",
        "category_metrics": "图4绘制所需的十类化合物样本数与四项指标数据.csv",
        "category_shap": "图4绘制所需的各类别SHAP均值及行内归一化数据.csv",
        "memberships": "图4绘制所需的十类非互斥类别逐样本归属与预测数据.csv",
        "history": "图S5绘制所需的最终模型逐轮训练与开发集指标数据.csv",
    }
    for key, filename in filenames.items():
        write_rows(data_dir / filename, data_rows[key])
    write_rows(data_dir / "图S5绘制所需的最终模型测试集预测散点数据.csv",
               [{"mode": mode, "model": labels[mode], **row}
                for mode in ("pos", "neg") for row in payloads[mode]["predictions"]])
    draw_figure3(payloads, figures)
    draw_figure4(payloads, figures)
    diagnostic_base = figures / "图S5_最终特征配置模型训练过程与测试集预测"
    _draw_figure_s5(diagnostic_base, histories,
                    {mode: payloads[mode]["predictions"] for mode in payloads}, labels)
    from PIL import Image
    with Image.open(str(diagnostic_base) + ".png") as picture:
        picture.thumbnail((2200, 1600))
        picture.save(str(diagnostic_base) + "_预览.png")
    notes = [
        "图3左侧为未掩码GP正模式和GPT负模式的完整测试集描述符SHAP，右侧为各步完整测试集四指标，不做Bootstrap，不绘制误差棒。",
        "图3的selected和stop只沿用已锁定的开发集MAE选择记录，不根据本图的测试集指标重新选择模型。",
        "图3显示的测试集SHAP排序仅用于描述。实际删项顺序固定来自开发集未掩码模型的SHAP排序，两者可能不同；不能根据测试集图中的排序反过来调整删项顺序或重选模型。开发集与测试集的重要性、排名及冻结删项顺序见配套对照CSV，重要性排名1表示最高，删项顺序1表示最先尝试删除。",
        "原测试集在前轮分析中已经查看，本轮固定样本报告不将其重新表述为从未查看的独立测试集。",
        "图S5与图4均使用3.3最终保留的特征配置、权重及其测试集预测，不混用未掩码模型。",
        "图4的十类化合物沿用原有标签，类别不互斥，同一化合物可以出现在多个类别。",
        "图4热图先对每类全部测试样本取各保留特征SHAP的绝对值均值，再在该类内部做(min,max)归一化。常数行归一化为0，CSV中constant_row列标记。",
        "当某个模式仅保留两个特征时，行内min-max归一化会把每个非恒定行映射为0和1。这是归一化方法的结果，不表示两项特征的绝对贡献差距很大；原始平均绝对SHAP保留在同一CSV中供对照。",
        "热图高值紫色、低值黄色。它反映同一类别内模型对保留描述符的相对依赖，不代表特征对预测准确度的因果作用，也不能跨类别比较绝对贡献。",
        "SHAP仍使用各化合物自己的分子图和训练集背景样本。背景采样用于定义归因参考，不是测试集指标Bootstrap重采样。",
        "图中SHAP采用标准化目标尺度。逐样本CSV同时给出原始特征值、标准化特征值和原始RTI尺度的SHAP贡献，shap_rti等于shap_value乘保存的目标标准差。所有误差指标采用原始RTI单位。",
    ]
    (output_dir / "图3图4图S5数据使用说明.txt").write_text("\n".join(notes) + "\n", encoding="utf-8-sig")
    report = {"metric_checks": "passed", "test_sample_counts": {mode: len(payloads[mode]["predictions"]) for mode in payloads},
              "selected_by": "development_mae", "error_bars": None, "bootstrap_used": False,
              "category_count_per_mode": len(CATEGORIES), "visual_review": "pending",
              "selected_steps": {mode: payloads[mode]["selected_step"] for mode in payloads}}
    (output_dir / "图3图4图S5数据核验.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build(args.run_root, args.output_dir), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
