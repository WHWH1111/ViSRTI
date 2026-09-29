#!/usr/bin/env python3
"""Render the original SHAP/masking panel layout from development-only evidence."""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import shap
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/data-val"))
from analyze_gp_gpt_masking import METRICS, metrics, read_csv, write_csv, write_json

LABELS = {
    "Monoiso_Mass": "Monoiso_Mass", "LogKow": "LogKow", "LogP": "LogP",
    "Koc": "Koc", "IC50": "IC₅₀", "48h-LC50": "48h-LC₅₀",
    "72h-EC50": "72h-EC₅₀", "96h-LC50": "96h-LC₅₀",
}
COLORS = ("#F4A6A6", "#A8C8DC", "#A8D5AE", "#CAB3D2")


def bootstrap(y, prediction, indices):
    target, pred = y[indices], prediction[indices]
    residual = target - pred
    centered_y, centered_p = target - target.mean(1)[:, None], pred - pred.mean(1)[:, None]
    return np.column_stack((
        np.mean(np.abs(residual), axis=1), np.sqrt(np.mean(residual ** 2, axis=1)),
        1 - np.sum(residual ** 2, axis=1) / np.sum(centered_y ** 2, axis=1),
        np.sum(centered_y * centered_p, axis=1) /
        np.sqrt(np.sum(centered_y ** 2, axis=1) * np.sum(centered_p ** 2, axis=1)),
    ))


def gather(run_root, data_dir):
    payload, summary_rows, prediction_rows, bootstrap_rows, shap_rows, rank_rows = {}, [], [], [], [], []
    for mode, code in (("pos", "GP"), ("neg", "GPT")):
        mode_dir = run_root / (mode + "_" + code)
        result = json.loads((mode_dir / "shap_result.json").read_text(encoding="utf-8"))
        if result["smoke_only"]:
            raise ValueError("Smoke results cannot be exported as a completed analysis")
        steps = json.loads((mode_dir / "masking_steps.json").read_text(encoding="utf-8"))
        baseline = read_csv(Path(steps[0]["prediction_file"]), "\t")
        y = np.asarray([float(row["Correct"]) for row in baseline])
        smiles = [row["Smiles"] for row in baseline]
        indices = np.random.RandomState(1234).randint(len(y), size=(1000, len(y)))
        selected_step = max(item["step"] for item in steps if item["accepted"])
        for item in steps:
            predictions = read_csv(Path(item["prediction_file"]), "\t")
            if [row["Smiles"] for row in predictions] != smiles:
                raise ValueError("Cannot compare masking stages with different compounds")
            values_y = np.asarray([float(row["Correct"]) for row in predictions])
            if not np.array_equal(y, values_y):
                raise ValueError("Masking stages use different responses")
            pred = np.asarray([float(row["Predict"]) for row in predictions])
            values = metrics(y, pred)
            for name in METRICS:
                if abs(values[name] - item["metrics"][name]) > 1e-7:
                    raise ValueError("Published metric differs from its predictions")
            draws = bootstrap(y, pred, indices)
            sd = draws.std(0, ddof=1)
            item["bootstrap_sd"] = sd.tolist()
            status = "selected" if item["step"] == selected_step else ("stopped" if not item["accepted"] else "accepted")
            summary_rows.append({
                "mode": mode, "model": "ViSRTI-" + code, "partition": "development",
                "step": item["step"], "removed_features": "+".join(item["removed_features"]) or "None",
                "n": len(y), "status": status, **values,
                **{name + "_bootstrap_sd": float(sd[j]) for j, name in enumerate(METRICS)},
            })
            for i, row in enumerate(predictions):
                prediction_rows.append({
                    "mode": mode, "model": "ViSRTI-" + code, "step": item["step"],
                    "smiles": row["Smiles"], "source_rti": float(y[i]), "prediction_rti": float(pred[i]),
                    "residual_source_minus_prediction": float(y[i] - pred[i]),
                })
            for i, draw in enumerate(draws):
                bootstrap_rows.append({"mode": mode, "step": item["step"], "replicate": i + 1,
                                       **{name: float(draw[j]) for j, name in enumerate(METRICS)}})
        shap_rows += read_csv(mode_dir / "图3绘制所需的逐样本SHAP散点数据.csv")
        rank_rows += read_csv(mode_dir / "图3绘制所需的SHAP重要性排序数据.csv")
        shutil.copy2(mode_dir / "SHAP背景样本.csv", data_dir / (mode + "模式_SHAP背景样本.csv"))
        payload[mode] = {"shap": result, "steps": steps, "selected_step": selected_step}
    write_csv(data_dir / "图3绘制所需的掩码柱高和误差棒数据.csv", summary_rows)
    write_csv(data_dir / "图3绘制所需的各步开发集逐样本预测数据.csv", prediction_rows)
    write_csv(data_dir / "图3绘制所需的Bootstrap逐次抽样指标数据.csv", bootstrap_rows)
    write_csv(data_dir / "图3绘制所需的SHAP散点数据.csv", shap_rows)
    write_csv(data_dir / "图3绘制所需的SHAP重要性排序数据.csv", rank_rows)
    return payload


def draw_shap(ax, payload, panel, mode):
    values = np.asarray(payload["shap_values"])
    features = np.asarray(payload["feature_values"])
    names = [LABELS[name] for name in payload["feature_names"]]
    importance = np.mean(np.abs(values), axis=0)
    order = np.argsort(importance)
    # The reference figure specifies green=High and purple=Low.
    cmap = LinearSegmentedColormap.from_list("purple_to_green", ["#CC33CC", "#33CC33"])
    plt.sca(ax)
    np.random.seed(1234)
    shap.summary_plot(values, features, feature_names=names, show=False, color_bar=False,
                      plot_size=None, max_display=len(names), cmap=cmap, alpha=0.82)
    top = ax.twiny()
    top.barh(np.arange(len(names)), importance[order], height=0.6,
             color="lightblue", alpha=0.30, edgecolor="none")
    top.set_xlim(0, float(importance.max()) * 1.13)
    top.set_xlabel("Feature Importance (absolute SHAP value)", fontsize=12, labelpad=9)
    top.tick_params(axis="x", labelsize=10)
    ax.set_zorder(top.get_zorder() + 1)
    ax.patch.set_visible(False)
    ax.set_yticks(np.arange(len(names)))
    ax.set_yticklabels([names[i] for i in order], fontsize=12)
    ax.tick_params(axis="x", labelsize=10)
    ax.set_xlabel("SHAP value (impact on model output)", fontsize=12)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("black")
    ax.text(0.02, 0.98, panel, transform=ax.transAxes, fontsize=17, va="top")
    ax.text(0.97, 0.04, mode + "-RTI", transform=ax.transAxes, fontsize=12, ha="right")
    ax.text(0.97, 0.13, payload["model"], transform=ax.transAxes, fontsize=11, ha="right")
    return cmap


def draw_bars(ax, payload, panel, mode):
    right = ax.twinx()
    steps = payload["steps"]
    x, width = np.arange(len(steps)), 0.20
    for j, (metric, color) in enumerate(zip(METRICS, COLORS)):
        axis = ax if j < 2 else right
        values = [step["metrics"][metric] for step in steps]
        sd = [step["bootstrap_sd"][j] for step in steps]
        bars = axis.bar(x + (j - 1.5) * width, values, width=width, color=color,
                        edgecolor="none", yerr=sd,
                        error_kw={"ecolor": "#777777", "elinewidth": 0.8, "capsize": 2.5})
        for bar, value in zip(bars, values):
            axis.text(bar.get_x() + width / 2, value * 0.47, "{:.4f}".format(value),
                      rotation=90, va="center", ha="center", fontsize=10)
    ymax = max(step["metrics"]["rmse"] + step["bootstrap_sd"][1] for step in steps)
    upper = math.ceil((ymax * 1.14) / 10) * 10
    ax.set_ylim(0, upper)
    ax.set_yticks(np.arange(0, upper + 1, 20))
    right.set_ylim(0, 1.1)
    right.set_yticks(np.arange(0, 1.01, 0.2))
    ax.set_ylabel("MAE & RMSE", fontsize=12)
    right.set_ylabel("R² & PCC", rotation=270, labelpad=18, fontsize=12)
    labels = []
    for step in steps:
        if not step["removed_features"]:
            text = payload["shap"]["model"]
        else:
            text = "mask " + "+".join(LABELS[name] for name in step["removed_features"])
            if len(text) > 23:
                text = text.replace("+", "+\n", 1)
        if step["step"] == payload["selected_step"]:
            text += "\n(selected)"
        elif not step["accepted"]:
            text += "\n(stop)"
        labels.append(text)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10 if len(steps) <= 3 else 8.5)
    ax.set_xlim(-0.55, len(steps) - 0.45)
    ax.set_xlabel(mode + "-RTI (development set)", fontsize=12)
    ax.text(0.02, 0.98, panel, transform=ax.transAxes, fontsize=17, va="top")
    ax.legend([Patch(facecolor=color) for color in COLORS], ["MAE", "RMSE", "R²", "PCC"],
              loc="upper right", ncol=4, fontsize=10, frameon=False,
              handlelength=1.2, columnspacing=0.7)


def render(payload, figures):
    plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman"],
                         "font.size": 11, "axes.linewidth": 0.9, "svg.fonttype": "none", "pdf.fonttype": 42})
    fig, axes = plt.subplots(2, 2, figsize=(12.8, 9.5), gridspec_kw={"width_ratios": [1, 1]})
    fig.subplots_adjust(left=0.12, right=0.94, top=0.92, bottom=0.13, wspace=0.48, hspace=0.56)
    for row, mode in enumerate(("pos", "neg")):
        cmap = draw_shap(axes[row, 0], payload[mode]["shap"], "(a)" if row == 0 else "(c)", mode)
        draw_bars(axes[row, 1], payload[mode], "(b)" if row == 0 else "(d)", mode)
    # SHAP calls tight_layout internally; restore the reference grid afterwards.
    fig.subplots_adjust(left=0.12, right=0.94, top=0.92, bottom=0.14, wspace=0.48, hspace=0.62)
    for row in range(2):
        box = axes[row, 0].get_position()
        cax = fig.add_axes([box.x1 + 0.013, box.y0, 0.004, box.height])
        cb = fig.colorbar(matplotlib.cm.ScalarMappable(norm=Normalize(0, 1), cmap=cmap), cax=cax)
        cb.set_ticks([0, 1])
        cb.set_ticklabels(["Low", "High"])
        cb.ax.tick_params(length=0, labelsize=11)
        cb.outline.set_visible(False)
    fig.text(0.53, 0.025,
             "Development set; bars = full-set metrics; error bars = Bootstrap SD (1,000 resamples).\n"
             "SHAP explains descriptor contributions with each compound's molecular graph held fixed.",
             ha="center", va="bottom", fontsize=10)
    base = figures / "图3_GP与GPT开发集SHAP及逐步掩码"
    for suffix, options in ((".png", {"dpi": 600}), (".pdf", {}), (".svg", {}),
                            (".tif", {"dpi": 600, "pil_kwargs": {"compression": "tiff_lzw"}})):
        fig.savefig(str(base) + suffix, facecolor="white", bbox_inches="tight", **options)
    fig.savefig(figures / "图3_预览.png", dpi=140, facecolor="white", bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    figures, data_dir, code_dir = [output / name for name in ("01_图片", "02_绘图数据", "03_代码及配置")]
    for path in (figures, data_dir, code_dir):
        path.mkdir(parents=True, exist_ok=True)
    payload = gather(args.run_root.resolve(), data_dir)
    render(payload, figures)
    for filename in ("scripts/run_gp_gpt_masking.py", "scripts/data-val/analyze_gp_gpt_masking.py",
                     "scripts/plt/plot_gp_gpt_masking.py", "train_visnet_v2.py", "core/data_preprocessor.py",
                     "models/visnet_v2.py", "tests/test_gp_gpt_masking.py", "docs/gp_gpt_masking_analysis.md"):
        target = code_dir / filename
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / filename, target)
    lines = [
        "本次只更新正模式 GP、负模式 GPT 的 SHAP 和逐步特征掩码图，未修改 Word 和后续图。",
        "正模式 GP 不使用毒性编码器。Figure 2 对应 GPT 的毒性输出维度为正模式 64、负模式 128；两者中间层都是 64。",
        "未掩码 P 为四项：Monoiso_Mass、LogKow、LogP、Koc；T 为四项：IC50、48h-LC50、72h-EC50、96h-LC50。G 是分子图。",
        "本轮使用既有训练集和开发集，正模式开发集 561 个样本，负模式 534 个样本。",
        "每个化合物使用自己的分子图；特征仅按训练时保存的参数标准化一次。SHAP 的背景取自训练集 50 个样本，随机种子 1234。",
        "穷举全部特征子集计算描述符的 Shapley 值，图结构固定。图中数值为标准化目标单位，CSV 另提供 RTI 单位。",
        "按未掩码模型开发集的平均绝对 SHAP 从低到高累计删项，每删一项重新训练。最高 150 轮、开发集 MAE 早停 20 轮，沿用原训练设置。",
        "候选模型开发集 MAE 比前一步变差就停止，保留前一步。初始 SHAP 排序固定，不根据试验结果改顺序。",
        "柱高直接由完整开发集预测计算。误差棒为 1,000 次有放回 Bootstrap 指标的标准差，所有步骤共用每次抽样的样本序号。",
        "Bootstrap 误差棒反映固定模型在该开发集上的抽样波动，不表示多次独立训练的波动，也不是模型筛选后的独立验证置信区间。",
        "正 GP、负 GPT 是客户本轮指定的候选组合。此前开发集负模式 GPTC 的四项指标均优于 GPT，不能改称 GPT 由开发集全候选比较胜出。",
        "原测试集结果已在前轮被查看，无法恢复为未被查看的独立测试集。本轮不读取测试响应，也不自动更新后续分析。",
        "已有 SHAP 实现的共用分子图和重复标准化问题已在本轮分析入口修正；旧稿及旧图保持原状，供作者逐项替换。",
        "负模式保留原训练计算标准化参数时使用的描述符记录；有效训练和开发样本按原分区冻结，已排除的失败分子直接跳过，不反复尝试构象生成，也不纳入性能评价。",
        "SHAP 表示模型归因，不能证明因果关系；低重要性或一次删项改善，不能单独证明特征在其他数据上无用。",
        "03_代码及配置中的代码按原仓库目录保存，用于原工程中的复现；正负模式保留模型文件夹包含实际权重、训练参数和本次删项选择记录。",
        "各步配置记录保留了基准和所有已尝试步骤。02_绘图数据中的完整SHAP结果包含背景期望值和归因数组，可复核每个样本的归因之和。",
        "",
        "逐步掩码结果",
    ]
    for mode in ("pos", "neg"):
        result = payload[mode]
        selected = next(row for row in result["steps"] if row["step"] == result["selected_step"])
        lines.append("{}：保留第 {} 步，删除 {}；开发集 MAE {:.4f}，RMSE {:.4f}，R² {:.4f}，PCC {:.4f}。".format(
            mode, selected["step"], "+".join(selected["removed_features"]) or "无",
            *[selected["metrics"][name] for name in METRICS]))
        mode_dir = args.run_root / (mode + "_" + result["shap"]["feature_code"])
        selected_dir = code_dir / (mode + "_" + result["shap"]["feature_code"] + "_保留模型")
        selected_dir.mkdir(exist_ok=True)
        shutil.copy2(Path(selected["checkpoint"]), selected_dir / "best.pt")
        shutil.copy2(Path(selected["training_params"]), selected_dir / "training_params.json")
        selected_record = json.loads((mode_dir / "selected_mask.json").read_text(encoding="utf-8"))
        selected_record.update({"checkpoint": "best.pt", "training_params": "training_params.json"})
        write_json(selected_dir / "保留配置.json", selected_record)
        step_configs = code_dir / (mode + "_各步配置")
        step_configs.mkdir(exist_ok=True)
        for step in result["steps"]:
            shutil.copy2(Path(step["training_params"]), step_configs / ("第{:02d}步_训练参数.json".format(step["step"])))
        shutil.copy2(mode_dir / "masking_steps.json", step_configs / "各步指标与选择记录.json")
        shutil.copy2(mode_dir / "shap_result.json", data_dir / (mode + "模式_完整SHAP结果及核验.json"))
    (output / "本次修改与数据使用说明.txt").write_text("\n".join(lines) + "\n", encoding="utf-8-sig")
    write_json(output / "数据核验.json", {
        "passed": True, "partitions": {mode: len(payload[mode]["shap"]["smiles"]) for mode in payload},
        "metric_sources": "all_development_predictions", "error_bars": "paired_bootstrap_sd_1000",
        "test_evaluation_performed": False, "visual_review": "pending",
    })
    print("Exported: " + str(output))


if __name__ == "__main__":
    main()
