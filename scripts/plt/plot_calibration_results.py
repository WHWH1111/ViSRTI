#!/usr/bin/env python3
"""Render the two calibration panels from checked final-model predictions."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.ticker import AutoMinorLocator, MultipleLocator
import numpy as np
from scipy.stats import t as student_t

PREDICTIONS = "图6绘制所需的23个物质正负模式预测RTI数据.csv"
COORDINATES = "图6a-b绘制所需的校准拟合坐标与残差数据.csv"
EQUATIONS = "图6a-b绘制所需的校准方程和拟合指标.json"
CASES = "图6案例所需的BaP-BeP预测RT与实测RT误差数据.csv"
STEM = "图6a-b_最终模型RT与RTI校准"
LINE_CSV = "图6a-b绘制所需的RT对预测RTI拟合线及区间数据.csv"
DIRECTION = "experimental_rt_min = intercept_min + slope_min_per_rti * predicted_rti"
COLORS = {"point": "#218eaa", "edge": "#4c8393", "line": "#ff5555", "confidence": "#ffaaaa", "prediction": "#ffe3e3"}


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def read_csv(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def number(value):
    result = float(value)
    if not math.isfinite(result):
        raise ValueError("Calibration contains a nonfinite numeric value")
    return result


def check_close(actual, expected, label):
    if not math.isclose(number(actual), number(expected), rel_tol=1e-7, abs_tol=1e-8):
        raise ValueError("Calibration inconsistency: " + label)


def true_false(value):
    if str(value).lower() not in ("true", "false"):
        raise ValueError("Calibration inclusion flags must be explicit true/false")
    return str(value).lower() == "true"


def validate_calibration(calibration_dir):
    """Do not render a partial run, stale equation, or case-contaminated fit."""
    root = Path(calibration_dir)
    status = read_json(root / "run_status.json")
    if status.get("status") != "completed" or status.get("prediction_generated") is not True:
        raise ValueError("Calibration predictions are not completed; no figure will be generated")
    if (status.get("sample_count"), status.get("calibration_standard_count"), status.get("case_count")) != (23, 21, 2):
        raise ValueError("Expected 21 calibrants plus two separate cases")
    for key, checksum in (("input", "input_sha256"), ("selection_manifest", "selection_sha256")):
        if hashlib.sha256(Path(status[key]).read_bytes()).hexdigest() != status.get(checksum):
            raise ValueError("Calibration source changed since prediction: " + key)
    selection = read_json(status["selection_manifest"])
    if selection.get("status") != "frozen" or selection.get("selection_partition") != "development":
        raise ValueError("A frozen development-selected model configuration is required")
    for mode in ("pos", "neg"):
        for key in ("checkpoint", "training_params"):
            selected = Path(selection["models"][mode][key])
            if not selected.is_absolute():
                selected = Path(status["selection_manifest"]).resolve().parent / selected
            if selected.resolve() != Path(status["model_files"][mode][key]).resolve():
                raise ValueError("Calibration predictions use a different selected model")
    predicted = read_csv(root / PREDICTIONS)
    by_id = {row["compound_id"]: row for row in predicted}
    if len(predicted) != 23 or len(by_id) != 23:
        raise ValueError("Predictions must contain 23 unique compounds")
    if sum(row["role"] == "calibrant" for row in predicted) != 21 or sum(row["role"] == "case" for row in predicted) != 2:
        raise ValueError("Prediction roles differ from 21 plus two")
    input_rows = read_csv(status["input"])
    original = {row["compound_id"]: row for row in input_rows}
    if len(input_rows) != 23 or set(original) != set(by_id):
        raise ValueError("Confirmed input and prediction sample identities differ")
    for identifier, row in by_id.items():
        sample = original[identifier]
        if str(sample["identity_confirmed"]).strip().lower() not in ("true", "1", "yes"):
            raise ValueError("Sample identity remains unconfirmed")
        if sample["role"] != row["role"] or sample["name"] != row["name"] or sample["smiles"] != row["smiles"]:
            raise ValueError("Prediction identity differs from confirmed sample input")
        check_close(sample["measured_rt"], row["experimental_rt_min"], "experimental RT")
    points = read_csv(root / COORDINATES)
    equations = read_json(root / EQUATIONS)
    cases = read_csv(root / CASES)
    if len(points) != 46 or len(equations) != 2 or len(cases) != 4 or {row["mode"] for row in equations} != {"pos", "neg"}:
        raise ValueError("Positive/negative calibration output coverage is incomplete")
    modes = {}
    included_union = set()
    for mode in ("pos", "neg"):
        equation = next(row for row in equations if row["mode"] == mode)
        if equation["direction"] != DIRECTION:
            raise ValueError("The fitted regression direction is not RT on predicted RTI")
        rows = [row for row in points if row["mode"] == mode]
        if len(rows) != 23 or {row["compound_id"] for row in rows} != set(by_id):
            raise ValueError("Missing/duplicate compounds in mode " + mode)
        included = [row for row in rows if true_false(row["included_in_fit"])]
        if len(included) < 2 or any(row["role"] != "calibrant" for row in included):
            raise ValueError("Insufficient calibrants or a case compound entered fitting")
        if int(equation["n_calibrants"]) != len(included) or equation["calibrant_ids"] != [row["compound_id"] for row in included]:
            raise ValueError("Fitted equation and calibration membership differ")
        included_union.update(row["compound_id"] for row in included)
        x = np.asarray([number(row["predicted_rti"]) for row in included])
        y = np.asarray([number(row["experimental_rt_min"]) for row in included])
        if np.ptp(x) <= 0:
            raise ValueError("Calibration RTI coordinates are constant")
        intercept, slope = np.linalg.lstsq(np.column_stack((np.ones(len(x)), x)), y, rcond=None)[0]
        check_close(equation["intercept_min"], intercept, "intercept")
        check_close(equation["slope_min_per_rti"], slope, "slope")
        residual = y - intercept - slope * x
        check_close(equation["calibration_mae_min"], np.mean(np.abs(residual)), "in-sample MAE")
        check_close(equation["calibration_rmse_min"], np.sqrt(np.mean(residual ** 2)), "in-sample RMSE")
        denominator = np.sum((y - y.mean()) ** 2)
        if denominator > 0:
            check_close(equation["calibration_r2"], 1 - np.sum(residual ** 2) / denominator, "in-sample R2")
        elif equation["calibration_r2"] is not None:
            raise ValueError("Constant experimental RT has undefined R2")
        for row in rows:
            saved, sample = by_id[row["compound_id"]], original[row["compound_id"]]
            if row["role"] != saved["role"] or row["name"] != saved["name"]:
                raise ValueError("Coordinate row and prediction identities differ")
            check_close(row["predicted_rti"], saved["predicted_rti_" + mode], "RTI coordinate")
            check_close(row["experimental_rt_min"], saved["experimental_rt_min"], "RT coordinate")
            flag = str(sample["calibration_" + mode]).strip().lower()
            if flag not in ("true", "1", "yes", "false", "0", "no"):
                raise ValueError("Author-confirmed calibration membership is missing")
            expected_flag = flag in ("true", "1", "yes")
            if true_false(row["included_in_fit"]) != expected_flag:
                raise ValueError("Calibration membership differs from author-confirmed input")
            converted = intercept + slope * number(row["predicted_rti"])
            actual = number(row["experimental_rt_min"])
            if actual <= 0:
                raise ValueError("Experimental RT must be positive")
            check_close(row["converted_rt_min"], converted, "converted RT")
            check_close(row["absolute_error_min"], abs(converted - actual), "absolute RT error")
            check_close(row["relative_error_percent"], abs(converted - actual) / actual * 100, "relative RT error")
        if [row for row in rows if row["role"] == "case"] != [row for row in cases if row["mode"] == mode]:
            raise ValueError("Case table and per-mode coordinates differ")
        modes[mode] = {"equation": equation, "points": included, "rti": x, "rt": y}
    if included_union != {row["compound_id"] for row in predicted if row["role"] == "calibrant"}:
        raise ValueError("The confirmed mode lists do not cover all 21 standards")
    return {"status": status, "modes": modes, "calibration_dir": str(root.resolve())}


def fitted_line(data, mode, count=201):
    """Forward OLS intervals are horizontal in the original RT/RTI axes."""
    rti, rt, equation = data["rti"], data["rt"], data["equation"]
    grid = np.linspace(rti.min(), rti.max(), count)
    fitted_rt = equation["intercept_min"] + equation["slope_min_per_rti"] * grid
    interval_ci = interval_pi = None
    if len(rti) > 2:
        residual = rt - equation["intercept_min"] - equation["slope_min_per_rti"] * rti
        variance = np.sum(residual ** 2) / (len(rti) - 2)
        leverage = 1 / len(rti) + (grid - rti.mean()) ** 2 / np.sum((rti - rti.mean()) ** 2)
        critical = student_t.ppf(0.975, len(rti) - 2)
        interval_ci = critical * np.sqrt(variance * leverage)
        interval_pi = critical * np.sqrt(variance * (1 + leverage))
    return [{"mode": mode, "n_calibrants": len(rti), "predicted_rti_y": float(grid[i]),
             "fitted_rt_min_x": float(fitted_rt[i]),
             "rt_confidence95_lower_min": float(fitted_rt[i] - interval_ci[i]) if interval_ci is not None else "",
             "rt_confidence95_upper_min": float(fitted_rt[i] + interval_ci[i]) if interval_ci is not None else "",
             "rt_prediction95_lower_min": float(fitted_rt[i] - interval_pi[i]) if interval_pi is not None else "",
             "rt_prediction95_upper_min": float(fitted_rt[i] + interval_pi[i]) if interval_pi is not None else ""}
            for i in range(count)]


def build_figure(payload, lines):
    plt.rcParams.update({"font.family": "Times New Roman", "mathtext.fontset": "stix", "font.size": 12,
                         "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
                         "axes.linewidth": 0.9, "legend.frameon": False})
    figure, axes = plt.subplots(1, 2, figsize=(12.3, 4.8))
    figure.subplots_adjust(left=0.075, right=0.985, bottom=0.16, top=0.96, wspace=0.24)
    for index, (axis, mode) in enumerate(zip(axes, ("pos", "neg"))):
        data = payload["modes"][mode]
        equation = data["equation"]
        curve = [row for row in lines if row["mode"] == mode]
        y = np.asarray([row["predicted_rti_y"] for row in curve])
        x = np.asarray([row["fitted_rt_min_x"] for row in curve])
        bands = curve[0]["rt_confidence95_lower_min"] != ""
        if bands:
            axis.fill_betweenx(y, [row["rt_prediction95_lower_min"] for row in curve],
                               [row["rt_prediction95_upper_min"] for row in curve], color=COLORS["prediction"], linewidth=0)
            axis.fill_betweenx(y, [row["rt_confidence95_lower_min"] for row in curve],
                               [row["rt_confidence95_upper_min"] for row in curve], color=COLORS["confidence"], linewidth=0)
        line, = axis.plot(x, y, color=COLORS["line"], linewidth=0.8, label="Linear fit")
        points = axis.scatter(data["rt"], data["rti"], s=28, color=COLORS["point"],
                              edgecolor=COLORS["edge"], linewidth=0.5, label="Data", zorder=3)
        axis.set_xlabel("Measured RT (min)", fontsize=14)
        axis.set_ylabel(mode + "-RTI", fontsize=14)
        axis.text(0.015, 0.98, "({})".format(chr(97 + index)), transform=axis.transAxes, fontsize=22, va="top")
        r2 = equation["calibration_r2"]
        annotation = "RT = {:.4f} {:+.6f} RTI\n".format(equation["intercept_min"], equation["slope_min_per_rti"])
        annotation += "$R^2$ = {:.5f}".format(r2) if r2 is not None else "$R^2$ = undefined"
        axis.text(0.12, 0.965, annotation, transform=axis.transAxes, fontsize=12, va="top", linespacing=1.35)
        handles, labels = [points, line], ["Data", "Linear fit"]
        if bands:
            handles += [Patch(facecolor=COLORS["confidence"]), Patch(facecolor=COLORS["prediction"])]
            labels += ["95% Confidence band", "95% Prediction band"]
        axis.legend(handles, labels, loc="lower right", fontsize=11, handlelength=2.2,
                    handletextpad=0.4, borderaxespad=0.65, labelspacing=0.25)
        base_x = (3, 21) if mode == "pos" else (6, 19)
        ranges = np.concatenate((data["rt"], x))
        if bands:
            ranges = np.concatenate((ranges, [row["rt_prediction95_lower_min"] for row in curve],
                                     [row["rt_prediction95_upper_min"] for row in curve]))
        axis.set_xlim(min(base_x[0], math.floor(ranges.min()) - 0.5), max(base_x[1], math.ceil(ranges.max()) + 0.5))
        lower, upper = (-150, 1000) if mode == "pos" else (0, 900)
        axis.set_ylim(min(lower, math.floor(y.min() / 100) * 100 - 50), max(upper, math.ceil(y.max() / 100) * 100 + 50))
        axis.xaxis.set_major_locator(MultipleLocator(2))
        axis.yaxis.set_major_locator(MultipleLocator(100))
        axis.xaxis.set_minor_locator(AutoMinorLocator(2))
        axis.yaxis.set_minor_locator(AutoMinorLocator(2))
        axis.tick_params(which="both", direction="out", top=False, right=False, width=0.8)
        axis.tick_params(which="major", length=4)
        axis.tick_params(which="minor", length=2)
    return figure


def render(calibration_dir, output_dir):
    payload = validate_calibration(calibration_dir)
    lines = sum([fitted_line(payload["modes"][mode], mode) for mode in ("pos", "neg")], [])
    figures, source_data = Path(output_dir) / "01_图片", Path(output_dir) / "02_绘图数据"
    figures.mkdir(parents=True, exist_ok=True)
    source_data.mkdir(parents=True, exist_ok=True)
    figure = build_figure(payload, lines)
    output_files = []
    for extension in ("pdf", "svg", "png", "tiff"):
        options = {"dpi": 600 if extension == "tiff" else 300, "bbox_inches": "tight", "facecolor": "white"}
        if extension == "tiff":
            options["pil_kwargs"] = {"compression": "tiff_lzw"}
        path = figures / (STEM + "." + extension)
        figure.savefig(path, **options)
        output_files.append(path.relative_to(output_dir).as_posix())
    preview = figures / (STEM + "_预览.png")
    figure.savefig(preview, dpi=140, facecolor="white", bbox_inches="tight")
    output_files.append(preview.relative_to(output_dir).as_posix())
    plt.close(figure)
    with (source_data / LINE_CSV).open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(lines[0]))
        writer.writeheader()
        writer.writerows(lines)
    output_files.append((source_data / LINE_CSV).relative_to(output_dir).as_posix())
    report = {
        "numerical_passed": True, "visual_review": "pending", "source": payload["calibration_dir"],
        "input": payload["status"]["input"], "selection_manifest": payload["status"]["selection_manifest"],
        "sample_count": 23, "calibrant_count": 21, "case_count": 2,
        "calibrants_by_mode": {mode: len(payload["modes"][mode]["points"]) for mode in ("pos", "neg")},
        "regression_direction": DIRECTION, "display_x": "measured_RT_min", "display_y": "predicted_RTI",
        "line_display": "x = intercept + slope * predicted_RTI; y = predicted_RTI; no reverse OLS fit",
        "intervals": "analytic t intervals for the RT response under OLS assumptions, displayed horizontally; omitted if n <= 2",
        "independent_validation": False, "figure_c": "not regenerated; existing workflow schematic is separate from numerical calibration",
        "figure_stem": STEM, "line_csv": LINE_CSV,
        "calibration_source_sha256": {name: hashlib.sha256((Path(calibration_dir) / name).read_bytes()).hexdigest()
                                      for name in (PREDICTIONS, COORDINATES, EQUATIONS, CASES)},
        "output_files": output_files,
    }
    (Path(output_dir) / "图6a-b数据核验.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    render(args.calibration_dir, args.output_dir)


if __name__ == "__main__":
    main()
