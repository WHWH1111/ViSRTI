#!/usr/bin/env python3
"""Package the frozen masked-model evidence without training or replacing older deliveries."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
METRICS = ("mae", "rmse", "r2", "pcc")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def read_csv(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        delimiter = "\t" if "\t" in handle.readline() else ","
        handle.seek(0)
        rows = list(csv.DictReader(handle, delimiter=delimiter))
    if not rows:
        raise ValueError("Empty evidence table: " + str(path))
    return rows


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def predictions(rows):
    output = []
    for row in rows:
        smiles = row.get("smiles", row.get("Smiles"))
        y = float(row.get("source_rti", row.get("Correct", row.get("correct"))))
        p = float(row.get("predicted_rti", row.get("Predict", row.get("predict"))))
        if not smiles or not np.isfinite([y, p]).all():
            raise ValueError("Missing identity or non-finite prediction")
        if "residual_source_minus_prediction" in row:
            if not math.isclose(float(row["residual_source_minus_prediction"]), y - p, abs_tol=1e-6):
                raise ValueError("Residual differs from source minus prediction")
        output.append({"smiles": smiles, "source_rti": y, "predicted_rti": p})
    if not output or len({row["smiles"] for row in output}) != len(output):
        raise ValueError("Predictions must have unique compound identities")
    return output


def metrics(rows):
    y = np.asarray([float(row["source_rti"]) for row in rows])
    p = np.asarray([float(row["predicted_rti"]) for row in rows])
    denominator = float(np.sum((y - y.mean()) ** 2))
    if len(y) < 2 or denominator <= 0 or np.std(p) == 0:
        raise ValueError("Four metrics require nonconstant targets and predictions")
    errors = y - p
    return {"mae": float(np.abs(errors).mean()), "rmse": float(np.sqrt(np.mean(errors ** 2))),
            "r2": float(1 - np.sum(errors ** 2) / denominator), "pcc": float(np.corrcoef(y, p)[0, 1])}


def compare_metrics(calculated, stored, source, checks, tolerance=1e-5):
    for metric in METRICS:
        value = float(stored[metric])
        delta = abs(calculated[metric] - value)
        if not math.isfinite(value) or delta > tolerance:
            raise ValueError("Metric mismatch in {}: {} (delta={})".format(source, metric, delta))
        checks.append({"source": str(source), "metric": metric, "absolute_delta": delta})


def comparison_rows(mode, model_label, held_out, folds):
    """Keep whole-held-out, unweighted fold mean/sample SD, and pooled OOF distinct."""
    if len(folds) != 5:
        raise ValueError("Exactly five scaffold folds are required")
    pooled = predictions([row for fold in folds for row in fold])
    fold_metrics = [metrics(fold) for fold in folds]
    rows = []
    for scope, n, estimates, sd in (
        ("固定随机划分完整测试集", len(held_out), metrics(held_out), {}),
        ("scaffold五折指标算术均值", len(pooled),
         {key: float(np.mean([row[key] for row in fold_metrics])) for key in METRICS},
         {key: float(np.std([row[key] for row in fold_metrics], ddof=1)) for key in METRICS}),
        ("scaffold全部折外预测合并计算", len(pooled), metrics(pooled), {}),
    ):
        rows.append({"mode": mode, "model": model_label, "统计口径": scope, "n": n,
                     "folds": 1 if scope.startswith("固定") else 5,
                     **estimates, **{key + "_sample_sd": sd.get(key, "") for key in METRICS}})
    return rows


def validate_scaffold(manifest, selection_path, scaffold, checks):
    state = read_json(scaffold / "run_manifest.json")
    if state["status"] != "completed" or len(state["tasks"]) != 10:
        raise ValueError("Final-mask scaffold run is not complete")
    if state.get("selection_provenance", {}).get("manifest_sha256", "").lower() != digest(selection_path):
        raise ValueError("Scaffold results do not belong to this frozen selection")
    metric_rows = read_csv(scaffold / "all_run_metrics.csv")
    raw_predictions = read_csv(scaffold / "all_test_predictions.csv")
    by_id = {row["task_id"]: row for row in metric_rows}
    if len(by_id) != 10 or set(by_id) != {task["id"] for task in state["tasks"]}:
        raise ValueError("Scaffold metric rows differ from the ten recorded tasks")
    folds = {mode: {} for mode in ("pos", "neg")}
    for task in state["tasks"]:
        mode, fold = task["mode"], int(task["fold"])
        selected = manifest["models"][mode]
        if task["status"] != "completed" or task["model"] != selected["model_label"]:
            raise ValueError("A scaffold task uses an incomplete or different model")
        if task["feature_code"] != selected["feature_code"] or state["model_labels"][mode] != selected["model_label"]:
            raise ValueError("Scaffold model labels differ from the canonical selection")
        for key in ("checkpoint_sha256", "training_params_sha256"):
            if task["selection"][key].lower() != selected[key].lower():
                raise ValueError("Scaffold checkpoint configuration provenance differs")
        actual = read_json(Path(task["log_dir"]) / "training_params.json")
        for key, expected in task["expected_parameters"].items():
            if actual.get(key) != expected:
                raise ValueError("Actual fold parameter differs: " + key)
        raw = [row for row in raw_predictions if row["task_id"] == task["id"]]
        if any(row["mode"] != mode or int(row["fold"]) != fold or row["kind"] != "cv" for row in raw):
            raise ValueError("Scaffold prediction identity differs from task")
        rows = predictions(raw)
        record = by_id[task["id"]]
        if int(record["predicted_test_count"]) != len(rows) or int(record["scaffold_overlap_count"]) != 0:
            raise ValueError("Invalid scaffold count or nonzero scaffold overlap")
        compare_metrics(metrics(rows), record, task["id"], checks)
        folds[mode][fold] = rows
    if any(set(group) != {1, 2, 3, 4, 5} for group in folds.values()):
        raise ValueError("Missing or duplicated scaffold fold")
    prefix = "figure" + str(state["figure_number"])
    source_rows = read_csv(scaffold / (prefix + "_source_data.csv"))
    if source_rows != metric_rows:
        raise ValueError("Scaffold figure source differs from authoritative fold metrics")
    summary = read_csv(scaffold / (prefix + "_summary.csv"))
    for mode in folds:
        records = [row for row in summary if row["mode"] == mode]
        if len(records) != 1:
            raise ValueError("Missing scaffold summary mode")
        record = records[0]
        values = [metrics(rows) for rows in folds[mode].values()]
        for key in METRICS:
            for suffix, value in (("mean", np.mean([row[key] for row in values])),
                                  ("sd", np.std([row[key] for row in values], ddof=1))):
                if not math.isclose(float(record[key + "_" + suffix]), value, abs_tol=1e-6):
                    raise ValueError("Scaffold mean/SD table differs from fold predictions")
    report = read_json(scaffold / "aggregation_report.json")
    if report["cv_run_count"] != 10 or not report["all_scaffold_overlaps_zero"]:
        raise ValueError("Scaffold aggregation has not passed its checks")
    return state, folds, prefix


def validate_categories(metric_rows, member_rows, source, checks):
    for record in metric_rows:
        members = [row for row in member_rows if row["mode"] == record["mode"] and row["category"] == record["category"]]
        rows = predictions(members)
        if len(rows) != int(record["n"]):
            raise ValueError("Category sample count differs: " + str(source))
        compare_metrics(metrics(rows), record, str(source) + "/" + record["category"], checks)


def validate_shap(payload, rows, mask):
    if payload["partition"] != "test" or payload.get("smoke_only") or payload["used_for_selection"]:
        raise ValueError("SHAP must describe full test predictions without model selection")
    if payload["smiles"] != [row["smiles"] for row in rows]:
        raise ValueError("SHAP compounds differ from their predictions")
    values = np.asarray(payload["shap_values"], dtype=float)
    if values.shape != (len(rows), len(mask)) or not np.isfinite(values).all():
        raise ValueError("SHAP shape or finite-value check failed")
    if list(payload["feature_mask"]) != list(mask):
        raise ValueError("SHAP mask differs from its prediction checkpoint")
    for index, keep in enumerate(mask):
        if not keep and np.max(np.abs(values[:, index])) > 1e-6:
            raise ValueError("A removed feature still has SHAP contribution")
    reconstructed = (np.asarray(payload["base_values"]) + values.sum(1)) * payload["property_std"] + payload["property_mean"]
    if np.max(np.abs(reconstructed - np.asarray([row["predicted_rti"] for row in rows]))) > 0.003:
        raise ValueError("SHAP does not add up to the corresponding predictions")


def validate_fixed_tables(run_root, data_dir, manifest, held_out, checks):
    step_table = read_csv(data_dir / "图3绘制所需的逐步掩码完整测试集四指标及开发集选择数据.csv")
    step_details = read_csv(data_dir / "图3绘制所需的逐步掩码各步测试集逐样本预测数据.csv")
    scatter = read_csv(data_dir / "图S5绘制所需的最终模型测试集预测散点数据.csv")
    category_table = read_csv(data_dir / "图4绘制所需的十类化合物样本数与四项指标数据.csv")
    memberships = read_csv(data_dir / "图4绘制所需的十类非互斥类别逐样本归属与预测数据.csv")
    category_shap = read_csv(data_dir / "图4绘制所需的各类别SHAP均值及行内归一化数据.csv")
    for mode in ("pos", "neg"):
        if predictions([row for row in scatter if row["mode"] == mode]) != held_out[mode]:
            raise ValueError("Diagnostic scatter CSV differs from final predictions")
        steps = read_json(run_root / mode / "masking_steps_test.json")
        if len([row for row in step_table if row["mode"] == mode]) != len(steps):
            raise ValueError("Figure 3 step table is incomplete")
        for step in steps:
            rows = predictions(read_csv(step["prediction_file"]))
            matches = [row for row in step_table if row["mode"] == mode and int(row["step"]) == step["step"]]
            details = predictions([row for row in step_details if row["mode"] == mode and int(row["step"]) == step["step"]])
            if len(matches) != 1 or details != rows:
                raise ValueError("Figure 3 CSV differs from its source step predictions")
            compare_metrics(metrics(rows), matches[0], "Figure3/" + mode + "/" + str(step["step"]), checks)
            compare_metrics(step["development_metrics"],
                            {key: matches[0]["development_" + key] for key in METRICS}, "Figure3/development", checks)
        selected = manifest["models"][mode]
        shap = read_json(run_root / mode / "final_test_shap.json")
        mask = list(selected["physchem_mask"]) + list(selected.get("toxicity_mask") or [])
        validate_shap(shap, held_out[mode], mask)
        base_shap = read_json(run_root / mode / "unmasked_test_shap.json")
        validate_shap(base_shap, predictions(read_csv(steps[0]["prediction_file"])), [1] * len(mask))
        tags = {row["SMILES"]: set(part.strip() for part in row["Tags"].split(","))
                for row in read_csv(ROOT / "data/MMF-3/tags" / (mode + "_compound_tags.csv"))}
        names = [name for name, keep in zip(shap["feature_names"], mask) if keep]
        values = np.asarray(shap["shap_values"])[:, np.asarray(mask, dtype=bool)]
        lookup = {smiles: index for index, smiles in enumerate(shap["smiles"])}
        categories = [row for row in category_table if row["mode"] == mode]
        if len(categories) != 10 or len({row["category"] for row in categories}) != 10:
            raise ValueError("Figure 4 does not contain exactly ten nonexclusive categories")
        for category in categories:
            expected = [row for row in held_out[mode] if category["category"] in tags[row["smiles"]]]
            actual = predictions([row for row in memberships if row["mode"] == mode and row["category"] == category["category"]])
            if actual != expected:
                raise ValueError("Category membership differs from frozen labels and predictions")
            mean_abs = np.abs(values[[lookup[row["smiles"]] for row in actual]]).mean(0)
            span = np.ptp(mean_abs)
            normalized = (mean_abs - mean_abs.min()) / span if span else np.zeros_like(mean_abs)
            heat = [row for row in category_shap if row["mode"] == mode and row["category"] == category["category"]]
            if [row["feature"] for row in heat] != names:
                raise ValueError("Heatmap feature order differs from selected SHAP")
            if not np.allclose([float(row["mean_absolute_shap"]) for row in heat], mean_abs, rtol=1e-7, atol=1e-9):
                raise ValueError("Category absolute SHAP means differ")
            if not np.allclose([float(row["normalized_mean_absolute_shap"]) for row in heat], normalized, rtol=1e-7, atol=1e-9):
                raise ValueError("Category row-normalized SHAP differs")
    validate_categories(category_table, memberships, "Figure4/category_metrics", checks)


def copy_checked(source, target):
    source, target = Path(source), Path(target)
    if not source.is_file():
        raise FileNotFoundError(source)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if digest(source) != digest(target):
            raise FileExistsError("Refusing to overwrite a different delivered artifact: " + str(target))
    else:
        shutil.copy2(source, target)


def calibration_report_targets(delivery):
    folder = Path(delivery).resolve() / "03_校准输入待补"
    return {folder / "本轮校准计算状态.json": "run_status.json",
            folder / "本轮校准计算所需的待补资料明细.csv": "missing_inputs.csv"}


def check_calibration_report_update(source, target, delivery, selection_sha256):
    """Only two status reports can change after the same selection was packaged."""
    source, target = Path(source), Path(target).resolve()
    allowed = calibration_report_targets(delivery)
    if target not in allowed or source.name != allowed[target]:
        raise ValueError("Not an explicitly mutable calibration report")
    if target.exists() and digest(source) != digest(target):
        qa_path = Path(delivery) / "数据核验.json"
        if not qa_path.is_file() or read_json(qa_path).get("selection_sha256") != selection_sha256:
            raise ValueError("Calibration report update requires the same frozen selection")


def copy_calibration_report(source, target, delivery, selection_sha256):
    check_calibration_report_update(source, target, delivery, selection_sha256)
    Path(target).parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)


def read_missing_inputs(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != ["compound_id", "field", "reason"]:
            raise ValueError("Unexpected missing-input report columns")
        return list(reader)


def calibration_artifacts(run_root, delivery, manifest):
    source = run_root / "calibration"
    state = read_json(source / "run_status.json")
    target = delivery / "03_校准输入待补"
    copies = [(source / "run_status.json", target / "本轮校准计算状态.json")]
    if state["status"] == "missing_inputs":
        if state["prediction_generated"] or state["issues"] <= 0:
            raise ValueError("Calibration missing-input status is inconsistent")
        issues = read_missing_inputs(source / "missing_inputs.csv")
        if len(issues) != int(state["issues"]):
            raise ValueError("Calibration missing-input count differs")
        copies.append((source / "missing_inputs.csv", target / "本轮校准计算所需的待补资料明细.csv"))
        note = "3.7 尚未完成。本轮校准输入核查仍有{}项缺项，详见03_校准输入待补中的本轮清单。没有生成预测或拟合结果，也没有采用旧预测填补。".format(len(issues))
        return state["status"], copies, note
    if state["status"] != "completed" or not state["prediction_generated"]:
        raise ValueError("Calibration stage must be completed or explicitly missing inputs")
    if read_missing_inputs(source / "missing_inputs.csv"):
        raise ValueError("Completed calibration still reports unresolved inputs")
    copies.append((source / "missing_inputs.csv", target / "本轮校准计算所需的待补资料明细.csv"))
    if state["sample_count"] != 23 or state["calibration_standard_count"] != 21 or state["case_count"] != 2:
        raise ValueError("Calibration sample roles differ from 21 plus 2")
    for mode in ("pos", "neg"):
        for key in ("checkpoint", "training_params"):
            if digest(state["model_files"][mode][key]) != manifest["models"][mode][key + "_sha256"].lower():
                raise ValueError("Calibration predictions use a different selected model")
    prediction_name = "图6绘制所需的23个物质正负模式预测RTI数据.csv"
    coordinate_name = "图6a-b绘制所需的校准拟合坐标与残差数据.csv"
    equation_name = "图6a-b绘制所需的校准方程和拟合指标.json"
    case_name = "图6案例所需的BaP-BeP预测RT与实测RT误差数据.csv"
    predicted = read_csv(source / prediction_name)
    by_id = {row["compound_id"]: row for row in predicted}
    if len(by_id) != 23 or len(predicted) != 23:
        raise ValueError("Calibration predictions do not contain 23 unique substances")
    coordinates = read_csv(source / coordinate_name)
    equations = read_json(source / equation_name)
    cases = read_csv(source / case_name)
    if len(coordinates) != 46 or len(cases) != 4 or len(equations) != 2:
        raise ValueError("Calibration output coverage is incomplete")
    for mode in ("pos", "neg"):
        points = [row for row in coordinates if row["mode"] == mode]
        equation = next(row for row in equations if row["mode"] == mode)
        included = [row for row in points if row["included_in_fit"].lower() == "true"]
        if any(row["role"] != "calibrant" for row in included) or len(included) != int(equation["n_calibrants"]):
            raise ValueError("Case compound entered calibration or calibrant counts differ")
        x = np.asarray([float(row["predicted_rti"]) for row in included])
        y = np.asarray([float(row["experimental_rt_min"]) for row in included])
        intercept, slope = np.linalg.lstsq(np.column_stack((np.ones(len(x)), x)), y, rcond=None)[0]
        if not np.allclose([intercept, slope], [equation["intercept_min"], equation["slope_min_per_rti"]], rtol=1e-7, atol=1e-8):
            raise ValueError("Calibration equation differs from its coordinates")
        for point in points:
            raw = by_id[point["compound_id"]]
            rti, rt = float(point["predicted_rti"]), float(point["experimental_rt_min"])
            if rti != float(raw["predicted_rti_" + mode]) or rt != float(raw["experimental_rt_min"]):
                raise ValueError("Calibration coordinates differ from per-compound predictions")
            converted = intercept + slope * rti
            if not math.isclose(float(point["converted_rt_min"]), converted, abs_tol=1e-8):
                raise ValueError("Converted RT differs from fitted equation")
            if not math.isclose(float(point["absolute_error_min"]), abs(converted - rt), abs_tol=1e-8):
                raise ValueError("Calibration case error differs")
        if [row for row in points if row["role"] == "case"] != [row for row in cases if row["mode"] == mode]:
            raise ValueError("Case CSV differs from calibration coordinate records")
    copies.extend((source / filename, delivery / "02_绘图数据" / filename)
                  for filename in (prediction_name, coordinate_name, equation_name, case_name))
    copies.append((Path(state["input"]), target / "图6计算所用的作者确认输入数据.csv"))
    note = "3.7 已完成23个物质的正负RTI预测和校准拟合数据核对。21个标准品按确认名单用于校准，BaP与BeP两个案例未进入拟合。图6绘图数据位于02_绘图数据，图片视觉检查另行确认。"
    return state["status"], copies, note


def write_csv(path, rows):
    with Path(path).open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def validate_calibration_figures(run_root, delivery):
    report = read_json(delivery / "图6a-b数据核验.json")
    if report.get("numerical_passed") is not True:
        raise ValueError("Calibration figure numerical checks have not passed")
    names = ("图6绘制所需的23个物质正负模式预测RTI数据.csv",
             "图6a-b绘制所需的校准拟合坐标与残差数据.csv",
             "图6a-b绘制所需的校准方程和拟合指标.json",
             "图6案例所需的BaP-BeP预测RT与实测RT误差数据.csv")
    expected = {name: digest(run_root / "calibration" / name) for name in names}
    if report.get("calibration_source_sha256") != expected:
        raise ValueError("Calibration figure sources differ from the current predictions")
    files = report.get("output_files", [])
    required = ["01_图片/图6a-b_最终模型RT与RTI校准" + suffix for suffix in (".png", ".pdf", ".svg")]
    required.append("02_绘图数据/图6a-b绘制所需的RT对预测RTI拟合线及区间数据.csv")
    if not set(required).issubset({str(Path(name).as_posix()) for name in files}):
        raise ValueError("Calibration figure exports are incomplete")
    for relative in files:
        path = (delivery / relative).resolve()
        if delivery.resolve() not in path.parents or not path.is_file():
            raise ValueError("Missing or out-of-delivery calibration figure export")
    return report


def build(run_root, delivery, scaffold=None):
    run_root, delivery = Path(run_root).resolve(), Path(delivery).resolve()
    scaffold = Path(scaffold or ROOT / "outputs/final_masked_scaffold_seed1234").resolve()
    selection_path = run_root / "selection_manifest.json"
    manifest = read_json(selection_path)
    if manifest["selection_partition"] != "development" or manifest["test_used_for_mask_stopping"]:
        raise ValueError("Selection must be frozen from development-only masking")
    qa_path = delivery / "数据核验.json"
    if qa_path.exists() and read_json(qa_path).get("selection_sha256") != digest(selection_path):
        raise ValueError("Delivery folder already belongs to a different selection")
    checks, comparisons, copies = [], [], []
    image_dir, data_dir = delivery / "01_图片", delivery / "02_绘图数据"
    evidence_dir, model_dir, code_dir = delivery / "06_逐步计算依据", delivery / "04_最终模型", delivery / "05_复现代码"
    fixed_report = read_json(delivery / "图3图4图S5数据核验.json")
    if fixed_report["metric_checks"] != "passed" or fixed_report["bootstrap_used"] or fixed_report["error_bars"] is not None:
        raise ValueError("Fixed-test figures are missing or use bootstrap/error bars")
    held_out = {}
    portable = json.loads(json.dumps(manifest))
    for mode in ("pos", "neg"):
        label = "正模式" if mode == "pos" else "负模式"
        selected = manifest["models"][mode]
        mode_root = run_root / mode
        dev_root = Path(manifest["mask_run"]) / (mode + "_" + selected["feature_code"])
        for key in ("checkpoint", "training_params"):
            if digest(selected[key]) != selected[key + "_sha256"].lower():
                raise ValueError("Frozen selected artifact changed: " + key)
        steps = read_json(mode_root / "masking_steps_test.json")
        dev_steps = read_json(dev_root / "masking_steps.json")
        if [row["step"] for row in steps] != [row["step"] for row in dev_steps]:
            raise ValueError("Test and development step chains differ")
        held_out[mode] = predictions(read_csv(mode_root / "test_predictions.csv"))
        baseline = Path(manifest["source"]) / "experiments" / (selected["feature_code"] + "_" + mode)
        frozen_test = predictions(read_csv(baseline / "train_prediction_test.txt"))
        identity = lambda rows: [(row["smiles"], row["source_rti"]) for row in rows]
        if identity(held_out[mode]) != identity(frozen_test):
            raise ValueError("Final held-out sample identities or targets changed")
        if fixed_report["test_sample_counts"][mode] != len(held_out[mode]):
            raise ValueError("Plot report sample counts differ")
        if fixed_report["selected_steps"][mode] != selected["selected_step"]:
            raise ValueError("Plot report selected steps differ")
        selected_records = [row for row in steps if row["selected"]]
        if len(selected_records) != 1 or selected_records[0]["step"] != selected["selected_step"]:
            raise ValueError("Selected test step differs from frozen manifest")
        compare_metrics(metrics(held_out[mode]), selected_records[0]["metrics"], mode + "/final_test", checks)
        stopped, previous = False, None
        for step, dev_step in zip(steps, dev_steps):
            if stopped or (previous is not None and step["accepted"] != (step["development_metrics"]["mae"] <= previous + 1e-8)):
                raise ValueError("Mask stopping does not follow development MAE")
            previous = step["development_metrics"]["mae"]
            stopped = not step["accepted"]
            test_rows = predictions(read_csv(step["prediction_file"]))
            dev_rows = predictions(read_csv(dev_step["prediction_file"]))
            if identity(test_rows) != identity(frozen_test):
                raise ValueError("A masking step changed the held-out samples")
            compare_metrics(metrics(test_rows), step["metrics"], mode + "/test_step" + str(step["step"]), checks)
            compare_metrics(metrics(dev_rows), step["development_metrics"], mode + "/dev_step" + str(step["step"]), checks)
            target = evidence_dir / label / ("掩码第{:02d}步".format(step["step"]))
            copies.extend([(Path(step["prediction_file"]), target / "图3绘制所需的本步测试集逐样本预测数据.csv"),
                           (Path(dev_step["prediction_file"]), target / "开发集选择所需的本步逐样本预测数据.tsv"),
                           (Path(step["training_params"]), target / "本步训练配置.json")])
        for filename, chinese in (("unmasked_test_shap.json", "图3绘制所需的未掩码测试集完整SHAP数据.json"),
                                  ("final_test_shap.json", "图4绘制所需的最终模型测试集完整SHAP数据.json"),
                                  ("masking_steps_test.json", "图3绘制所需的逐步掩码测试指标与开发集选择记录.json")):
            copies.append((mode_root / filename, evidence_dir / label / chinese))
        for filename, chinese in (("shap_result.json", "开发集特征排序所需的完整SHAP数据.json"),
                                  ("masking_steps.json", "开发集逐步掩码选择记录.json"),
                                  ("selected_mask.json", "开发集最终保留配置.json"),
                                  ("训练及开发集特征.csv", "SHAP计算所需的训练及开发集特征数据.csv"),
                                  ("SHAP背景样本.csv", "SHAP计算所需的训练集背景样本数据.csv")):
            copies.append((dev_root / filename, evidence_dir / label / chinese))
        copies.append((baseline / "partition_assignments.csv", evidence_dir / label / "固定随机划分全部样本归属数据.csv"))
        copies.append((ROOT / "data/MMF-3/tags" / (mode + "_compound_tags.csv"),
                       data_dir / ("图4及图5绘制所需的" + label + "全部化合物非互斥类别标签数据.csv")))
        for key, filename in (("checkpoint", "最终模型权重.pt"), ("training_params", "最终模型训练配置.json")):
            copies.append((Path(selected[key]), model_dir / label / filename))
            portable["models"][mode][key] = str(Path(label) / filename)
    scaffold_state, folds, prefix = validate_scaffold(manifest, selection_path, scaffold, checks)
    validate_fixed_tables(run_root, data_dir, manifest, held_out, checks)
    for mode in held_out:
        comparisons.extend(comparison_rows(mode, manifest["models"][mode]["model_label"], held_out[mode],
                                           [folds[mode][index] for index in range(1, 6)]))
    category_file = scaffold / (prefix + "_category_metrics.csv")
    category_predictions = scaffold / (prefix + "_category_predictions.csv")
    validate_categories(read_csv(category_file), read_csv(category_predictions), category_file.name, checks)
    figure_number = str(scaffold_state["figure_number"])
    figure_label = "图" + figure_number
    names = {"all_run_metrics.csv": "结构五折每折完整指标及重算核对数据.csv",
             "all_test_predictions.csv": "结构五折全部折外逐样本预测数据.csv",
             prefix + "_source_data.csv": "结构五折绘图指标数据.csv",
             prefix + "_summary.csv": "结构五折指标均值及样本标准差数据.csv",
             prefix + "_category_metrics.csv": "十类化合物折外预测样本数及指标数据.csv",
             prefix + "_category_predictions.csv": "十类非互斥化合物折外归属及逐样本预测数据.csv",
             prefix + "_category_coverage.csv": "全部类别标签与有效折外预测覆盖数据.csv"}
    for filename, chinese in names.items():
        copies.append((scaffold / filename, data_dir / (figure_label + "绘制所需的" + chinese)))
    for mode, panel in (("pos", "a"), ("neg", "b")):
        for suffix in (".png", ".pdf", ".svg", ".tiff"):
            filename = "Figure" + figure_number + panel + "_scaffold_oof_categories_" + mode + suffix
            copies.append((scaffold / filename, image_dir / (figure_label + panel + "_" + mode + "模式最终模型结构五折十类化合物残差图" + suffix)))
    for suffix in (".png", ".pdf", ".svg", ".tiff"):
        copies.append((scaffold / ("scaffold_cv_fold_metrics" + suffix), image_dir / ("补充图_最终模型结构五折逐折指标" + suffix)))
    copies.extend([(selection_path, evidence_dir / "最终模型原始来源配置.json"),
                   (scaffold / "run_manifest.json", evidence_dir / "结构五折训练任务与模型来源记录.json"),
                   (scaffold / "aggregation_report.json", evidence_dir / "结构五折原始聚合核对报告.json")])
    calibration_status, calibration_copies, calibration_note = calibration_artifacts(run_root, delivery, manifest)
    if calibration_status == "completed":
        validate_calibration_figures(run_root, delivery)
    copies.extend(calibration_copies)
    scripts = ["train_visnet_v2.py", "train_visnet_v2_5.py", "environment.yml", "docs/final_model_updates.md", "scripts/run_gp_gpt_masking.py",
               "scripts/run_final_model_updates.py", "scripts/run_selected_model_scaffold_cv.ps1",
               "scripts/data-val/analyze_gp_gpt_masking.py", "scripts/data-val/evaluate_final_masked_models.py",
               "scripts/data-val/predict_calibration_samples.py",
               "scripts/data-val/package_final_model_results.py", "scripts/data-val/aggregate_scaffold_results.py",
               "scripts/data-val/build_selected_model_fixed_split_results.py",
               "scripts/plt/plot_gp_gpt_masking.py", "scripts/plt/plot_final_masked_models.py",
               "scripts/plt/plot_calibration_results.py"]
    scripts += [str(path.relative_to(ROOT)) for folder in ("core", "models", "utils") for path in (ROOT / folder).rglob("*.py")]
    copies.extend((ROOT / relative, code_dir / relative) for relative in scripts)
    for stem in ("图3_GP与GPT完整测试集SHAP及逐步掩码", "图4_最终模型十类化合物性能与SHAP热图",
                 "图S5_最终特征配置模型训练过程与测试集预测"):
        for suffix in (".png", ".pdf", ".svg", ".tif"):
            if not (image_dir / (stem + suffix)).is_file():
                raise FileNotFoundError("Required final figure is missing: " + str(image_dir / (stem + suffix)))
    mutable_reports = calibration_report_targets(delivery)
    selection_sha256 = digest(selection_path)
    for source, target in copies:
        if not source.is_file():
            raise FileNotFoundError("Required final artifact is missing: " + str(source))
        if target.resolve() in mutable_reports:
            check_calibration_report_update(source, target, delivery, selection_sha256)
        elif target.exists() and digest(source) != digest(target):
            raise FileExistsError("A different delivery artifact already exists: " + str(target))
    for source, target in copies:
        if target.resolve() in mutable_reports:
            copy_calibration_report(source, target, delivery, selection_sha256)
        else:
            copy_checked(source, target)
    write_csv(data_dir / "补充表所需的固定随机测试集与结构五折均值及合并折外指标对照数据.csv", comparisons)
    (model_dir / "最终模型配置.json").write_text(json.dumps(portable, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    notes = [
        "图3、图4、图S5和结构五折结果均使用最终特征配置，图表沿用论文中的视觉样式。",
        "图3柱高为固定测试集全部有效样本一次性计算的四项指标，没有Bootstrap，也没有误差棒。掩码的排序和停止只由开发集决定。",
        "模型筛选范围为正模式GP和负模式GPT；负模式GPT不能称为开发集全部候选比较的胜出者。测试集结果此前已被查看，不能视为从未查看的独立测试集。",
        "图4与图S5使用最终保留的同一套特征、权重及测试预测。十类化合物不互斥，类别SHAP热图表示模型相对依赖，不作因果解释。",
        "结构五折采用相同最终特征配置，每折重新训练。它是固定配置的结构稳健性验证，没有在各外折内部重新选择特征，因此不是nested CV。",
        "对照表分别列出完整随机测试集指标、五个结构折指标的算术均值及样本标准差、全部折外预测合并计算指标。三种统计口径不能混用。",
        "06_逐步计算依据保存每步开发集与测试集预测、训练配置、完整SHAP和最终选择记录。04_最终模型保存两个最终权重与训练配置。05_复现代码保持仓库中的相对目录结构；运行仍需原仓库数据和冻结划分。",
        calibration_note,
        "本项目Methods使用模型预测RTI与实测RT拟合，新模型预测变化后应重新拟合，实测RT保持原值。21个标准品用于校准，BaP和BeP是另两个案例，不能称23个独立验证样本。",
        "数值与图片视觉检查分别记录；视觉检查状态见数据核验.json。",
    ]
    (delivery / "最终模型结果与数据说明.txt").write_text("\n\n".join(notes) + "\n", encoding="utf-8-sig")
    code_notes = (
        "本目录提供最终模型分析所用代码的交付副本，不是独立的数据集或独立安装环境。\n"
        "请在 ViSRTI 仓库根目录和 pytorch38-gpu 环境运行，保留 data 中的原始数据与冻结划分。\n"
        "scripts/run_final_model_updates.py 按已完成阶段接续；不要同时启动重复训练。\n"
        "scripts/data-val/evaluate_final_masked_models.py 负责最终权重的完整测试集预测和逐样本图缓存SHAP。\n"
        "scripts/plt/plot_final_masked_models.py 负责图3、图4、图S5；scripts/run_selected_model_scaffold_cv.ps1 使用最终SelectionManifest完成结构五折。\n"
        "scripts/data-val/predict_calibration_samples.py 负责校准样品资料检查及补齐后的无标签预测，缺项时不能生成正式预测。\n"
        "scripts/plt/plot_calibration_results.py 负责图6a-b，校验23物质预测及校准数据后沿用原轴方向绘图，不重做反方向回归。\n"
        "core、models、utils 保持原仓库结构，图缓存仍由原分子处理入口及 analyze_gp_gpt_masking.py 读取。\n"
        "交付的04_最终模型/最终模型配置.json 使用该目录内相对权重路径；06_逐步计算依据中的原始配置保留计算时路径用于追溯。\n"
    )
    (code_dir / "复现代码使用说明.txt").write_text(code_notes, encoding="utf-8-sig")
    status_notes = [
        "本轮3.3至3.6计算及数值核对已经完成，图片仍待逐张视觉检查。论文Word未修改。",
        "3.1与3.2已经确认的架构和融合结果保留，没有重新训练这些部分。",
    ]
    for mode, label in (("pos", "正模式"), ("neg", "负模式")):
        selected = manifest["models"][mode]
        status_notes.append("3.3 " + label + "保留 " + selected["model_label"] + "，开发集选择第" + str(selected["selected_step"]) + "步。")
    status_notes.extend([
        "3.4 图S5已更新为最终掩码模型自己的训练曲线及固定测试集预测。",
        "3.5 图4已更新为相同最终模型的十类化合物性能及类别内归一化SHAP。",
        "3.6 正负各五折scaffold已重新训练，图和随机/结构对照表均由本轮结果生成，不使用旧GP/GPTC折外结果。",
        calibration_note,
        "图3使用完整固定测试集指标，不做Bootstrap、不加误差棒。筛选仍依据开发集。历史测试集已经查看，结构五折也不是嵌套特征选择CV。",
        "当前状态为已完成可执行计算、视觉检查待确认" + ("、校准输入待补。" if calibration_status == "missing_inputs" else "。") + "不能在视觉检查前称整个交付全部完成。",
    ])
    (delivery / "当前修改进度与待补资料.txt").write_text("\n\n".join(status_notes) + "\n", encoding="utf-8-sig")
    report = {"selection_sha256": digest(selection_path), "numerical_passed": True,
              "checks_performed_this_run": checks, "max_metric_absolute_delta": max(row["absolute_delta"] for row in checks),
              "test_sample_counts": {mode: len(rows) for mode, rows in held_out.items()},
              "selected_steps": {mode: model["selected_step"] for mode, model in manifest["models"].items()},
              "copied_artifacts": len(copies), "visual_review": "pending",
              "calibration_23_status": calibration_status,
              "overall_delivery_complete": False}
    qa_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--delivery", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build(args.run_root, args.delivery), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
