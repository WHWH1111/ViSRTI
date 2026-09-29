#!/usr/bin/env python3
"""Predict confirmed calibration/case samples using the frozen final models."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
FEATURES = ("Monoiso_Mass", "LogKow", "LogP", "Koc", "IC50", "48h-LC50", "72h-EC50", "96h-LC50")
FIELDS = ("compound_id", "name", "role", "measured_rt", "smiles") + FEATURES + (
    "calibration_pos", "calibration_neg", "identity_confirmed", "source_note")
PREDICTION_NAME = "图6绘制所需的23个物质正负模式预测RTI数据.csv"
RESULT_FILES = (PREDICTION_NAME, "图6a-b绘制所需的校准拟合坐标与残差数据.csv",
                "图6a-b绘制所需的校准方程和拟合指标.json", "图6案例所需的BaP-BeP预测RT与实测RT误差数据.csv")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def reusable_prediction(output_dir, input_path, selection_manifest):
    state_file = output_dir / "run_status.json"
    if not state_file.is_file():
        return False
    state = json.loads(state_file.read_text(encoding="utf-8"))
    return (state.get("status") == "completed" and state.get("prediction_generated") is True
            and state.get("input_sha256") == digest(input_path)
            and state.get("selection_sha256") == digest(selection_manifest)
            and all((output_dir / filename).is_file() for filename in RESULT_FILES))


def read_csv(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows, fields=None):
    with Path(path).open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields or list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def boolean(value):
    value = str(value).strip().lower()
    if value in ("true", "1", "yes"):
        return True
    if value in ("false", "0", "no"):
        return False
    return None


def finite_number(value):
    try:
        result = float(value)
    except (ValueError, TypeError):
        return None
    return result if math.isfinite(result) else None


def active_features(params):
    level = params["visnet_v2_feature_level"]
    if level not in ("graph_physchem", "graph_physchem_toxicity"):
        raise ValueError("Calibration requires the selected GP/GPT models, not a substituted model")
    mask_p = params.get("visnet_v2_physchem_mask") or [1] * 4
    mask_t = params.get("visnet_v2_toxicity_mask") or [1] * 4
    if len(mask_p) != 4 or len(mask_t) != 4 or any(value not in (0, 1) for value in mask_p + mask_t):
        raise ValueError("Expected four-element binary feature masks")
    names = [name for name, keep in zip(FEATURES[:4], mask_p) if keep]
    if level == "graph_physchem_toxicity":
        names += [name for name, keep in zip(FEATURES[4:], mask_t) if keep]
    return names


def validate_samples(rows, required_features=FEATURES):
    """Reject absent inputs; ion adducts never imply calibration membership."""
    issues = []

    def issue(compound_id, field, reason):
        issues.append({"compound_id": compound_id, "field": field, "reason": reason})

    if len(rows) != 23:
        issue("all", "row_count", "Expected 23 samples: 21 calibrants and 2 separate cases")
    if sum(row.get("role") == "calibrant" for row in rows) != 21:
        issue("all", "role", "Exactly 21 calibration standards must be identified")
    if sum(row.get("role") == "case" for row in rows) != 2:
        issue("all", "role", "Exactly two case compounds must remain outside calibration")
    ids = set()
    for row in rows:
        sample = row.get("compound_id", "").strip()
        if not sample or sample in ids:
            issue(sample, "compound_id", "Missing or duplicate sample identity")
        ids.add(sample)
        for field in ("name", "smiles", "source_note"):
            if not row.get(field, "").strip():
                issue(sample, field, "Required input is missing")
        if row.get("role") not in ("calibrant", "case"):
            issue(sample, "role", "Use calibrant or case")
        if boolean(row.get("identity_confirmed")) is not True:
            issue(sample, "identity_confirmed", "Author confirmation of structure/standard identity is required")
        rt = finite_number(row.get("measured_rt"))
        if rt is None or rt <= 0:
            issue(sample, "measured_rt", "A positive experimental RT in minutes is required")
        for field in required_features:
            if finite_number(row.get(field)) is None:
                issue(sample, field, "A finite source value is required for an active model input")
        flags = [boolean(row.get("calibration_" + mode)) for mode in ("pos", "neg")]
        for mode, flag in zip(("pos", "neg"), flags):
            if flag is None:
                issue(sample, "calibration_" + mode, "Explicit true/false membership is required; do not infer from adduct")
            if row.get("role") == "case" and flag is True:
                issue(sample, "calibration_" + mode, "Case compounds must not enter calibration fitting")
        if row.get("role") == "calibrant" and flags == [False, False]:
            issue(sample, "calibration_membership", "A calibration standard must belong to at least one confirmed mode")
    for mode in ("pos", "neg"):
        if sum(row.get("role") == "calibrant" and boolean(row.get("calibration_" + mode)) is True for row in rows) < 2:
            issue("all", "calibration_" + mode, "At least two explicitly assigned calibration standards are required")
    return issues


def standardize_inputs(row, params):
    """Use stored training statistics once; masked-out fields are never consumed."""
    active = set(active_features(params))
    arrays = []
    for group, names in (("physchem", FEATURES[:4]), ("toxicity", FEATURES[4:])):
        if group == "toxicity" and params["visnet_v2_feature_level"] == "graph_physchem":
            arrays.append(None)
            continue
        statistics = params["standardization_info"][group]
        means = np.asarray(statistics["mean"], dtype=np.float32)
        stds = np.asarray(statistics["std"], dtype=np.float32)
        if means.shape != (4,) or stds.shape != (4,) or not np.isfinite(means).all() or not np.isfinite(stds).all() or (stds <= 0).any():
            raise ValueError("Invalid stored training standardization for " + group)
        values = means.copy()
        for index, name in enumerate(names):
            if name in active:
                value = finite_number(row.get(name))
                if value is None:
                    raise ValueError("Missing active descriptor: " + name)
                values[index] = value
        arrays.append((values - means) / stds)
    return arrays


def fit_calibration(rows, predictions, mode):
    """OLS in the stated prediction direction: RT(min) = intercept + slope * RTI."""
    included = [row for row in rows if row["role"] == "calibrant" and boolean(row.get("calibration_" + mode)) is True]
    if len(included) < 2:
        raise ValueError("Insufficient confirmed calibrants for " + mode)
    x = np.asarray([predictions[row["compound_id"]] for row in included], dtype=float)
    y = np.asarray([float(row["measured_rt"]) for row in included], dtype=float)
    if not np.isfinite(x).all() or not np.isfinite(y).all() or np.ptp(x) <= 0:
        raise ValueError("Nonfinite or constant calibration coordinates")
    design = np.column_stack((np.ones(len(x)), x))
    intercept, slope = np.linalg.lstsq(design, y, rcond=None)[0]
    fitted = intercept + slope * x
    residual = y - fitted
    denominator = float(np.sum((y - y.mean()) ** 2))
    summary = {
        "mode": mode, "direction": "experimental_rt_min = intercept_min + slope_min_per_rti * predicted_rti",
        "n_calibrants": len(x), "calibrant_ids": [row["compound_id"] for row in included],
        "intercept_min": float(intercept), "slope_min_per_rti": float(slope),
        "calibration_r2": float(1 - np.sum(residual ** 2) / denominator) if denominator > 0 else None,
        "calibration_mae_min": float(np.mean(np.abs(residual))),
        "calibration_rmse_min": float(np.sqrt(np.mean(residual ** 2))),
        "interpretation": "in-sample laboratory calibration; not independent experimental validation",
    }
    coordinates, cases = [], []
    for row in rows:
        pred = float(predictions[row["compound_id"]])
        converted = float(intercept + slope * pred)
        actual = float(row["measured_rt"])
        record = {"mode": mode, "compound_id": row["compound_id"], "name": row["name"], "role": row["role"],
                  "included_in_fit": row in included, "predicted_rti": pred, "experimental_rt_min": actual,
                  "converted_rt_min": converted, "absolute_error_min": abs(converted - actual),
                  "relative_error_percent": abs(converted - actual) / actual * 100}
        coordinates.append(record)
        if row["role"] == "case":
            cases.append(record.copy())
    return summary, coordinates, cases


def load_selection(path):
    manifest = json.loads(Path(path).read_text(encoding="utf-8"))
    if manifest.get("status") != "frozen" or manifest.get("selection_partition") != "development":
        raise ValueError("A frozen development-selected manifest is required")
    result = {}
    for mode in ("pos", "neg"):
        record = dict(manifest["models"][mode])
        for key in ("checkpoint", "training_params"):
            target = Path(record[key])
            if not target.is_absolute():
                target = Path(path).resolve().parent / target
            if not target.is_file():
                raise FileNotFoundError(str(target))
            record[key] = str(target.resolve())
        record["params"] = json.loads(Path(record["training_params"]).read_text(encoding="utf-8"))
        if record["params"].get("standardize") is not True or record["params"].get("standardize_features") is not True:
            raise ValueError("This entry point expects the selected standardized GP/GPT training configuration")
        if finite_number(record["params"].get("property_mean")) is None or (finite_number(record["params"].get("property_std")) or 0) <= 0:
            raise ValueError("Valid stored target standardization is required")
        active_features(record["params"])
        result[mode] = record
    return result


def molecular_graph(smiles, seed):
    """Training ETKDG route, failing instead of using an unseeded fallback."""
    import torch
    from rdkit import Chem
    from rdkit.Chem import AllChem

    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        raise ValueError("Invalid SMILES")
    molecule = Chem.AddHs(molecule)
    options = AllChem.ETKDGv3() if hasattr(AllChem, "ETKDGv3") else AllChem.ETKDGv2()
    options.randomSeed = seed
    code = AllChem.EmbedMolecule(molecule, options)
    if code < 0:
        Chem.RemoveStereochemistry(molecule)
        code = AllChem.EmbedMolecule(molecule, options)
    if code < 0:
        raise ValueError("Seeded training ETKDG conformer generation failed")
    z = torch.tensor([atom.GetAtomicNum() for atom in molecule.GetAtoms()], dtype=torch.long)
    positions = torch.tensor(molecule.GetConformer().GetPositions(), dtype=torch.float32)
    return z, positions


def predict(rows, selection, device_name="cpu"):
    import torch
    sys.path.insert(0, str(ROOT))
    from train_visnet_v2 import get_model
    from utils.molecule import ETKDG_PARAMS

    device = torch.device(device_name)
    graphs, outputs = {}, {}
    for mode in ("pos", "neg"):
        record, predictions = selection[mode], {}
        params = record["params"]
        seed = int(params.get("etkdg_seed", ETKDG_PARAMS.randomSeed))
        if seed < 0:
            raise ValueError("An explicit conformer seed is required")
        model = get_model(SimpleNamespace(**params), device)
        model.load_state_dict(torch.load(record["checkpoint"], map_location=device), strict=True)
        model.eval()
        with torch.no_grad():
            for row in rows:
                key = (row["smiles"], seed)
                if key not in graphs:
                    graphs[key] = molecular_graph(*key)
                z, positions = [value.to(device) for value in graphs[key]]
                physchem, toxicity = standardize_inputs(row, params)
                physchem = torch.as_tensor(physchem[None, :], device=device)
                toxicity = torch.as_tensor(toxicity[None, :], device=device) if toxicity is not None else None
                result, _ = model(z, positions, torch.zeros(len(z), dtype=torch.long, device=device), physchem, toxicity)
                value = float(result.reshape(-1)[0].cpu()) * params["property_std"] + params["property_mean"]
                if not math.isfinite(value):
                    raise ValueError("Nonfinite prediction for " + row["compound_id"])
                predictions[row["compound_id"]] = value
        outputs[mode] = predictions
        del model
    return outputs


def prepare_template(output):
    """Extract only supplied experimental fields; candidate identities stay unconfirmed."""
    source = ROOT / "Supplementary Material.docx"
    ns = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
    with zipfile.ZipFile(source) as document:
        xml = ET.fromstring(document.read("word/document.xml"))
    for deletion in xml.findall(".//w:del", ns):
        deletion.clear()
    tables = [[ ["".join(node.text or "" for node in cell.findall(".//w:t", ns))
                 for cell in row.findall("w:tc", ns)] for row in table.findall("w:tr", ns)]
              for table in xml.findall(".//w:tbl", ns)]
    standards = next(table for table in tables if table and table[0][0] == "RT")
    cases = next(table for table in tables if table and table[0][0] == "Compound")
    if len(standards) != 22 or len(cases) != 3:
        raise ValueError("SI does not contain the expected 21 plus 2 sample tables")
    rows = []
    for index, source_row in enumerate(standards[1:], 1):
        row = dict.fromkeys(FIELDS, "")
        row.update(compound_id="calibrant_{:02d}".format(index), name=source_row[2], role="calibrant",
                   measured_rt=source_row[0], identity_confirmed="false",
                   source_note="Supplementary Material.docx Table S1; 实测RT单位min；结构和逐模式校准名单待作者确认")
        rows.append(row)
    for source_row in cases[1:]:
        row = dict.fromkeys(FIELDS, "")
        row.update(compound_id=source_row[0], name=source_row[0], role="case", measured_rt=source_row[3],
                   calibration_pos="false", calibration_neg="false", identity_confirmed="false",
                   source_note="Supplementary Material.docx Table S2; 实测RT单位min；旧预测RTI不作为新模型输入或目标")
        rows.append(row)
    evidence_ids = {
        "Citalopram": "NS00000035", "Acetochlor": "NS00000067", "Oxytetracycline": "NS00000179",
        "Metronidazole": "NS00000210", "Heptafluorobutyric acid": "NS00000364",
        "Nonafluoropentanoic acid": "NS00000365", "Perfluoroheptanoic acid": "NS00000367",
        "Malathion": "NS00001740", "Perfluorooctanoic acid": "NS00009389", "Triclosan": "NS00010282",
        "Chlorpyrifos": "NS00000445",
    }
    source_rows = {}
    for filename in ("data/MMF/MMF-GNN-valid-smiles.csv", "data/MMF-3/data1028_sheet0.csv"):
        for item in read_csv(ROOT / filename):
            if item["Norman_SusDat_ID"] in evidence_ids.values():
                source_rows[item["Norman_SusDat_ID"]] = (filename, item)
    source_fields = ("Norman_SusDat_ID", "Name", "CAS_RN", "SMILES", "Molecular_Formula", "Monoiso_Mass",
                     "logKow_EPISuite", "Exp_logKow_EPISuite", "alogp_ChemSpider", "xlogp_ChemSpider",
                     "Koc_min_predicted (L/kg)", "Koc_max_predicted (L/kg)", "Tetrahymena_pyriformis_toxicity",
                     "Daphnia_toxicity", "Algae_toxicity", "Pimephales_promelas_toxicity")
    evidence = []
    for row in rows:
        identifier = evidence_ids.get(row["name"])
        if identifier:
            filename, item = source_rows[identifier]
            evidence.append(dict({"compound_id": row["compound_id"], "requested_name": row["name"],
                                  "identity_confirmed": "false", "source_file": filename},
                                 **{field: item.get(field, "") for field in source_fields}))
    output.mkdir(parents=True, exist_ok=True)
    template = output / "图6绘制所需的23个物质输入_待作者确认.csv"
    if template.exists():
        raise FileExistsError("Do not overwrite the author's partially completed input template")
    write_csv(template, rows, FIELDS)
    write_csv(output / "图6输入候选证据_11个数据库记录_未经身份确认.csv", evidence)
    write_csv(output / "missing_inputs.csv", validate_samples(rows), ("compound_id", "field", "reason"))
    (output / "校准数据补充说明.txt").write_text(
        "这份表共有23行，21行是校准标准品，BaP和BeP是两个案例，不能放进校准拟合。\n"
        "实测RT已经从补充材料S1和S2逐行提取，单位为分钟。未写入任何旧模型预测充当真实值。\n"
        "请核对物质身份并补齐SMILES、所需描述符及来源。确认后将identity_confirmed填写true。\n"
        "另表提供11个本地数据库候选，仅供核对CAS、盐型和立体化学；主输入表没有自动采用这些候选。\n"
        "Oxytetracycline候选的alogp和xlogp均缺失，不能随意填0。Koc原始表有min/max，需沿用训练定义明确取值。\n"
        "P4依次为Monoiso_Mass、LogKow、LogP、Koc；T4为IC50、48h-LC50、72h-EC50、96h-LC50。\n"
        "描述符应与NORMAN训练数据同单位同变换，不要把毒性质量浓度直接填进已变换的毒性列。\n"
        "calibration_pos和calibration_neg需逐行明确true或false，可同时为true；案例两列必须false。\n"
        "S1加合物为13条正离子和8条负离子，旧S2文字却写正模式14个标准品。不能据此猜拟合名单，请提供实际名单。\n"
        "本文Methods用模型预测RTI与实测RT建立校准，新模型预测改变后须重新拟合；实测RT保持不变。\n"
        "入口明确拟合RT(min)=intercept+slope×predictedRTI，不将反方向回归的斜率直接倒数。\n"
        "这是实验室校准加两个案例，不是23个独立验证样本。\n"
        "填写完成后运行原仓库脚本，唯一模型来源为outputs/final_masked_models_seed1234/selection_manifest.json。\n"
        "python scripts/data-val/predict_calibration_samples.py --selection-manifest outputs/final_masked_models_seed1234/selection_manifest.json --input <已确认输入CSV> --output-dir <校准结果目录>\n"
        "缺项或身份未确认时只输出missing_inputs.csv，不生成预测数据或拟合图。输入中已被最终模型掩掉的描述符不作为必填项。\n",
        encoding="utf-8-sig")
    return template


def run(selection_manifest, input_path, output_dir, device="cpu"):
    rows = read_csv(input_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    if (output_dir / PREDICTION_NAME).exists():
        if reusable_prediction(output_dir, input_path, selection_manifest):
            return 0
        state_path = output_dir / "run_status.json"
        state = json.loads(state_path.read_text(encoding="utf-8")) if state_path.is_file() else {}
        if not (state.get("status") == "running"
                and state.get("input_sha256") == digest(input_path)
                and state.get("selection_sha256") == digest(selection_manifest)):
            raise FileExistsError("Existing calibration does not match the unchanged inputs and frozen model selection")
    # Validate before loading a model or generating any molecular conformer.
    issues = validate_samples(rows, required_features=())
    if issues:
        issues = validate_samples(rows)
        write_csv(output_dir / "missing_inputs.csv", issues, ("compound_id", "field", "reason"))
        write_json(output_dir / "run_status.json", {"status": "missing_inputs", "prediction_generated": False, "issues": len(issues)})
        return 2
    selection = load_selection(selection_manifest)
    required = set(active_features(selection["pos"]["params"]) + active_features(selection["neg"]["params"]))
    issues = validate_samples(rows, required)
    if issues:
        write_csv(output_dir / "missing_inputs.csv", issues, ("compound_id", "field", "reason"))
        write_json(output_dir / "run_status.json", {"status": "missing_inputs", "prediction_generated": False, "issues": len(issues)})
        return 2
    write_json(output_dir / "run_status.json", {
        "status": "running", "prediction_generated": False,
        "input_sha256": digest(input_path), "selection_sha256": digest(selection_manifest),
    })
    predictions = predict(rows, selection, device)
    summaries, coordinates, cases = [], [], []
    for mode in ("pos", "neg"):
        summary, points, example = fit_calibration(rows, predictions[mode], mode)
        summaries.append(summary)
        coordinates.extend(points)
        cases.extend(example)
    write_csv(output_dir / PREDICTION_NAME, [
        {"compound_id": row["compound_id"], "name": row["name"], "role": row["role"],
         "smiles": row["smiles"], "experimental_rt_min": float(row["measured_rt"]),
         "predicted_rti_pos": predictions["pos"][row["compound_id"]],
         "predicted_rti_neg": predictions["neg"][row["compound_id"]]}
        for row in rows])
    write_csv(output_dir / "图6a-b绘制所需的校准拟合坐标与残差数据.csv", coordinates)
    write_json(output_dir / "图6a-b绘制所需的校准方程和拟合指标.json", summaries)
    write_csv(output_dir / "图6案例所需的BaP-BeP预测RT与实测RT误差数据.csv", cases)
    write_csv(output_dir / "missing_inputs.csv", [], ("compound_id", "field", "reason"))
    write_json(output_dir / "run_status.json", {
        "status": "completed", "selection_manifest": str(selection_manifest.resolve()),
        "input": str(input_path.resolve()), "prediction_generated": True,
        "input_sha256": digest(input_path), "selection_sha256": digest(selection_manifest),
        "sample_count": 23, "calibration_standard_count": 21, "case_count": 2,
        "model_files": {mode: {key: value for key, value in selection[mode].items() if key != "params"} for mode in selection},
        "test_metrics_generated": False, "device": device,
    })
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection-manifest", type=Path)
    parser.add_argument("--input", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--prepare-input-template", action="store_true")
    args = parser.parse_args()
    if args.prepare_input_template:
        print(prepare_template(args.output_dir))
        return 0
    if args.selection_manifest is None or args.input is None:
        parser.error("--selection-manifest and --input are required for prediction")
    return run(args.selection_manifest, args.input, args.output_dir, args.device)


if __name__ == "__main__":
    sys.exit(main())
