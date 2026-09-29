#!/usr/bin/env python3
"""Freeze development-selected masks, then report the fixed held-out predictions."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def load_runtime():
    """Load scientific DLLs only for inference, never for selection freezing."""
    global np, torch, FEATURES, exact_descriptor_shap, graph_embeddings, metrics
    global write_csv, write_json, load_development_inputs, load_additional_features, get_model
    import numpy as np
    import torch
    from analyze_gp_gpt_masking import (
        FEATURES, exact_descriptor_shap, graph_embeddings, metrics, write_csv,
        write_json, load_development_inputs,
    )
    from core.data_preprocessor import load_additional_features
    from train_visnet_v2 import get_model


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def read_csv(path, delimiter=","):
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle, delimiter=delimiter))


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_selected_steps(steps, selected):
    if not steps or steps[0]["step"] != 0 or not steps[0]["accepted"]:
        raise ValueError("Missing unmasked development baseline")
    for previous, current in zip(steps, steps[1:]):
        if current["step"] != previous["step"] + 1 or not previous["accepted"]:
            raise ValueError("Candidates continued after the first worsening step")
        expected = current["metrics"]["mae"] <= previous["metrics"]["mae"] + 1e-8
        if bool(current["accepted"]) != expected:
            raise ValueError("Selection does not follow development MAE")
    winner = next(row for row in reversed(steps) if row["accepted"])
    if winner["step"] != selected["selected_step"]:
        raise ValueError("Selected mask differs from the development stopping rule")
    return winner


def freeze_selection(mask_run, output):
    state = read_json(mask_run / "run_status.json")
    if state["status"] != "completed":
        raise ValueError("Wait for both development masking searches to complete")
    models = {}
    for mode, code in (("pos", "GP"), ("neg", "GPT")):
        mode_root = mask_run / (mode + "_" + code)
        selected = read_json(mode_root / "selected_mask.json")
        steps = read_json(mode_root / "masking_steps.json")
        feature_names = read_json(mode_root / "shap_result.json")["feature_names"]
        winner = validate_selected_steps(steps, selected)
        params = read_json(winner["training_params"])
        p_mask = params.get("visnet_v2_physchem_mask") or [1] * 4
        t_mask = (params.get("visnet_v2_toxicity_mask") or [1] * 4) if code == "GPT" else None
        active = p_mask + (t_mask or [])
        if len(feature_names) != len(active):
            raise ValueError("SHAP feature names and selected input masks differ in width")
        removed = [name for name, keep in zip(feature_names, active) if not keep]
        if set(removed) != set(winner["removed_features"]):
            raise ValueError("Recorded removals do not match the checkpoint configuration")
        label = "ViSRTI-" + code
        if removed:
            label += " (mask " + "+".join(winner["removed_features"]) + ")"
        models[mode] = {
            "feature_code": code, "model_label": label,
            "selected_step": winner["step"], "removed_features": winner["removed_features"],
            "retained_features": [name for name, keep in zip(feature_names, active) if keep],
            "physchem_mask": p_mask, "toxicity_mask": t_mask,
            "checkpoint": winner["checkpoint"], "training_params": winner["training_params"],
            "experiment": str(Path(winner["checkpoint"]).parent),
            "checkpoint_sha256": sha256(winner["checkpoint"]),
            "training_params_sha256": sha256(winner["training_params"]),
            "development_metrics": winner["metrics"],
        }
    manifest = {
        "status": "frozen", "selection_partition": "development", "selection_metric": "mae",
        "candidate_scope": "user_requested_GP_positive_GPT_negative",
        "candidate_scope_chosen_after_prior_test_inspection": True,
        "test_previously_inspected": True, "test_used_for_mask_stopping": False,
        "scaffold_interpretation": "fixed_configuration_structural_robustness_not_nested_selection_cv",
        "mask_run": str(mask_run), "source": state["source"], "models": models,
    }
    output.mkdir(parents=True, exist_ok=True)
    path = output / "selection_manifest.json"
    if path.exists() and read_json(path) != manifest:
        raise ValueError("Existing frozen selection differs; do not overwrite it")
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return manifest


def load_inputs(source, mode, code, params):
    baseline = source / "experiments" / (code + "_" + mode)
    partition = read_csv(baseline / "partition_assignments.csv")
    train = [row["smiles"] for row in partition if row["subset"] == "train"]
    test = read_csv(baseline / "train_prediction_test.txt", "\t")
    smiles = [row["Smiles"] for row in test]
    if smiles != [row["smiles"] for row in partition if row["subset"] == "test"]:
        raise ValueError("Archived predictions and frozen test partition differ")
    if set(train) & set(smiles) or len(set(smiles)) != len(smiles):
        raise ValueError("Duplicated or overlapping held-out compounds")
    physchem, toxicity, _, _ = load_additional_features(
        train + smiles, "MMF_GNN_" + mode, standardize=False,
        input_file=str(source / "input_data" / ("MMF_GNN_" + mode + ".csv")),
        feature_dataset_type=mode)
    feature_count = 4 if code == "GP" else 8
    groups = ["physchem"] if code == "GP" else ["physchem", "toxicity"]
    means = np.asarray([x for key in groups for x in params["standardization_info"][key]["mean"]], np.float32)
    stds = np.asarray([x for key in groups for x in params["standardization_info"][key]["std"]], np.float32)

    def raw(items):
        return np.stack([np.concatenate([physchem[s]] + ([toxicity[s]] if code == "GPT" else [])) for s in items])

    raw_test, raw_train = raw(smiles), raw(train)
    return {
        "train_smiles": train, "smiles": smiles, "raw_features": raw_test,
        "features": (raw_test - means) / stds, "train_features": (raw_train - means) / stds,
        "y": np.asarray([float(row["Correct"]) for row in test]),
        "saved_prediction": np.asarray([float(row["Predict"]) for row in test]),
        "feature_names": list(FEATURES[:feature_count]),
    }


def load_model(params, checkpoint, device):
    model = get_model(SimpleNamespace(**params), device)
    model.load_state_dict(torch.load(checkpoint, map_location=device), strict=True)
    model.eval()
    return model


def predict(model, params, mode, data, device):
    cache = ROOT / ".cache/molecule_graphs" / ("visnet_test_v2_MMF_GNN_" + mode + ".pt")
    embedding, standardized = graph_embeddings(
        model, mode, data["smiles"], data["features"], device, cache_path=cache)
    prediction = standardized * params["property_std"] + params["property_mean"]
    return embedding, standardized, prediction


def prediction_rows(data, prediction):
    return [{"smiles": s, "source_rti": float(y), "predicted_rti": float(p),
             "residual_source_minus_prediction": float(y - p)}
            for s, y, p in zip(data["smiles"], data["y"], prediction)]


def explain(model, params, mode, code, data, embedding, standardized, label, path, device):
    indices = np.random.RandomState(1234).choice(len(data["train_smiles"]), 50, replace=False)
    values, base = exact_descriptor_shap(
        model, embedding, data["features"], data["train_features"][indices], device,
        path.with_name(path.stem + "_progress.json"))
    error = float(np.max(np.abs(base + values.sum(1) - standardized)))
    if error > 1e-5:
        raise ValueError("Final SHAP attribution does not sum to the model output")
    mask = list(params.get("visnet_v2_physchem_mask") or [1] * 4)
    if code == "GPT":
        mask += params.get("visnet_v2_toxicity_mask") or [1] * 4
    for column, keep in enumerate(mask):
        if not keep and np.max(np.abs(values[:, column])) > 1e-6:
            raise ValueError("A masked feature has nonzero SHAP attribution")
    result = {
        "mode": mode, "feature_code": code, "model": label, "partition": "test",
        "feature_names": data["feature_names"], "feature_values": data["features"].tolist(),
        "raw_feature_values": data["raw_features"].tolist(), "smiles": data["smiles"],
        "shap_values": values.tolist(), "base_values": base.tolist(),
        "mean_absolute_shap": np.abs(values).mean(0).tolist(), "feature_mask": mask,
        "property_mean": params["property_mean"], "property_std": params["property_std"],
        "background_partition": "train", "background_samples": len(indices),
        "background_smiles": [data["train_smiles"][int(i)] for i in indices],
        "method": "exact_coalition_interventional_descriptor_SHAP",
        "graph_policy": "each_sample_own_graph_fixed_during_descriptor_perturbation",
        "shap_unit": "standardized_target", "graph_attribution_included": False,
        "additivity_max_abs_error": error, "used_for_selection": False,
        "test_previously_inspected": True, "smoke_only": False,
    }
    write_json(path, result)


def evaluate_mode(manifest, mode, output, device):
    selected = manifest["models"][mode]
    code = selected["feature_code"]
    mode_dir = output / mode
    mode_dir.mkdir(exist_ok=True)
    completed = mode_dir / "evaluation_complete.json"
    if completed.exists():
        record = read_json(completed)
        if record["checkpoint_sha256"] != selected["checkpoint_sha256"]:
            raise ValueError("Completed test evaluation belongs to another model")
        for filename in record["required_files"]:
            if not (mode_dir / filename).exists():
                raise ValueError("Incomplete persisted final evaluation: " + filename)
        return
    steps = read_json(Path(manifest["mask_run"]) / (mode + "_" + code) / "masking_steps.json")
    test_steps = []
    baseline_params = read_json(steps[0]["training_params"])
    data = load_inputs(Path(manifest["source"]), mode, code, baseline_params)
    for step in steps:
        params = read_json(step["training_params"])
        for key in ("standardization_info", "property_mean", "property_std"):
            if params[key] != baseline_params[key]:
                raise ValueError("Candidate standardization differs from the frozen training data")
        model = load_model(params, step["checkpoint"], device)
        embedding, standardized, prediction = predict(model, params, mode, data, device)
        if step["step"] == 0:
            delta = float(np.max(np.abs(prediction - data["saved_prediction"])))
            if delta > 0.003:
                raise ValueError("Baseline no longer reproduces Figure 2 predictions")
            # Keep the Figure 2 source values, after checking numerical reproduction.
            prediction = data["saved_prediction"]
        target = mode_dir / ("mask_step_{:02d}_test_predictions.csv".format(step["step"]))
        write_csv(target, prediction_rows(data, prediction))
        item = dict(step)
        item.update({"development_metrics": step["metrics"], "metrics": metrics(data["y"], prediction),
                     "partition": "test", "n": len(data["smiles"]), "prediction_file": str(target),
                     "selected": step["step"] == selected["selected_step"],
                     "status": "selected" if step["step"] == selected["selected_step"] else
                               ("accepted" if step["accepted"] else "stopped")})
        test_steps.append(item)
        if step["step"] == 0:
            explain(model, params, mode, code, data, embedding, standardized, "ViSRTI-" + code,
                    mode_dir / "unmasked_test_shap.json", device)
        if step["step"] == selected["selected_step"]:
            shutil.copy2(target, mode_dir / "test_predictions.csv")
            if step["step"] == 0:
                shutil.copy2(mode_dir / "unmasked_test_shap.json", mode_dir / "final_test_shap.json")
            else:
                explain(model, params, mode, code, data, embedding, standardized, selected["model_label"],
                        mode_dir / "final_test_shap.json", device)
        del model, embedding
        if device.type == "cuda":
            torch.cuda.empty_cache()
    write_json(mode_dir / "masking_steps_test.json", test_steps)
    required = ["test_predictions.csv", "masking_steps_test.json", "unmasked_test_shap.json", "final_test_shap.json"]
    required += [Path(item["prediction_file"]).name for item in test_steps]
    write_json(completed, {"checkpoint_sha256": selected["checkpoint_sha256"],
                          "n": len(data["smiles"]), "required_files": required,
                          "selection_unchanged": True, "bootstrap": False,
                          "completed_at": datetime.now().astimezone().isoformat()})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mask-run", type=Path, default=ROOT / "outputs/gp_gpt_masking_dev_seed1234")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/final_masked_models_seed1234")
    parser.add_argument("--freeze-only", action="store_true")
    parser.add_argument("--preflight-development", action="store_true")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.preflight_development:
        load_runtime()
        torch.set_num_threads(2)
        device = torch.device(args.device)
        source = ROOT / "outputs/feature_fusion_random_seed1234"
        output = ROOT / "tmp/final_model_development_preflight"
        output.mkdir(parents=True, exist_ok=True)
        checks = []
        for mode, code in (("pos", "GP"), ("neg", "GPT")):
            directory = output / mode
            directory.mkdir(exist_ok=True)
            data = load_development_inputs(source, mode, code, directory)
            params = data["params"]
            model = load_model(params, data["experiment"] / "best.pt", device)
            cache = ROOT / ".cache/molecule_graphs" / ("visnet_train_v2_MMF_GNN_" + mode + ".pt")
            _, prediction = graph_embeddings(model, mode, data["dev_smiles"][:2], data["dev_features"][:2],
                                              device, cache_path=cache)
            prediction = prediction * params["property_std"] + params["property_mean"]
            delta = float(np.max(np.abs(prediction - data["saved_prediction"][:2])))
            print("{} {} two-sample prediction difference: {:.8f} RTI".format(mode, device, delta), flush=True)
            if delta > 0.003:
                raise ValueError("Preflight does not reproduce archived development predictions")
            checks.append({"mode": mode, "device": str(device), "n": 2,
                           "max_abs_rti_delta": delta, "test_responses_read": False})
        write_json(output / "preflight.json", {"passed": True, "checks": checks, "smoke_only": True})
        print(json.dumps(checks, indent=2))
        return
    manifest = freeze_selection(args.mask_run.resolve(), args.output_dir.resolve())
    if args.freeze_only:
        return
    load_runtime()
    torch.set_num_threads(6)
    torch.manual_seed(1234)
    for mode in ("pos", "neg"):
        evaluate_mode(manifest, mode, args.output_dir.resolve(), torch.device(args.device))


if __name__ == "__main__":
    main()
