#!/usr/bin/env python3
"""Development-only SHAP and ordered feature-removal experiments for GP/GPT."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from core.data_preprocessor import load_additional_features
from train_visnet_v2 import get_model

FEATURES = (
    "Monoiso_Mass", "LogKow", "LogP", "Koc",
    "IC50", "48h-LC50", "72h-EC50", "96h-LC50",
)
METRICS = ("mae", "rmse", "r2", "pcc")


def read_csv(path, delimiter=","):
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle, delimiter=delimiter))


def write_csv(path, rows):
    rows = list(rows)
    with Path(path).open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def metrics(y, prediction):
    y, prediction = np.asarray(y, dtype=float), np.asarray(prediction, dtype=float)
    residual = y - prediction
    return dict(zip(METRICS, (
        float(np.mean(np.abs(residual))), float(np.sqrt(np.mean(residual ** 2))),
        float(1 - np.sum(residual ** 2) / np.sum((y - y.mean()) ** 2)),
        float(np.corrcoef(y, prediction)[0, 1]),
    )))


def retained_step(previous_mae, candidate_mae, tolerance=1e-8):
    """Accept only a non-worsening development MAE; never inspect other partitions."""
    return bool(candidate_mae <= previous_mae + tolerance)


def exact_shap_from_values(values, feature_count):
    """Shapley values from every coalition, ordered by integer bit mask."""
    values = np.asarray(values, dtype=float)
    result = np.zeros(feature_count)
    for feature in range(feature_count):
        bit = 1 << feature
        for subset in range(1 << feature_count):
            if not subset & bit:
                size = bin(subset).count("1")
                weight = 1.0 / (feature_count * math.comb(feature_count - 1, size))
                result[feature] += weight * (values[subset | bit] - values[subset])
    return result


def load_development_inputs(source, mode, feature_code, output):
    experiment = source / "experiments" / (feature_code + "_" + mode)
    params = json.loads((experiment / "training_params.json").read_text(encoding="utf-8"))
    partitions = read_csv(experiment / "partition_assignments.csv")
    train = [row["smiles"] for row in partitions if row["subset"] == "train"]
    dev = read_csv(experiment / "train_prediction_dev.txt", "\t")
    dev_smiles = [row["Smiles"] for row in dev]
    expected_dev = [row["smiles"] for row in partitions if row["subset"] == "development"]
    test_smiles = {row["smiles"] for row in partitions if row["subset"] == "test"}
    assert dev_smiles == expected_dev
    assert not (set(train) & set(dev_smiles))
    assert not ((set(train) | set(dev_smiles)) & test_smiles)

    # Keep only development/training descriptors; target columns never enter this file.
    allowed = set(train) | set(dev_smiles)
    descriptor_rows = []
    with (source / "input_data" / ("MMF_GNN_" + mode + ".csv")).open(
            encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["SMILES"] in allowed:
                descriptor_rows.append({k: v for k, v in row.items() if "RTI" not in k})
    descriptor_path = output / "训练及开发集特征.csv"
    write_csv(descriptor_path, descriptor_rows)
    physchem, toxicity, _, _ = load_additional_features(
        train + dev_smiles, "MMF_GNN_" + mode, standardize=False,
        input_file=str(descriptor_path), feature_dataset_type=mode)
    feature_count = 4 if feature_code == "GP" else 8
    means = params["standardization_info"]["physchem"]["mean"][:]
    stds = params["standardization_info"]["physchem"]["std"][:]
    if feature_count == 8:
        means += params["standardization_info"]["toxicity"]["mean"]
        stds += params["standardization_info"]["toxicity"]["std"]

    def values(smiles):
        arrays = [physchem[smiles]]
        if feature_count == 8:
            arrays.append(toxicity[smiles])
        return np.concatenate(arrays)

    raw_dev = np.stack([values(smiles) for smiles in dev_smiles])
    raw_train = np.stack([values(smiles) for smiles in train])
    # Match training float32 standardization exactly, once.
    means, stds = np.asarray(means, np.float32), np.asarray(stds, np.float32)
    return {
        "experiment": experiment, "params": params, "train_smiles": train,
        "dev_smiles": dev_smiles, "raw_dev": raw_dev,
        "dev_features": (raw_dev - means) / stds,
        "train_features": (raw_train - means) / stds,
        "y": np.asarray([float(row["Correct"]) for row in dev]),
        "saved_prediction": np.asarray([float(row["Predict"]) for row in dev]),
        "feature_names": FEATURES[:feature_count],
    }


def graph_embeddings(model, mode, smiles, features, device, batch_size=32, cache_path=None):
    if cache_path is None:
        cache_path = ROOT / ".cache" / "molecule_graphs" / ("visnet_train_v2_MMF_GNN_" + mode + ".pt")
    cache = torch.load(cache_path, map_location="cpu")
    cache = cache.get("success", cache)
    by_smiles = {key.rsplit("_", 1)[0]: value for key, value in cache.items()}
    embeddings, full_predictions = [], []
    with torch.no_grad():
        for start in range(0, len(smiles), batch_size):
            selected = smiles[start:start + batch_size]
            graphs = [by_smiles[item] for item in selected]
            z = torch.cat([graph["z"] for graph in graphs]).to(device)
            pos = torch.cat([graph["pos"] for graph in graphs]).to(device)
            batch = torch.cat([
                torch.full((graph["z"].numel(),), i, dtype=torch.long)
                for i, graph in enumerate(graphs)
            ]).to(device)
            _, representation = model.visnet(z, pos, batch)
            graph_features = model.graph_processor(representation)
            x = torch.as_tensor(features[start:start + len(graphs)], device=device)
            prediction, _ = model.forward_from_graph_features(
                graph_features, x[:, :4], x[:, 4:] if x.shape[1] == 8 else None)
            # Verify the cached route against the ordinary full model on actual graphs.
            ordinary, _ = model(z, pos, batch, x[:, :4], x[:, 4:] if x.shape[1] == 8 else None)
            if not torch.allclose(prediction, ordinary, atol=2e-6, rtol=2e-6):
                raise ValueError("Cached graph inference differs from the full model")
            embeddings.append(graph_features.detach())
            full_predictions.append(prediction.cpu().numpy())
    return torch.cat(embeddings), np.concatenate(full_predictions)


def exact_descriptor_shap(model, embeddings, features, background, device, progress):
    feature_count = features.shape[1]
    bits = torch.as_tensor([
        [(subset >> j) & 1 for j in range(feature_count)]
        for subset in range(1 << feature_count)
    ], dtype=torch.bool, device=device)
    background = torch.as_tensor(background, device=device)
    x = torch.as_tensor(features, device=device)
    shap_values, base_values = [], []
    started = time.time()
    with torch.no_grad():
        for i in range(len(x)):
            mixed = torch.where(bits[:, None, :], x[i], background[None, :, :])
            mixed = mixed.reshape(-1, feature_count)
            predictions = []
            for start in range(0, len(mixed), 4096):
                block = mixed[start:start + 4096]
                graph = embeddings[i:i + 1].expand(len(block), -1)
                pred, _ = model.forward_from_graph_features(
                    graph, block[:, :4], block[:, 4:] if feature_count == 8 else None)
                predictions.append(pred)
            values = torch.cat(predictions).reshape(len(bits), len(background)).mean(1).cpu().numpy()
            attribution = exact_shap_from_values(values, feature_count)
            if abs(attribution.sum() - (values[-1] - values[0])) > 1e-5:
                raise ValueError("SHAP additivity check failed")
            shap_values.append(attribution)
            base_values.append(float(values[0]))
            if (i + 1) % 50 == 0 or i + 1 == len(x):
                print("SHAP {}/{} ({:.1f}s)".format(i + 1, len(x), time.time() - started), flush=True)
                write_json(progress, {"completed": i + 1, "total": len(x), "elapsed_seconds": time.time() - started})
    return np.asarray(shap_values), np.asarray(base_values)


def run_shap(args):
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if (output / "shap_result.json").exists():
        raise FileExistsError("Completed SHAP result already exists")
    torch.set_num_threads(6)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = load_development_inputs(args.source.resolve(), args.mode, args.feature_code, output)
    params = data["params"]
    model = get_model(SimpleNamespace(**params), device)
    model.load_state_dict(torch.load(data["experiment"] / "best.pt", map_location=device), strict=True)
    model.eval()
    count = len(data["dev_smiles"]) if args.limit is None else min(args.limit, len(data["dev_smiles"]))
    features = data["dev_features"][:count]
    smiles = data["dev_smiles"][:count]
    graph, prediction = graph_embeddings(model, args.mode, smiles, features, device)
    raw_prediction = prediction * params["property_std"] + params["property_mean"]
    baseline_delta = float(np.max(np.abs(raw_prediction - data["saved_prediction"][:count])))
    print("Development prediction reproduction max delta = {:.7f} RTI".format(baseline_delta), flush=True)
    if baseline_delta > 0.003:
        raise ValueError("Unmasked prediction does not reproduce the source checkpoint")
    background_indices = np.random.RandomState(args.seed).choice(
        len(data["train_smiles"]), min(args.background, len(data["train_smiles"])), replace=False)
    background = data["train_features"][background_indices]
    write_csv(output / "SHAP背景样本.csv", [
        {"subset": "train", "smiles": data["train_smiles"][int(i)]}
        for i in background_indices
    ])
    shap_values, base_values = exact_descriptor_shap(
        model, graph, features, background, device, output / "shap_progress.json")
    errors = np.abs(base_values + shap_values.sum(1) - prediction)
    if errors.max() > 1e-5:
        raise ValueError("SHAP does not explain the unmasked model prediction")
    importance = np.mean(np.abs(shap_values), axis=0)
    order = np.argsort(importance, kind="stable").tolist()
    result = {
        "mode": args.mode, "feature_code": args.feature_code,
        "model": "ViSRTI-" + args.feature_code, "partition": "development",
        "feature_names": list(data["feature_names"]), "smiles": smiles,
        "feature_values": features.tolist(), "raw_feature_values": data["raw_dev"][:count].tolist(),
        "shap_values": shap_values.tolist(), "base_values": base_values.tolist(),
        "mean_absolute_shap": importance.tolist(), "removal_order": order,
        "property_mean": params["property_mean"], "property_std": params["property_std"],
        "graph_policy": "each_sample_own_graph_fixed_during_descriptor_perturbation",
        "background_partition": "train", "background_samples": len(background_indices),
        "method": "exact_coalition_interventional_descriptor_SHAP",
        "shap_unit": "standardized_target", "graph_attribution_included": False,
        "test_responses_loaded": False, "seed": args.seed,
        "additivity_max_abs_error": float(errors.max()),
        "baseline_reproduction_max_abs_rti_error": baseline_delta,
        "baseline_metrics": metrics(data["y"][:count], raw_prediction),
        "source_checkpoint": str(data["experiment"] / "best.pt"),
        "source_checkpoint_sha256": hashlib.sha256((data["experiment"] / "best.pt").read_bytes()).hexdigest(),
        "toxicity_hidden_dim": params["visnet_v2_toxicity_hidden_dim"] if args.feature_code == "GPT" else None,
        "primary_stop_metric": "development_mae", "retrain_after_removal": True,
        "candidate_selection": "user_requested_GP_pos_GPT_neg_after_prior_test_inspection",
        "smoke_only": args.limit is not None,
    }
    write_json(output / "shap_result.json", result)
    write_csv(output / "图3绘制所需的SHAP重要性排序数据.csv", [
        {"mode": args.mode, "model": result["model"], "feature": data["feature_names"][j],
         "removal_order": i + 1, "mean_absolute_shap_standardized": float(importance[j]),
         "mean_absolute_shap_rti": float(importance[j] * params["property_std"])}
        for i, j in enumerate(order)
    ])
    write_csv(output / "图3绘制所需的逐样本SHAP散点数据.csv", [
        {"mode": args.mode, "model": result["model"], "partition": "development",
         "smiles": smiles[i], "feature": name,
         "raw_feature_value": float(data["raw_dev"][i, j]), "standardized_feature_value": float(features[i, j]),
         "shap_standardized": float(shap_values[i, j]),
         "shap_rti": float(shap_values[i, j] * params["property_std"])}
        for i in range(count) for j, name in enumerate(data["feature_names"])
    ])
    write_csv(output / "图3绘制所需的未掩码开发集逐样本预测数据.csv", [
        {"mode": args.mode, "smiles": smiles[i], "source_rti": float(data["y"][i]),
         "prediction_rti": float(raw_prediction[i]), "archived_prediction_rti": float(data["saved_prediction"][i])}
        for i in range(count)
    ])
    print("SHAP completed: " + str(output), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "outputs/feature_fusion_random_seed1234")
    parser.add_argument("--mode", choices=("pos", "neg"), required=True)
    parser.add_argument("--feature-code", choices=("GP", "GPT"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--background", type=int, default=50)
    parser.add_argument("--limit", type=int)
    run_shap(parser.parse_args())


if __name__ == "__main__":
    main()
