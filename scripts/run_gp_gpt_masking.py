#!/usr/bin/env python3
"""Run GP/positive and GPT/negative SHAP-guided removal on development data."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("mask_analysis", ROOT / "scripts/data-val/analyze_gp_gpt_masking.py")
analysis = importlib.util.module_from_spec(spec)
spec.loader.exec_module(analysis)


def now():
    return datetime.now().astimezone().isoformat(timespec="seconds")


def execute(command, log_path, state, state_path):
    state["command"] = [str(value) for value in command]
    state["log_file"] = str(log_path)
    state["updated_at"] = now()
    analysis.write_json(state_path, state)
    with log_path.open("w", encoding="utf-8") as handle:
        process = subprocess.Popen(command, cwd=str(ROOT), stdout=handle, stderr=subprocess.STDOUT)
        state["child_pid"] = process.pid
        analysis.write_json(state_path, state)
        code = process.wait()
    if code:
        raise RuntimeError("Command failed with exit code {}: {}".format(code, log_path))
    state["child_pid"] = None


def prepare_train_inputs(source, output, mode):
    name = "MMF_GNN_" + mode
    target = output / name
    target.mkdir(parents=True, exist_ok=True)
    origin = source / "input_data" / name
    for filename in (
        name + "_train_set.txt", name + "-" + name + "_train_set-cache.pkl",
        name + "-atom_dict.pickle", name + "-bond_dict.pickle",
        name + "-edge_dict.pickle", name + "-fingerprint_dict.pickle", "split_manifest.json",
    ):
        shutil.copy2(origin / filename, target / filename)
    # Original feature statistics precede failed-graph filtering. Preserve those
    # descriptor rows while still excluding every held-out test compound.
    with (origin / (name + "_train_set.txt")).open(encoding="utf-8") as handle:
        train_pool = {line.split("\t", 1)[0] for line in handle if line.strip()}
    descriptors = [
        {key: value for key, value in row.items() if "RTI" not in key}
        for row in analysis.read_csv(source / "input_data" / (name + ".csv"))
        if row["SMILES"] in train_pool
    ]
    analysis.write_csv(output / (name + ".csv"), descriptors)
    code = "GP" if mode == "pos" else "GPT"
    reference = analysis.read_csv(source / "experiments" / (code + "_" + mode) / "partition_assignments.csv")
    analysis.write_csv(target / "development_partition_assignments.csv",
                       [row for row in reference if row["subset"] != "test"])


def training_command(params, input_root, output, physchem_mask, toxicity_mask):
    command = [sys.executable, "-u", str(ROOT / "train_visnet_v2.py")]
    keys = (
        "dataname", "split_strategy", "split_seed", "dev_split_seed", "batch_train", "batch_test",
        "learning_rate", "iteration", "early_stopping_patience", "scheduler_patience",
        "scheduler_factor", "min_lr", "weight_decay", "loss_function",
        "visnet_v2_node_feature_dim", "visnet_v2_physchem_feature_dim",
        "visnet_v2_toxicity_feature_dim", "visnet_v2_chromato_feature_dim",
        "visnet_v2_graph_hidden_dim", "visnet_v2_physchem_hidden_dim",
        "visnet_v2_toxicity_hidden_dim", "visnet_v2_toxicity_intermediate_dim",
        "visnet_v2_chromato_hidden_dim", "visnet_v2_dropout_rate",
        "visnet_v2_feature_level", "visnet_v2_output_head",
    )
    for key in keys:
        command += ["--" + key.replace("_", "-"), str(params[key])]
    command += ["--visnet-v2-fusion-hidden-dims"] + [str(value) for value in params["visnet_v2_fusion_hidden_dims"]]
    for key in ("standardize", "standardize_features", "visnet_v2_use_attention", "visnet_v2_use_gating"):
        if params[key]:
            command.append("--" + key.replace("_", "-"))
    command += ["--data-root", str(input_root), "--output-dir", str(output), "--development-only"]
    command += ["--visnet-v2-physchem-mask"] + [str(value) for value in physchem_mask]
    if params["visnet_v2_feature_level"] == "graph_physchem_toxicity":
        command += ["--visnet-v2-toxicity-mask"] + [str(value) for value in toxicity_mask]
    return command


def compare_step(source_experiment, candidate_dir):
    original = analysis.read_csv(source_experiment / "train_prediction_dev.txt", "\t")
    candidate = analysis.read_csv(candidate_dir / "train_prediction_dev.txt", "\t")
    if [(r["Smiles"], r["Correct"]) for r in original] != [(r["Smiles"], r["Correct"]) for r in candidate]:
        raise ValueError("The development compounds, order, or responses changed")
    original_parts = analysis.read_csv(source_experiment / "partition_assignments.csv")
    candidate_parts = analysis.read_csv(candidate_dir / "partition_assignments.csv")
    if [r for r in original_parts if r["subset"] != "test"] != candidate_parts:
        raise ValueError("Training or development assignments changed")
    if list(candidate_dir.glob("*prediction_test*")) or (candidate_dir / "predictions_test.csv").exists():
        raise ValueError("A removal candidate unexpectedly evaluated test data")
    p1 = json.loads((source_experiment / "training_params.json").read_text(encoding="utf-8"))
    p2 = json.loads((candidate_dir / "training_params.json").read_text(encoding="utf-8"))
    for key in ("property_mean", "property_std", "standardization_info"):
        if p1[key] != p2[key]:
            raise ValueError("Training standardization changed: " + key)
    return analysis.metrics([float(r["Correct"]) for r in candidate], [float(r["Predict"]) for r in candidate])


def run(args):
    run_root = args.output_dir.resolve()
    run_root.mkdir(parents=True, exist_ok=True)
    state_path = run_root / "run_status.json"
    if state_path.exists():
        raise FileExistsError("This experiment already has a run status; inspect it before restarting")
    source = args.source.resolve()
    state = {
        "status": "running", "started_at": now(), "supervisor_pid": os.getpid(),
        "scope": "GP_pos_and_GPT_neg_development_SHAP_and_retrained_sequential_masks_only",
        "stop_metric": "development_mae", "stop_on_first_increase": True,
        "candidate_choice": "user_requested_after_previous_test_inspection",
        "held_out_test_used_this_run": False, "removal_order": "fixed_unmasked_development_mean_absolute_SHAP_ascending",
        "background_samples": 50, "seed": 1234, "source": str(source),
        "mode_results": {},
    }
    analysis.write_json(state_path, state)
    logs = run_root / "logs"
    logs.mkdir()
    input_root = run_root / "input_data"
    input_root.mkdir()
    os.environ.update({
        "PYTHONIOENCODING": "utf-8", "PYTHONUNBUFFERED": "1", "PYTHONHASHSEED": "1234",
        "OMP_NUM_THREADS": "6", "MKL_NUM_THREADS": "6", "CUDA_VISIBLE_DEVICES": "0",
    })
    try:
        # Both SHAP panels are available before the longer retraining steps start.
        for mode, code in (("pos", "GP"), ("neg", "GPT")):
            analysis_dir = run_root / (mode + "_" + code)
            state.update({"current_stage": "SHAP", "current_mode": mode})
            execute([
                sys.executable, "-u", str(ROOT / "scripts/data-val/analyze_gp_gpt_masking.py"),
                "--source", str(source), "--mode", mode, "--feature-code", code,
                "--output-dir", str(analysis_dir),
            ], logs / (mode + "_SHAP.log"), state, state_path)
            prepare_train_inputs(source, input_root, mode)

        for mode, code in (("pos", "GP"), ("neg", "GPT")):
            mode_root = run_root / (mode + "_" + code)
            shap_result = json.loads((mode_root / "shap_result.json").read_text(encoding="utf-8"))
            experiment = source / "experiments" / (code + "_" + mode)
            params = json.loads((experiment / "training_params.json").read_text(encoding="utf-8"))
            original = analysis.read_csv(experiment / "train_prediction_dev.txt", "\t")
            baseline = analysis.metrics([float(r["Correct"]) for r in original], [float(r["Predict"]) for r in original])
            steps = [{"step": 0, "removed_features": [], "accepted": True,
                      "metrics": baseline, "prediction_file": str(experiment / "train_prediction_dev.txt"),
                      "checkpoint": str(experiment / "best.pt"), "training_params": str(experiment / "training_params.json")}]
            mask = [1] * len(shap_result["feature_names"])
            removed = []
            for step, index in enumerate(shap_result["removal_order"], 1):
                # Keep the molecular graph; its learned dimensions are not individual descriptors.
                mask[index] = 0
                removed.append(shap_result["feature_names"][index])
                candidate_dir = mode_root / ("mask_step_{:02d}".format(step))
                state.update({"current_stage": "retraining", "current_mode": mode,
                              "current_step": step, "removed_features": removed[:],
                              "current_experiment": str(candidate_dir), "stage_started_at": now()})
                execute(training_command(params, input_root, candidate_dir, mask[:4], mask[4:]),
                        logs / ("{}_mask_step_{:02d}.log".format(mode, step)), state, state_path)
                values = compare_step(experiment, candidate_dir)
                accepted = analysis.retained_step(steps[-1]["metrics"]["mae"], values["mae"])
                steps.append({"step": step, "removed_features": removed[:], "accepted": accepted,
                              "metrics": values, "prediction_file": str(candidate_dir / "train_prediction_dev.txt"),
                              "checkpoint": str(candidate_dir / "best.pt"),
                              "training_params": str(candidate_dir / "training_params.json")})
                analysis.write_json(mode_root / "masking_steps.json", steps)
                print("{} step {} MAE {:.6f}: {}".format(mode, step, values["mae"], "retain" if accepted else "stop"), flush=True)
                if not accepted:
                    break
            selected = next(item for item in reversed(steps) if item["accepted"])
            state["mode_results"][mode] = {
                "feature_code": code, "selected_step": selected["step"],
                "removed_features": selected["removed_features"], "checkpoint": selected["checkpoint"],
                "training_params": selected["training_params"], "metrics": selected["metrics"],
                "evaluated_candidates": len(steps) - 1, "test_evaluation": "not_performed",
            }
            analysis.write_json(mode_root / "selected_mask.json", state["mode_results"][mode])
            analysis.write_json(state_path, state)

        state["current_stage"] = "render_and_export"
        execute([sys.executable, str(ROOT / "scripts/plt/plot_gp_gpt_masking.py"),
                 "--run-root", str(run_root), "--output-dir", str(args.delivery.resolve())],
                logs / "render_and_export.log", state, state_path)
        state.update({"status": "completed", "completed_at": now(), "current_stage": None,
                      "delivery": str(args.delivery.resolve())})
        analysis.write_json(state_path, state)
        shutil.copy2(state_path, args.delivery / "运行完成状态.json")
    except Exception as exc:
        state.update({"status": "failed", "failure": str(exc), "failed_at": now()})
        analysis.write_json(state_path, state)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "outputs/feature_fusion_random_seed1234")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/gp_gpt_masking_dev_seed1234")
    parser.add_argument("--delivery", type=Path, default=ROOT / "修改部分内容/06_GP_GPT逐步掩码分析")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
