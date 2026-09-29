#!/usr/bin/env python3
"""Continue final-model analyses after the existing development masking run."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def now():
    return datetime.now().astimezone().isoformat(timespec="seconds")


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


@contextmanager
def controller_lock(directory):
    """Allow only one supervisor to advance this experiment."""
    directory.mkdir(parents=True, exist_ok=True)
    handle = (directory / "controller.lock").open("a+b")
    try:
        if handle.tell() == 0:
            handle.write(b"0")
            handle.flush()
        handle.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise RuntimeError("The final-model controller is already running") from exc
        try:
            yield
        finally:
            handle.seek(0)
            if os.name == "nt":
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    finally:
        handle.close()


def run(args):
    output = ROOT / "outputs/final_masked_models_seed1234"
    delivery = ROOT / "修改部分内容/07_最终模型后续分析"
    mask_run = ROOT / "outputs/gp_gpt_masking_dev_seed1234"
    state_path = output / "run_status.json"
    output.mkdir(parents=True, exist_ok=True)
    logs = output / "logs"
    logs.mkdir(exist_ok=True)
    if state_path.exists():
        if not args.resume:
            raise FileExistsError("Existing pipeline; inspect its process before using --resume")
        state = read_json(state_path)
    else:
        state = {"started_at": now(), "completed_stages": [], "delivery": str(delivery)}
    state.update({"status": "running", "supervisor_pid": os.getpid(), "child_pid": None})
    state.pop("failure", None)
    state.pop("failed_at", None)
    os.environ.update({"PYTHONIOENCODING": "utf-8", "PYTHONUNBUFFERED": "1",
                       "PYTHONHASHSEED": "1234", "OMP_NUM_THREADS": "6", "MKL_NUM_THREADS": "6",
                       "CUDA_VISIBLE_DEVICES": "0", "CUBLAS_WORKSPACE_CONFIG": ":4096:8"})

    def save():
        state["updated_at"] = now()
        state_path.write_text(json.dumps(state, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    def execute(name, command, accepted_codes=(0,)):
        if name in state["completed_stages"]:
            return 0
        log = logs / (name + ".log")
        state.update({"stage": name, "stage_started_at": now(), "command": [str(x) for x in command],
                      "log_file": str(log)})
        save()
        with log.open("a", encoding="utf-8") as handle:
            process = subprocess.Popen(command, cwd=str(ROOT), stdout=handle, stderr=subprocess.STDOUT)
            state["child_pid"] = process.pid
            save()
            code = process.wait()
        state["child_pid"] = None
        if code not in accepted_codes:
            raise RuntimeError("{} failed with exit code {}; see {}".format(name, code, log))
        if code == 0:
            state["completed_stages"].append(name)
        else:
            state["waiting_for_inputs"] = name
        save()
        return code

    try:
        state["stage"] = "waiting_for_development_masks"
        save()
        while True:
            mask_state = read_json(mask_run / "run_status.json")
            state["mask_progress"] = {key: mask_state.get(key) for key in
                                      ("current_mode", "current_step", "status", "current_experiment")}
            save()
            if mask_state["status"] == "completed":
                break
            if mask_state["status"] == "failed":
                raise RuntimeError("Development masking failed: " + mask_state.get("failure", "unknown"))
            time.sleep(60)
        evaluate = [sys.executable, "-u", str(ROOT / "scripts/data-val/evaluate_final_masked_models.py"),
                    "--mask-run", str(mask_run), "--output-dir", str(output)]
        execute("freeze_selection", evaluate + ["--freeze-only"])
        execute("held_out_predictions_and_shap", evaluate)
        execute("fixed_test_figures", [sys.executable, "-u", str(ROOT / "scripts/plt/plot_final_masked_models.py"),
                                       "--run-root", str(output), "--output-dir", str(delivery)])
        previously_missing_calibration = (output / "calibration/run_status.json").exists() and (
            read_json(output / "calibration/run_status.json")["status"] == "missing_inputs")
        calibration_code = execute("calibration_input_check_and_prediction", [
            sys.executable, "-u", str(ROOT / "scripts/data-val/predict_calibration_samples.py"),
            "--selection-manifest", str(output / "selection_manifest.json"),
            "--input", str(delivery / "03_校准输入待补/图6绘制所需的23个物质输入_待作者确认.csv"),
            "--output-dir", str(output / "calibration"), "--device", "cuda"], accepted_codes=(0, 2))
        if calibration_code == 0:
            state.pop("waiting_for_inputs", None)
            if previously_missing_calibration and "assemble_delivery" in state["completed_stages"]:
                state["completed_stages"].remove("assemble_delivery")
            save()
            execute("calibration_figures", [
                sys.executable, "-u", str(ROOT / "scripts/plt/plot_calibration_results.py"),
                "--calibration-dir", str(output / "calibration"), "--output-dir", str(delivery)])
        scaffold = ROOT / "outputs/final_masked_scaffold_seed1234"
        scaffold_command = ["pwsh", "-NoLogo", "-NoProfile", "-File",
                            str(ROOT / "scripts/run_selected_model_scaffold_cv.ps1"),
                            "-SelectionManifest", str(output / "selection_manifest.json")]
        if (scaffold / "run_manifest.json").exists():
            scaffold_command.append("-Resume")
        execute("scaffold_five_fold", scaffold_command)
        execute("assemble_delivery", [sys.executable, "-u", str(ROOT / "scripts/data-val/package_final_model_results.py"),
                                      "--run-root", str(output), "--delivery", str(delivery)])
        calibration = read_json(output / "calibration/run_status.json")
        state.update({"status": "computed_pending_visual_review" if calibration["status"] == "completed"
                                   else "computed_pending_visual_review_and_calibration_inputs",
                      "calibration_status": calibration["status"],
                      "stage": None, "computed_at": now(), "child_pid": None})
        save()
    except Exception as exc:
        state.update({"status": "failed", "failure": str(exc), "failed_at": now()})
        save()
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    with controller_lock(ROOT / "outputs/final_masked_models_seed1234"):
        run(args)


if __name__ == "__main__":
    main()
