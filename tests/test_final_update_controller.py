import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture
def controller(monkeypatch, tmp_path):
    path = Path(__file__).resolve().parents[1] / "scripts/run_final_model_updates.py"
    spec = importlib.util.spec_from_file_location("final_controller", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "ROOT", tmp_path)
    mask = tmp_path / "outputs/gp_gpt_masking_dev_seed1234"
    mask.mkdir(parents=True)
    (mask / "run_status.json").write_text(json.dumps({"status": "completed"}), encoding="utf-8")
    return module


def fake_processes(controller, monkeypatch, calibration_complete, fail_stage=None):
    calls = []

    def popen(command, **kwargs):
        stage = Path(command[2]).name if command[1] == "-u" else "scaffold"
        calls.append(stage)
        if stage == "predict_calibration_samples.py":
            folder = controller.ROOT / "outputs/final_masked_models_seed1234/calibration"
            folder.mkdir(parents=True, exist_ok=True)
            (folder / "run_status.json").write_text(json.dumps({
                "status": "completed" if calibration_complete else "missing_inputs"}), encoding="utf-8")
            return SimpleNamespace(pid=12345, wait=lambda: 0 if calibration_complete else 2)
        return SimpleNamespace(pid=12345, wait=lambda: 9 if fail_stage == stage else 0)

    monkeypatch.setattr(controller.subprocess, "Popen", popen)
    return calls


@pytest.mark.parametrize("resume", (False, True))
def test_missing_calibration_does_not_block_scaffold(controller, monkeypatch, resume):
    calls = fake_processes(controller, monkeypatch, False)
    controller.run(SimpleNamespace(resume=resume))
    state = controller.read_json(controller.ROOT / "outputs/final_masked_models_seed1234/run_status.json")
    assert "scaffold" in calls and "package_final_model_results.py" in calls
    assert "predict_calibration_samples.py" not in calls
    assert "plot_calibration_results.py" not in calls
    assert "calibration_input_check_and_prediction" not in state["completed_stages"]
    assert state["status"] == "computed_pending_visual_review_and_calibration_inputs"
    calibration = controller.ROOT / "outputs/final_masked_models_seed1234/calibration"
    assert controller.read_json(calibration / "run_status.json")["status"] == "missing_inputs"
    assert "input_file" in (calibration / "missing_inputs.csv").read_text(encoding="utf-8-sig")


def test_resume_after_inputs_only_runs_calibration_and_repackages(controller, monkeypatch):
    fake_processes(controller, monkeypatch, False)
    controller.run(SimpleNamespace(resume=False))
    input_file = controller.ROOT / "修改部分内容/07_最终模型后续分析/03_校准输入待补/图6绘制所需的23个物质输入_待作者确认.csv"
    input_file.parent.mkdir(parents=True)
    input_file.write_text("compound_id\n", encoding="utf-8")
    calls = fake_processes(controller, monkeypatch, True)
    controller.run(SimpleNamespace(resume=True))
    assert calls == ["predict_calibration_samples.py", "plot_calibration_results.py", "package_final_model_results.py"]
    state = controller.read_json(controller.ROOT / "outputs/final_masked_models_seed1234/run_status.json")
    assert state["status"] == "computed_pending_visual_review"
    assert "waiting_for_inputs" not in state


def test_stage_failure_resumes_without_repeating_predictions(controller, monkeypatch):
    fake_processes(controller, monkeypatch, True, fail_stage="plot_final_masked_models.py")
    with pytest.raises(RuntimeError, match="fixed_test_figures failed"):
        controller.run(SimpleNamespace(resume=False))
    input_file = controller.ROOT / "修改部分内容/07_最终模型后续分析/03_校准输入待补/图6绘制所需的23个物质输入_待作者确认.csv"
    input_file.parent.mkdir(parents=True)
    input_file.write_text("compound_id\n", encoding="utf-8")
    calls = fake_processes(controller, monkeypatch, True)
    controller.run(SimpleNamespace(resume=True))
    assert "evaluate_final_masked_models.py" not in calls
    state = controller.read_json(controller.ROOT / "outputs/final_masked_models_seed1234/run_status.json")
    assert "failure" not in state


def test_lock_rejects_duplicate_supervisor_and_releases_on_exit(controller, tmp_path):
    directory = tmp_path / "locked"
    with controller.controller_lock(directory):
        with pytest.raises(RuntimeError, match="already running"):
            with controller.controller_lock(directory):
                pytest.fail("Duplicate supervisor acquired the experiment")
    with controller.controller_lock(directory):
        pass
