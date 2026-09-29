import importlib.util
import json
import sys
import subprocess
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture(scope="module")
def evaluation():
    directory = Path(__file__).resolve().parents[1] / "scripts/data-val"
    sys.path.insert(0, str(directory))
    spec = importlib.util.spec_from_file_location("final_evaluation", directory / "evaluate_final_masked_models.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def step(index, mae, accepted):
    return {"step": index, "metrics": {"mae": mae}, "accepted": accepted}


def test_selection_uses_development_mae_and_previous_step(evaluation):
    steps = [step(0, 5, True), step(1, 4, True), step(2, 4.5, False)]
    assert evaluation.validate_selected_steps(steps, {"selected_step": 1})["step"] == 1
    with pytest.raises(ValueError, match="Selected mask"):
        evaluation.validate_selected_steps(steps, {"selected_step": 2})


def test_no_candidates_after_first_decline(evaluation):
    with pytest.raises(ValueError, match="continued"):
        evaluation.validate_selected_steps(
            [step(0, 5, True), step(1, 6, False), step(2, 4, True)], {"selected_step": 2})


def test_selection_cannot_silently_change_metric(evaluation):
    with pytest.raises(ValueError, match="development MAE"):
        evaluation.validate_selected_steps([step(0, 5, True), step(1, 6, True)], {"selected_step": 1})


def test_running_search_cannot_open_test_data(evaluation, tmp_path, monkeypatch):
    run = tmp_path / "mask"
    run.mkdir()
    (run / "run_status.json").write_text(json.dumps({"status": "running"}))
    monkeypatch.setattr(evaluation, "read_csv", lambda *a: pytest.fail("Test data opened before freeze"))
    with pytest.raises(ValueError, match="both development"):
        evaluation.freeze_selection(run, tmp_path / "final")
    assert not (tmp_path / "final").exists()


def test_prediction_source_and_residual_are_preserved(evaluation):
    rows = evaluation.prediction_rows({"smiles": ["CCO", "CO"], "y": np.array([10., 20.])}, [12., 17.])
    assert [r["residual_source_minus_prediction"] for r in rows] == [-2., 3.]
    assert [r["source_rti"] for r in rows] == [10., 20.]


def test_freeze_cli_does_not_import_scientific_libraries(tmp_path):
    mask = tmp_path / "mask"
    mask.mkdir()
    (mask / "run_status.json").write_text(json.dumps({"status": "completed", "source": "frozen_source"}))
    p_names = ["Monoiso_Mass", "LogKow", "LogP", "Koc"]
    t_names = ["IC50", "48h-LC50", "72h-EC50", "96h-LC50"]
    for mode, code in (("pos", "GP"), ("neg", "GPT")):
        directory = mask / (mode + "_" + code)
        directory.mkdir()
        checkpoint, params = directory / "best.pt", directory / "training_params.json"
        checkpoint.write_bytes(b"unit-test-only-checkpoint")
        params.write_text(json.dumps({"visnet_v2_physchem_mask": [1, 1, 1, 0],
                                      "visnet_v2_toxicity_mask": [1, 1, 1, 1] if code == "GPT" else None}))
        record = {"step": 0, "metrics": {"mae": 5}, "accepted": True, "removed_features": ["Koc"],
                  "training_params": str(params), "checkpoint": str(checkpoint)}
        (directory / "masking_steps.json").write_text(json.dumps([record]))
        (directory / "selected_mask.json").write_text(json.dumps({"selected_step": 0}))
        (directory / "shap_result.json").write_text(json.dumps({"feature_names": p_names + (t_names if code == "GPT" else [])}))
    output = tmp_path / "frozen"
    script = Path(__file__).resolve().parents[1] / "scripts/data-val/evaluate_final_masked_models.py"
    command = (
        "import sys,runpy,importlib.abc; "
        "\nclass BlockScientific(importlib.abc.MetaPathFinder):"
        "\n def find_spec(self, fullname, path=None, target=None):"
        "\n  if fullname.split('.')[0] in ('numpy','torch','rdkit','analyze_gp_gpt_masking','core','train_visnet_v2'):"
        "\n   raise AssertionError('Unnecessary scientific import during freeze: '+fullname)"
        "\nsys.meta_path.insert(0,BlockScientific()); "
        "sys.argv=[sys.argv[1],'--mask-run',sys.argv[2],'--output-dir',sys.argv[3],'--freeze-only']; "
        "runpy.run_path(sys.argv[0],run_name='__main__')"
    )
    result = subprocess.run([sys.executable, "-c", command, str(script), str(mask), str(output)],
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    manifest = json.loads((output / "selection_manifest.json").read_text())
    assert manifest["models"]["pos"]["toxicity_mask"] is None
    assert manifest["models"]["neg"]["retained_features"] == p_names[:3] + t_names
