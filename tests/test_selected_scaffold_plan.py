"""Check final-mask scaffold commands without loading a model or starting training."""

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest


REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts/run_selected_model_scaffold_cv.ps1"


@pytest.fixture
def selection_manifest(tmp_path):
    if not shutil.which("pwsh"):
        pytest.skip("PowerShell 7 is needed for scaffold command planning")
    source = REPO / "results/final_masked_models_seed1234/selection_manifest.json"
    selection = json.loads(source.read_text(encoding="utf-8"))
    for record in selection["models"].values():
        for key in ("checkpoint", "training_params"):
            record[key] = str((source.parent / record[key]).resolve())
    path = tmp_path / "selection.json"
    path.write_text(json.dumps(selection), encoding="utf-8")
    return path


def plan(path, *extra):
    return subprocess.run(
        ["pwsh", "-NoLogo", "-NoProfile", "-File", str(SCRIPT), "-RepoRoot", str(REPO),
         "-SelectionManifest", str(path), "-DryRun", *extra],
        capture_output=True, text=True, encoding="utf-8", check=False,
    )


def arguments_by_name(arguments):
    output = {}
    current = None
    for item in arguments:
        if item.startswith("--"):
            current = item
            output[current] = []
        else:
            output[current].append(item)
    return output


def test_final_scaffold_plan_carries_selected_hyperparameters_and_masks(selection_manifest):
    result = plan(selection_manifest)
    assert result.returncode == 0, result.stderr
    tasks = json.loads(result.stdout)
    assert len(tasks) == 10
    selection = json.loads(selection_manifest.read_text(encoding="utf-8"))
    for task in tasks:
        selected = selection["models"][task["mode"]]
        params = json.loads(Path(selected["training_params"]).read_text(encoding="utf-8"))
        args = arguments_by_name(task["arguments"])
        assert task["feature_code"] == selected["feature_code"]
        assert task["model"] == selected["model_label"]
        assert args["--split-strategy"] == ["scaffold"]
        assert args["--split-seed"] == ["1234"]
        assert args["--fold"] == [str(task["fold"])]
        assert args["--visnet-v2-output-head"] == ["nested_mlp"]
        for key, expected in task["expected_parameters"].items():
            if key in ("split_strategy", "split_seed", "debug_size"):
                continue
            recorded = params[key]
            if key in ("visnet_v2_physchem_mask", "visnet_v2_toxicity_mask") and recorded is None and expected is not None:
                recorded = [1, 1, 1, 1]
            assert expected == recorded
            flag = "--" + key.replace("_", "-")
            if isinstance(expected, bool):
                assert (flag in args) == expected
            elif expected is None:
                assert flag not in args
            elif isinstance(expected, list):
                assert args[flag] == [str(value) for value in expected]
            else:
                if isinstance(expected, float):
                    assert float(args[flag][0]) == expected
                else:
                    assert args[flag] == [str(expected)]
        assert len(task["frozen_split_files"]) == 2
        assert all(len(item["sha256"]) == 64 for item in task["frozen_split_files"])
        assert task["selection"]["removed_features"] == selected["removed_features"]


def test_final_scaffold_plan_rejects_mask_mismatch(selection_manifest):
    selection = json.loads(selection_manifest.read_text(encoding="utf-8"))
    selection["models"]["pos"]["physchem_mask"] = [1, 1, 1, 1]
    selection_manifest.write_text(json.dumps(selection), encoding="utf-8")
    result = plan(selection_manifest)
    assert result.returncode != 0
    assert "mask differs" in result.stderr


def test_final_scaffold_plan_does_not_reuse_legacy_run_name(selection_manifest):
    result = plan(selection_manifest, "-RunName", "selected_models_scaffold_seed1234")
    assert result.returncode != 0
    assert "final_masked_scaffold_seed1234" in result.stderr


def test_final_scaffold_smoke_only_changes_iteration_and_debug_size(selection_manifest):
    result = plan(selection_manifest, "-Smoke")
    assert result.returncode == 0, result.stderr
    tasks = json.loads(result.stdout)
    assert len(tasks) == 10
    for task in tasks:
        assert task["expected_parameters"]["iteration"] == 1
        assert task["expected_parameters"]["debug_size"] == 128


def test_cv_entry_point_accepts_every_generated_argument(selection_manifest):
    result = plan(selection_manifest)
    assert result.returncode == 0, result.stderr
    task = json.loads(result.stdout)[0]
    help_result = subprocess.run(
        [sys.executable, str(REPO / task["script"]), *task["arguments"], "--help"],
        capture_output=True, text=True, encoding="utf-8", check=False,
    )
    assert help_result.returncode == 0, help_result.stderr
    assert "--visnet-v2-output-head" in help_result.stdout


def test_completed_output_validation_rejects_legacy_masks(selection_manifest, tmp_path):
    result = plan(selection_manifest)
    assert result.returncode == 0, result.stderr
    task = json.loads(result.stdout)[0]
    task_path = tmp_path / "task.json"
    task_path.write_text(json.dumps(task), encoding="utf-8")
    actual = dict(task["expected_parameters"], cv_dataname="MMF_GNN_pos", fold=1)
    params_path = tmp_path / "training_params.json"
    params_path.write_text(json.dumps(actual), encoding="utf-8")
    check_script = tmp_path / "check.ps1"
    check_script.write_text(
        "param($SourceScript, $TaskPath, $OutputPath)\n"
        "$ast = [System.Management.Automation.Language.Parser]::ParseFile($SourceScript, [ref]$null, [ref]$null)\n"
        "$function = $ast.Find({ param($node) $node -is [System.Management.Automation.Language.FunctionDefinitionAst] "
        "-and $node.Name -eq 'Test-SelectedParameters' }, $true)\n"
        "Invoke-Expression $function.Extent.Text\n"
        "$task = Get-Content -LiteralPath $TaskPath -Raw | ConvertFrom-Json\n"
        "Test-SelectedParameters -Directory (Get-Item -LiteralPath $OutputPath) -Task $task\n",
        encoding="utf-8",
    )

    def validate():
        completed = subprocess.run(
            ["pwsh", "-NoLogo", "-NoProfile", "-File", str(check_script),
             str(SCRIPT), str(task_path), str(tmp_path)],
            capture_output=True, text=True, encoding="utf-8", check=False,
        )
        assert completed.returncode == 0, completed.stderr
        return completed.stdout.strip()

    assert validate() == "True"
    actual["visnet_v2_physchem_mask"] = None
    params_path.write_text(json.dumps(actual), encoding="utf-8")
    assert validate() == "False"


def test_default_scaffold_plan_uses_frozen_gp_and_gpt():
    if not shutil.which("pwsh"):
        pytest.skip("PowerShell 7 is needed for scaffold command planning")
    result = subprocess.run(
        ["pwsh", "-NoLogo", "-NoProfile", "-File", str(SCRIPT), "-RepoRoot", str(REPO), "-DryRun"],
        capture_output=True, text=True, encoding="utf-8", check=False,
    )
    assert result.returncode == 0, result.stderr
    tasks = json.loads(result.stdout)
    assert len(tasks) == 10
    assert {task["mode"]: task["feature_code"] for task in tasks} == {"pos": "GP", "neg": "GPT"}


def test_selected_scaffold_figure_number_does_not_collide_with_calibration():
    script = SCRIPT.read_text(encoding="utf-8-sig")
    assert "$state.figure_number = '5'" in script
    assert "$state.figure_number = '6'" not in script
