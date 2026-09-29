import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from core import data_preprocessor
from models.visnet_v2 import VisNetV2


@pytest.fixture(scope="module")
def analysis():
    path = Path(__file__).resolve().parents[1] / "scripts/data-val/analyze_gp_gpt_masking.py"
    spec = importlib.util.spec_from_file_location("gp_gpt_analysis", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_development_only_loader_never_reads_test_responses(monkeypatch):
    calls = []

    def create_dataset(filename, *args):
        calls.append(filename)
        assert "test_set" not in filename
        return [(str(i), float(i)) for i in range(20)]

    monkeypatch.setattr(data_preprocessor.pp, "create_dataset", create_dataset)
    args = SimpleNamespace(
        model="visnet_v2", split_strategy="random", dev_split_seed=42,
        development_only=True, debug_size=None,
    )
    train, dev, test, *_ = data_preprocessor.load_datasets(args, "unused", "example", None)
    assert len(train) == 18 and len(dev) == 2 and test == []
    assert calls == ["example_train_set.txt"]


@pytest.mark.parametrize("level", ["graph_physchem", "graph_physchem_toxicity"])
def test_cached_graph_route_matches_full_prediction(level):
    torch.manual_seed(7)
    model = VisNetV2(feature_level=level, dropout_rate=0).eval()
    z = torch.tensor([6, 8, 6, 7], dtype=torch.long)
    positions = torch.tensor([[0., 0., 0.], [1.2, 0., 0.], [0., 0., 0.], [0., 1.1, 0.]])
    batch = torch.tensor([0, 0, 1, 1])
    physchem = torch.randn(2, 4)
    toxicity = torch.randn(2, 4) if level.endswith("toxicity") else None
    with torch.no_grad():
        prediction, _ = model(z, positions, batch, physchem, toxicity)
        _, representation = model.visnet(z, positions, batch)
        cached, _ = model.forward_from_graph_features(model.graph_processor(representation), physchem, toxicity)
    torch.testing.assert_close(prediction, cached)


def test_exact_shap_has_correct_main_and_interaction_effects(analysis):
    values = []
    for subset in range(8):
        a, b, c = [(subset >> j) & 1 for j in range(3)]
        values.append(3 + 2 * a + 4 * b + 6 * c + 8 * a * b)
    result = analysis.exact_shap_from_values(values, 3)
    np.testing.assert_allclose(result, [6, 8, 6])
    assert result.sum() == values[-1] - values[0]


def test_masking_stops_at_first_development_mae_increase(analysis):
    assert analysis.retained_step(42.0, 41.5)
    assert analysis.retained_step(42.0, 42.0)
    assert not analysis.retained_step(42.0, 42.1)


def test_unused_descriptor_has_zero_exact_shap(analysis):
    values = [float(mask & 1) for mask in range(16)]
    np.testing.assert_allclose(analysis.exact_shap_from_values(values, 4), [1, 0, 0, 0])


def test_training_descriptor_pool_keeps_failed_graph_rows_but_excludes_test(tmp_path):
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location("mask_runner", root / "scripts/run_gp_gpt_masking.py")
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    source, output = tmp_path / "source", tmp_path / "prepared"
    name = "MMF_GNN_neg"
    origin = source / "input_data" / name
    origin.mkdir(parents=True)
    output.mkdir()
    (origin / (name + "_train_set.txt")).write_text("CCO\t10\nfailed_graph\t20\n", encoding="utf-8")
    for filename in (name + "-" + name + "_train_set-cache.pkl", name + "-atom_dict.pickle",
                     name + "-bond_dict.pickle", name + "-edge_dict.pickle",
                     name + "-fingerprint_dict.pickle", "split_manifest.json"):
        (origin / filename).write_bytes(b"frozen")
    runner.analysis.write_csv(source / "input_data" / (name + ".csv"), [
        {"SMILES": "CCO", "Monoiso_Mass": 46, "Pred_RTI_Negative_ESI": 10},
        {"SMILES": "failed_graph", "Monoiso_Mass": 200, "Pred_RTI_Negative_ESI": 20},
        {"SMILES": "held_out", "Monoiso_Mass": 80, "Pred_RTI_Negative_ESI": 999},
    ])
    experiment = source / "experiments/GPT_neg"
    experiment.mkdir(parents=True)
    runner.analysis.write_csv(experiment / "partition_assignments.csv", [
        {"subset": "train", "smiles": "CCO"}, {"subset": "test", "smiles": "held_out"},
    ])
    runner.prepare_train_inputs(source, output, "neg")
    descriptors = runner.analysis.read_csv(output / (name + ".csv"))
    assert [row["SMILES"] for row in descriptors] == ["CCO", "failed_graph"]
    assert all(not any("RTI" in column for column in row) for row in descriptors)
    assert runner.analysis.read_csv(output / name / "development_partition_assignments.csv") == [
        {"subset": "train", "smiles": "CCO"}]


def test_frozen_excluded_molecules_are_not_reprocessed():
    def unexpected_reprocessing(*args, **kwargs):
        raise AssertionError("Known failed molecule was processed again")
    cache = SimpleNamespace(failed_smiles=set(), process_smiles=unexpected_reprocessing)
    dataset = [("CCO", 10), ("failed_graph", 20)]
    result = data_preprocessor.filter_failed_molecules(
        dataset, {"CCO": {}}, cache, {}, "training", allowed_smiles={"CCO"})
    assert result == [("CCO", 10)]
