import json
import sys

import numpy as np
import pandas as pd
import pytest
import torch

import core.trainer_tester as trainer_tester
import predict_visnet_v2 as prediction


def test_unlabeled_csv_saves_predictions_without_evaluation(tmp_path, monkeypatch, capsys):
    input_file = tmp_path / "unlabeled.csv"
    input_file.write_text("SMILES,Compound_ID\nCCO,compound-1\n", encoding="utf-8")
    params = {
        "dataname": "MMF_GNN_pos",
        "model_type": "visnet_v2",
        "visnet_v2_feature_level": "graph",
        "property_mean": 10.0,
        "property_std": 2.0,
    }
    params_file = tmp_path / "params.json"
    params_file.write_text(json.dumps(params), encoding="utf-8")

    class Cache:
        failed_smiles = set()

        def __init__(self, name):
            self.name = name

        def get(self, smiles, types):
            return tuple(np.ones(1) for _ in range(5))

    class Model:
        def eval(self):
            pass

        def modules(self):
            return []

    class Tester:
        def __init__(self, *args, **kwargs):
            pass

        def set_preprocessed_data(self, data):
            assert data["CCO"]["target_available"] is False

        def set_standardization_params(self, mean, std):
            pass

        def test_regressor(self, dataset):
            assert dataset[0]["property_value"] == -5.0
            return 0.0, 0.0, 0.0, 0.0, "CCO\t0.0\t123.4"

    monkeypatch.setattr(prediction, "MoleculeCache", Cache)
    monkeypatch.setattr(prediction, "Tester", Tester)
    monkeypatch.setattr(prediction, "load_additional_features", lambda *args, **kwargs: ({}, {}, {}, None))
    monkeypatch.setattr(prediction, "load_model", lambda *args: (Model(), params))
    monkeypatch.setattr(prediction.torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(sys, "argv", [
        "predict_visnet_v2.py",
        "--model_path", str(tmp_path / "model.pt"),
        "--params_path", str(params_file),
        "--input_file", str(input_file),
        "--output_dir", str(tmp_path / "output"),
    ])

    prediction.main()

    output_dir = next((tmp_path / "output").iterdir())
    output_files = list(output_dir.iterdir())
    assert len(output_files) == 1
    result = pd.read_csv(output_files[0])
    assert result.loc[0, "Predicted"] == 123.4
    assert pd.isna(result.loc[0, "Actual"])
    assert "Preprocessed all dataset" in capsys.readouterr().out


def test_single_item_visnet_batch_keeps_target_array(monkeypatch):
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.visnet = torch.nn.Identity()
            self.feature_level = "graph"

        def forward(self, z, pos, batch, physchem_features=None,
                    toxicity_features=None, chromato_features=None):
            return torch.tensor([[0.5]])

    monkeypatch.setattr(trainer_tester, "MoleculeCache", lambda name: object())
    tester = trainer_tester.Tester(
        Model(), visnet=True, batch_test=1, device=torch.device("cpu"),
        MoleculeCacheName="test-single-item",
    )
    tester.set_preprocessed_data({"CCO": {}})
    tester.set_standardization_params(10.0, 2.0)
    monkeypatch.setattr(
        tester, "_prepare_visnet_batch_from_preprocessed",
        lambda batch, train: (
            torch.tensor([6]), torch.zeros((1, 3)), torch.tensor([0]),
            torch.tensor([0.0]), ["CCO"],
        ),
    )

    *_, rows = tester.test_regressor([{"smiles": "CCO", "property_value": 0.0}])

    assert rows == "CCO\t10.0\t11.0"


def test_prediction_cli_fails_when_no_rows_are_predicted(tmp_path, monkeypatch):
    input_file = tmp_path / "unlabeled.csv"
    input_file.write_text("SMILES\nCCO\n", encoding="utf-8")
    params_file = tmp_path / "params.json"
    params_file.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(prediction, "load_model", lambda *args: (object(), {}))
    monkeypatch.setattr(prediction, "load_and_process_dataset_with_features", lambda *args: [{}])
    monkeypatch.setattr(prediction, "predict_dataset", lambda *args: [])
    monkeypatch.setattr(prediction.torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(sys, "argv", [
        "predict_visnet_v2.py",
        "--model_path", str(tmp_path / "model.pt"),
        "--params_path", str(params_file),
        "--input_file", str(input_file),
        "--output_dir", str(tmp_path / "output"),
    ])

    with pytest.raises(RuntimeError, match="No predictions were generated"):
        prediction.main()


def test_single_labeled_prediction_has_no_undefined_metrics_or_plot(tmp_path, monkeypatch):
    input_file = tmp_path / "labeled.csv"
    input_file.write_text("SMILES,Pred_RTI_Positive_ESI\nCCO,100\n", encoding="utf-8")
    params_file = tmp_path / "params.json"
    params_file.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(prediction, "load_model", lambda *args: (object(), {}))
    monkeypatch.setattr(prediction, "load_and_process_dataset_with_features", lambda *args: [{}])
    monkeypatch.setattr(prediction, "predict_dataset", lambda *args: [("CCO", 123.4, 100.0)])
    monkeypatch.setattr(prediction.torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(sys, "argv", [
        "predict_visnet_v2.py",
        "--model_path", str(tmp_path / "model.pt"),
        "--params_path", str(params_file),
        "--input_file", str(input_file),
        "--output_dir", str(tmp_path / "output"),
    ])

    prediction.main()

    output_dir = next((tmp_path / "output").iterdir())
    output_files = list(output_dir.iterdir())
    assert len(output_files) == 1
    result = pd.read_csv(output_files[0])
    assert result.loc[0, "Predicted"] == 123.4
    assert result.loc[0, "Actual"] == 100.0
