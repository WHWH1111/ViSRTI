#!/usr/bin/env python3
"""Train one baseline on the exact ViSRTI-G train/dev/test partitions."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from torch_geometric.data import Batch, Data

from models.baseline_models import (
    GINRegressor,
    SmilesBiLSTMRegressor,
    WLGraphRegressor,
    model_identity,
)


@dataclass(frozen=True)
class Record:
    row_id: int
    smiles: str
    target: float


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_records(path: Path, subset: str) -> List[Record]:
    delimiter = "\t" if path.suffix.lower() == ".txt" else ","
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter=delimiter))
    if subset == "test":
        smiles_key, target_key = "smiles", "source_rti"
    else:
        smiles_key, target_key = "Smiles", "Correct"
    records = [
        Record(index, row[smiles_key], float(row[target_key]))
        for index, row in enumerate(rows)
    ]
    if not records:
        raise ValueError("No records in {}".format(path))
    return records


def _visnet_cache(path: Path) -> Dict[str, Mapping[str, torch.Tensor]]:
    stored = torch.load(path, map_location="cpu")["success"]
    result = {}
    for key, graph in stored.items():
        marker = key.rfind("_[('")
        smiles = key[:marker] if marker >= 0 else key
        result[smiles] = graph
    return result


class TokenLookup:
    def __init__(self) -> None:
        self.values: Dict[object, int] = {"<UNK>": 0}
        self.frozen = False

    def index(self, value: object) -> int:
        if value in self.values:
            return self.values[value]
        if self.frozen:
            return 0
        self.values[value] = len(self.values)
        return self.values[value]


class CachedWLEncoder:
    """Build GNN-RT WL fingerprints from already validated molecular graphs."""

    def __init__(self) -> None:
        self.atoms = TokenLookup()
        self.fingerprints = TokenLookup()

    def freeze(self) -> None:
        self.atoms.frozen = True
        self.fingerprints.frozen = True

    def encode(self, graph: Mapping[str, torch.Tensor]) -> Data:
        node_features = graph["x"]
        atom_keys = [
            (int(row[-6].item()), int(row[-5].item()))
            for row in node_features
        ]
        atom_ids = [self.atoms.index(key) for key in atom_keys]
        neighbours: Dict[int, List[Tuple[int, int]]] = {
            index: [] for index in range(len(atom_ids))
        }
        edge_index = graph["edge_index"].long()
        edge_types = graph["edge_attr"].argmax(dim=1).long()
        for edge_position in range(edge_index.shape[1]):
            left = int(edge_index[0, edge_position])
            right = int(edge_index[1, edge_position])
            neighbours[left].append((right, int(edge_types[edge_position])))
        fingerprints = []
        for index, atom_id in enumerate(atom_ids):
            adjacent = tuple(sorted((atom_ids[right], bond) for right, bond in neighbours[index]))
            fingerprints.append(self.fingerprints.index((atom_id, adjacent)))
        return Data(x=torch.tensor(fingerprints, dtype=torch.long), edge_index=edge_index)


def cached_graph_sets(
    model_name: str,
    mode: str,
    repo_root: Path,
    subsets: Mapping[str, Sequence[Record]],
) -> Tuple[Dict[str, List[Data]], Dict[str, object]]:
    dataset_name = "MMF_GNN_{}".format(mode)
    cache_root = repo_root / ".cache" / "molecule_graphs"
    train_path = cache_root / ("visnet_train_v2_" + dataset_name + ".pt")
    test_path = cache_root / ("visnet_test_v2_" + dataset_name + ".pt")
    train_cache, test_cache = _visnet_cache(train_path), _visnet_cache(test_path)
    if model_name == "gnn_rt":
        encoder = CachedWLEncoder()
        graph_sets = {"train": []}
        for record in subsets["train"]:
            if record.smiles not in train_cache:
                raise KeyError("Missing cached molecular graph for {}".format(record.smiles))
            graph_sets["train"].append(encoder.encode(train_cache[record.smiles]))
        encoder.freeze()
        for subset in ("dev", "test"):
            source = test_cache if subset == "test" else train_cache
            graph_sets[subset] = []
            for record in subsets[subset]:
                if record.smiles not in source:
                    raise KeyError("Missing cached molecular graph for {}".format(record.smiles))
                graph_sets[subset].append(encoder.encode(source[record.smiles]))
        state = {
            "source": "train-fitted GNN-RT WL fingerprints derived from author-generated molecular graph caches",
            "train_cache": str(train_path),
            "test_cache": str(test_path),
            "atom_token_count": len(encoder.atoms.values),
            "fingerprint_count": len(encoder.fingerprints.values),
            "fit_scope": "train only; unknown index 0 for dev/test",
            "rdkit_runtime_required": False,
        }
        return graph_sets, state

    graph_sets = {}
    input_dim = None
    for subset, records in subsets.items():
        source = test_cache if subset == "test" else train_cache
        graphs = []
        for record in records:
            if record.smiles not in source:
                raise KeyError("Missing cached molecular graph for {}".format(record.smiles))
            cached = source[record.smiles]
            input_dim = int(cached["x"].shape[1])
            graphs.append(Data(x=cached["x"].float(), edge_index=cached["edge_index"].long()))
        graph_sets[subset] = graphs
    state = {
        "source": "author-generated ViSRTI molecular graph caches",
        "train_cache": str(train_path),
        "test_cache": str(test_path),
        "input_dim": input_dim,
        "rdkit_runtime_required": False,
    }
    return graph_sets, state


class GraphDataset(Dataset):
    def __init__(self, records: Sequence[Record], graphs: Sequence[Data]) -> None:
        self.records = list(records)
        self.graphs = list(graphs)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> Tuple[Data, float, int]:
        return self.graphs[index], self.records[index].target, self.records[index].row_id


class SequenceDataset(Dataset):
    def __init__(self, records: Sequence[Record], vocabulary: Mapping[str, int]) -> None:
        self.records = list(records)
        self.tokens = [
            torch.tensor([vocabulary.get(char, 1) for char in record.smiles], dtype=torch.long)
            for record in records
        ]

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, float, int]:
        return self.tokens[index], self.records[index].target, self.records[index].row_id


def graph_collate(items: Sequence[Tuple[Data, float, int]]) -> Tuple[Batch, torch.Tensor, torch.Tensor]:
    graphs, targets, row_ids = zip(*items)
    return (
        Batch.from_data_list(list(graphs)),
        torch.tensor(targets, dtype=torch.float32),
        torch.tensor(row_ids, dtype=torch.long),
    )


def sequence_collate(
    items: Sequence[Tuple[torch.Tensor, float, int]]
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    tokens, targets, row_ids = zip(*items)
    lengths = torch.tensor([len(item) for item in tokens], dtype=torch.long)
    return (
        pad_sequence(tokens, batch_first=True, padding_value=0),
        lengths,
        torch.tensor(targets, dtype=torch.float32),
        torch.tensor(row_ids, dtype=torch.long),
    )


def metrics(actual: np.ndarray, predicted: np.ndarray) -> Dict[str, float]:
    residual = predicted - actual
    denominator = float(np.sum((actual - np.mean(actual)) ** 2))
    return {
        "r2": float(1.0 - np.sum(residual ** 2) / denominator),
        "mae": float(np.mean(np.abs(residual))),
        "rmse": float(np.sqrt(np.mean(residual ** 2))),
        "pcc": float(np.corrcoef(actual, predicted)[0, 1]),
    }


def predict(
    model: nn.Module,
    loader: DataLoader,
    model_name: str,
    device: torch.device,
    target_mean: float,
    target_std: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    actual, predicted, row_ids = [], [], []
    with torch.no_grad():
        for batch in loader:
            if model_name == "smiles_bilstm":
                tokens, lengths, targets, identifiers = batch
                output = model(tokens.to(device), lengths.to(device))
            else:
                graphs, targets, identifiers = batch
                graphs = graphs.to(device)
                output = model(graphs.x, graphs.edge_index, graphs.batch)
            predicted.extend((output.cpu().numpy() * target_std + target_mean).tolist())
            actual.extend(targets.numpy().tolist())
            row_ids.extend(identifiers.numpy().tolist())
    return np.asarray(actual), np.asarray(predicted), np.asarray(row_ids)


def build_loaders(
    model_name: str,
    mode: str,
    repo_root: Path,
    subsets: Mapping[str, Sequence[Record]],
    batch_size: int,
) -> Tuple[Dict[str, DataLoader], Dict[str, object]]:
    if model_name == "smiles_bilstm":
        chars = sorted({char for record in subsets["train"] for char in record.smiles})
        vocabulary = {"<PAD>": 0, "<UNK>": 1}
        vocabulary.update({char: index + 2 for index, char in enumerate(chars)})
        datasets = {
            subset: SequenceDataset(records, vocabulary) for subset, records in subsets.items()
        }
        collate = sequence_collate
        state = {"vocabulary": vocabulary, "fit_scope": "train only"}
    else:
        graph_sets, state = cached_graph_sets(model_name, mode, repo_root, subsets)
        datasets = {
            subset: GraphDataset(records, graph_sets[subset])
            for subset, records in subsets.items()
        }
        collate = graph_collate
    loaders = {
        subset: DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(subset == "train"),
            collate_fn=collate,
            num_workers=0,
        )
        for subset, dataset in datasets.items()
    }
    return loaders, state


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("gnn_rt", "gin", "smiles_bilstm"), required=True)
    parser.add_argument("--mode", choices=("pos", "neg"), required=True)
    parser.add_argument("--split-source", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--loss-function", choices=("mse", "l1"), default="mse")
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--scheduler-patience", type=int, default=10)
    parser.add_argument("--scheduler-factor", type=float, default=0.5)
    parser.add_argument("--minimum-learning-rate", type=float, default=1e-6)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--debug-size", type=int)
    args = parser.parse_args()

    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError("Refusing to overwrite experiment: {}".format(output_dir))
    output_dir.mkdir(parents=True)
    seed_everything(args.seed)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the baseline reproduction run")
    device = torch.device("cuda:0")

    split_source = args.split_source.resolve()
    paths = {
        "train": split_source / "train_prediction_train.txt",
        "dev": split_source / "train_prediction_dev.txt",
        "test": split_source / "predictions_test.csv",
    }
    subsets = {subset: read_records(path, subset) for subset, path in paths.items()}
    if args.debug_size:
        subsets = {
            subset: records[: min(args.debug_size, len(records))]
            for subset, records in subsets.items()
        }
    train_targets = np.asarray([record.target for record in subsets["train"]])
    target_mean, target_std = float(train_targets.mean()), float(train_targets.std())
    loaders, preprocessing = build_loaders(
        args.model, args.mode, args.repo_root.resolve(), subsets, args.batch_size
    )

    if args.model == "gnn_rt":
        model = WLGraphRegressor(int(preprocessing["fingerprint_count"]))
    elif args.model == "gin":
        model = GINRegressor(int(preprocessing["input_dim"]))
    else:
        model = SmilesBiLSTMRegressor(len(preprocessing["vocabulary"]))
    model = model.to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        patience=args.scheduler_patience,
        factor=args.scheduler_factor,
        min_lr=args.minimum_learning_rate,
    )
    loss_function = nn.MSELoss() if args.loss_function == "mse" else nn.L1Loss()
    best_dev_mae = math.inf
    patience_left = args.patience
    history: List[Dict[str, float]] = []
    best_path = output_dir / "best.pt"

    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for batch in loaders["train"]:
            optimizer.zero_grad()
            if args.model == "smiles_bilstm":
                tokens, lengths, targets, _ = batch
                output = model(tokens.to(device), lengths.to(device))
            else:
                graphs, targets, _ = batch
                graphs = graphs.to(device)
                output = model(graphs.x, graphs.edge_index, graphs.batch)
            standardized = (targets.to(device) - target_mean) / target_std
            loss = loss_function(output, standardized)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        dev_actual, dev_predicted, _ = predict(
            model, loaders["dev"], args.model, device, target_mean, target_std
        )
        dev_mae = metrics(dev_actual, dev_predicted)["mae"]
        scheduler.step(dev_mae)
        history.append({
            "epoch": epoch,
            "train_loss": float(np.mean(losses)),
            "dev_mae": dev_mae,
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
        })
        print(
            "epoch={} train_{}={:.6f} dev_mae={:.6f} lr={:.8f}".format(
                epoch,
                args.loss_function,
                np.mean(losses),
                dev_mae,
                optimizer.param_groups[0]["lr"],
            ),
            flush=True,
        )
        if dev_mae < best_dev_mae:
            best_dev_mae = dev_mae
            patience_left = args.patience
            torch.save(model.state_dict(), best_path)
        else:
            patience_left -= 1
            if patience_left == 0:
                break

    model.load_state_dict(torch.load(best_path, map_location=device))
    test_actual, test_predicted, row_ids = predict(
        model, loaders["test"], args.model, device, target_mean, target_std
    )
    calculated = metrics(test_actual, test_predicted)
    label, qualification = model_identity(args.model)
    configuration = {
        "model_key": args.model,
        "model_label": label,
        "model_qualification": qualification,
        "mode": args.mode,
        "seed": args.seed,
        "epochs_requested": args.epochs,
        "epochs_completed": len(history),
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "scheduler_patience": args.scheduler_patience,
        "scheduler_factor": args.scheduler_factor,
        "minimum_learning_rate": args.minimum_learning_rate,
        "early_stopping_patience": args.patience,
        "loss": "{} on train-standardized target".format(args.loss_function.upper()),
        "target_mean_train_only": target_mean,
        "target_std_train_only": target_std,
        "device": str(device),
        "cuda_device": torch.cuda.get_device_name(0),
        "preprocessing": preprocessing,
        "split_source": str(split_source),
        "debug_size": args.debug_size,
    }
    split_manifest = {
        "source_experiment": str(split_source),
        "subsets": {
            subset: {"path": str(paths[subset]), "sha256": sha256(paths[subset]), "n": len(records)}
            for subset, records in subsets.items()
        },
        "identity_rule": "exact row order from ViSRTI-G model-used files",
    }
    with (output_dir / "predictions_test.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("row_id", "smiles", "source_rti", "predicted_rti", "residual_source_minus_prediction"),
        )
        writer.writeheader()
        for row_id, actual, predicted in zip(row_ids, test_actual, test_predicted):
            record = subsets["test"][int(row_id)]
            writer.writerow({
                "row_id": int(row_id),
                "smiles": record.smiles,
                "source_rti": float(actual),
                "predicted_rti": float(predicted),
                "residual_source_minus_prediction": float(actual - predicted),
            })
    (output_dir / "experiment_config.json").write_text(
        json.dumps(configuration, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (output_dir / "split_manifest.json").write_text(
        json.dumps(split_manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (output_dir / "metrics.json").write_text(
        json.dumps(calculated, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    with (output_dir / "training_history.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=("epoch", "train_loss", "dev_mae", "learning_rate")
        )
        writer.writeheader()
        writer.writerows(history)
    torch.save({"state_dict": model.state_dict(), "configuration": configuration}, output_dir / "model.pt")
    print(json.dumps(calculated, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
