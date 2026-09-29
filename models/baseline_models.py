"""Self-contained baseline models used by the unified reproduction runner."""

from __future__ import annotations

from typing import List, Sequence, Tuple

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch_geometric.nn import GINConv, global_add_pool, global_mean_pool


class WLGraphRegressor(nn.Module):
    """GNN-RT-style Weisfeiler-Lehman fingerprint regressor."""

    def __init__(
        self,
        fingerprint_count: int,
        hidden_dim: int = 48,
        message_layers: int = 6,
        output_layers: int = 6,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(fingerprint_count, hidden_dim)
        self.message_layers = nn.ModuleList(
            nn.Linear(hidden_dim, hidden_dim) for _ in range(message_layers)
        )
        self.output_layers = nn.ModuleList(
            nn.Linear(hidden_dim, hidden_dim) for _ in range(output_layers)
        )
        self.output = nn.Linear(hidden_dim, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        fingerprints: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor,
    ) -> torch.Tensor:
        vectors = self.embedding(fingerprints)
        for layer in self.message_layers:
            transformed = torch.relu(layer(vectors))
            aggregated = torch.zeros_like(transformed)
            if edge_index.numel():
                aggregated.index_add_(0, edge_index[1], transformed[edge_index[0]])
            vectors = torch.nn.functional.normalize(transformed + aggregated, p=2, dim=1)
        molecules = global_add_pool(vectors, batch)
        for layer in self.output_layers:
            molecules = self.dropout(torch.relu(layer(molecules)))
        return self.output(molecules).view(-1)


class GINRegressor(nn.Module):
    """Five-layer graph-isomorphism-network regression baseline."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        layers: int = 5,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.input = nn.Linear(input_dim, hidden_dim)
        self.convolutions = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(layers):
            mlp = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            self.convolutions.append(GINConv(mlp, train_eps=True))
            self.norms.append(nn.BatchNorm1d(hidden_dim))
        self.dropout = nn.Dropout(dropout)
        self.output = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        atom_features: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor,
    ) -> torch.Tensor:
        vectors = torch.relu(self.input(atom_features.float()))
        for convolution, norm in zip(self.convolutions, self.norms):
            residual = vectors
            vectors = torch.relu(norm(convolution(vectors, edge_index)))
            vectors = self.dropout(vectors) + residual
        return self.output(global_mean_pool(vectors, batch)).view(-1)


class SmilesBiLSTMRegressor(nn.Module):
    """Train-set-tokenized SMILES sequence baseline without external weights."""

    def __init__(
        self,
        vocabulary_size: int,
        embedding_dim: int = 64,
        hidden_dim: int = 96,
        layers: int = 2,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocabulary_size, embedding_dim, padding_idx=0)
        self.encoder = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers=layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if layers > 1 else 0.0,
        )
        self.output = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, tokens: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        embedded = self.embedding(tokens)
        packed = pack_padded_sequence(
            embedded, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        packed_output, _ = self.encoder(packed)
        encoded, _ = pad_packed_sequence(packed_output, batch_first=True)
        mask = (
            torch.arange(encoded.shape[1], device=encoded.device)[None, :]
            < lengths[:, None]
        )
        pooled = (encoded * mask.unsqueeze(-1)).sum(dim=1) / lengths.unsqueeze(1)
        return self.output(pooled).view(-1)


def model_identity(model_name: str) -> Tuple[str, str]:
    """Return the publication label and the reproducibility qualification."""
    identities = {
        "gnn_rt": (
            "GNN-RT",
            "local reimplementation of the published WL-fingerprint architecture",
        ),
        "gin": (
            "GIN",
            "local from-scratch GIN; no unavailable external pretraining weights claimed",
        ),
        "smiles_bilstm": (
            "SMILES-BiLSTM",
            "local train-set-tokenized sequence baseline; not RoBERTa/ChemBERTa",
        ),
    }
    return identities[model_name]
