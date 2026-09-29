#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Deterministic Bemis-Murcko scaffold splitting utilities."""

import random
from collections import defaultdict

from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold


def bemis_murcko_scaffold(smiles, include_chirality=False):
    """Return a stable scaffold key for one SMILES string.

    Bemis-Murcko scaffolds are empty for acyclic molecules. Those compounds are
    keyed by canonical SMILES so duplicate acyclic structures stay together
    without forcing every acyclic compound into one partition.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("Invalid SMILES: {}".format(smiles))

    scaffold = MurckoScaffold.MurckoScaffoldSmiles(
        mol=mol,
        includeChirality=include_chirality,
    )
    if scaffold:
        return "BM::{}".format(scaffold)

    canonical = Chem.MolToSmiles(mol, isomericSmiles=include_chirality)
    return "ACYCLIC::{}".format(canonical)


def scaffold_partition_indices(smiles_list, fractions, seed=1234):
    """Partition row indices while keeping each scaffold in one partition."""
    if not smiles_list:
        raise ValueError("Cannot split an empty SMILES collection")
    if not fractions or any(fraction <= 0 for fraction in fractions):
        raise ValueError("All split fractions must be positive")
    if abs(sum(fractions) - 1.0) > 1e-8:
        raise ValueError("Split fractions must sum to 1.0")

    scaffold_groups = defaultdict(list)
    for index, smiles in enumerate(smiles_list):
        scaffold_groups[bemis_murcko_scaffold(smiles)].append(index)

    if len(scaffold_groups) < len(fractions):
        raise ValueError(
            "Need at least {} distinct scaffolds, found {}".format(
                len(fractions), len(scaffold_groups)
            )
        )

    groups = sorted(scaffold_groups.items(), key=lambda item: item[0])
    random.Random(seed).shuffle(groups)
    groups.sort(key=lambda item: len(item[1]), reverse=True)

    targets = [len(smiles_list) * fraction for fraction in fractions]
    partitions = [[] for _ in fractions]
    counts = [0 for _ in fractions]

    for _, group_indices in groups:
        # Fill the partition with the largest proportional deficit. The larger
        # target wins exact ties, keeping the first large group in training.
        destination = max(
            range(len(fractions)),
            key=lambda i: (
                (targets[i] - counts[i]) / targets[i],
                targets[i],
                -i,
            ),
        )
        partitions[destination].extend(group_indices)
        counts[destination] += len(group_indices)

    for partition in partitions:
        partition.sort()
    return partitions


def scaffold_kfold_indices(smiles_list, n_splits=5, seed=1234):
    """Return deterministic scaffold-disjoint train/test indices for K-fold CV."""
    if n_splits < 2:
        raise ValueError("n_splits must be at least 2")

    test_partitions = scaffold_partition_indices(
        smiles_list,
        fractions=[1.0 / n_splits] * n_splits,
        seed=seed,
    )
    all_indices = set(range(len(smiles_list)))
    folds = []
    for test_indices in test_partitions:
        test_set = set(test_indices)
        train_indices = sorted(all_indices - test_set)
        folds.append((train_indices, test_indices))
    return folds
