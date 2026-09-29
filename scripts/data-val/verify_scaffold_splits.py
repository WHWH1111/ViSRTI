#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Verify scaffold separation in the generated holdout and CV files."""

import argparse
import json
import os
import sys

import pandas as pd


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from core.scaffold_split import (  # noqa: E402
    bemis_murcko_scaffold,
    scaffold_partition_indices,
)


def _read_smiles(file_path):
    frame = pd.read_csv(file_path, sep='\t', header=None, usecols=[0])
    return frame.iloc[:, 0].astype(str).tolist()


def _scaffolds(smiles_list):
    return {bemis_murcko_scaffold(smiles) for smiles in smiles_list}


def verify_holdout(base_path, dataname, seed):
    data_dir = os.path.join(base_path, dataname)
    train_smiles = _read_smiles(
        os.path.join(data_dir, '{}_train_set.txt'.format(dataname))
    )
    test_smiles = _read_smiles(
        os.path.join(data_dir, '{}_test_set.txt'.format(dataname))
    )
    train_indices, dev_indices = scaffold_partition_indices(
        train_smiles, [0.9, 0.1], seed=seed
    )
    final_train = [train_smiles[index] for index in train_indices]
    dev = [train_smiles[index] for index in dev_indices]

    scaffold_sets = {
        'train': _scaffolds(final_train),
        'dev': _scaffolds(dev),
        'test': _scaffolds(test_smiles),
    }
    return {
        'counts': {
            'train': len(final_train),
            'dev': len(dev),
            'test': len(test_smiles),
        },
        'scaffold_counts': {
            name: len(values) for name, values in scaffold_sets.items()
        },
        'overlap_counts': {
            'train_dev': len(scaffold_sets['train'] & scaffold_sets['dev']),
            'train_test': len(scaffold_sets['train'] & scaffold_sets['test']),
            'dev_test': len(scaffold_sets['dev'] & scaffold_sets['test']),
        },
    }


def verify_folds(base_path, dataname, n_folds):
    results = []
    cv_dir = os.path.join(base_path, '{}_{}'.format(dataname, n_folds))
    for fold in range(1, n_folds + 1):
        fold_dir = os.path.join(cv_dir, 'fold_{}'.format(fold))
        train = _read_smiles(
            os.path.join(fold_dir, 'fold_{}_train_set.txt'.format(fold))
        )
        test = _read_smiles(
            os.path.join(fold_dir, 'fold_{}_test_set.txt'.format(fold))
        )
        train_scaffolds = _scaffolds(train)
        test_scaffolds = _scaffolds(test)
        results.append({
            'fold': fold,
            'train_count': len(train),
            'test_count': len(test),
            'scaffold_overlap_count': len(train_scaffolds & test_scaffolds),
        })
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-path', default='./data/MMF-3')
    parser.add_argument('--dataname', required=True)
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--folds', type=int, default=5)
    parser.add_argument('--output', default=None)
    args = parser.parse_args()

    result = {
        'dataname': args.dataname,
        'split_strategy': 'bemis_murcko_scaffold',
        'random_seed': args.seed,
        'acyclic_scaffold_policy': 'canonical_smiles',
        'holdout': verify_holdout(args.base_path, args.dataname, args.seed),
        'cross_validation': verify_folds(
            args.base_path, args.dataname, args.folds
        ),
    }
    overlap_counts = list(result['holdout']['overlap_counts'].values())
    overlap_counts.extend(
        fold['scaffold_overlap_count'] for fold in result['cross_validation']
    )
    result['passed'] = all(count == 0 for count in overlap_counts)

    output_path = args.output or os.path.join(
        args.base_path, args.dataname, 'split_verification.json'
    )
    with open(output_path, 'w', encoding='utf-8') as handle:
        json.dump(result, handle, indent=2, ensure_ascii=False)
        handle.write('\n')
    print(json.dumps(result, indent=2, ensure_ascii=False))
    if not result['passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
