import json
from pathlib import Path

import pandas as pd
import pytest

from core.data_preprocessor import split_raw_data_if_needed
from core.scaffold_split import (
    bemis_murcko_scaffold,
    scaffold_kfold_indices,
    scaffold_partition_indices,
)


SMILES = [
    'c1ccccc1',
    'Oc1ccccc1',
    'Nc1ccccc1',
    'C1CCCCC1',
    'CC1CCCCC1',
    'c1ccncc1',
    'Cc1ccncc1',
    'CCO',
    'OCC',
    'CCN',
]


def _scaffolds(indices):
    return {bemis_murcko_scaffold(SMILES[index]) for index in indices}


def test_scaffold_partition_is_reproducible_and_disjoint():
    first = scaffold_partition_indices(SMILES, [0.7, 0.2, 0.1], seed=1234)
    second = scaffold_partition_indices(SMILES, [0.7, 0.2, 0.1], seed=1234)

    assert first == second
    assert sorted(index for part in first for index in part) == list(range(len(SMILES)))
    assert _scaffolds(first[0]).isdisjoint(_scaffolds(first[1]))
    assert _scaffolds(first[0]).isdisjoint(_scaffolds(first[2]))
    assert _scaffolds(first[1]).isdisjoint(_scaffolds(first[2]))


def test_equivalent_acyclic_smiles_stay_together():
    assert bemis_murcko_scaffold('CCO') == bemis_murcko_scaffold('OCC')


def test_scaffold_kfold_has_no_leakage_and_covers_each_sample_once():
    folds = scaffold_kfold_indices(SMILES, n_splits=3, seed=1234)
    test_indices = []

    for train, test in folds:
        assert set(train).isdisjoint(test)
        assert _scaffolds(train).isdisjoint(_scaffolds(test))
        test_indices.extend(test)

    assert sorted(test_indices) == list(range(len(SMILES)))


def test_invalid_smiles_fails_closed():
    with pytest.raises(ValueError, match='Invalid SMILES'):
        bemis_murcko_scaffold('not-a-smiles')


def test_split_manifest_prevents_stale_cache_reuse(tmp_path):
    dataname = 'example_pos'
    target = 'Pred_RTI_Positive_ESI'
    data_dir = tmp_path / dataname
    data_dir.mkdir()
    pd.DataFrame({'SMILES': SMILES, target: range(len(SMILES))}).to_csv(
        tmp_path / '{}.csv'.format(dataname), index=False
    )
    stale_cache = data_dir / '{}-{}_train_set-cache.pkl'.format(
        dataname, dataname
    )
    stale_cache.write_bytes(b'stale')

    train_path, test_path = split_raw_data_if_needed(
        str(tmp_path),
        dataname,
        target,
        split_strategy='scaffold',
        random_state=1234,
    )

    manifest = json.loads((data_dir / 'split_manifest.json').read_text('utf-8'))
    assert Path(train_path).exists()
    assert Path(test_path).exists()
    assert not stale_cache.exists()
    assert manifest['split_strategy'] == 'scaffold'
    assert manifest['random_seed'] == 1234
    assert manifest['scaffold_overlap_count'] == 0
    assert manifest['train_count'] + manifest['test_count'] == len(SMILES)
    assert manifest['train_sha256']
    assert manifest['test_sha256']

    original_test = Path(test_path).read_text('utf-8')
    Path(test_path).write_text('tampered\t0\n', encoding='utf-8')
    split_raw_data_if_needed(
        str(tmp_path),
        dataname,
        target,
        split_strategy='scaffold',
        random_state=1234,
    )
    assert Path(test_path).read_text('utf-8') == original_test


def test_random_split_matches_author_shuffle_then_slice(tmp_path):
    dataname = 'example_pos'
    target = 'Pred_RTI_Positive_ESI'
    pd.DataFrame({'SMILES': SMILES, target: range(len(SMILES))}).to_csv(
        tmp_path / '{}.csv'.format(dataname), index=False
    )

    train_path, test_path = split_raw_data_if_needed(
        str(tmp_path),
        dataname,
        target,
        split_strategy='random',
        random_state=1234,
    )

    expected = (
        pd.DataFrame({'SMILES': SMILES, target: range(len(SMILES))})
        .sample(frac=1, random_state=1234)
        .reset_index(drop=True)
    )
    actual_train = pd.read_csv(train_path, sep='\t', names=['SMILES', target])
    actual_test = pd.read_csv(test_path, sep='\t', names=['SMILES', target])
    pd.testing.assert_frame_equal(actual_train, expected.iloc[:9].reset_index(drop=True))
    pd.testing.assert_frame_equal(actual_test, expected.iloc[9:].reset_index(drop=True))
