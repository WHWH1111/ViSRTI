#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据预处理模块

处理分子数据，生成图结构表示，创建数据集
"""

import glob
import hashlib
import json
import os
import pandas as pd
import numpy as np

import preprocess as pp
from tqdm import tqdm
from utils.molecule import MoleculeCache
from sklearn.model_selection import train_test_split
from utils.atom_types import ATOM_TYPES
from utils.feature_utils import get_feature_config
from .scaffold_split import (
    bemis_murcko_scaffold,
    scaffold_partition_indices,
)

# 物化特征名称到索引的映射，用于特征掩码设置
# 当前物化特征处理逻辑:
# 1. Monoiso_Mass (索引 0)
# 2. logKow/Exp_logKow (索引 1) - 优先使用Exp_logKow_EPISuite，否则使用logKow_EPISuite
# 3. alogp/xlogp (索引 2) - 优先使用xlogp_ChemSpider，否则使用alogp_ChemSpider
# 4. Koc_predicted (索引 3)
PHYSICOCHEMICAL_FEATURE_MAP = {
    'Monoiso_Mass': 0,
    'logKow/Exp_logKow': 1,  # 优先使用Exp_logKow_EPISuite，否则使用logKow_EPISuite
    'alogp/xlogp': 2,         # 优先使用xlogp_ChemSpider，否则使用alogp_ChemSpider
    'Koc_predicted': 3
}

# 毒性特征名称到索引的映射
TOXICITY_FEATURE_MAP = {
    'Tetrahymena_pyriformis_toxicity': 0,
    'Daphnia_toxicity': 1,
    'Algae_toxicity': 2,
    'Pimephales_promelas_toxicity': 3
}

# 色谱特征名称到索引的映射
CHROMATO_FEATURE_MAP = {
    'Prob. +ESI': 0,
    'Prob. -ESI': 1
}


def split_dataset(dataset, split_ratio=0.9, split_strategy='scaffold', random_state=1234):
    """Split the dataset into training and development sets."""
    if split_strategy == 'scaffold':
        partitions = scaffold_partition_indices(
            [data[0] for data in dataset],
            fractions=[split_ratio, 1.0 - split_ratio],
            seed=random_state,
        )
        train_data = [dataset[index] for index in partitions[0]]
        dev_data = [dataset[index] for index in partitions[1]]
    elif split_strategy == 'random':
        train_data, dev_data = train_test_split(
            dataset,
            train_size=split_ratio,
            random_state=random_state,
        )
    else:
        raise ValueError("Unsupported split strategy: {}".format(split_strategy))
    return train_data, dev_data

def standardize_dataset(dataset_train, mean_val=None, std_val=None):
    """
    标准化训练数据集中的属性值
    """
    # 如果没有提供均值和标准差，则从训练集计算
    if mean_val is None or std_val is None:
        properties = [data[-1] for data in dataset_train]
        mean_val = np.mean(properties)
        std_val = np.std(properties)
        if std_val == 0:
            std_val = 1  # 避免除零错误
    
    # 标准化数据集
    standardized_dataset = []
    for data in dataset_train:
        if len(data) == 5:  # 原始格式 (smiles, fingerprints, adjacency, molecular_size, property)
            smiles, fingerprints, adjacency, molecular_size, property_value = data
            standardized_property = (property_value - mean_val) / std_val
            standardized_dataset.append((smiles, fingerprints, adjacency, molecular_size, standardized_property))
        elif len(data) == 7:  # 新格式 (smiles, fingerprints, adjacency, molecular_size, physchem_features, toxicity_features, chromato_features, property)
            smiles, fingerprints, adjacency, molecular_size, physchem_features, toxicity_features, chromato_features, property_value = data
            standardized_property = (property_value - mean_val) / std_val
            standardized_dataset.append((smiles, fingerprints, adjacency, molecular_size, physchem_features, toxicity_features, chromato_features, standardized_property))
        else:  # 扩展格式 (smiles, fingerprints, adjacency, molecular_size, additional_features, property)
            smiles, fingerprints, adjacency, molecular_size, additional_features, property_value = data
            standardized_property = (property_value - mean_val) / std_val
            standardized_dataset.append((smiles, fingerprints, adjacency, molecular_size, additional_features, standardized_property))
    
    return standardized_dataset, mean_val, std_val

def standardize_additional_features_in_dataset(dataset, mean_vals, std_vals):
    """
    使用给定的均值和标准差对数据集中的额外特征进行标准化
    
    Args:
        dataset: 数据集
        mean_vals: 特征均值列表
        std_vals: 特征标准差列表
        
    Returns:
        standardized_dataset: 标准化后的数据集
    """
    if mean_vals is None or std_vals is None:
        return dataset
        
    standardized_dataset = []
    for data in dataset:
        if len(data) >= 6:  # 扩展格式 (smiles, fingerprints, adjacency, molecular_size, additional_features, property)
            smiles, fingerprints, adjacency, molecular_size, additional_features, property_value = data
            # 标准化额外特征
            standardized_features = (additional_features - mean_vals) / std_vals
            standardized_dataset.append((smiles, fingerprints, adjacency, molecular_size, standardized_features, property_value))
        elif len(data) == 8:  # 新格式 (smiles, fingerprints, adjacency, molecular_size, physchem_features, toxicity_features, chromato_features, property)
            smiles, fingerprints, adjacency, molecular_size, physchem_features, toxicity_features, chromato_features, property_value = data
            # 标准化额外特征
            standardized_physchem = (physchem_features - mean_vals[0]) / std_vals[0] if mean_vals[0] is not None else physchem_features
            standardized_toxicity = (toxicity_features - mean_vals[1]) / std_vals[1] if mean_vals[1] is not None else toxicity_features
            standardized_chromato = (chromato_features - mean_vals[2]) / std_vals[2] if mean_vals[2] is not None else chromato_features
            standardized_dataset.append((smiles, fingerprints, adjacency, molecular_size, standardized_physchem, standardized_toxicity, standardized_chromato, property_value))
        else:
            # 如果不是扩展格式，直接添加
            standardized_dataset.append(data)
            
    return standardized_dataset

def _sha256_file(file_path):
    digest = hashlib.sha256()
    with open(file_path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _invalidate_split_caches(data_path, dataname):
    """Remove only caches derived from the generated split files."""
    patterns = [
        '{}-{}_train_set*-cache.pkl'.format(dataname, dataname),
        '{}-{}_test_set*-cache.pkl'.format(dataname, dataname),
    ]
    removed = []
    for pattern in patterns:
        for cache_path in glob.glob(os.path.join(data_path, pattern)):
            os.remove(cache_path)
            removed.append(cache_path)
    return removed


def split_raw_data_if_needed(
        path,
        dataname,
        target_column_name,
        split_strategy='scaffold',
        random_state=1234,
        test_fraction=0.10,
        force_resplit=False):
    """Create one authoritative train/test split and record its provenance.

    Existing files are reused only when the split manifest matches the raw CSV
    hash, target column, strategy, seed, and requested test fraction.
    """
    if split_strategy not in {'random', 'scaffold'}:
        raise ValueError("Unsupported split strategy: {}".format(split_strategy))
    if not 0 < test_fraction < 1:
        raise ValueError("test_fraction must be between 0 and 1")

    raw_data_path = os.path.join(path, '{}.csv'.format(dataname))
    if not os.path.exists(raw_data_path):
        raise FileNotFoundError("Raw data file {} not found!".format(raw_data_path))

    data_path = os.path.join(path, dataname)
    os.makedirs(data_path, exist_ok=True)
    train_file_path = os.path.join(data_path, '{}_train_set.txt'.format(dataname))
    test_file_path = os.path.join(data_path, '{}_test_set.txt'.format(dataname))
    manifest_path = os.path.join(data_path, 'split_manifest.json')

    expected_manifest = {
        'schema_version': 1,
        'source_file': os.path.abspath(raw_data_path),
        'source_sha256': _sha256_file(raw_data_path),
        'target_column': target_column_name,
        'split_strategy': split_strategy,
        'random_seed': int(random_state),
        'test_fraction': float(test_fraction),
        'acyclic_scaffold_policy': 'canonical_smiles',
    }
    current_manifest = None
    if os.path.exists(manifest_path):
        try:
            with open(manifest_path, 'r', encoding='utf-8') as handle:
                current_manifest = json.load(handle)
        except (OSError, ValueError):
            current_manifest = None

    manifest_matches = current_manifest is not None and all(
        current_manifest.get(key) == value
        for key, value in expected_manifest.items()
    )
    split_files_exist = os.path.exists(train_file_path) and os.path.exists(test_file_path)
    output_hashes_match = (
        split_files_exist
        and current_manifest is not None
        and current_manifest.get('train_sha256') == _sha256_file(train_file_path)
        and current_manifest.get('test_sha256') == _sha256_file(test_file_path)
    )
    if split_files_exist and manifest_matches and output_hashes_match and not force_resplit:
        print("Reusing verified {} split (seed={}).".format(
            split_strategy, random_state
        ))
        return train_file_path, test_file_path

    print("Creating {} train/test split from raw CSV (seed={})...".format(
        split_strategy, random_state
    ))
    df = pd.read_csv(raw_data_path)
    df_clean = df[['SMILES', target_column_name]].dropna().reset_index(drop=True)

    if split_strategy == 'scaffold':
        train_indices, test_indices = scaffold_partition_indices(
            df_clean['SMILES'].tolist(),
            fractions=[1.0 - test_fraction, test_fraction],
            seed=random_state,
        )
        rng = np.random.RandomState(random_state)
        train_indices = rng.permutation(train_indices).tolist()
        test_indices = rng.permutation(test_indices).tolist()
        train_data = df_clean.iloc[train_indices]
        test_data = df_clean.iloc[test_indices]
    else:
        # Preserve the author's original random holdout exactly: shuffle the
        # cleaned table once with random_state=1234, then take the first 90%.
        shuffled = df_clean.sample(frac=1, random_state=random_state).reset_index(drop=True)
        train_count = int((1.0 - test_fraction) * len(shuffled))
        train_data = shuffled.iloc[:train_count]
        test_data = shuffled.iloc[train_count:]

    train_data.to_csv(train_file_path, sep='\t', index=False, header=False)
    test_data.to_csv(test_file_path, sep='\t', index=False, header=False)
    removed_caches = _invalidate_split_caches(data_path, dataname)

    train_scaffolds = {
        bemis_murcko_scaffold(smiles) for smiles in train_data['SMILES']
    }
    test_scaffolds = {
        bemis_murcko_scaffold(smiles) for smiles in test_data['SMILES']
    }
    manifest = dict(expected_manifest)
    manifest.update({
        'sample_count': int(len(df_clean)),
        'train_count': int(len(train_data)),
        'test_count': int(len(test_data)),
        'train_sha256': _sha256_file(train_file_path),
        'test_sha256': _sha256_file(test_file_path),
        'train_scaffold_count': len(train_scaffolds),
        'test_scaffold_count': len(test_scaffolds),
        'scaffold_overlap_count': len(train_scaffolds & test_scaffolds),
    })
    with open(manifest_path, 'w', encoding='utf-8') as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=False)
        handle.write('\n')

    print("Created train={} and test={}; scaffold overlap={}.".format(
        len(train_data), len(test_data), manifest['scaffold_overlap_count']
    ))
    if removed_caches:
        print("Removed {} stale split cache file(s).".format(len(removed_caches)))
    return train_file_path, test_file_path


def load_datasets(args, path, dataname, max_data):
    """
    根据是否使用扩展模型加载不同的数据集
    """
    # 根据是否使用扩展模型加载不同的数据集
    if args.model == 'extended':  # 修复：使用args.model == 'extended'而不是args.extended_model
        # 根据特征配置选择特征
        feature_config = args.feature_config
            
        target_col_name = 'Pred_RTI_Negative_ESI' if 'neg' in dataname else 'Pred_RTI_Positive_ESI'
        # 如果需要标准化额外特征，则获取标准化参数
        if hasattr(args, 'standardize') and args.standardize:
            dataset_train_result = pp.create_extended_dataset(
                f'{dataname}_train_set.txt', 
                os.path.join(path, dataname), 
                dataname, 
                target_col_name, 
                max_data,
                feature_config,
                standardize=True
            )
            # 如果返回的是元组，则包含数据集和标准化参数
            if isinstance(dataset_train_result, tuple) and len(dataset_train_result) == 3:
                dataset_train, additional_features_mean, additional_features_std = dataset_train_result
            else:
                dataset_train = dataset_train_result
                additional_features_mean, additional_features_std = None, None
        else:
            dataset_train = pp.create_extended_dataset(
                f'{dataname}_train_set.txt', 
                os.path.join(path, dataname), 
                dataname, 
                target_col_name, 
                max_data,
                feature_config
            )
            additional_features_mean, additional_features_std = None, None
            
        # 自动推断额外特征维度
        if args.additional_features_dim is None and len(dataset_train) > 0:
            # 从第一个样本推断额外特征维度
            sample = dataset_train[0]
            if len(sample) >= 5:  # 扩展格式
                additional_features_dim = len(sample[4]) if hasattr(sample[4], '__len__') else 1
            else:
                additional_features_dim = 0
        else:
            additional_features_dim = args.additional_features_dim or 0
            
        print(f"Using extended dataset with additional features. Feature dim: {additional_features_dim}")
    else:
        dataset_train = pp.create_dataset(f'{dataname}_train_set.txt', os.path.join(path, dataname), dataname, max_data)
        additional_features_dim = 0
        additional_features_mean, additional_features_std = None, None

    # 使用工具函数替换原来的数据集拆分函数
    dev_split_seed = getattr(args, 'dev_split_seed', None)
    if dev_split_seed is None:
        dev_split_seed = (
            42 if getattr(args, 'split_strategy', 'scaffold') == 'random'
            else getattr(args, 'split_seed', 1234)
        )
    dataset_train, dataset_dev = split_dataset(
        dataset_train,
        0.9,
        split_strategy=getattr(args, 'split_strategy', 'scaffold'),
        random_state=dev_split_seed,
    )
    if getattr(args, 'development_only', False):
        return (dataset_train, dataset_dev, [], additional_features_dim,
                additional_features_mean, additional_features_std)
    
    # 对于测试集，如果在调试模式下，限制测试集大小
    if args.debug_size is not None:
        test_size = max(10, int(args.debug_size * 0.1))  # 至少使用10个样本
        if args.model == 'extended':  # 修复：使用args.model == 'extended'而不是args.extended_model
            target_col_name = 'Pred_RTI_Negative_ESI' if 'neg' in dataname else 'Pred_RTI_Positive_ESI'
            dataset_test = pp.create_extended_dataset(
                f'{dataname}_test_set.txt', 
                os.path.join(path, dataname), 
                dataname,
                target_col_name,
                test_size,
                feature_config if args.model == 'extended' else None,  # 修复：使用args.model == 'extended'而不是args.extended_model
                standardize=False  # 测试集不进行标准化，使用训练集的参数
            )
        else:
            dataset_test = pp.create_dataset(f'{dataname}_test_set.txt', os.path.join(path, dataname), dataname, test_size)
    else:
        if args.model == 'extended':  # 修复：使用args.model == 'extended'而不是args.extended_model
            target_col_name = 'Pred_RTI_Negative_ESI' if 'neg' in dataname else 'Pred_RTI_Positive_ESI'
            dataset_test = pp.create_extended_dataset(
                f'{dataname}_test_set.txt', 
                os.path.join(path, dataname), 
                dataname,
                target_col_name,
                None,  # 不限制测试集大小
                feature_config if args.model == 'extended' else None,  # 修复：使用args.model == 'extended'而不是args.extended_model
                standardize=False  # 测试集不进行标准化，使用训练集的参数
            )
        else:
            dataset_test = pp.create_dataset(f'{dataname}_test_set.txt', os.path.join(path, dataname), dataname)
    
    # 返回额外特征的标准化参数
    return dataset_train, dataset_dev, dataset_test, additional_features_dim, additional_features_mean, additional_features_std


def standardize_datasets_if_needed(args, dataset_train, dataset_dev, dataset_test, mean_val=None, std_val=None, 
                                  additional_features_mean=None, additional_features_std=None):
    """
    标准化数据（如果需要）
    """
    property_mean, property_std = None, None
    if args.standardize:
        print("Standardizing property values...")
        dataset_train, property_mean, property_std = standardize_dataset(dataset_train, mean_val, std_val)
        # 对开发集和测试集应用相同的标准化参数
        dataset_dev, _, _ = standardize_dataset(dataset_dev, property_mean, property_std)
        dataset_test, _, _ = standardize_dataset(dataset_test, property_mean, property_std)
        
        # 如果有额外特征的标准化参数，对开发集和测试集也进行标准化
        if additional_features_mean is not None and additional_features_std is not None:
            print("Standardizing additional features...")
            dataset_dev = standardize_additional_features_in_dataset(dataset_dev, additional_features_mean, additional_features_std)
            dataset_test = standardize_additional_features_in_dataset(dataset_test, additional_features_mean, additional_features_std)
    
    return dataset_train, dataset_dev, dataset_test, property_mean, property_std


def filter_failed_molecules(dataset, preprocessed_data, cache, types, dataset_name, allowed_smiles=None):
    """
    过滤掉无法生成3D坐标的分子
    
    Args:
        dataset: 原始数据集
        preprocessed_data: 预处理后的数据
        cache: 分子缓存对象
        types: 原子类型映射
        dataset_name: 数据集名称（用于日志输出）
        
    Returns:
        filtered_dataset: 过滤后的数据集
    """
    filtered_dataset = []
    failed_smiles = []
    
    for data in dataset:
        smiles = data[0]
        if allowed_smiles is not None and smiles not in allowed_smiles:
            failed_smiles.append(smiles)
            continue
        # 检查分子是否成功预处理
        if smiles in preprocessed_data:
            filtered_dataset.append(data)
        else:
            # 检查是否在失败列表中
            if smiles in cache.failed_smiles:
                failed_smiles.append(smiles)
            else:
                # 尝试重新处理
                result = cache.process_smiles(smiles, types)
                if result is not None and len(result) == 5:
                    x, z, pos, edge_index, edge_attr = result
                    if (x is not None and z is not None and pos is not None and 
                        edge_index is not None and edge_attr is not None):
                        filtered_dataset.append(data)
                    else:
                        failed_smiles.append(smiles)
                else:
                    failed_smiles.append(smiles)
    
    print(f"Filtered {dataset_name} dataset: {len(dataset)} -> {len(filtered_dataset)}, "
          f"removed {len(failed_smiles)} failed molecules")
    if len(failed_smiles) > 0:
        print(f"Failed molecules in {dataset_name}: {failed_smiles[:5]}...")  # 只显示前5个
    
    return filtered_dataset


def load_additional_features(smiles_list, dataname, standardize=False, input_file=None, feature_dataset_type=None):  # INFO
    """
    为VisNetV2模型加载额外特征
    
    Args:
        smiles_list: SMILES字符串列表
        dataname: 数据集名称
        standardize: 是否对每个子维度进行标准化
        
    Returns:
        physchem_features_dict: 物化特征字典
        toxicity_features_dict: 毒性特征字典
        chromato_features_dict: 色谱特征字典
        standardization_info: 标准化信息（均值和标准差），如果standardize为False则为None
    """
    # 确定数据集路径
    if input_file is not None:
        csv_path = input_file
    # 原本的解析方式
    elif 'neg' in dataname or feature_dataset_type and 'neg' in feature_dataset_type:  # 兼容旧数据集
        csv_path = f'data/MMF-3/MMF_GNN_neg.csv'
    else:
        csv_path = f'data/MMF-3/MMF_GNN_pos.csv'
    
    # 读取CSV文件
    df = pd.read_csv(csv_path)
    
    # 创建SMILES到行的映射
    smiles_to_row = {row['SMILES']: row for _, row in df.iterrows()}
    print(f"Loaded {len(smiles_to_row)} rows from {csv_path}")
    
    # 初始化特征字典
    physchem_features_dict = {}
    toxicity_features_dict = {}
    chromato_features_dict = {}
    
    # 定义特征列
    # 物化特征 (4个)
    physchem_columns = [
        'Monoiso_Mass',
        # logKow_EPISuite和Exp_logKow_EPISuite二选一，优先用Exp_logKow_EPISuite
        # alogp_ChemSpider和xlogp_ChemSpider二选一，优先用xlogp_ChemSpider
        'Koc_predicted (L/kg)'
    ]
    
    # 毒性特征 (4个)
    toxicity_columns = [
        'Tetrahymena_pyriformis_toxicity',
        'Daphnia_toxicity',
        'Algae_toxicity',
        'Pimephales_promelas_toxicity'
    ]
    
    # 色谱特征 (2个)
    chromato_columns = [
        'Prob. +ESI',
        'Prob. -ESI'
    ]
    
    # 收集所有特征值用于标准化
    all_physchem_features = []
    all_toxicity_features = []
    all_chromato_features = []
    
    # 处理每个SMILES
    for smiles in smiles_list:
        if smiles in smiles_to_row:
            row = smiles_to_row[smiles]
            
            # 处理物化特征
            physchem_features = []
            
            # 添加Monoiso_Mass
            monoiso_mass = row.get('Monoiso_Mass', 0.0)
            try:
                physchem_features.append(float(monoiso_mass) if pd.notna(monoiso_mass) else 0.0)
            except (ValueError, TypeError):
                physchem_features.append(0.0)
            
            # 处理logKow特征，优先使用Exp_logKow_EPISuite，否则使用logKow_EPISuite
            exp_logkow = row.get('Exp_logKow_EPISuite')
            logkow = row.get('logKow_EPISuite')
            
            if pd.notna(exp_logkow):
                try:
                    physchem_features.append(float(exp_logkow))
                except (ValueError, TypeError):
                    physchem_features.append(0.0)
            elif pd.notna(logkow):
                try:
                    physchem_features.append(float(logkow))
                except (ValueError, TypeError):
                    physchem_features.append(0.0)
            else:
                physchem_features.append(0.0)
                
            # 处理logp特征，优先使用xlogp_ChemSpider，否则使用alogp_ChemSpider
            xlogp = row.get('xlogp_ChemSpider')
            alogp = row.get('alogp_ChemSpider')
            
            if pd.notna(xlogp):
                try:
                    physchem_features.append(float(xlogp))
                except (ValueError, TypeError):
                    physchem_features.append(0.0)
            elif pd.notna(alogp):
                try:
                    physchem_features.append(float(alogp))
                except (ValueError, TypeError):
                    physchem_features.append(0.0)
            else:
                physchem_features.append(0.0)
                
            # 添加Koc_predicted，优先使用koc_predicted，否则使用koc_max_predicted
            koc_predicted = row.get('Koc_predicted (L/kg)', 0.0)
            koc_max_predicted = row.get('Koc_max_predicted (L/kg)', 0.0)
            if pd.notna(koc_predicted):
                try:
                    physchem_features.append(float(koc_predicted))
                except (ValueError, TypeError):
                    physchem_features.append(0.0)
            elif pd.notna(koc_max_predicted):
                try:
                    physchem_features.append(float(koc_max_predicted))
                except (ValueError, TypeError):
                    physchem_features.append(0.0)
            else:
                physchem_features.append(0.0)
            physchem_features = np.array(physchem_features, dtype=np.float32)
            physchem_features_dict[smiles] = physchem_features
            
            # 处理毒性特征
            toxicity_features = []
            for col in toxicity_columns:
                val = row.get(col, 0.0)
                try:
                    toxicity_features.append(float(val) if pd.notna(val) else 0.0)
                except (ValueError, TypeError):
                    toxicity_features.append(0.0)
            toxicity_features = np.array(toxicity_features, dtype=np.float32)
            toxicity_features_dict[smiles] = toxicity_features
            
            # 处理色谱特征
            chromato_features = []
            for col in chromato_columns:
                val = row.get(col, 0.0)
                try:
                    chromato_features.append(float(val) if pd.notna(val) else 0.0)
                except (ValueError, TypeError):
                    chromato_features.append(0.0)
            chromato_features = np.array(chromato_features, dtype=np.float32)
            chromato_features_dict[smiles] = chromato_features
            
            # 收集特征值用于标准化
            if standardize:
                all_physchem_features.append(physchem_features)
                all_toxicity_features.append(toxicity_features)
                all_chromato_features.append(chromato_features)
        else:
            # 如果找不到SMILES，直接报错而不是填充零值
            raise ValueError(f"Missing features for molecule {smiles}. Cannot find physchem, toxicity, or chromato features.")
            
            # 收集特征值用于标准化
            if standardize:
                all_physchem_features.append(np.zeros(4, dtype=np.float32))  # 修正为4个特征
                all_toxicity_features.append(np.zeros(4, dtype=np.float32))
                all_chromato_features.append(np.zeros(2, dtype=np.float32))
    
    # 标准化处理
    standardization_info = None
    if standardize and len(all_physchem_features) > 0:
        # 计算均值和标准差
        physchem_mean = np.mean(all_physchem_features, axis=0)
        physchem_std = np.std(all_physchem_features, axis=0)
        # 避免除零错误
        physchem_std = np.where(physchem_std == 0, 1.0, physchem_std)
        
        toxicity_mean = np.mean(all_toxicity_features, axis=0)
        toxicity_std = np.std(all_toxicity_features, axis=0)
        # 避免除零错误
        toxicity_std = np.where(toxicity_std == 0, 1.0, toxicity_std)
        
        chromato_mean = np.mean(all_chromato_features, axis=0)
        chromato_std = np.std(all_chromato_features, axis=0)
        # 避免除零错误
        chromato_std = np.where(chromato_std == 0, 1.0, chromato_std)
        
        # 应用标准化
        for smiles in physchem_features_dict:
            # 确保特征是numpy数组
            if not isinstance(physchem_features_dict[smiles], np.ndarray):
                physchem_features_dict[smiles] = np.array(physchem_features_dict[smiles], dtype=np.float32)
            if not isinstance(toxicity_features_dict[smiles], np.ndarray):
                toxicity_features_dict[smiles] = np.array(toxicity_features_dict[smiles], dtype=np.float32)
            if not isinstance(chromato_features_dict[smiles], np.ndarray):
                chromato_features_dict[smiles] = np.array(chromato_features_dict[smiles], dtype=np.float32)
                
            # 应用标准化并确保结果为float32类型
            physchem_features_dict[smiles] = ((physchem_features_dict[smiles] - physchem_mean) / physchem_std).astype(np.float32)
            toxicity_features_dict[smiles] = ((toxicity_features_dict[smiles] - toxicity_mean) / toxicity_std).astype(np.float32)
            chromato_features_dict[smiles] = ((chromato_features_dict[smiles] - chromato_mean) / chromato_std).astype(np.float32)
        
        # 保存标准化信息
        standardization_info = {
            'physchem': {'mean': physchem_mean.tolist(), 'std': physchem_std.tolist()},
            'toxicity': {'mean': toxicity_mean.tolist(), 'std': toxicity_std.tolist()},
            'chromato': {'mean': chromato_mean.tolist(), 'std': chromato_std.tolist()}
        }
    
    return physchem_features_dict, toxicity_features_dict, chromato_features_dict, standardization_info


def resolve_feature_source_file(args, dataname, feature_dataset_type=None):
    """Return the mode-level feature table used by fixed and CV datasets.

    Cross-validation folds are named ``fold_1`` ... ``fold_5`` but their
    auxiliary features still come from the corresponding mode-level CSV.
    """
    source_name = dataname
    if feature_dataset_type in {'pos', 'neg'}:
        source_name = 'MMF_GNN_{}'.format(feature_dataset_type)
    return os.path.join(
        getattr(args, 'data_root', './data/MMF-3/'),
        '{}.csv'.format(source_name),
    )


def preprocess_visnet_data(args, dataset_train, dataset_dev, dataset_test, dataname, _visnet_train, _visnet_test, feature_dataset_type=None):
    """
    如果使用VisNet模型，预先处理所有数据
    """
    # 使用从atom_types.py导入的原子类型映射
    types = ATOM_TYPES
    frozen_partitions = getattr(args, 'effective_partition_smiles', {})
    allowed_train = frozen_partitions.get('train')
    allowed_dev = frozen_partitions.get('development')
    
    # 预处理训练数据
    train_preprocessed_data = {}
    train_smiles_list = [data[0] for data in dataset_train]
    print("Preprocessing training data...")
    train_cache = MoleculeCache(_visnet_train)

    # 原有的串行处理方式
    successful_train_count = 0
    failed_train_count = 0
    skipped_train_count = 0
    
    # 预先统计缓存命中情况
    cached_hits = 0
    known_failures = 0
    for smiles in train_smiles_list:
        if smiles in train_cache.failed_smiles:
            known_failures += 1
        elif train_cache.get(smiles, types) is not None:
            cached_hits += 1
    
    print(f"Training data cache stats - Total: {len(train_smiles_list)}, "
          f"Cached hits: {cached_hits}, Known failures: {known_failures}")
    
    # 使用tqdm显示处理进度
    pbar = tqdm(train_smiles_list, desc="Preprocessing train data")
    for smiles in pbar:
        if allowed_train is not None and smiles not in allowed_train:
            failed_train_count += 1
            skipped_train_count += 1
            continue
        # 首先检查是否之前已经标记为失败
        if smiles in train_cache.failed_smiles:
            failed_train_count += 1
            skipped_train_count += 1
            continue
            
        # 检查是否已经在缓存中（包括成功和失败的情况）
        cached_result = train_cache.get(smiles, types)
        if cached_result is not None:
            # 成功缓存命中
            x, z, pos, edge_index, edge_attr = cached_result
            if (x is not None and z is not None and pos is not None and 
                edge_index is not None and edge_attr is not None):
                train_preprocessed_data[smiles] = {
                    'x': x,
                    'z': z,
                    'pos': pos,
                    'edge_index': edge_index,
                    'edge_attr': edge_attr
                }
                successful_train_count += 1
            else:
                failed_train_count += 1
            continue
        
        # 缓存未命中，计算分子图结构
        result = train_cache.process_smiles(smiles, types)
        if result is not None and len(result) == 5:
            x, z, pos, edge_index, edge_attr = result
            # 确保所有必要数据都存在且不为None
            if (x is not None and z is not None and pos is not None and 
                edge_index is not None and edge_attr is not None):
                train_preprocessed_data[smiles] = {
                    'x': x,
                    'z': z,
                    'pos': pos,
                    'edge_index': edge_index,
                    'edge_attr': edge_attr
                }
                successful_train_count += 1
            else:
                failed_train_count += 1
        else:
            failed_train_count += 1
        
        # 更新进度条描述信息
        pbar.set_postfix({
            'Success': successful_train_count,
            'Failed': failed_train_count,
            'Skipped': skipped_train_count
        })
    
    # 过滤掉训练集中无法生成3D坐标的分子
    dataset_train_filtered = filter_failed_molecules(
        dataset_train, train_preprocessed_data, train_cache, types, "training", allowed_smiles=allowed_train)
    
    feature_source_file = resolve_feature_source_file(
        args, dataname, feature_dataset_type=feature_dataset_type
    )

    # 为VisNetV2加载额外特征
    train_physchem_features, train_toxicity_features, train_chromato_features, standardization_info = load_additional_features(
        train_smiles_list, dataname,
        standardize=getattr(args, 'standardize_features', False),
        input_file=feature_source_file,
        feature_dataset_type=feature_dataset_type)
    
    # 预处理开发数据
    dev_preprocessed_data = {}
    dev_smiles_list = [data[0] for data in dataset_dev]
    print("Preprocessing development data...")
    
    # 原有的串行处理方式
    successful_dev_count = 0
    failed_dev_count = 0
    skipped_dev_count = 0
    
    # 预先统计缓存命中情况
    cached_hits = 0
    known_failures = 0
    for smiles in dev_smiles_list:
        if smiles in train_cache.failed_smiles:
            known_failures += 1
        elif train_cache.get(smiles, types) is not None:
            cached_hits += 1
    
    print(f"Development data cache stats - Total: {len(dev_smiles_list)}, "
          f"Cached hits: {cached_hits}, Known failures: {known_failures}")
    
    # 使用tqdm显示处理进度
    pbar = tqdm(dev_smiles_list, desc="Preprocessing dev data")
    for smiles in pbar:
        if allowed_dev is not None and smiles not in allowed_dev:
            failed_dev_count += 1
            skipped_dev_count += 1
            continue
        # 首先检查是否之前已经标记为失败
        if smiles in train_cache.failed_smiles:
            failed_dev_count += 1
            skipped_dev_count += 1
            continue
            
        # 检查是否已经在缓存中（包括成功和失败的情况）
        cached_result = train_cache.get(smiles, types)
        if cached_result is not None:
            # 成功缓存命中
            x, z, pos, edge_index, edge_attr = cached_result
            if (x is not None and z is not None and pos is not None and 
                edge_index is not None and edge_attr is not None):
                dev_preprocessed_data[smiles] = {
                    'x': x,
                    'z': z,
                    'pos': pos,
                    'edge_index': edge_index,
                    'edge_attr': edge_attr
                }
                successful_dev_count += 1
            else:
                failed_dev_count += 1
            continue
        
        # 缓存未命中，计算分子图结构
        result = train_cache.process_smiles(smiles, types)
        if result is not None and len(result) == 5:
            x, z, pos, edge_index, edge_attr = result
            # 确保所有必要数据都存在且不为None
            if (x is not None and z is not None and pos is not None and 
                edge_index is not None and edge_attr is not None):
                dev_preprocessed_data[smiles] = {
                    'x': x,
                    'z': z,
                    'pos': pos,
                    'edge_index': edge_index,
                    'edge_attr': edge_attr
                }
                successful_dev_count += 1
            else:
                failed_dev_count += 1
        else:
            failed_dev_count += 1
        
        # 更新进度条描述信息
        pbar.set_postfix({
            'Success': successful_dev_count,
            'Failed': failed_dev_count,
            'Skipped': skipped_dev_count
        })
    
    # 过滤掉开发集中无法生成3D坐标的分子
    dataset_dev_filtered = filter_failed_molecules(
        dataset_dev, dev_preprocessed_data, train_cache, types, "development", allowed_smiles=allowed_dev)
    
    # 为VisNetV2加载额外特征
    dev_physchem_features, dev_toxicity_features, dev_chromato_features, _ = load_additional_features(
        dev_smiles_list, dataname, standardize=False,
        input_file=feature_source_file,
        feature_dataset_type=feature_dataset_type)  # 开发集不进行标准化，使用训练集的标准化参数
    
    # 如果有训练集标准化信息，对开发集应用相同的标准化参数
    if standardization_info is not None:
        # 对开发集特征应用训练集的标准化参数
        physchem_mean = np.array(standardization_info['physchem']['mean'])
        physchem_std = np.array(standardization_info['physchem']['std'])
        for smiles in dev_physchem_features:
            dev_physchem_features[smiles] = ((dev_physchem_features[smiles] - physchem_mean) / physchem_std).astype(np.float32)
            
        toxicity_mean = np.array(standardization_info['toxicity']['mean'])
        toxicity_std = np.array(standardization_info['toxicity']['std'])
        for smiles in dev_toxicity_features:
            dev_toxicity_features[smiles] = ((dev_toxicity_features[smiles] - toxicity_mean) / toxicity_std).astype(np.float32)
            
        chromato_mean = np.array(standardization_info['chromato']['mean'])
        chromato_std = np.array(standardization_info['chromato']['std'])
        for smiles in dev_chromato_features:
            dev_chromato_features[smiles] = ((dev_chromato_features[smiles] - chromato_mean) / chromato_std).astype(np.float32)
    
    # 预处理测试数据
    test_cache = MoleculeCache(_visnet_test)
    test_preprocessed_data = {}
    test_smiles_list = [data[0] for data in dataset_test]
    print("Preprocessing test data...")
    
    # 原有的串行处理方式
    successful_test_count = 0
    failed_test_count = 0
    skipped_test_count = 0
    
    # 预先统计缓存命中情况
    cached_hits = 0
    known_failures = 0
    for smiles in test_smiles_list:
        if smiles in test_cache.failed_smiles:
            known_failures += 1
        elif test_cache.get(smiles, types) is not None:
            cached_hits += 1
    
    print(f"Test data cache stats - Total: {len(test_smiles_list)}, "
          f"Cached hits: {cached_hits}, Known failures: {known_failures}")
    
    # 使用tqdm显示处理进度
    pbar = tqdm(test_smiles_list, desc="Preprocessing test data")
    for smiles in pbar:
        # 首先检查是否之前已经标记为失败
        if smiles in test_cache.failed_smiles:
            failed_test_count += 1
            skipped_test_count += 1
            continue
            
        # 检查是否已经在缓存中（包括成功和失败的情况）
        cached_result = test_cache.get(smiles, types)
        if cached_result is not None:
            # 成功缓存命中
            x, z, pos, edge_index, edge_attr = cached_result
            if (x is not None and z is not None and pos is not None and 
                edge_index is not None and edge_attr is not None):
                test_preprocessed_data[smiles] = {
                    'x': x,
                    'z': z,
                    'pos': pos,
                    'edge_index': edge_index,
                    'edge_attr': edge_attr
                }
                successful_test_count += 1
            else:
                failed_test_count += 1
            continue
        
        # 缓存未命中，计算分子图结构
        result = test_cache.process_smiles(smiles, types)
        if result is not None and len(result) == 5:
            x, z, pos, edge_index, edge_attr = result
            # 确保所有必要数据都存在且不为None
            if (x is not None and z is not None and pos is not None and 
                edge_index is not None and edge_attr is not None):
                test_preprocessed_data[smiles] = {
                    'x': x,
                    'z': z,
                    'pos': pos,
                    'edge_index': edge_index,
                    'edge_attr': edge_attr
                }
                successful_test_count += 1
            else:
                failed_test_count += 1
        else:
            failed_test_count += 1
        
        # 更新进度条描述信息
        pbar.set_postfix({
            'Success': successful_test_count,
            'Failed': failed_test_count,
            'Skipped': skipped_test_count
        })
    
    # 过滤掉测试集中无法生成3D坐标的分子
    dataset_test_filtered = filter_failed_molecules(
        dataset_test, test_preprocessed_data, test_cache, types, "test")
    
    # 为VisNetV2加载额外特征
    test_physchem_features, test_toxicity_features, test_chromato_features, _ = load_additional_features(
        test_smiles_list, dataname, standardize=False,
        input_file=feature_source_file,
        feature_dataset_type=feature_dataset_type)  # 测试集不进行标准化，使用训练集的标准化参数
    
    # 如果有训练集标准化信息，对测试集应用相同的标准化参数
    if standardization_info is not None:
        # 对测试集特征应用训练集的标准化参数
        physchem_mean = np.array(standardization_info['physchem']['mean'])
        physchem_std = np.array(standardization_info['physchem']['std'])
        for smiles in test_physchem_features:
            test_physchem_features[smiles] = ((test_physchem_features[smiles] - physchem_mean) / physchem_std).astype(np.float32)
            
        toxicity_mean = np.array(standardization_info['toxicity']['mean'])
        toxicity_std = np.array(standardization_info['toxicity']['std'])
        for smiles in test_toxicity_features:
            test_toxicity_features[smiles] = ((test_toxicity_features[smiles] - toxicity_mean) / toxicity_std).astype(np.float32)
            
        chromato_mean = np.array(standardization_info['chromato']['mean'])
        chromato_std = np.array(standardization_info['chromato']['std'])
        for smiles in test_chromato_features:
            test_chromato_features[smiles] = ((test_chromato_features[smiles] - chromato_mean) / chromato_std).astype(np.float32)
    
    # 将额外特征添加到预处理数据中
    for smiles in train_preprocessed_data:
        train_preprocessed_data[smiles]['physchem_features'] = train_physchem_features.get(smiles, np.zeros(4, dtype=np.float32))
        train_preprocessed_data[smiles]['toxicity_features'] = train_toxicity_features.get(smiles, np.zeros(4, dtype=np.float32))
        train_preprocessed_data[smiles]['chromato_features'] = train_chromato_features.get(smiles, np.zeros(2, dtype=np.float32))
        
        # 移除重复的特征检查逻辑，将在后面统一处理
    
    for smiles in dev_preprocessed_data:
        dev_preprocessed_data[smiles]['physchem_features'] = dev_physchem_features.get(smiles, np.zeros(4, dtype=np.float32))
        dev_preprocessed_data[smiles]['toxicity_features'] = dev_toxicity_features.get(smiles, np.zeros(4, dtype=np.float32))
        dev_preprocessed_data[smiles]['chromato_features'] = dev_chromato_features.get(smiles, np.zeros(2, dtype=np.float32))
        
        # 移除重复的特征检查逻辑，将在后面统一处理
    
    for smiles in test_preprocessed_data:
        test_preprocessed_data[smiles]['physchem_features'] = test_physchem_features.get(smiles, np.zeros(4, dtype=np.float32))
        test_preprocessed_data[smiles]['toxicity_features'] = test_toxicity_features.get(smiles, np.zeros(4, dtype=np.float32))
        test_preprocessed_data[smiles]['chromato_features'] = test_chromato_features.get(smiles, np.zeros(2, dtype=np.float32))
    
    # 根据特征级别设置不需要的特征为None
    if args.visnet_v2_feature_level == 'graph':
        # 只使用图特征，将其他特征设置为None
        for data_dict in [train_preprocessed_data, dev_preprocessed_data, test_preprocessed_data]:
            for smiles in data_dict:
                data_dict[smiles]['physchem_features'] = None
                data_dict[smiles]['toxicity_features'] = None
                data_dict[smiles]['chromato_features'] = None
    elif args.visnet_v2_feature_level == 'graph_physchem':
        # 使用图特征和物化特征，将其他特征设置为None
        for data_dict in [train_preprocessed_data, dev_preprocessed_data, test_preprocessed_data]:
            for smiles in data_dict:
                data_dict[smiles]['toxicity_features'] = None
                data_dict[smiles]['chromato_features'] = None
    elif args.visnet_v2_feature_level == 'graph_physchem_toxicity':
        # 使用图特征、物化特征和毒性特征，将色谱特征设置为None
        for data_dict in [train_preprocessed_data, dev_preprocessed_data, test_preprocessed_data]:
            for smiles in data_dict:
                data_dict[smiles]['chromato_features'] = None
                
    # 特征的有效性检查将在模型中统一进行，此处不再重复检查
    
    print(f"Preprocessed data loaded: "
          f"Train: {len(train_preprocessed_data)} (success: {successful_train_count}, failed: {failed_train_count}, skipped: {skipped_train_count}), "
          f"Dev: {len(dev_preprocessed_data)} (success: {successful_dev_count}, failed: {failed_dev_count}, skipped: {skipped_dev_count}), "
          f"Test: {len(test_preprocessed_data)} (success: {successful_test_count}, failed: {failed_test_count}, skipped: {skipped_test_count})")
    
    # 如果有标准化信息，也一并返回
    if standardization_info is not None:
        return train_preprocessed_data, dev_preprocessed_data, test_preprocessed_data, dataset_train_filtered, dataset_dev_filtered, dataset_test_filtered, standardization_info
    
    return train_preprocessed_data, dev_preprocessed_data, test_preprocessed_data, dataset_train_filtered, dataset_dev_filtered, dataset_test_filtered
