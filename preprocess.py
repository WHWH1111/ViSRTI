#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据预处理模块

用于将SMILES格式的分子数据转换为模型可处理的格式
"""

import os
from collections import defaultdict
import numpy as np
from rdkit import Chem
import torch
import pickle
import pandas as pd
from utils.feature_utils import DEFAULT_FEATURE_CONFIG, ALL_FEATURES_CONFIG, get_feature_config

# 物化特征名称到索引的映射，用于特征掩码设置
PHYSICOCHEMICAL_FEATURE_MAP = {
    'Monoiso_Mass': 0,
    'average_mass': 1, 
    'M+H+': 2,
    'M-H-': 3
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


def load_existing_dicts(dir_dataset, dataname):
    """加载已存在的字典文件，确保字典在不同运行中保持一致"""
    atom_dict = defaultdict(lambda: len(atom_dict))
    bond_dict = defaultdict(lambda: len(bond_dict))
    fingerprint_dict = defaultdict(lambda: len(fingerprint_dict))
    edge_dict = defaultdict(lambda: len(edge_dict))
    
    dict_files = {
        'atom': (atom_dict, f'{dir_dataset}/{dataname}-atom_dict.pickle'),
        'bond': (bond_dict, f'{dir_dataset}/{dataname}-bond_dict.pickle'),
        'fingerprint': (fingerprint_dict, f'{dir_dataset}/{dataname}-fingerprint_dict.pickle'),
        'edge': (edge_dict, f'{dir_dataset}/{dataname}-edge_dict.pickle')
    }
    
    for dict_name, (dict_obj, file_path) in dict_files.items():
        if os.path.exists(file_path):
            try:
                with open(file_path, 'rb') as f:
                    existing_dict = pickle.load(f)
                    # 更新默认字典
                    dict_obj.update(existing_dict)
                print(f"Loaded existing {dict_name} dictionary from {file_path}")
            except Exception as e:
                print(f"Warning: Could not load {dict_name} dictionary from {file_path}: {e}")
        else:
            print(f"Note: {dict_name} dictionary file {file_path} does not exist yet.")
    
    return atom_dict, bond_dict, fingerprint_dict, edge_dict

# 初始化字典，尝试加载已存在的字典
atom_dict = defaultdict(lambda: len(atom_dict))
bond_dict = defaultdict(lambda: len(bond_dict))
fingerprint_dict = defaultdict(lambda: len(fingerprint_dict))
edge_dict = defaultdict(lambda: len(edge_dict))
radius = 1

def dump_dictionary(dictionary, filename):
    with open(filename, 'wb') as f:
        pickle.dump(dict(dictionary), f)
        
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
	
def create_atoms(mol, atom_dict):
    """Transform the atom types in a molecule (e.g., H, C, and O)
    into the indices (e.g., H=0, C=1, and O=2).
    Note that each atom index considers the aromaticity.
    """
    atoms = [a.GetSymbol() for a in mol.GetAtoms()]
    for a in mol.GetAromaticAtoms():
        i = a.GetIdx()
        atoms[i] = (atoms[i], 'aromatic')
    atoms = [atom_dict[a] for a in atoms]
    return np.array(atoms)


def create_ijbonddict(mol, bond_dict):
    """Create a dictionary, in which each key is a node ID
    and each value is the tuples of its neighboring node
    and chemical bond (e.g., single and double) IDs.
    """
    i_jbond_dict = defaultdict(lambda: [])
    for b in mol.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        bond = bond_dict[str(b.GetBondType())]
        i_jbond_dict[i].append((j, bond))
        i_jbond_dict[j].append((i, bond))
    return i_jbond_dict


def extract_fingerprints(radius, atoms, i_jbond_dict,
                         fingerprint_dict, edge_dict):
    """Extract the fingerprints from a molecular graph
    based on Weisfeiler-Lehman algorithm.
    """
    if (len(atoms) == 1) or (radius == 0):
        nodes = [fingerprint_dict[a] for a in atoms]

    else:
        nodes = atoms
        i_jedge_dict = i_jbond_dict

        for _ in range(radius):

            """Update each node ID considering its neighboring nodes and edges.
            The updated node IDs are the fingerprint IDs.
            """
            nodes_ = []
            for i, j_edge in i_jedge_dict.items():
                neighbors = [(nodes[j], edge) for j, edge in j_edge]
                fingerprint = (nodes[i], tuple(sorted(neighbors)))
                nodes_.append(fingerprint_dict[fingerprint])

            """Also update each edge ID considering
            its two nodes on both sides.
            """
            i_jedge_dict_ = defaultdict(lambda: [])
            for i, j_edge in i_jedge_dict.items():
                for j, edge in j_edge:
                    both_side = tuple(sorted((nodes[i], nodes[j])))
                    edge = edge_dict[(both_side, edge)]
                    i_jedge_dict_[i].append((j, edge))

            nodes = nodes_
            i_jedge_dict = i_jedge_dict_

    return np.array(nodes)


def extract_additional_features(df_row, feature_config=None):
    """从数据行中提取额外特征
    包括物理化学特征、毒性特征和色谱质谱特征
    """
    features = []
    
    # 如果没有提供特征配置，则使用默认的基本特征配置
    if feature_config is None:
        feature_config = DEFAULT_FEATURE_CONFIG['basic']
    elif isinstance(feature_config, str):
        # 如果是字符串，则使用预定义的配置组
        # 支持逗号分隔的多个配置
        if ',' in feature_config:
            config_names = [name.strip() for name in feature_config.split(',')]
            feature_config_list = []
            for name in config_names:
                if name in DEFAULT_FEATURE_CONFIG:
                    feature_config_list.extend(DEFAULT_FEATURE_CONFIG[name])
            # 去重但保持顺序
            seen = set()
            unique_features = []
            for item in feature_config_list:
                if item not in seen:
                    seen.add(item)
                    unique_features.append(item)
            feature_config = unique_features
        else:
            feature_config = DEFAULT_FEATURE_CONFIG.get(feature_config, DEFAULT_FEATURE_CONFIG['basic'])
    
    # 提取指定的特征，缺失值用0填充
    for feature_name in feature_config:
        if feature_name in df_row and pd.notna(df_row[feature_name]):
            try:
                # 尝试转换为浮点数
                val = float(df_row[feature_name])
                features.append(val)
            except (ValueError, TypeError):
                # 如果转换失败，用0填充
                features.append(0.0)
        else:
            # 如果列不存在或为空，用0填充
            features.append(0.0)
    
    return np.array(features, dtype=np.float32)


def standardize_additional_features(features_list, feature_config=None, mean_vals=None, std_vals=None):
    """
    对额外特征进行标准化处理
    
    Args:
        features_list: 特征列表
        feature_config: 特征配置
        mean_vals: 均值列表（如果提供，则使用这些值进行标准化）
        std_vals: 标准差列表（如果提供，则使用这些值进行标准化）
        
    Returns:
        standardized_features_list: 标准化后的特征列表
        mean_vals: 均值列表
        std_vals: 标准差列表
    """
    if len(features_list) == 0:
        return features_list, mean_vals, std_vals
    
    # 转换为numpy数组以便处理
    features_array = np.array(features_list)
    
    # 如果没有提供均值和标准差，则计算
    if mean_vals is None or std_vals is None:
        mean_vals = np.mean(features_array, axis=0)
        std_vals = np.std(features_array, axis=0)
        # 避免除零错误
        std_vals = np.where(std_vals == 0, 1.0, std_vals)
    
    # 标准化特征
    standardized_features_array = (features_array - mean_vals) / std_vals
    
    # 转换回列表格式
    standardized_features_list = [np.array(f, dtype=np.float32) for f in standardized_features_array]
    
    return standardized_features_list, mean_vals, std_vals


def create_extended_dataset(filename, path, dataname, target_column='Pred_RTI_Negative_ESI', 
                           max_data=None, feature_config=None, standardize=False):
    """创建包含额外特征的扩展数据集"""
    dir_dataset = path
    raw_data_path = path + '.csv'
    
    # 在处理数据前，加载已存在的字典以确保一致性
    global atom_dict, bond_dict, fingerprint_dict, edge_dict
    atom_dict, bond_dict, fingerprint_dict, edge_dict = load_existing_dicts(dir_dataset, dataname)
    
    # 构造缓存文件名，包含特征配置信息
    feature_suffix = "basic" if feature_config is None else (
        feature_config if isinstance(feature_config, str) else "custom"
    )
    # 添加标准化标识到缓存文件名
    standardize_suffix = "-standardized" if standardize else ""
    sample_suffix = f'-n{max_data}' if max_data is not None else ''
    cache_filename = dir_dataset + '/' + dataname + '-' + filename.replace('.txt', '') + f'-extended-{feature_suffix}{standardize_suffix}{sample_suffix}-cache.pkl'
    
    # 如果缓存文件存在，尝试加载缓存
    if os.path.exists(cache_filename):
        print(f"Loading cached extended dataset from {cache_filename}")
        with open(cache_filename, 'rb') as f:
            cached_data = pickle.load(f)
            # 检查缓存数据的数量是否符合要求
            if max_data is None:
                print("Cache hit! Loading extended dataset from cache.")
                return cached_data
            elif len(cached_data) >= max_data:
                # 如果缓存数据量大于等于所需数据量，截取前max_data个样本
                print(f"Cache hit! Loading first {max_data} samples from cache.")
                return cached_data[:max_data]
            else:
                print(f"Cache size ({len(cached_data)}) is less than required size ({max_data}), regenerating...")
    
    # 读取原始CSV数据
    print(f"Loading raw data from {raw_data_path}")
    df = pd.read_csv(raw_data_path)
    
    # 读取文本文件中的SMILES和目标值索引
    with open(os.path.join(dir_dataset, filename), 'r') as f:
        data_original = f.read().strip().split('\n')
        
    """Exclude the data contains '.' in its smiles."""
    data_original = [data for data in data_original if '.' not in data.split()[0]]
    
    # 如果指定了最大数据量，则只处理这部分数据
    if max_data is not None:
        data_original = data_original[:max_data]
        print(f"Processing only first {max_data} samples for debugging")
    
    dataset = []
    skipped_count = 0
    additional_features_list = []  # 用于收集额外特征以进行标准化
    
    for data in data_original:
        try:
            # 处理可能包含多个空格分隔的数据行
            parts = data.strip().split()
            if len(parts) < 2:
                print(f"Skipping invalid data line (insufficient parts): {data}")
                skipped_count += 1
                continue
                
            smiles = parts[0]
            property = parts[1]
            
            # 在DataFrame中查找对应的行
            df_rows = df[df['SMILES'] == smiles]
            if len(df_rows) == 0:
                print(f"Skipping SMILES not found in CSV: {smiles}")
                skipped_count += 1
                continue
                
            df_row = df_rows.iloc[0]  # 取第一行
            
            """Create each data with the above defined functions."""
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                print(f"Skipping invalid SMILES: {smiles}")
                skipped_count += 1
                continue
            mol = Chem.AddHs(mol)
            atoms = create_atoms(mol, atom_dict)
            molecular_size = len(atoms)
            i_jbond_dict = create_ijbonddict(mol, bond_dict)
            fingerprints = extract_fingerprints(radius, atoms, i_jbond_dict,
                                                    fingerprint_dict, edge_dict)
            adjacency = np.float32((Chem.GetAdjacencyMatrix(mol)))
            
            # 检查维度一致性
            if len(fingerprints) != molecular_size or adjacency.shape[0] != molecular_size:
                print(f"Dimension mismatch for SMILES: {smiles}")
                print(f"  Fingerprints length: {len(fingerprints)}")
                print(f"  Adjacency matrix shape: {adjacency.shape}")
                print(f"  Molecular size: {molecular_size}")
                skipped_count += 1
                continue
            
            # 提取额外特征
            additional_features = extract_additional_features(df_row, feature_config)
            additional_features_list.append(additional_features)
            
            # 仅存储numpy数组，不预先转换为tensor并移动到GPU，减少显存占用
            dataset.append((smiles, fingerprints, adjacency, molecular_size, additional_features, float(property)))
        except Exception as e:
            print(f"Error processing data line '{data}': {e}")
            skipped_count += 1
            continue
    
    # 如果需要标准化额外特征
    if standardize and len(additional_features_list) > 0:
        print("Standardizing additional features...")
        standardized_features_list, mean_vals, std_vals = standardize_additional_features(additional_features_list, feature_config)
        
        # 更新数据集中的额外特征
        for i, (data, standardized_features) in enumerate(zip(dataset, standardized_features_list)):
            smiles, fingerprints, adjacency, molecular_size, _, property_value = data
            dataset[i] = (smiles, fingerprints, adjacency, molecular_size, standardized_features, property_value)
        
        # 返回标准化参数
        return dataset, mean_vals, std_vals
    
    if skipped_count > 0:
        print(f"Skipped {skipped_count} samples due to errors")
        
    # 保存处理后的数据到缓存文件
    print(f"Saving extended dataset to cache: {cache_filename}")
    with open(cache_filename, 'wb') as f:
        pickle.dump(dataset, f)
    
    dir_dataset=path
    dump_dictionary(fingerprint_dict, dir_dataset+'/'+dataname+ '-fingerprint_dict.pickle')
    dump_dictionary(atom_dict, dir_dataset+'/'+dataname+ '-atom_dict.pickle')
    dump_dictionary(bond_dict, dir_dataset+'/'+dataname+ '-bond_dict.pickle')
    dump_dictionary(edge_dict, dir_dataset+'/'+dataname+ '-edge_dict.pickle')
    return dataset


def create_dataset(filename, path, dataname, max_data=None):
    dir_dataset = path
    
    # 在处理数据前，加载已存在的字典以确保一致性
    global atom_dict, bond_dict, fingerprint_dict, edge_dict
    atom_dict, bond_dict, fingerprint_dict, edge_dict = load_existing_dicts(dir_dataset, dataname)
    
    # 构造缓存文件名
    sample_suffix = f'-n{max_data}' if max_data is not None else ''
    cache_filename = dir_dataset + '/' + dataname + '-' + filename.replace('.txt', '') + sample_suffix + '-cache.pkl'
    
    # 如果缓存文件存在，尝试加载缓存
    if os.path.exists(cache_filename):
        print(f"Loading cached dataset from {cache_filename}")
        with open(cache_filename, 'rb') as f:
            cached_data = pickle.load(f)
            # 检查缓存数据的数量是否符合要求
            if max_data is None:
                print("Cache hit! Loading dataset from cache.")
                return cached_data
            elif len(cached_data) >= max_data:
                # 如果缓存数据量大于等于所需数据量，截取前max_data个样本
                print(f"Cache hit! Loading first {max_data} samples from cache.")
                return cached_data[:max_data]
            else:
                print(f"Cache size ({len(cached_data)}) is less than required size ({max_data}), regenerating...")
    
    """Load a dataset."""
    with open(os.path.join(dir_dataset, filename), 'r') as f:
        data_original = f.read().strip().split('\n')

    """Exclude the data contains '.' in its smiles."""
    data_original = [data for data in data_original if '.' not in data.split()[0]]
    
    # 如果指定了最大数据量，则只处理这部分数据
    if max_data is not None:
        data_original = data_original[:max_data]
        print(f"Processing only first {max_data} samples for debugging")
    
    dataset = []
    for data in data_original:
        try:
            # 处理可能包含多个空格分隔的数据行
            parts = data.strip().split()
            if len(parts) < 2:
                print(f"Warning: skipping invalid data line (insufficient parts): {data}")
                continue
                
            smiles = parts[0]
            property = parts[1]
            
            # 检查property是否为有效的数字
            try:
                float(property)
            except ValueError:
                print(f"Warning: skipping invalid data line (invalid property value): {data}, property: {property}")
                continue

            """Create each data with the above defined functions."""
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                print(f"Skipping invalid SMILES: {smiles}")
                continue
            mol = Chem.AddHs(mol)
            atoms = create_atoms(mol, atom_dict)
            molecular_size = len(atoms)
            i_jbond_dict = create_ijbonddict(mol, bond_dict)
            fingerprints = extract_fingerprints(radius, atoms, i_jbond_dict,
                                                    fingerprint_dict, edge_dict)
            adjacency = np.float32((Chem.GetAdjacencyMatrix(mol)))
            
            # 检查维度一致性
            if len(fingerprints) != molecular_size or adjacency.shape[0] != molecular_size:
                print(f"Warning: dimension mismatch for SMILES: {smiles}")
                print(f"  Fingerprints length: {len(fingerprints)}")
                print(f"  Adjacency matrix shape: {adjacency.shape}")
                print(f"  Molecular size: {molecular_size}")
                continue
            
            # 仅存储numpy数组，不预先转换为tensor并移动到GPU，减少显存占用
            dataset.append((smiles, fingerprints, adjacency, molecular_size, float(property)))
        except Exception as e:
            print(f"Error processing data line '{data}': {e}")
            continue
        
    # 保存处理后的数据到缓存文件
    print(f"Saving dataset to cache: {cache_filename}")
    with open(cache_filename, 'wb') as f:
        pickle.dump(dataset, f)
    
    dir_dataset=path
    dump_dictionary(fingerprint_dict, dir_dataset+'/'+dataname+ '-fingerprint_dict.pickle')
    dump_dictionary(atom_dict, dir_dataset+'/'+dataname+ '-atom_dict.pickle')
    dump_dictionary(bond_dict, dir_dataset+'/'+dataname+ '-bond_dict.pickle')
    dump_dictionary(edge_dict, dir_dataset+'/'+dataname+ '-edge_dict.pickle')
    return dataset
	
def create_dataset_randomsplit(x,y,path,dataname):
    dir_input = path + 'SMRT-'
    with open(dir_input + 'atom_dict.pickle', 'rb') as f:
        c=pickle.load(f)
        for k in c.keys():
            atom_dict.get(k)
            atom_dict[k]=c[k]
    with open(dir_input+ 'bond_dict.pickle', 'rb') as f:
        c=pickle.load(f)
        for k in c.keys():
            bond_dict.get(k)
            bond_dict[k]=c[k]
        
    with open(dir_input + 'edge_dict.pickle', 'rb') as f:
        c=pickle.load(f)
        for k in c.keys():
            edge_dict.get(k)
            edge_dict[k]=c[k]
        
    with open(dir_input + 'fingerprint_dict.pickle', 'rb') as f:
        c=pickle.load(f)
        for k in c.keys():
            fingerprint_dict.get(k)
            fingerprint_dict[k]=c[k]    
    dataset = []  
    for i in range(len(x)):
        smiles=x[i]
        property=y[i]         
        """Create each data with the above defined functions."""
        mol = Chem.MolFromInchi(smiles)     
        mol = Chem.AddHs(Chem.MolFromInchi(smiles))
        atoms = create_atoms(mol, atom_dict)
        molecular_size = len(atoms)
        i_jbond_dict = create_ijbonddict(mol, bond_dict)
        fingerprints = extract_fingerprints(radius, atoms, i_jbond_dict,
                                                fingerprint_dict, edge_dict)
        adjacency = np.float32((Chem.GetAdjacencyMatrix(mol)))
        # 仅存储numpy数组，不预先转换为tensor并移动到GPU，减少显存占用
        dataset.append((smiles, fingerprints, adjacency, molecular_size, float(property)))
    dir_dataset=path
    dump_dictionary(fingerprint_dict, dir_dataset +dataname+ '-fingerprint_dict.pickle')
    dump_dictionary(atom_dict, dir_dataset +dataname+ '-atom_dict.pickle')
    dump_dictionary(bond_dict, dir_dataset  +dataname+ '-bond_dict.pickle')
    dump_dictionary(edge_dict, dir_dataset +dataname+ '-edge_dict.pickle')
    return dataset
	
def create_dataset_kfold(x,y,path,dataname):
    dir_input =path+'SMRT-'
    with open(dir_input + 'atom_dict.pickle', 'rb') as f:
        c=pickle.load(f)
        for k in c.keys():
            atom_dict.get(k)
            atom_dict[k]=c[k]
    with open(dir_input+ 'bond_dict.pickle', 'rb') as f:
        c=pickle.load(f)
        for k in c.keys():
            bond_dict.get(k)
            bond_dict[k]=c[k]
        
    with open(dir_input + 'edge_dict.pickle', 'rb') as f:
        c=pickle.load(f)
        for k in c.keys():
            edge_dict.get(k)
            edge_dict[k]=c[k]
        
    with open(dir_input + 'fingerprint_dict.pickle', 'rb') as f:
        c=pickle.load(f)
        for k in c.keys():
            fingerprint_dict.get(k)
            fingerprint_dict[k]=c[k]   
    dataset = []
    for i in range(len(x)):
        smiles=x[i]
        property=y[i]
        """Create each data with the above defined functions."""
        mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
        atoms = create_atoms(mol, atom_dict)
        molecular_size = len(atoms)
        i_jbond_dict = create_ijbonddict(mol, bond_dict)
        fingerprints = extract_fingerprints(radius, atoms, i_jbond_dict,
                                                fingerprint_dict, edge_dict)
        adjacency = np.float32((Chem.GetAdjacencyMatrix(mol)))
        # 仅存储numpy数组，不预先转换为tensor并移动到GPU，减少显存占用
        dataset.append((smiles, fingerprints, adjacency, molecular_size, float(property)))
    dir_dataset=path
    dump_dictionary(fingerprint_dict, dir_dataset +dataname+ '-fingerprint_dict.pickle')
    dump_dictionary(atom_dict, dir_dataset +dataname+ '-atom_dict.pickle')
    dump_dictionary(bond_dict, dir_dataset  +dataname+ '-bond_dict.pickle')
    dump_dictionary(edge_dict, dir_dataset +dataname+ '-edge_dict.pickle')
    return dataset

def transferlearning_dataset_predict(x,path):
    dir_input = path+'SMRT-'
    with open(dir_input + 'atom_dict.pickle', 'rb') as f:
        c=pickle.load(f)
        for k in c.keys():
            atom_dict.get(k)
            atom_dict[k]=c[k]
    with open(dir_input+ 'bond_dict.pickle', 'rb') as f:
        c=pickle.load(f)
        for k in c.keys():
            bond_dict.get(k)
            bond_dict[k]=c[k]
        
    with open(dir_input + 'edge_dict.pickle', 'rb') as f:
        c=pickle.load(f)
        for k in c.keys():
            edge_dict.get(k)
            edge_dict[k]=c[k]
        
    with open(dir_input + 'fingerprint_dict.pickle', 'rb') as f:
        c=pickle.load(f)
        for k in c.keys():
            fingerprint_dict.get(k)
            fingerprint_dict[k]=c[k]
    dataset = []
    for i in range(len(x)):
        smiles=x[i]
        """Create each data with the above defined functions."""       
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            continue           
        else:
            smi = Chem.MolToSmiles(mol)            
        mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
        atoms = create_atoms(mol, atom_dict)
        molecular_size = len(atoms)
        i_jbond_dict = create_ijbonddict(mol, bond_dict)
        fingerprints = extract_fingerprints(radius, atoms, i_jbond_dict,
                                                fingerprint_dict, edge_dict)
        adjacency = np.float32((Chem.GetAdjacencyMatrix(mol)))
# 仅存储numpy数组，不预先转换为tensor并移动到GPU，减少显存占用
        dataset.append((smiles, fingerprints, adjacency, molecular_size)) 
    return dataset
