#!/usr/bin/env python3
"""
测试RDKit的ETKDG算法的随机性
"""

# 添加系统路径以确保能导入RDKit
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem
    # 正确导入Random模块
    from rdkit.Chem import rdMolDescriptors
    from rdkit import DataStructs
    import rdkit.RDLogger as RDLogger
    RDLogger.DisableLog('rdApp.*')  # 禁用RDKit日志
except ImportError as e:
    print(f"无法导入RDKit模块: {e}")
    print("请确保在安装了RDKit的环境中运行此脚本")
    sys.exit(1)

import numpy as np

def test_etkdg_randomness():
    """测试ETKDG算法是否具有随机性"""
    # 测试分子 SMILES
    smiles = "CCN(CCOC(=O)c1ccccc1)c1ccccc1"
    print(f"Testing ETKDG randomness with molecule: {smiles}")
    
    # 创建分子
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        print("Failed to parse SMILES")
        return
    
    mol = Chem.AddHs(mol)
    
    # 多次生成3D坐标并比较结果
    coords_list = []
    for i in range(5):
        # 尝试使用ETKDGv3
        try:
            params = AllChem.ETKDGv3()
        except AttributeError:
            params = AllChem.ETKDGv2()
        
        # 生成3D坐标
        AllChem.EmbedMolecule(mol, params)
        conf = mol.GetConformer()
        pos = conf.GetPositions()
        coords_list.append(pos.copy())
        print(f"Run {i+1}: Generated coordinates with shape {pos.shape}")
    
    # 比较坐标是否相同
    print("\nComparing coordinates:")
    all_same = True
    for i in range(1, 5):
        same = np.allclose(coords_list[0], coords_list[i], atol=1e-6)
        print(f"Run 1 vs Run {i+1}: {'Same' if same else 'Different'}")
        if not same:
            all_same = False
    
    if all_same:
        print("\nAll coordinates are identical - ETKDG is deterministic")
    else:
        print("\nCoordinates differ - ETKDG has randomness")

def test_etkdg_with_seed():
    """测试设置随机种子后ETKDG的行为"""
    print("\n" + "="*50)
    print("Testing ETKDG with fixed random seed")
    
    # 测试分子 SMILES
    smiles = "CCN(CCOC(=O)c1ccccc1)c1ccccc1"
    print(f"Testing ETKDG with fixed seed for molecule: {smiles}")
    
    coords_list = []
    
    for run in range(3):
        # 设置随机种子
        # 使用NumPy的种子设置，这是RDKit ETKDG算法通常依赖的
        # np.random.seed(1234)
        print("Set NumPy random seed to 1234")
        
        # 创建分子
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            print("Failed to parse SMILES")
            return
        
        mol = Chem.AddHs(mol)
        
        # 生成3D坐标
        try:
            params = AllChem.ETKDGv3()
            # 尝试设置参数的随机种子
            params.randomSeed = 1234
        except AttributeError:
            params = AllChem.ETKDGv2()
            params.randomSeed = 1234
        
        AllChem.EmbedMolecule(mol, params)
        conf = mol.GetConformer()
        pos = conf.GetPositions()
        coords_list.append(pos.copy())
        print(f"Run {run+1} with seed 1234: Generated coordinates with shape {pos.shape}")
    
    # 比较坐标是否相同
    print("\nComparing coordinates with fixed seed:")
    all_same = True
    for i in range(1, 3):
        same = np.allclose(coords_list[0], coords_list[i], atol=1e-6)
        print(f"Run 1 vs Run {i+1}: {'Same' if same else 'Different'}")
        if not same:
            all_same = False
    
    if all_same:
        print("\nAll coordinates are identical with fixed seed - ETKDG is reproducible")
    else:
        print("\nCoordinates still differ even with fixed seed")

if __name__ == "__main__":
    test_etkdg_randomness()
    test_etkdg_with_seed()
