#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
分子处理工具函数
"""

import torch
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.rdchem import HybridizationType, BondType
import os
import hashlib
import threading

# 定义需要的全局变量和辅助函数
try:
    ETKDG_PARAMS = AllChem.ETKDGv3()
    ETKDG_PARAMS.randomSeed = 1234
    ETKDG_VERSION_USED = "ETKDGv3"
except AttributeError:
    ETKDG_PARAMS = AllChem.ETKDGv2()
    ETKDG_PARAMS.randomSeed = 1234
    ETKDG_VERSION_USED = "ETKDGv2 (fallback)"

# 更新bonds字典以包含更多键类型，未知键类型将被视为单键
bonds = {BondType.SINGLE: 0, BondType.DOUBLE: 1, BondType.TRIPLE: 2, BondType.AROMATIC: 3, BondType.DATIVE: 0}


def one_hot(tensor, num_classes):
    """
    将整数张量转换为 one-hot 编码。
    
    Args:
        tensor: 输入张量。
        num_classes: 类别数量。
        
    Returns:
        one-hot 编码张量。
    """
    return torch.eye(num_classes, dtype=torch.float)[tensor]


def smile_to_graph_xyz(smile, types):
    """
    将SMILES字符串转换为图结构表示，包括3D坐标信息。
    
    Args:
        smile (str): SMILES字符串
        types (dict): 原子类型映射字典
        
    Returns:
        tuple: (x, z, pos, edge_index, edge_attr) 其中:
            - x: 节点特征矩阵
            - z: 原子序数
            - pos: 3D坐标
            - edge_index: 边索引
            - edge_attr: 边属性: 边类型 one-hot 编码
    """
    try:
        # print(f"正在处理分子 {smile}")
        # 更严格的输入验证，防止float类型或其他无效类型传递给Chem.MolFromSmiles
        if not smile or not isinstance(smile, str):  # 如果SMILES为空或不是字符串，则返回None
            return None, None, None, None, None 
        # 去除字符串两端的空白字符
        smile = smile.strip()
        if not smile:  # 如果去除空白后为空，则返回None
            return None, None, None, None, None
            
        mol = Chem.MolFromSmiles(smile)
        
        if mol is None:
            print(f"无法解析SMILES: {smile}")
            return None, None, None, None, None
        
        mol = Chem.AddHs(mol)
        try:
            AllChem.EmbedMolecule(mol, ETKDG_PARAMS)
            conf = mol.GetConformer()
            pos = conf.GetPositions()
            pos = torch.tensor(pos, dtype=torch.float)
        except Exception:
            # 第一次尝试失败，尝试去除立体构型后重新生成
            try:
                # 去除立体构型
                Chem.RemoveStereochemistry(mol)
                AllChem.EmbedMolecule(mol, ETKDG_PARAMS)
                conf = mol.GetConformer()
                pos = conf.GetPositions()
                pos = torch.tensor(pos, dtype=torch.float)
            except Exception:
                # 如果还是失败，尝试使用MMFF优化
                try:
                    AllChem.EmbedMolecule(mol, AllChem.ETKDG())
                    AllChem.MMFFOptimizeMolecule(mol)
                    conf = mol.GetConformer()
                    pos = conf.GetPositions()
                    pos = torch.tensor(pos, dtype=torch.float)
                except Exception:
                    print(f'无法为 {smile} 生成3D坐标，所有尝试均失败')
                    pos = None  # 明确设置为None
        
        # 检查是否成功生成坐标
        if pos is None:
            # 返回空的结果而不是未定义的变量
            return None, None, None, None, None
            
        # 获取原子特征
        type_idx = []
        atomic_number = []
        aromatic = []
        sp = []
        sp2 = []
        sp3 = []
        num_hs = []
        for atom in mol.GetAtoms():
            symbol = atom.GetSymbol()
            # 如果原子类型不在预定义的类型中，则跳过该原子而不是整个分子
            if symbol not in types:
                print(f"警告: 原子类型 {symbol} 不在类型映射中，将使用默认类型0")
                type_idx.append(0)  # 使用默认类型
            else:
                type_idx.append(types[symbol])
            atomic_number.append(atom.GetAtomicNum())                       # 原子序数
            aromatic.append(1 if atom.GetIsAromatic() else 0)               # 芳香性
            hybridization = atom.GetHybridization()                         # 杂化类型 * 3
            sp.append(1 if hybridization == HybridizationType.SP else 0)    
            sp2.append(1 if hybridization == HybridizationType.SP2 else 0)
            sp3.append(1 if hybridization == HybridizationType.SP3 else 0)

        z = torch.tensor(atomic_number, dtype=torch.long)

        # 获取键信息
        row, col, edge_type = [], [], []
        for bond in mol.GetBonds():
            start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            row += [start, end]
            col += [end, start]
            # 获取键类型，如果遇到未知键类型则使用默认值 SINGLE (0)
            bond_type = bond.GetBondType()
            if bond_type not in bonds:
                print(f"警告: 发现未知键类型 {bond_type}，将使用单键处理")
                edge_type += 2 * [0]  # 默认使用单键类型
            else:
                edge_type += 2 * [bonds[bond_type]]

        if len(row) == 0 or len(col) == 0:
            print(f"警告: 分子 {smile} 没有有效键连接")
            return None, None, None, None, None

        edge_index = torch.tensor([row, col], dtype=torch.long)
        edge_type = torch.tensor(edge_type, dtype=torch.long)
        edge_attr = one_hot(edge_type, num_classes=len(bonds))

        # 排序边
        N = mol.GetNumAtoms()
        if N == 0:
            print(f"警告: 分子 {smile} 没有原子")
            return None, None, None, None, None

        perm = (edge_index[0] * N + edge_index[1]).argsort()
        edge_index = edge_index[:, perm]
        edge_type = edge_type[perm]
        edge_attr = edge_attr[perm]

        # 计算氢原子数量
        row, col = edge_index
        hs = (z == 1).to(torch.float)
        num_hs = torch.zeros(N, dtype=torch.float).scatter_add_(0, col, hs[row]).tolist()

        # 构建节点特征
        x1 = one_hot(torch.tensor(type_idx), num_classes=len(types))        # 原子类型 one-hot 编码
        x2 = torch.tensor([atomic_number, aromatic, sp, sp2, sp3, num_hs],  # 直接将各种属性信息作为节点特征
                            dtype=torch.float).t().contiguous()
        x = torch.cat([x1, x2], dim=-1)
        
        # 添加调试信息
        # print(f"成功处理分子 {smile}: x.shape={x.shape}, z.shape={z.shape}, pos.shape={pos.shape}")
        
        return x, z, pos, edge_index, edge_attr
    except Exception as e:
        print(f"处理分子 {smile} 时发生错误: {e}")
        return None, None, None, None, None


class MoleculeCache:
    """
    分子图结构缓存类，支持将多个SMILES存储在同一个pt文件中
    """
    
    def __init__(self, cache_name="default", csv_file=None, logger=None, dataset_dir=None):
        """
        初始化缓存
        
        Args:
            cache_name (str): 缓存文件名（不包含扩展名）
            csv_file (str): 关联的CSV文件路径，如果提供则会基于此生成缓存文件名
            logger: 日志记录器
            dataset_dir (str): 数据集目录路径，如果提供则将缓存文件存储在此目录下
        """
        if csv_file:
            # 根据CSV文件路径生成唯一的缓存名称
            self.cache_name = self._generate_cache_name_from_csv(csv_file)
        else:
            self.cache_name = cache_name
            
        self.logger = logger
        self.dataset_dir = dataset_dir
        self.cache_dir = self._get_cache_dir()
        self.cache_file = os.path.join(self.cache_dir, f"{self.cache_name}.pt")
        print(f"Using cache file: {self.cache_file}")
        self.cache_data = self._load_cache()
        self.cache_hits = 0
        self.cache_misses = 0
        self.added_count = 0  # 新增计数器
        self.failed_smiles = set()  # 用于记录处理失败的 SMILES
        
        # 添加线程锁以支持多线程操作
        self._lock = threading.RLock()
        self._save_counter = 0  # 独立的保存计数器，避免多线程环境下的竞争条件
    
    def _generate_cache_name_from_csv(self, csv_file):
        """
        根据CSV文件路径生成缓存名称
        
        Args:
            csv_file (str): CSV文件路径
            
        Returns:
            str: 基于CSV文件生成的缓存名称
        """
        # 获取文件的绝对路径和基本信息
        abs_path = os.path.abspath(csv_file)
        file_hash = hashlib.md5(abs_path.encode('utf-8')).hexdigest()[:16]
        basename = os.path.basename(csv_file)
        name_without_ext = os.path.splitext(basename)[0]
        
        # 生成缓存名称
        return f"csv_{name_without_ext}_{file_hash}"
    
    def _get_cache_dir(self):
        """
        获取缓存目录路径
        
        Returns:
            str: 缓存目录路径
        """
        # 如果提供了dataset_dir，则将缓存文件存储在数据集目录下
        if self.dataset_dir is not None:
            cache_dir = self.dataset_dir
        else:
            # 使用项目根目录下的cache目录
            script_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.join(script_dir, '..')
            cache_dir = os.path.join(project_root, '.cache', 'molecule_graphs')
        os.makedirs(cache_dir, exist_ok=True)
        return cache_dir
    
    def _log(self, message):
        """
        统一日志输出方法
        
        Args:
            message (str): 日志消息
        """
        # 只有在logger明确设置为非None且非False时才输出日志
        if self.logger is not None and self.logger is not False:
            self.logger.info(message)
        elif self.logger is None and not hasattr(self, '_log_disabled'):
            # 只在第一次时输出，避免重复日志
            print(message)
            self._log_disabled = True
    
    def _load_cache(self):
        """
        加载缓存文件
        
        Returns:
            dict: 缓存数据字典
        """
        if os.path.exists(self.cache_file):
            try:
                loaded_data = torch.load(self.cache_file)
                # 只有在需要输出日志时才输出
                if self.logger is not None and self.logger is not False:
                    self._log(f"加载缓存文件: {self.cache_file}")
                elif self.logger is None and not hasattr(self, '_load_logged'):
                    print(f"加载缓存文件: {self.cache_file}")
                    self._load_logged = True
                    
                # 兼容旧格式和新格式
                if isinstance(loaded_data, dict) and 'success' in loaded_data:
                    # 新格式
                    self.cache_data = loaded_data['success']
                    self.failed_smiles = set(loaded_data.get('failed', []))
                    if self.logger is not None and self.logger is not False:
                        self._log(f"  成功: {len(self.cache_data)}, 失败: {len(self.failed_smiles)}")
                    elif self.logger is None and not hasattr(self, '_load_logged'):
                        print(f"  成功: {len(self.cache_data)}, 失败: {len(self.failed_smiles)}")
                else:
                    # 旧格式，只有成功记录
                    self.cache_data = loaded_data
                    self.failed_smiles = set()
                    if self.logger is not None and self.logger is not False:
                        self._log(f"  成功: {len(self.cache_data)} (旧格式)")
                    elif self.logger is None and not hasattr(self, '_load_logged'):
                        print(f"  成功: {len(self.cache_data)} (旧格式)")
                
                return self.cache_data
            except Exception as e:
                if self.logger is not None and self.logger is not False:
                    self._log(f"加载缓存文件失败: {e}，将创建新的缓存")
                elif self.logger is None and not hasattr(self, '_load_error_logged'):
                    print(f"加载缓存文件失败: {e}，将创建新的缓存")
                    self._load_error_logged = True
                # 初始化为空
                self.cache_data = {}
                self.failed_smiles = set()
                return self.cache_data
        else:
            # 文件不存在，初始化为空
            self.cache_data = {}
            self.failed_smiles = set()
        return self.cache_data
    
    def _save_cache(self):
        """
        保存缓存到文件
        """
        try:
            # 同时保存成功和失败的记录
            save_data = {
                'success': self.cache_data,
                'failed': list(self.failed_smiles)
            }
            torch.save(save_data, self.cache_file)
            # 只有在需要输出详细日志时才输出
            if self.logger is not None and self.logger is not False and hasattr(self, '_verbose_log') and self._verbose_log:
                self._log(f"已保存缓存文件: {self.cache_file}，成功: {len(self.cache_data)}, 失败: {len(self.failed_smiles)}")
        except Exception as e:
            if self.logger is not None and self.logger is not False:
                self._log(f"保存缓存文件失败: {e}")
            elif self.logger is None and not hasattr(self, '_save_error_logged'):
                print(f"保存缓存文件失败: {e}")
                self._save_error_logged = True
    
    def get(self, smile, types):
        """
        从缓存中获取分子图结构
        
        Args:
            smile (str): SMILES字符串
            types (dict): 原子类型映射字典
            
        Returns:
            tuple or None: 分子图结构数据，如果不存在则返回None
        """
        # 检查是否之前处理失败
        with self._lock:
            if smile in self.failed_smiles:
                return None
                
            # 创建唯一的键
            types_str = str(sorted(types.items()))
            key = f"{smile}_{types_str}"
            
            # if key.startswith('CCN(CCOC(=O)c1ccccc1)c1ccccc1'):  # TEST
            #     print('🐘', self.cache_file, key, self.cache_data[key])
            
            if key in self.cache_data:
                self.cache_hits += 1
                cached_item = self.cache_data[key]
                # 确保所有必要字段都存在
                required_fields = ['x', 'z', 'pos', 'edge_index', 'edge_attr']
                if all(field in cached_item for field in required_fields):
                    return (cached_item['x'], cached_item['z'], cached_item['pos'], 
                           cached_item['edge_index'], cached_item['edge_attr'])
                else:
                    # 数据不完整，从缓存中移除
                    del self.cache_data[key]
                    return None
            else:
                self.cache_misses += 1
                return None
    
    def put(self, smile, types, data):
        """
        将分子图结构存入缓存
        
        Args:
            smile (str): SMILES字符串
            types (dict): 原子类型映射字典
            data (tuple): 分子图结构数据
        """
        # 创建唯一的键
        types_str = str(sorted(types.items()))
        key = f"{smile}_{types_str}"
        
        # 使用线程锁保护共享资源
        with self._lock:
            # 存储数据
            self.cache_data[key] = {
                'x': data[0],
                'z': data[1], 
                'pos': data[2],
                'edge_index': data[3],
                'edge_attr': data[4]
            }
            
            # 每增加10个分子保存一次缓存，避免过于频繁的磁盘I/O
            self.added_count += 1
            self._save_counter += 1
            if self._save_counter % 10 == 0:
                self._save_cache()
    
    def mark_failed(self, smile):
        """
        标记处理失败的 SMILES
        
        Args:
            smile (str): 处理失败的 SMILES 字符串
        """
        with self._lock:
            self.failed_smiles.add(smile)
    
    def get_stats(self):
        """
        获取缓存统计信息
        
        Returns:
            dict: 包含缓存命中率等统计信息
        """
        with self._lock:
            total_requests = self.cache_hits + self.cache_misses
            hit_rate = self.cache_hits / total_requests if total_requests > 0 else 0
            return {
                'hits': self.cache_hits,
                'misses': self.cache_misses,
                'total': total_requests,
                'hit_rate': hit_rate,
                'cache_size': len(self.cache_data),
                'added_count': self.added_count,
                'failed_count': len(self.failed_smiles)
            }
    
    def __del__(self):
        """
        析构函数，确保在对象销毁前保存缓存
        """
        with self._lock:
            if hasattr(self, 'added_count') and self.added_count > 0:
                self._save_cache()
    
    def process_smiles(self, smile, types):
        """
        处理单个SMILES，带缓存支持
        
        Args:
            smile (str): SMILES字符串
            types (dict): 原子类型映射字典
            
        Returns:
            tuple: 分子图结构数据
        """
        # 尝试从缓存获取
        cached_result = self.get(smile, types)
        if cached_result is not None:
            return cached_result
        
        # 缓存未命中，计算分子图结构
        result = smile_to_graph_xyz(smile, types)
        
        # 如果计算成功，存入缓存
        if result is not None and len(result) == 5 and result[0] is not None:
            self.put(smile, types, result)
        else:
            # 如果计算失败，标记为失败
            self.mark_failed(smile)
        
        return result
    
    def get_data_size(self):
        """
        获取缓存中存储的分子数量
        
        Returns:
            int: 分子数量
        """
        with self._lock:
            return len(self.cache_data)
