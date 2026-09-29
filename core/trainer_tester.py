import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import r2_score, mean_squared_error
from tqdm import tqdm
import traceback
import sys
import os
# 添加导入语句
from utils.atom_types import ATOM_TYPES

# 定义VisNetV2模型额外特征的维度
PHYS_CHEM_DIM = 4    # 物化特征维度
TOXICITY_DIM = 4     # 毒性特征维度
CHROMATO_DIM = 2     # 色谱特征维度

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 导入工具函数
from utils.molecule import MoleculeCache

class Trainer(object):
    def __init__(self, model, visnet=False, batch_train=256, device=None, loss_function='mse', MoleculeCacheName=None):
        self.model = model
        self.visnet = visnet
        self.batch_train = batch_train
        self.device = device
        self.optimizer = None
        self.scheduler = None
        self.clip_grad = True
        # 设置损失函数
        if loss_function == 'l1':
            self.criterion = nn.L1Loss()
        elif loss_function == 'mse':
            self.criterion = nn.MSELoss()
        else:
            raise ValueError("Unsupported loss function. Use 'l1' or 'mse'.")
        if visnet:
            # 初始化分子缓存
            self.mol_cache = MoleculeCache(MoleculeCacheName)
            self.cache_stats_printed = False  # 添加标志位，用于控制缓存统计信息只打印一次
            # 预处理好的数据
            self.preprocessed_data = None
        # 注意：优化器的学习率需要在训练开始前设置
        # 如果是VisNetV2模型且有预处理数据，设置特征级别
        if hasattr(model, 'feature_level'):
            self.feature_level = model.feature_level
        else:
            self.feature_level = 'all'  # 默认为'all'

    def set_optimizer(self, optimizer):
        """设置优化器"""
        self.optimizer = optimizer

    def set_scheduler(self, scheduler):
        """设置学习率调度器"""
        self.scheduler = scheduler

    def set_preprocessed_data(self, preprocessed_data):
        """设置预处理好的VisNet数据"""
        if self.visnet:
            self.preprocessed_data = preprocessed_data

    def train(self, dataset, lr):
        if self.optimizer is None:
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=1e-5)
            
        np.random.shuffle(dataset)
        N = len(dataset)
        loss_total = 0
        
        # 添加进度条显示训练过程
        batch_count = (N + self.batch_train - 1) // self.batch_train
        valid_batches = 0  # 记录有效批次数量
        
        with tqdm(total=batch_count, desc="Training", leave=False) as pbar:
            for i in range(0, N, self.batch_train):
                # 动态将数据加载到GPU上
                data_batch_raw = dataset[i:i+self.batch_train]

                if self.visnet:
                    try:
                        if self.preprocessed_data is not None:
                            # 使用预处理好的数据
                            data_batch = self._prepare_visnet_batch_from_preprocessed(data_batch_raw, train=True)
                        else:
                            # 原有方式处理数据
                            # data_batch = self._prepare_visnet_batch(data_batch_raw, train=True)
                            raise ValueError("Preprocessed data not available")
                        
                        # 检查批次数据是否有效
                        if data_batch is not None:  # 只有当批次不为空时才进行训练
                            loss = self._train_visnet_batch(data_batch)
                            # 只有在损失值有效时才更新loss_total
                            if not torch.isnan(loss):
                                loss_total += loss.item()
                                valid_batches += 1
                            else:
                                # NaN损失值不计入统计
                                print("Warning: NaN loss encountered in batch")
                        else:
                            # 跳过空批次
                            print("Skipping empty batch")
                            pbar.update(1)
                            continue
                    except Exception as e:
                        print(f"Error: Error processing batch: {e}")
                        print(f"Full traceback:\n{traceback.format_exc()}")
                        # 直接抛出异常，停止程序执行
                        raise e
                else:
                    data_batch = self._prepare_batch(data_batch_raw, train=True)
                    loss = self.model.forward_regressor(data_batch, train=True)
                    loss_total += loss.item()
                    valid_batches += 1
                
                try:
                    # 只有在不是NaN时才执行反向传播
                    if not torch.isnan(loss):
                        self.optimizer.zero_grad()
                        loss.backward()
                        
                        # 梯度裁剪，防止梯度爆炸
                        if self.clip_grad:
                            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                            
                        self.optimizer.step()
                        
                        # 清理GPU内存缓存，防止内存碎片
                        if self.device.type == 'cuda':
                            torch.cuda.empty_cache()
                except RuntimeError as e:
                    if "out of memory" in str(e):
                        print(f"CUDA out of memory error occurred during backpropagation: {e}")
                        # 清理GPU内存并重新抛出异常
                        if self.device.type == 'cuda':
                            torch.cuda.empty_cache()
                        raise e
                    else:
                        raise e
                pbar.update(1)
                # 只显示有效损失值
                if not torch.isnan(loss):
                    pbar.set_postfix({"Loss": f"{loss.item():.4f}"})
        
        # 打印缓存统计信息（只在训练结束后打印一次）
        if self.visnet and not self.cache_stats_printed and self.preprocessed_data is None:
            stats = self.mol_cache.get_stats()
            print(f"Cache stats - Hits: {stats['hits']}, Misses: {stats['misses']}, "
                  f"Hit Rate: {stats['hit_rate']:.2%}, Failed: {stats['failed_count']}")
            self.cache_stats_printed = True  # 设置标志位，避免重复打印
        
        # 计算平均损失
        avg_loss = loss_total / valid_batches if valid_batches > 0 else 0.0
        
        # 更新学习率调度器
        if self.scheduler is not None:
            self.scheduler.step(avg_loss)
        
        # 返回平均损失，避免除零错误
        return avg_loss
        
    def _train_visnet_batch(self, data_batch):
        """训练 VisNet 模型的一个批次"""
        # 解包数据，根据返回值数量进行不同处理
        if len(data_batch) == 5:
            # 包含SMILES列表的预处理数据格式
            z_batch, pos_batch, batch_indices, y_batch, smiles_list = data_batch
        elif len(data_batch) == 4:
            # 不包含SMILES列表的格式
            z_batch, pos_batch, batch_indices, y_batch = data_batch
            smiles_list = None
        else:
            raise ValueError(f"Unexpected number of elements in data_batch: {len(data_batch)}")
            
        # 检查输入数据是否有效
        if z_batch is None or pos_batch is None or y_batch is None:
            raise ValueError("Invalid input data: one or more tensors are None")
            
        # 确保所有张量在同一设备上
        z_batch = z_batch.to(self.device)
        pos_batch = pos_batch.to(self.device)
        batch_indices = batch_indices.to(self.device)
        y_batch = y_batch.to(self.device)
            
        # 前向传播
        # 根据模型类型决定如何调用forward方法
        if hasattr(self.model, 'visnet') and hasattr(self.model.visnet, 'forward'):
            # VisNetV1和VisNetV2模型需要构造Data对象
            from torch_geometric.data import Data
            data = Data(z=z_batch, pos=pos_batch, batch=batch_indices)
            
            # 检查是否为VisNetV2模型并传递额外特征
            if hasattr(self.model, 'forward') and 'physchem_features' in self.model.forward.__code__.co_varnames:
                # 这是VisNetV2模型，需要额外特征
                physchem_features = None
                toxicity_features = None
                chromato_features = None
                
                # 如果有预处理数据和SMILES列表，提取额外特征
                if self.preprocessed_data is not None and smiles_list is not None:
                    physchem_features_list = []
                    toxicity_features_list = []
                    chromato_features_list = []
                    
                    # 从预处理数据中提取特征
                    for smiles in smiles_list:
                        if smiles in self.preprocessed_data:
                            data = self.preprocessed_data[smiles]
                            
                            # 根据模型期望的特征检查是否存在相应特征
                            # 注意：VisNetV2模型根据feature_level可能只需要部分特征
                            # 只有当模型需要该特征且特征级别包含该特征时才检查
                            feature_level = getattr(self.model, 'feature_level', 'all')  # 默认为'all'
                            
                            # 获取特征数据
                            physchem_feat = data.get('physchem_features')
                            toxicity_feat = data.get('toxicity_features')
                            chromato_feat = data.get('chromato_features')
                            
                            # 物化特征检查
                            if 'physchem_features' in self.model.forward.__code__.co_varnames:
                                # graph_physchem及以上级别需要物化特征
                                if feature_level in ['graph_physchem', 'graph_physchem_toxicity', 'all']:
                                    if 'physchem_features' not in data:
                                        raise ValueError(f"Missing physchem features for molecule {smiles}")
                                    if physchem_feat is None:
                                        raise ValueError(f"Physchem features is None for molecule {smiles}")
                                    physchem_features_list.append(torch.FloatTensor(physchem_feat))
                            
                            # 毒性特征检查
                            if 'toxicity_features' in self.model.forward.__code__.co_varnames:
                                # graph_physchem_toxicity及以上级别需要毒性特征
                                if feature_level in ['graph_physchem_toxicity', 'all']:
                                    if 'toxicity_features' not in data:
                                        raise ValueError(f"Missing toxicity features for molecule {smiles}")
                                    if toxicity_feat is None:
                                        raise ValueError(f"Toxicity features is None for molecule {smiles}")
                                    toxicity_features_list.append(torch.FloatTensor(toxicity_feat))
                            
                            # 色谱特征检查
                            if 'chromato_features' in self.model.forward.__code__.co_varnames:
                                # all级别需要色谱特征
                                if feature_level == 'all':
                                    if 'chromato_features' not in data:
                                        raise ValueError(f"Missing chromato features for molecule {smiles}")
                                    if chromato_feat is None:
                                        raise ValueError(f"Chromato features is None for molecule {smiles}")
                                    chromato_features_list.append(torch.FloatTensor(chromato_feat))
                        else:
                            # 如果没有找到特征，直接报错而不是填充零值
                            raise ValueError(f"Missing features for molecule {smiles}. Expected physchem_feature_dim={self.model.physchem_feature_dim}, toxicity_feature_dim={self.model.toxicity_feature_dim}, chromato_feature_dim={self.model.chromato_feature_dim}")
                    
                    # 堆叠特征（只有当列表不为空时）
                    if physchem_features_list:
                        physchem_features = torch.stack(physchem_features_list).to(z_batch.device)
                        # 检查维度是否正确
                        if physchem_features.shape[1] != self.model.physchem_feature_dim:
                            raise ValueError(f"Physchem features dimension mismatch. Expected {self.model.physchem_feature_dim}, got {physchem_features.shape[1]}")
                    if toxicity_features_list:
                        toxicity_features = torch.stack(toxicity_features_list).to(z_batch.device)
                        # 检查维度是否正确
                        if toxicity_features.shape[1] != self.model.toxicity_feature_dim:
                            raise ValueError(f"Toxicity features dimension mismatch. Expected {self.model.toxicity_feature_dim}, got {toxicity_features.shape[1]}")
                    if chromato_features_list:
                        chromato_features = torch.stack(chromato_features_list).to(z_batch.device)
                        # 检查维度是否正确
                        if chromato_features.shape[1] != self.model.chromato_feature_dim:
                            raise ValueError(f"Chromato features dimension mismatch. Expected {self.model.chromato_feature_dim}, got {chromato_features.shape[1]}")
                
                predicted_batch = self.model(z_batch, pos_batch, batch_indices, 
                                           physchem_features, toxicity_features, chromato_features)
            else:
                # VisNetV1模型
                predicted_batch = self.model(data)
        else:
            # VisNet基础模型可以直接调用
            predicted_batch, _ = self.model(z_batch, pos_batch, batch_indices)
        
        # 处理模型返回值（VisNetV2返回两个值）
        if isinstance(predicted_batch, tuple):
            predicted_batch = predicted_batch[0]
        
        # 确保预测结果和真实值在同一设备上，并且维度匹配
        predicted_batch = predicted_batch.to(self.device)
        # 确保y_batch和predicted_batch都是正确的形状 (batch_size,)
        y_batch = y_batch.squeeze() if y_batch.dim() > 1 else y_batch
        predicted_batch = predicted_batch.squeeze() if predicted_batch.dim() > 1 else predicted_batch
        
        # 计算损失
        loss = self.criterion(predicted_batch, y_batch)
        
        # 检查损失是否为NaN
        if torch.isnan(loss):
            print("Warning: NaN loss detected, skipping backward pass")
            return loss
            
        return loss

    def _prepare_batch(self, data_batch_raw, train):
        """将原始数据转换为模型所需的格式并移动到GPU"""
        # 解包原始数据
        # smiles_list：一批SMILES字符串列表
        # fingerprints_list：一批指纹张量列表
        # adjacencies_list：一批邻接矩阵张量列表
        # molecular_sizes_list：一批分子大小列表
        # properties_list：一批目标性质值张量列表

        if len(data_batch_raw[0]) == 5:  # 原始格式 (smiles, fingerprints, adjacencies, molecular_sizes, properties)
            smiles_list, fingerprints_list, adjacencies_list, molecular_sizes_list, properties_list = zip(*data_batch_raw)
            
            # 转换为张量并移动到GPU
            fingerprints_list = [torch.LongTensor(f).to(self.device) for f in fingerprints_list]
            adjacencies_list = [torch.FloatTensor(a).to(self.device) for a in adjacencies_list]
            if train:
                properties_list = [torch.FloatTensor([p]).to(self.device) for p in properties_list]
                return [smiles_list, fingerprints_list, adjacencies_list, molecular_sizes_list, properties_list]
            else:
                properties_list = [torch.FloatTensor([p]).to(self.device) for p in properties_list]
                return [smiles_list, fingerprints_list, adjacencies_list, molecular_sizes_list, properties_list]
        else:  # 扩展格式 (smiles, fingerprints, adjacencies, molecular_sizes, additional_features, properties)
            smiles_list, fingerprints_list, adjacencies_list, molecular_sizes_list, additional_features_list, properties_list = zip(*data_batch_raw)
            
            # 转换为张量并移动到GPU
            fingerprints_list = [torch.LongTensor(f).to(self.device) for f in fingerprints_list]
            adjacencies_list = [torch.FloatTensor(a).to(self.device) for a in adjacencies_list]
            additional_features_list = [torch.FloatTensor(a).to(self.device) for a in additional_features_list]
            if train:
                properties_list = [torch.FloatTensor([p]).to(self.device) for p in properties_list]
                return [smiles_list, fingerprints_list, adjacencies_list, molecular_sizes_list, additional_features_list, properties_list]
            else:
                properties_list = [torch.FloatTensor([p]).to(self.device) for p in properties_list]
                return [smiles_list, fingerprints_list, adjacencies_list, molecular_sizes_list, additional_features_list, properties_list]

    def _prepare_visnet_batch(self, data_batch_raw, train):
        """将原始数据转换为 VisNet 模型所需的格式"""
        # 使用统一的原子类型映射
        types = ATOM_TYPES
        
        z_list = []
        pos_list = []
        batch_indices_list = []
        y_list = []
        
        batch_idx = 0
        atom_offset = 0
        
        valid_count = 0
        failed_count = 0
        
        for data in data_batch_raw:
            if len(data) == 5:  # 原始格式 (smiles, fingerprints, adjacencies, molecular_sizes, properties)
                smiles, _, _, _, property_value = data
            else:  # 扩展格式 (smiles, fingerprints, adjacencies, molecular_sizes, additional_features, properties)
                smiles, _, _, _, _, property_value = data
                
            # 使用缓存处理 SMILES
            result = self.mol_cache.process_smiles(smiles, types)
            # 检查是否成功生成了图结构 (smile_to_graph_xyz 返回5个值)
            if result is not None and len(result) == 5:
                x, z, pos, edge_index, edge_attr = result
                if z is not None and pos is not None and x is not None:
                    z_list.append(z)
                    pos_list.append(pos)
                    batch_indices_list.append(torch.full((z.size(0),), batch_idx, dtype=torch.long))
                    y_list.append(torch.FloatTensor([property_value]))
                    batch_idx += 1
                    atom_offset += z.size(0)
                    valid_count += 1
                else:
                    failed_count += 1
            else:
                failed_count += 1
        
        if failed_count > 0:
            print(f"Test batch processing: {valid_count} valid molecules, {failed_count} failed molecules")
        
        # 如果没有有效的分子，则返回 None
        if len(z_list) == 0:
            return None
            
        # 合并批次数据
        z_batch = torch.cat(z_list, dim=0).to(self.device)
        pos_batch = torch.cat(pos_list, dim=0).to(self.device)
        batch_indices = torch.cat(batch_indices_list, dim=0).to(self.device)
        y_batch = torch.stack(y_list, dim=0).to(self.device)
        
        return z_batch, pos_batch, batch_indices, y_batch

    def _prepare_visnet_batch_from_preprocessed(self, data_batch_raw, train):
        """从预处理好的数据中构建VisNet模型所需的批次数据"""
        z_list = []
        pos_list = []
        batch_indices_list = []
        y_list = []
        smiles_list = []  # 添加SMILES列表以返回
        
        batch_idx = 0
        
        valid_count = 0
        failed_count = 0
        
        for data in data_batch_raw:
            if len(data) == 5:  # 原始格式 (smiles, fingerprints, adjacencies, molecular_sizes, properties)
                smiles, _, _, _, property_value = data
            else:  # 扩展格式 (smiles, fingerprints, adjacencies, molecular_sizes, additional_features, properties)
                smiles, _, _, _, _, property_value = data
                
            # 从预处理数据中获取分子图结构
            if smiles in self.preprocessed_data:
                graph_data = self.preprocessed_data[smiles]
                x, z, pos, edge_index, edge_attr = (
                    graph_data['x'], graph_data['z'], graph_data['pos'], 
                    graph_data['edge_index'], graph_data['edge_attr']
                )
                # 确保所有必要数据都存在且不为None
                if (z is not None and pos is not None and x is not None and 
                    edge_index is not None and edge_attr is not None and
                    hasattr(z, 'size') and hasattr(pos, 'size') and hasattr(x, 'size')):
                    try:
                        # 确保张量在CPU上，训练时再移动到设备
                        z_tensor = z.cpu() if hasattr(z, 'cpu') else torch.tensor(z)
                        pos_tensor = pos.cpu() if hasattr(pos, 'cpu') else torch.tensor(pos)
                        
                        # 检查张量是否有效
                        if z_tensor.numel() > 0 and pos_tensor.numel() > 0:
                            z_list.append(z_tensor)
                            pos_list.append(pos_tensor)
                            batch_indices_list.append(torch.full((z_tensor.size(0) if hasattr(z_tensor, 'size') else len(z_tensor),), batch_idx, dtype=torch.long))
                            y_list.append(torch.FloatTensor([float(property_value)]).squeeze())  # 确保 property_value 转换为 float
                            smiles_list.append(smiles)  # 添加SMILES到列表
                            batch_idx += 1
                            valid_count += 1
                        else:
                            failed_count += 1
                    except Exception as e:
                        print(f"Error processing molecule {smiles}: {e}")
                        failed_count += 1
                else:
                    failed_count += 1
            else:
                failed_count += 1
        
        if failed_count > 0:
            print_func = print if train else lambda *args, **kwargs: None  # 训练时才打印
            print_func(f"Trainer: _prepare_visnet_batch_from_preprocessed - Batch processing: {valid_count} valid molecules, {failed_count} failed molecules")

        # 如果没有有效的分子，则返回 None
        if len(z_list) == 0:
            print("No valid molecules in batch, returning None")
            return None
            
        # 合并批次数据
        z_batch = torch.cat(z_list, dim=0).to(self.device)
        pos_batch = torch.cat(pos_list, dim=0).to(self.device)
        batch_indices = torch.cat(batch_indices_list, dim=0).to(self.device)
        y_batch = torch.stack(y_list, dim=0).to(self.device)
        
        # 返回与_prepare_visnet_batch相同格式的数据，额外包含smiles_list
        return z_batch, pos_batch, batch_indices, y_batch, smiles_list


class Tester(object):
    def __init__(self, model, visnet=False, batch_test=128, device=None, name="", MoleculeCacheName=None):
        self.model = model
        self.visnet = visnet
        self.batch_test = batch_test
        self.device = device
        self.name = name
        if visnet:
            # 初始化分子缓存
            print(f"Initializing MoleculeCache for Tester: {MoleculeCacheName}")
            self.mol_cache = MoleculeCache(MoleculeCacheName)
            self.cache_stats_printed = False  # 添加标志位，用于控制缓存统计信息只打印一次
            # 预处理好的数据
            self.preprocessed_data = None
        # 初始化标准化参数
        self.property_mean = None
        self.property_std = None
    
    def set_preprocessed_data(self, preprocessed_data):
        """设置预处理好的VisNet数据"""
        if self.visnet:
            self.preprocessed_data = preprocessed_data

    def set_standardization_params(self, property_mean, property_std):
        """设置标准化参数"""
        self.property_mean = property_mean
        self.property_std = property_std

    def test_regressor(self, dataset):
        N = len(dataset)
        # 添加对空数据集的检查
        if N == 0:
            # 返回默认值，避免后续计算出错
            return 0.0, 0.0, 0.0, 0.0, ""
        
        if self.visnet:
            return self._test_visnet(dataset)
        
        SMILES, Ts, Ys = '', [], []
        SAE = 0  # sum absolute error.
        
        # 添加进度条显示测试过程
        batch_count = (N + self.batch_test - 1) // self.batch_test
        with tqdm(total=batch_count, desc=f"Testing {self.name}", leave=False) as pbar:
            for i in range(0, N, self.batch_test):
                # 动态将数据加载到GPU上
                data_batch_raw = dataset[i:i+self.batch_test]
                try:
                    data_batch = self._prepare_batch(data_batch_raw, train=False)
                    (Smiles,  predicted_values,correct_values) = self.model.forward_regressor(
                                                    data_batch, train=False)
                    SMILES += ' '.join(Smiles) + ' '
                    Ts.append(correct_values)
                    Ys.append(predicted_values)
                    
                    SAE += sum(np.abs(predicted_values-correct_values))
                except RuntimeError as e:
                    if "out of memory" in str(e):
                        print(f"CUDA out of memory error occurred during testing: {e}")
                        raise e
                    else:
                        print(f"Warning: Error processing test batch: {e}")
                        # 跳过这个批次
                        pass
                except Exception as e:
                    import traceback
                    print(f"Warning: Error processing test batch: {e}")
                    print(f"Full traceback:\n{traceback.format_exc()}")
                    # 跳过这个批次
                    pass
                pbar.update(1)
        SMILES = SMILES.strip().split()
        
        # 检查处理后的数据是否为空
        if len(SMILES) == 0:
            return 0.0, 0.0, 0.0, 0.0, ""
        
        # 直接使用数组而不是转换为字符串再转回来
        T = np.concatenate(Ts)  # 真实值
        Y = np.concatenate(Ys)  # 预测值
        
        predictions = '\n'.join(['\t'.join([smi, str(t), str(y)]) for smi, t, y in zip(SMILES, T, Y)])
            
        MAEs = SAE / N  # mean absolute error.
        
        # 检查转换后的数据是否为空
        if len(T) == 0 or len(Y) == 0:
            return 0.0, 0.0, 0.0, 0.0, ""
        
        # Calculate additional metrics
        mse = mean_squared_error(T, Y)
        r2 = r2_score(T, Y)
        pcc = np.corrcoef(T, Y)[0, 1]
        
        # Return all metrics
        return MAEs, mse, r2, pcc, predictions
    
    def _test_visnet(self, dataset):
        """测试 VisNet 模型"""
        N = len(dataset)
        if N == 0:
            return 0.0, 0.0, 0.0, 0.0, ""
        
        SMILES, Ts, Ys = '', [], []
        SAE = 0  # sum absolute error.
        
        # 使用字典存储预测结果，确保数据一致性
        prediction_results = []  # 存储 (smiles, true_value, pred_value) 元组
        
        # 添加进度条显示测试过程
        batch_count = (N + self.batch_test - 1) // self.batch_test
        processed_samples = 0
        with tqdm(total=batch_count, desc=f"Testing {self.name}", leave=False) as pbar:
            for i in range(0, N, self.batch_test):
                data_batch_raw = dataset[i:i+self.batch_test]

                # 添加调试信息
                # if i == 0:  # 只在第一个批次打印调试信息
                #     print(f"=== Testing Batch Info ===")
                #     print(f"Batch index: {i}, Batch size: {len(data_batch_raw)}")
                #     if len(data_batch_raw) > 0:
                #         sample = data_batch_raw[0]
                #         if isinstance(sample, dict):
                #             print(f"Sample keys: {list(sample.keys())}")
                #             if 'smiles' in sample:
                #                 print(f"First sample SMILES: {sample['smiles']}")
                #             if 'property_value' in sample:
                #                 print(f"First sample property value: {sample['property_value']}")
                #         else:
                #             print(f"Sample type: {type(sample)}, length: {len(sample) if hasattr(sample, '__len__') else 'N/A'}")
                
                try:
                    if self.preprocessed_data is not None:
                        # 使用预处理好的数据
                        batch_data = self._prepare_visnet_batch_from_preprocessed(data_batch_raw, train=False)
                    else:
                        raise RuntimeError("Preprocessed data is not available")
                        # 原有方式处理数据
                        # batch_data = self._prepare_visnet_batch(data_batch_raw, train=False)
                    
                    if batch_data is None:
                        print("Skipping empty batch in testing")
                        pbar.update(1)
                        continue
                        
                    if self.preprocessed_data is not None:
                        # 从预处理数据返回的是包含SMILES列表的元组
                        if len(batch_data) == 5:
                            z_batch, pos_batch, batch_indices, y_batch, valid_smiles_list = batch_data
                        else:
                            raise ValueError(f"Unexpected number of values in batch_data: {len(batch_data)}")
                    else:
                        raise RuntimeError("Preprocessed data is not available")
                    
                    # # 添加调试信息
                    # if i == 0:  # 只在第一个批次打印调试信息
                    #     print(f"=== Batch Data Info ===")
                    #     print(f"z_batch shape: {z_batch.shape}")
                    #     print(f"pos_batch shape: {pos_batch.shape}")
                    #     print(f"batch_indices shape: {batch_indices.shape}")
                    #     print(f"y_batch shape: {y_batch.shape}")
                    #     print(f"Valid SMILES list length: {len(valid_smiles_list)}")
                    #     if valid_smiles_list:
                    #         print(f"First SMILES in batch: {valid_smiles_list[0]}")
                    
                    # 确保所有张量在同一设备上
                    z_batch = z_batch.to(self.device)
                    pos_batch = pos_batch.to(self.device)
                    batch_indices = batch_indices.to(self.device)
                    y_batch = y_batch.to(self.device)
                    
                    # 获取预测值
                    with torch.no_grad():
                        # 根据模型类型决定如何调用forward方法
                        if hasattr(self.model, 'visnet') and hasattr(self.model.visnet, 'forward'):
                            # VisNetV1和VisNetV2模型需要构造Data对象
                            from torch_geometric.data import Data
                            data = Data(z=z_batch, pos=pos_batch, batch=batch_indices)
                            
                            # 检查是否为VisNetV2模型并传递额外特征
                            if (hasattr(self.model, 'forward') and 'physchem_features' in self.model.forward.__code__.co_varnames):
                                # 这是VisNetV2模型，需要额外特征
                                physchem_features = None
                                toxicity_features = None
                                chromato_features = None
                                
                                # 如果有预处理数据和SMILES列表，提取额外特征
                                if self.preprocessed_data is not None and valid_smiles_list is not None:
                                    physchem_features_list = []
                                    toxicity_features_list = []
                                    chromato_features_list = []
                                    
                                    for smiles in valid_smiles_list:
                                        if smiles in self.preprocessed_data:
                                            data = self.preprocessed_data[smiles]
                                            # 获取特征数据
                                            physchem_feat = data.get('physchem_features')
                                            toxicity_feat = data.get('toxicity_features')
                                            chromato_feat = data.get('chromato_features')
                                            
                                            # 根据模型参数和特征级别判断是否需要检查特征
                                            # 注意：VisNetV2模型根据feature_level可能只需要部分特征
                                            # 只有当模型需要该特征且特征级别包含该特征时才检查
                                            # 优先使用Trainer中设置的feature_level，保持与训练阶段一致
                                            feature_level = getattr(self, 'feature_level', getattr(self.model, 'feature_level', 'all'))  # 默认为'all'
                                            
                                            # 添加调试信息
                                            # if i == 0 and smiles == valid_smiles_list[0]:  # 只在第一个分子打印调试信息
                                            #     print(f"=== Feature Processing Info ===")
                                            #     print(f"Feature level: {feature_level}")
                                            #     print(f"Model feature level: {getattr(self.model, 'feature_level', 'N/A')}")
                                            #     print(f"Physchem feature available: {'physchem_features' in data}")
                                            #     print(f"Toxicity feature available: {'toxicity_features' in data}")
                                            #     print(f"Chromato feature available: {'chromato_features' in data}")
                                            
                                            # 物化特征检查
                                            if 'physchem_features' in self.model.forward.__code__.co_varnames:
                                                # graph_physchem及以上级别需要物化特征
                                                if feature_level in ['graph_physchem', 'graph_physchem_toxicity', 'all']:
                                                    if 'physchem_features' not in data:
                                                        raise ValueError(f"Missing physchem features for molecule {smiles}")
                                                    if physchem_feat is None:
                                                        raise ValueError(f"Physchem features is None for molecule {smiles}")
                                                    physchem_features_list.append(torch.from_numpy(physchem_feat).float())
                                            
                                            # 毒性特征检查
                                            if 'toxicity_features' in self.model.forward.__code__.co_varnames:
                                                # graph_physchem_toxicity及以上级别需要毒性特征
                                                if feature_level in ['graph_physchem_toxicity', 'all']:
                                                    if 'toxicity_features' not in data:
                                                        raise ValueError(f"Missing toxicity features for molecule {smiles}")
                                                    toxicity_feat = data.get('toxicity_features')
                                                    if toxicity_feat is None:
                                                        raise ValueError(f"Toxicity features is None for molecule {smiles}")
                                                    toxicity_features_list.append(torch.from_numpy(toxicity_feat).float())
                                            
                                            # 色谱特征检查
                                            if 'chromato_features' in self.model.forward.__code__.co_varnames:
                                                # all级别需要色谱特征
                                                if feature_level == 'all':
                                                    if 'chromato_features' not in data:
                                                        raise ValueError(f"Missing chromato features for molecule {smiles}")
                                                    chromato_feat = data.get('chromato_features')
                                                    if chromato_feat is None:
                                                        raise ValueError(f"Chromato features is None for molecule {smiles}")
                                                    chromato_features_list.append(torch.from_numpy(chromato_feat).float())
                                        else:
                                            # 如果没有找到特征，直接报错而不是填充零值
                                            raise ValueError(f"Missing features for molecule {smiles}. Expected physchem_feature_dim={self.model.physchem_feature_dim}, toxicity_feature_dim={self.model.toxicity_feature_dim}, chromato_feature_dim={self.model.chromato_feature_dim}")
                                    
                                    # 堆叠特征（只有当列表不为空时）
                                    if physchem_features_list:
                                        physchem_features = torch.stack(physchem_features_list).to(self.device)
                                    if toxicity_features_list:
                                        toxicity_features = torch.stack(toxicity_features_list).to(self.device)
                                    if chromato_features_list:
                                        chromato_features = torch.stack(chromato_features_list).to(self.device)
                                
                                # # 添加调试信息
                                # if i == 0:  # 只在第一个批次打印调试信息
                                #     print(f"=== Feature Stack Info ===")
                                #     if physchem_features is not None:
                                #         print(f"Physchem features shape: {physchem_features.shape}")
                                #         print(f"Physchem features sample: {physchem_features[0][:5] if physchem_features.numel() > 0 else 'N/A'}")
                                #     if toxicity_features is not None:
                                #         print(f"Toxicity features shape: {toxicity_features.shape}")
                                #         print(f"Toxicity features sample: {toxicity_features[0][:5] if toxicity_features.numel() > 0 else 'N/A'}")
                                #     if chromato_features is not None:
                                #         print(f"Chromato features shape: {chromato_features.shape}")
                                #         print(f"Chromato features sample: {chromato_features[0][:5] if chromato_features.numel() > 0 else 'N/A'}")
                                
                                pred = self.model(z_batch, pos_batch, batch_indices, 
                                                physchem_features, toxicity_features, chromato_features)
                                
                                # # 添加预测值调试信息
                                # if i == 0:  # 只在第一个批次打印调试信息
                                #     print(f"=== Prediction Info ===")
                                #     print(f"Prediction shape: {pred[0].shape if isinstance(pred, tuple) else pred.shape}")
                                #     print(f"Prediction sample: {pred[0][:5] if isinstance(pred, tuple) else pred[:5]}")
                            else:
                                # VisNetV1模型
                                pred = self.model(data)
                                
                            # 处理模型返回值（VisNetV2返回两个值）
                            if isinstance(pred, tuple):
                                pred = pred[0]
                                
                            predicted_values = pred.cpu().numpy().flatten()
                        else:
                            # VisNet基础模型可以直接调用
                            pred, _ = self.model(z_batch, pos_batch, batch_indices)
                            predicted_values = pred.cpu().numpy().flatten()
                        
                        # 保持单样本批次和多样本批次的真实值均为一维数组。
                        correct_values = y_batch.reshape(-1).cpu().numpy()

                        # 反标准化预测值和真实值
                        if self.property_mean is not None and self.property_std is not None:
                            predicted_values = predicted_values * self.property_std + self.property_mean
                            correct_values = correct_values * self.property_std + self.property_mean
                        
                        # 检查预测值和真实值的对应性，过滤掉任何包含NaN或无穷大的样本
                        valid_mask = np.isfinite(predicted_values) & np.isfinite(correct_values)
                        
                        # 为每个样本添加结果，保持数据一致性
                        for idx, smiles in enumerate(valid_smiles_list):
                            true_val = correct_values[idx]
                            pred_val = predicted_values[idx]
                            
                            # 只有当预测值和真实值都是有限数值时才添加到结果中
                            if np.isfinite(true_val) and np.isfinite(pred_val):
                                prediction_results.append((smiles, true_val, pred_val))
                                SMILES += smiles + ' '
                                Ts.append(true_val)
                                Ys.append(pred_val)
                                SAE += abs(pred_val - true_val)
                                processed_samples += 1
                            else:
                                print(f"Warning: Skipping molecule {smiles} due to NaN or infinite values")
                    
                    # 清理GPU内存缓存
                    if self.device.type == 'cuda':
                        torch.cuda.empty_cache()

                except Exception as e:
                    import traceback
                    print(f"Error: Error processing batch: {e}")
                    print(f"Full traceback:\n{traceback.format_exc()}")
                    # 直接抛出异常，停止程序执行
                    # raise e
            
                pbar.update(1)
        
        # 打印缓存统计信息（只打印一次）
        if self.visnet and not self.cache_stats_printed and self.preprocessed_data is None:
            stats = self.mol_cache.get_stats()
            print(f"Test cache stats - Hits: {stats['hits']}, Misses: {stats['misses']}, "
                  f"Hit Rate: {stats['hit_rate']:.2%}, Failed: {stats['failed_count']}")
            self.cache_stats_printed = True  # 设置标志位，避免重复打印
        
        # 检查处理后的数据是否为空
        if len(prediction_results) == 0:
            return 0.0, 0.0, 0.0, 0.0, ""
        
        # 构造预测结果字符串
        predictions = '\n'.join(['\t'.join([smi, str(t), str(y)]) for smi, t, y in prediction_results])
            
        MAEs = SAE / processed_samples if processed_samples > 0 else 0  # mean absolute error.
        
        # Calculate additional metrics
        if len(Ts) > 0 and len(Ys) > 0:
            T = np.array(Ts)
            Y = np.array(Ys)
            mse = mean_squared_error(T, Y)
            r2 = r2_score(T, Y) if len(T) > 1 else float('nan')
            pcc = np.corrcoef(T, Y)[0, 1] if len(T) > 1 else float('nan')
        else:
            mse = r2 = pcc = 0.0
        
        # Return all metrics
        return MAEs, mse, r2, pcc, predictions
        
    def _prepare_visnet_batch(self, data_batch_raw, train):
        """将原始数据转换为 VisNet 模型所需的格式"""
        # 使用统一的原子类型映射
        types = ATOM_TYPES
        
        z_list = []
        pos_list = []
        batch_indices_list = []
        y_list = []
        
        batch_idx = 0
        atom_offset = 0
        
        valid_count = 0
        failed_count = 0
        
        for data in data_batch_raw:
            if len(data) == 5:  # 原始格式 (smiles, fingerprints, adjacencies, molecular_sizes, properties)
                smiles, _, _, _, property_value = data
            else:  # 扩展格式 (smiles, fingerprints, adjacencies, molecular_sizes, additional_features, properties)
                smiles, _, _, _, _, property_value = data
                
            # 使用缓存处理 SMILES
            result = self.mol_cache.process_smiles(smiles, types)
            # 检查是否成功生成了图结构 (smile_to_graph_xyz 返回5个值)
            if result is not None and len(result) == 5:
                x, z, pos, edge_index, edge_attr = result
                if z is not None and pos is not None and x is not None:
                    z_list.append(z)
                    pos_list.append(pos)
                    batch_indices_list.append(torch.full((z.size(0),), batch_idx, dtype=torch.long))
                    y_list.append(torch.FloatTensor([property_value]))
                    batch_idx += 1
                    atom_offset += z.size(0)
                    valid_count += 1
                else:
                    failed_count += 1
            else:
                failed_count += 1
        
        if failed_count > 0:
            print(f"Test batch processing: {valid_count} valid molecules, {failed_count} failed molecules")
        
        # 如果没有有效的分子，则返回 None
        if len(z_list) == 0:
            return None
            
        # 合并批次数据
        z_batch = torch.cat(z_list, dim=0).to(self.device)
        pos_batch = torch.cat(pos_list, dim=0).to(self.device)
        batch_indices = torch.cat(batch_indices_list, dim=0).to(self.device)
        y_batch = torch.stack(y_list, dim=0).to(self.device)
        
        return z_batch, pos_batch, batch_indices, y_batch

    def _prepare_visnet_batch_from_preprocessed(self, data_batch_raw, train):
        """从预处理好的数据中构建VisNet模型所需的批次数据"""
        z_list = []
        pos_list = []
        batch_indices_list = []
        y_list = []
        smiles_list = []  # 添加SMILES列表以返回
        
        batch_idx = 0
        
        valid_count = 0
        failed_count = 0
        
        for data in data_batch_raw:
            if type(data) == dict:
                smiles = data.get('smiles', None)
                property_value = data.get('property_value', None)
            else:
                if len(data) == 5:  # 原始格式 (smiles, fingerprints, adjacencies, molecular_sizes, properties)
                    smiles, _, _, _, property_value = data
                else:  # 扩展格式 (smiles, fingerprints, adjacencies, molecular_sizes, additional_features, properties)
                    smiles, _, _, _, _, property_value = data
                
            # 从预处理数据中获取分子图结构
            if smiles in self.preprocessed_data:
                graph_data = self.preprocessed_data[smiles]
                x, z, pos, edge_index, edge_attr = (
                    graph_data['x'], graph_data['z'], graph_data['pos'], 
                    graph_data['edge_index'], graph_data['edge_attr']
                )
                # 确保所有必要数据都存在且不为None
                if (z is not None and pos is not None and
                    # x is not None and edge_index is not None and edge_attr is not None and
                    # and hasattr(x, 'size')
                    hasattr(z, 'size') and hasattr(pos, 'size')):
                    try:
                        # 确保张量在CPU上，训练时再移动到设备
                        z_tensor = z.cpu() if hasattr(z, 'cpu') else torch.tensor(z)
                        pos_tensor = pos.cpu() if hasattr(pos, 'cpu') else torch.tensor(pos)
                        
                        # 检查张量是否有效
                        if z_tensor.numel() > 0 and pos_tensor.numel() > 0:
                            z_list.append(z_tensor)
                            # 注意：pos张量需要特别处理，我们只需要原子坐标部分，即pos[:, :3]
                            if pos_tensor.dim() == 2 and pos_tensor.size(1) >= 3:
                                pos_list.append(pos_tensor[:, :3])  # 只取前3列（x, y, z坐标）
                            else:
                                pos_list.append(pos_tensor)
                            batch_indices_list.append(torch.full((z_tensor.size(0) if hasattr(z_tensor, 'size') else len(z_tensor),), batch_idx, dtype=torch.long))
                            y_list.append(torch.FloatTensor([property_value]))
                            smiles_list.append(smiles)  # 添加SMILES到列表
                            batch_idx += 1
                            valid_count += 1
                        else:
                            failed_count += 1
                    except Exception as e:
                        print(f"Error processing molecule {smiles}: {e}")
                        failed_count += 1
                else:
                    # print(f"Molecule {smiles} has incomplete graph data")
                    failed_count += 1
            else:
                print(f"Molecule {smiles} not found in preprocessed data")
                failed_count += 1
        
        if failed_count > 0:
            print(f"Tester: _prepare_visnet_batch_from_preprocessed - Batch processing: {valid_count} valid molecules, {failed_count} failed molecules")
        
        # 如果没有有效的分子，则返回 None
        if len(z_list) == 0:
            return None
            
        # 合并批次数据
        try:
            z_batch = torch.cat(z_list, dim=0).to(self.device)
            pos_batch = torch.cat(pos_list, dim=0).to(self.device)
            batch_indices = torch.cat(batch_indices_list, dim=0).to(self.device)
            y_batch = torch.stack(y_list, dim=0).to(self.device)
        except Exception as e:
            print(f"Error concatenating tensors: {e}")
            # 添加调试信息
            print("Debug info for tensor concatenation:")
            print(f"  z_list: {len(z_list)} tensors")
            for i, z in enumerate(z_list):
                print(f"    z_list[{i}].shape={z.shape}")
            print(f"  pos_list: {len(pos_list)} tensors")
            for i, pos in enumerate(pos_list):
                print(f"    pos_list[{i}].shape={pos.shape}")
            print(f"  batch_indices_list: {len(batch_indices_list)} tensors")
            for i, bi in enumerate(batch_indices_list):
                print(f"    batch_indices_list[{i}].shape={bi.shape}")
            print(f"  y_list: {len(y_list)} tensors")
            for i, y in enumerate(y_list):
                print(f"    y_list[{i}].shape={y.shape}")
            if smiles_list:
                print(f"  First molecule SMILES: {smiles_list[0]}")
            return None
        
        # 返回与_prepare_visnet_batch相同格式的数据，额外包含smiles_list
        return z_batch, pos_batch, batch_indices, y_batch, smiles_list

    def _prepare_batch(self, data_batch_raw, train):
        """将原始数据转换为模型所需的格式并移动到GPU"""
        # 解包原始数据
        if len(data_batch_raw[0]) == 5:  # 原始格式 (smiles, fingerprints, adjacencies, molecular_sizes, properties)
            smiles_list, fingerprints_list, adjacencies_list, molecular_sizes_list, properties_list = zip(*data_batch_raw)
            
            # 转换为张量并移动到GPU
            fingerprints_list = [torch.LongTensor(f).to(self.device) for f in fingerprints_list]
            adjacencies_list = [torch.FloatTensor(a).to(self.device) for a in adjacencies_list]
            if train:
                properties_list = [torch.FloatTensor([p]).to(self.device) for p in properties_list]
                return [smiles_list, fingerprints_list, adjacencies_list, molecular_sizes_list, properties_list]
            else:
                properties_list = [torch.FloatTensor([p]).to(self.device) for p in properties_list]
                return [smiles_list, fingerprints_list, adjacencies_list, molecular_sizes_list, properties_list]
        elif len(data_batch_raw[0]) == 7:  # 新的扩展格式 (smiles, fingerprints, adjacencies, molecular_sizes, physchem_features, toxicity_features, chromato_features, properties)
            smiles_list, fingerprints_list, adjacencies_list, molecular_sizes_list, physchem_features_list, toxicity_features_list, chromato_features_list, properties_list = zip(*data_batch_raw)
            
            # 转换为张量并移动到GPU
            fingerprints_list = [torch.LongTensor(f).to(self.device) for f in fingerprints_list]
            adjacencies_list = [torch.FloatTensor(a).to(self.device) for a in adjacencies_list]
            physchem_features_list = [torch.FloatTensor(a).to(self.device) for a in physchem_features_list]
            toxicity_features_list = [torch.FloatTensor(a).to(self.device) for a in toxicity_features_list]
            chromato_features_list = [torch.FloatTensor(a).to(self.device) for a in chromato_features_list]
            if train:
                properties_list = [torch.FloatTensor([p]).to(self.device) for p in properties_list]
                return [smiles_list, fingerprints_list, adjacencies_list, molecular_sizes_list, physchem_features_list, toxicity_features_list, chromato_features_list, properties_list]
            else:
                properties_list = [torch.FloatTensor([p]).to(self.device) for p in properties_list]
                return [smiles_list, fingerprints_list, adjacencies_list, molecular_sizes_list, physchem_features_list, toxicity_features_list, chromato_features_list, properties_list]
        else:  # 扩展格式 (smiles, fingerprints, adjacencies, molecular_sizes, additional_features, properties)
            smiles_list, fingerprints_list, adjacencies_list, molecular_sizes_list, additional_features_list, properties_list = zip(*data_batch_raw)
            
            # 转换为张量并移动到GPU
            fingerprints_list = [torch.LongTensor(f).to(self.device) for f in fingerprints_list]
            adjacencies_list = [torch.FloatTensor(a).to(self.device) for a in adjacencies_list]
            additional_features_list = [torch.FloatTensor(a).to(self.device) for a in additional_features_list]
            if train:
                properties_list = [torch.FloatTensor([p]).to(self.device) for p in properties_list]
                return [smiles_list, fingerprints_list, adjacencies_list, molecular_sizes_list, additional_features_list, properties_list]
            else:
                properties_list = [torch.FloatTensor([p]).to(self.device) for p in properties_list]
                return [smiles_list, fingerprints_list, adjacencies_list, molecular_sizes_list, additional_features_list, properties_list]

    def save_MAEs(self, MAEs, filename):
        with open(filename, 'a') as f:
            f.write(MAEs + '\n')
            
    def save_predictions(self, predictions, filename):
        with open(filename, 'w') as f:
            f.write('Smiles\tCorrect\tPredict\n')
            f.write(predictions + '\n')
            
    def save_model(self, model, filename):
        torch.save(model.state_dict(), filename)
