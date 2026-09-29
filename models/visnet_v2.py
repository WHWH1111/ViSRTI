import torch
import torch.nn as nn
from .visnet_core import ViSNet


class VisNetV2(nn.Module):
    """
    多模态RT预测器 - 简化版
    只输出单个维度的RT值
    支持逐步添加特征的模块化设计
    """
    
    def __init__(self, 
                 # 图网络参数
                 node_feature_dim: int = 64,
                 
                 # 各模态特征维度
                 physchem_feature_dim: int = 4,  # 物化特征维度为4
                 toxicity_feature_dim: int = 4,
                 chromato_feature_dim: int = 2,
                 
                 # 隐藏层维度
                 graph_hidden_dim: int = 512,
                 physchem_hidden_dim: int = 64,
                 toxicity_hidden_dim: int = 64,
                 chromato_hidden_dim: int = 32,
                 
                 # 中间层维度
                 toxicity_intermediate_dim: int = 32,  # 毒性特征处理的中间维度
                 
                 # 融合层维度
                 fusion_hidden_dims: list = [512, 256, 128],
                 dropout_rate: float = 0.3,
                 
                 # 注意力机制参数
                 use_attention: bool = False,  # 默认关闭注意力机制
                 attention_heads: int = 4,
                 use_gating: bool = False,  # 默认关闭门控机制

                 # 输出头配置
                 output_head: str = 'nested_mlp',  # 'nested_mlp' or 'original_visnet'
                 
                 # 特征级别参数
                 feature_level: str = 'all',  # 'graph', 'graph_physchem', 'graph_physchem_toxicity', 'all'
                 
                 # 物化特征掩码，用于控制特定特征的使用
                 physchem_feature_mask: list = None,  # 默认None表示使用所有特征
                 
                 # 毒性特征掩码，用于控制特定特征的使用
                 toxicity_feature_mask: list = None,  # 默认None表示使用所有特征
                 
                 # 色谱特征掩码，用于控制特定特征的使用
                 chromato_feature_mask: list = None):  # 默认None表示使用所有特征
        
        super(VisNetV2, self).__init__()
        
        # 保存配置参数
        self.use_attention = use_attention
        self.use_gating = use_gating
        self.feature_level = feature_level
        if output_head not in ('nested_mlp', 'original_visnet'):
            raise ValueError(f"Unsupported output_head: {output_head}")
        if output_head == 'original_visnet' and feature_level != 'graph':
            raise ValueError("The original ViSNet output head supports graph-only input")
        self.output_head = output_head
        
        # 保存物化特征掩码
        self.physchem_feature_mask = physchem_feature_mask
        # 如果提供了掩码，则计算实际使用的特征维度
        if physchem_feature_mask is not None:
            self.physchem_effective_dim = sum(physchem_feature_mask)
        else:
            self.physchem_effective_dim = physchem_feature_dim
            
        # 保存毒性特征掩码
        self.toxicity_feature_mask = toxicity_feature_mask
        # 如果提供了掩码，则计算实际使用的特征维度
        if toxicity_feature_mask is not None:
            self.toxicity_effective_dim = sum(toxicity_feature_mask)
        else:
            self.toxicity_effective_dim = toxicity_feature_dim
            
        # 保存色谱特征掩码
        self.chromato_feature_mask = chromato_feature_mask
        # 如果提供了掩码，则计算实际使用的特征维度
        if chromato_feature_mask is not None:
            self.chromato_effective_dim = sum(chromato_feature_mask)
        else:
            self.chromato_effective_dim = chromato_feature_dim
            
        # 1. 图结构特征提取器 (VisNet) - 始终需要
        self.visnet = ViSNet(
            hidden_channels=128,
            num_layers=6, 
            num_rbf=32,
            cutoff=5.0
        )
        
        # 保存各特征维度以便后续使用
        self.physchem_feature_dim = physchem_feature_dim
        self.toxicity_feature_dim = toxicity_feature_dim
        self.chromato_feature_dim = chromato_feature_dim
        self.physchem_hidden_dim = physchem_hidden_dim
        self.toxicity_hidden_dim = toxicity_hidden_dim
        self.chromato_hidden_dim = chromato_hidden_dim
        
        # 如果未指定毒性特征中间维度，则默认使用毒性特征的有效维度
        if toxicity_intermediate_dim is None:
            self.toxicity_intermediate_dim = self.toxicity_effective_dim
        else:
            self.toxicity_intermediate_dim = toxicity_intermediate_dim

        # 2. 图特征处理器 - 始终需要
        if self.output_head == 'nested_mlp':
            self.graph_processor = nn.Sequential(
                nn.Linear(64, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
                
                nn.Linear(256, graph_hidden_dim),
                nn.BatchNorm1d(graph_hidden_dim),
                nn.ReLU()
            )
        else:
            self.graph_processor = None
        
        # 3. 物理化学特征处理器 - 根据特征级别决定是否创建
        if self.output_head == 'nested_mlp' and self.feature_level in ['graph_physchem', 'graph_physchem_toxicity', 'all']:
            # 使用有效维度创建处理器
            self.physchem_processor = nn.Sequential(
                nn.Linear(self.physchem_effective_dim, self.physchem_hidden_dim),
                nn.BatchNorm1d(self.physchem_hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
                
                nn.Linear(self.physchem_hidden_dim, self.physchem_hidden_dim),
                nn.ReLU()
            )
        else:
            self.physchem_processor = None
        
        # 4. 毒性特征处理器 - 根据特征级别决定是否创建
        if self.output_head == 'nested_mlp' and self.feature_level in ['graph_physchem_toxicity', 'all']:
            # 使用有效维度创建处理器
            self.toxicity_processor = nn.Sequential(
                # nn.Linear(self.toxicity_effective_dim, 32),
                # nn.BatchNorm1d(32),
                # nn.ReLU(),
                # nn.Linear(32, self.toxicity_hidden_dim),
                # nn.ReLU()
                
                nn.Linear(self.toxicity_effective_dim, self.toxicity_intermediate_dim),
                nn.BatchNorm1d(self.toxicity_intermediate_dim),
                nn.ReLU(),
                nn.Linear(self.toxicity_intermediate_dim, self.toxicity_hidden_dim),
                nn.ReLU()
            )
        else:
            self.toxicity_processor = None
        
        # 5. 色谱质谱特征处理器 - 根据特征级别决定是否创建
        if self.output_head == 'nested_mlp' and self.feature_level == 'all':
            # 使用有效维度创建处理器
            self.chromato_processor = nn.Sequential(
                nn.Linear(self.chromato_effective_dim, 16),
                nn.ReLU(),
                
                nn.Linear(16, self.chromato_hidden_dim),
                nn.ReLU()
            )
        else:
            self.chromato_processor = None
        
        # 6. 多模态注意力机制 - 根据配置决定是否创建
        if self.output_head == 'nested_mlp' and self.use_attention:
            self.modal_attention = MultiModalAttention(
                graph_dim=graph_hidden_dim,
                physchem_dim=physchem_hidden_dim if self.physchem_processor is not None else 0,
                toxicity_dim=toxicity_hidden_dim if self.toxicity_processor is not None else 0,
                chromato_dim=chromato_hidden_dim if self.chromato_processor is not None else 0,
                num_heads=attention_heads,
                dropout=dropout_rate
            )
        
        # 7. 门控融合机制 - 根据配置决定是否创建
        if self.output_head == 'nested_mlp' and self.use_gating:
            self.modal_gating = ModalGating(
                graph_dim=graph_hidden_dim,
                physchem_dim=physchem_hidden_dim if self.physchem_processor is not None else 0,
                toxicity_dim=toxicity_hidden_dim if self.toxicity_processor is not None else 0,
                chromato_dim=chromato_hidden_dim if self.chromato_processor is not None else 0
            )
        
        # 8. 特征融合层 - 支持不同组合的动态输入维度
        self.graph_dim = graph_hidden_dim
        self.physchem_dim = physchem_hidden_dim if self.physchem_processor is not None else 0
        self.toxicity_dim = toxicity_hidden_dim if self.toxicity_processor is not None else 0
        self.chromato_dim = chromato_hidden_dim if self.chromato_processor is not None else 0
        
        # 初始化融合网络但不构建，将在forward中动态构建
        self.fusion_hidden_dims = fusion_hidden_dims
        
        self.dropout_rate = dropout_rate
        self.fusion_net = None
        if self.output_head == 'nested_mlp':
            fusion_input_dim = self.get_fusion_net_input_dim()
            self._build_fusion_net(fusion_input_dim)
            self._last_feature_dim = fusion_input_dim
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """初始化权重"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
    
    def _build_fusion_net(self, input_dim):
        """动态构建融合网络"""
        if self.output_head != 'nested_mlp':
            raise RuntimeError("The original ViSNet output head does not use fusion_net")
        self.fusion_net = nn.Sequential(
            nn.Linear(input_dim, self.fusion_hidden_dims[0]),
            nn.BatchNorm1d(self.fusion_hidden_dims[0]),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate),
            
            nn.Linear(self.fusion_hidden_dims[0], self.fusion_hidden_dims[1]),
            nn.BatchNorm1d(self.fusion_hidden_dims[1]),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate * 0.5),
            
            nn.Linear(self.fusion_hidden_dims[1], self.fusion_hidden_dims[2]),
            nn.ReLU(),
            
            nn.Linear(self.fusion_hidden_dims[2], 1)  # 输出单个RT值
        )
        
        #  确保BatchNorm层在正确模式下
        if hasattr(self, 'training'):
            for module in self.fusion_net.modules():
                if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d)):
                    if self.training:
                        module.train()
                    else:
                        module.eval()
    
    def forward(self, z, pos, batch, physchem_features=None, toxicity_features=None, chromato_features=None):
        """
        前向传播 - 输出单个RT值
        
        Args:
            z: 原子序数 [num_atoms]
            pos: 原子3D坐标 [num_atoms, 3]
            batch: 批次索引 [num_atoms]
            physchem_features: 物理化学特征 [batch_size, physchem_feature_dim]
            toxicity_features: 毒性特征 [batch_size, 4] 
            chromato_features: 色谱特征 [batch_size, 2]
            
        Returns:
            rt_pred: RT预测值 [batch_size]
        """
        # 确保所有输入张量在同一设备上
        device = z.device
        
        # 1. 图结构特征提取
        visnet_prediction, graph_rep = self.visnet(z, pos, batch)  # [batch_size, 1], [batch_size, 64]
        if self.output_head == 'original_visnet':
            return visnet_prediction.squeeze(-1), None
        graph_rep = graph_rep.to(device)
        graph_features = self.graph_processor(graph_rep)  # [batch_size, 512]
        return self.forward_from_graph_features(
            graph_features, physchem_features, toxicity_features, chromato_features)

    def forward_from_graph_features(
            self, graph_features, physchem_features=None,
            toxicity_features=None, chromato_features=None):
        """Evaluate auxiliary features with an already computed graph embedding."""
        if self.output_head != 'nested_mlp':
            raise ValueError('Cached graph features require the nested MLP head')
        device = graph_features.device
        
        # 初始化特征列表
        features_list = [graph_features]
        feature_dims = [self.graph_dim]
        
        # 2. 物理化学特征处理 - 如果提供了且处理器存在则使用
        if physchem_features is not None and self.physchem_processor is not None:
            physchem_features = physchem_features.to(device)
            # 应用特征掩码（如果提供了掩码）
            if self.physchem_feature_mask is not None and physchem_features.shape[1] != self.physchem_effective_dim:
                # 使用布尔索引选择特征
                mask = torch.tensor(self.physchem_feature_mask, dtype=torch.bool, device=device)
                physchem_features = physchem_features[:, mask]
            physchem_processed = self.physchem_processor(physchem_features)  # [batch_size, 64]
            features_list.append(physchem_processed)
            feature_dims.append(self.physchem_dim)
        
        # 3. 毒性特征处理 - 如果提供了且处理器存在则使用
        if toxicity_features is not None and self.toxicity_processor is not None:
            toxicity_features = toxicity_features.to(device)
            # 应用特征掩码（如果提供了掩码）
            if self.toxicity_feature_mask is not None:
                # 使用布尔索引选择特征
                mask = torch.tensor(self.toxicity_feature_mask, dtype=torch.bool, device=device)
                toxicity_features = toxicity_features[:, mask]
            toxicity_processed = self.toxicity_processor(toxicity_features)  # [batch_size, 64]
            features_list.append(toxicity_processed)
            feature_dims.append(self.toxicity_dim)
        
        # 4. 色谱特征处理 - 如果提供了且处理器存在则使用
        if chromato_features is not None and self.chromato_processor is not None:
            chromato_features = chromato_features.to(device)
            # 应用特征掩码（如果提供了掩码）
            if self.chromato_feature_mask is not None:
                # 使用布尔索引选择特征
                mask = torch.tensor(self.chromato_feature_mask, dtype=torch.bool, device=device)
                chromato_features = chromato_features[:, mask]
            chromato_processed = self.chromato_processor(chromato_features)  # [batch_size, 32]
            features_list.append(chromato_processed)
            feature_dims.append(self.chromato_dim)
        
        # 5. 应用注意力机制动态调节特征贡献（如果启用了注意力机制且有多种特征）
        if self.use_attention and len(features_list) > 1:
            # 动态填充缺失的模态
            all_features = [features_list[0]]  # 图特征总是存在
            modal_inputs = [features_list[0]]  # 用于注意力网络的实际输入
            
            # 添加物化特征
            if len(features_list) > 1:
                modal_inputs.append(features_list[1])
                all_features.append(features_list[1])
            else:
                # 只有当physchem_processor存在时才填充零张量
                if self.physchem_processor is not None:
                    all_features.append(torch.zeros_like(features_list[0][:, :1], device=device))
                else:
                    all_features.append(None)
                
            # 添加毒性特征
            if len(features_list) > 2:
                modal_inputs.append(features_list[2])
                all_features.append(features_list[2])
            else:
                # 只有当toxicity_processor存在时才填充零张量
                if self.toxicity_processor is not None:
                    all_features.append(torch.zeros_like(features_list[0][:, :1], device=device))
                else:
                    all_features.append(None)
                
            # 添加色谱特征
            if len(features_list) > 3:
                modal_inputs.append(features_list[3])
                all_features.append(features_list[3])
            else:
                # 只有当chromato_processor存在时才填充零张量
                if self.chromato_processor is not None:
                    all_features.append(torch.zeros_like(features_list[0][:, :1], device=device))
                else:
                    all_features.append(None)
            
            # 只将存在的特征传入注意力网络
            attended_out = self.modal_attention(*[f for f in modal_inputs if f is not None])
            
            # 将注意力输出映射回原始特征列表
            attended_features_list = []
            attended_index = 0
            for feature in all_features:
                if feature is not None:
                    attended_features_list.append(attended_out[attended_index])
                    attended_index += 1
                else:
                    attended_features_list.append(None)
            
            # 过滤掉None值，只保留实际存在的特征
            features_list = [f for f in attended_features_list if f is not None]
        
        # 6. 应用门控机制进一步调节特征融合（如果启用了门控机制且有多种特征）
        if self.use_gating and len(features_list) > 1:
            # 直接传入所有存在的特征，由ModalGating内部处理缺失情况
            combined_features = self.modal_gating(*features_list)
        else:
            # 原始的特征拼接方式
            combined_features = torch.cat(features_list, dim=1)  # 根据实际提供的特征拼接
        
        # 7. 动态构建回归网络（如果尚未构建或输入维度发生变化）
        total_feature_dim = sum(feature_dims)
        if self.fusion_net is None or hasattr(self, '_last_feature_dim') and self._last_feature_dim != total_feature_dim:
            self._build_fusion_net(total_feature_dim)
            self._last_feature_dim = total_feature_dim
            
            # 确保fusion_net在正确的设备上
            self.fusion_net = self.fusion_net.to(device)
        
        # 8. 回归预测
        rt_pred = self.fusion_net(combined_features).squeeze(-1)  # [batch_size]
        
        # 返回两个值以匹配trainer_tester.py的期望格式
        return rt_pred, None

    def get_fusion_net_input_dim(self):
        """
        获取fusion_net的输入维度，用于确保模型保存时fusion_net已正确构建
        """
        # 计算特征维度
        graph_dim = self.graph_dim
        physchem_dim = self.physchem_dim if self.physchem_processor is not None else 0
        toxicity_dim = self.toxicity_dim if self.toxicity_processor is not None else 0
        chromato_dim = self.chromato_dim if self.chromato_processor is not None else 0
        
        return graph_dim + physchem_dim + toxicity_dim + chromato_dim


class MultiModalAttention(nn.Module):
    """
    多模态注意力机制
    动态学习不同模态特征的重要性权重
    """
    
    def __init__(self, graph_dim, physchem_dim, toxicity_dim, chromato_dim, num_heads=4, dropout=0.1):
        super(MultiModalAttention, self).__init__()
        
        self.num_heads = num_heads
        # 只包含实际存在的模态维度
        self.modal_dims = [dim for dim in [graph_dim, physchem_dim, toxicity_dim, chromato_dim] if dim > 0]
        self.total_dim = sum(self.modal_dims)
        
        # 多头注意力层
        self.attention = nn.MultiheadAttention(
            embed_dim=self.total_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # 层归一化
        self.layer_norm = nn.LayerNorm(self.total_dim)
        
        # 模态特定的投影层（可选，用于统一维度）- 只为实际存在的模态创建
        dim_index = 0
        if graph_dim > 0:
            self.graph_proj = nn.Linear(graph_dim, graph_dim)
            dim_index += 1
            
        if physchem_dim > 0:
            self.physchem_proj = nn.Linear(physchem_dim, physchem_dim)
            dim_index += 1
            
        if toxicity_dim > 0:
            self.toxicity_proj = nn.Linear(toxicity_dim, toxicity_dim)
            dim_index += 1
            
        if chromato_dim > 0:
            self.chromato_proj = nn.Linear(chromato_dim, chromato_dim)
        
        # 注意力权重dropout
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, *modal_features):
        batch_size = modal_features[0].size(0)
        
        # 投影特征 - 只处理实际传入且有对应投影层的特征
        projected_features = []
        for i, feat in enumerate(modal_features):
            if i == 0 and hasattr(self, 'graph_proj'):
                projected_features.append(self.graph_proj(feat))
            elif i == 1 and hasattr(self, 'physchem_proj'):
                projected_features.append(self.physchem_proj(feat))
            elif i == 2 and hasattr(self, 'toxicity_proj'):
                projected_features.append(self.toxicity_proj(feat))
            elif i == 3 and hasattr(self, 'chromato_proj'):
                projected_features.append(self.chromato_proj(feat))
            else:
                projected_features.append(feat)
        
        # 拼接所有模态特征 [batch_size, total_dim]
        combined_features = torch.cat(projected_features, dim=1)
        
        # 重塑为序列形式 [batch_size, 1, total_dim]
        query = key = value = combined_features.unsqueeze(1)
        
        # 应用多头注意力
        attended_features, attention_weights = self.attention(
            query, key, value, 
            need_weights=True
        )
        attended_features = attended_features.squeeze(1)  # [batch_size, total_dim]
        
        # 残差连接和层归一化
        attended_features = self.layer_norm(combined_features + attended_features)
        
        # 应用dropout
        attended_features = self.dropout(attended_features)
        
        # 重新分割回各模态特征
        output_features = []
        start_idx = 0
        for dim in self.modal_dims:
            output_features.append(attended_features[:, start_idx:start_idx+dim])
            start_idx += dim
        
        return tuple(output_features)


class ModalGating(nn.Module):
    """
    模态门控机制
    使用门控单元动态控制各模态特征的融合权重
    """
    
    def __init__(self, graph_dim, physchem_dim, toxicity_dim, chromato_dim):
        super(ModalGating, self).__init__()
        
        # 只包含实际存在的模态维度
        self.modal_dims = [dim for dim in [graph_dim, physchem_dim, toxicity_dim, chromato_dim] if dim > 0]
        self.total_dim = sum(self.modal_dims)
        
        if self.total_dim == 0:
            raise ValueError("At least one modal dimension must be greater than 0")
        
        # 门控权重生成网络
        self.gate_network = nn.Sequential(
            nn.Linear(self.total_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, len(self.modal_dims)),  # 模态数量动态确定
            nn.Softmax(dim=1)   # 确保权重和为1
        )
        
        # 模态特定的变换网络（可选）- 只为实际存在的模态创建
        dim_index = 0
        if graph_dim > 0:
            self.graph_transform = nn.Linear(graph_dim, graph_dim)
            dim_index += 1
            
        if physchem_dim > 0:
            self.physchem_transform = nn.Linear(physchem_dim, physchem_dim)
            dim_index += 1
            
        if toxicity_dim > 0:
            self.toxicity_transform = nn.Linear(toxicity_dim, toxicity_dim)
            dim_index += 1
            
        if chromato_dim > 0:
            self.chromato_transform = nn.Linear(chromato_dim, chromato_dim)
        
    def forward(self, *modal_features):
        device = modal_features[0].device
        
        # 拼接所有原始特征用于门控计算
        combined_for_gate = torch.cat(modal_features, dim=1)
        
        # 计算各模态的融合权重 [batch_size, num_modals]
        gate_weights = self.gate_network(combined_for_gate)
        
        # 应用门控权重到对应的特征
        weighted_features = []
        for i, feat in enumerate(modal_features):
            weight = gate_weights[:, i:i+1]  # [batch_size, 1]
            weighted_features.append(feat * weight)
        
        # 拼接加权后的特征
        fused_features = torch.cat(weighted_features, dim=1)
        
        return fused_features


class CrossModalAttention(nn.Module):
    """
    跨模态注意力机制（可选扩展）
    允许不同模态之间进行交互
    """
    
    def __init__(self, modal_dims, hidden_dim=256, num_heads=4):
        super(CrossModalAttention, self).__init__()
        
        self.num_modals = len(modal_dims)
        self.modal_dims = modal_dims
        self.hidden_dim = hidden_dim
        
        # 为每个模态创建投影层
        self.modal_projections = nn.ModuleList([
            nn.Linear(dim, hidden_dim) for dim in modal_dims
        ])
        
        # 跨模态注意力层
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            batch_first=True
        )
        
        # 输出投影层
        self.output_projections = nn.ModuleList([
            nn.Linear(hidden_dim, dim) for dim in modal_dims
        ])
        
    def forward(self, *modal_features):
        assert len(modal_features) == self.num_modals
        
        # 投影到统一维度
        projected_features = []
        for i, feat in enumerate(modal_features):
            projected = self.modal_projections[i](feat).unsqueeze(1)  # [batch_size, 1, hidden_dim]
            projected_features.append(projected)
        
        # 拼接所有模态特征 [batch_size, num_modals, hidden_dim]
        all_features = torch.cat(projected_features, dim=1)
        
        # 应用跨模态注意力
        attended_features, _ = self.cross_attention(
            all_features, all_features, all_features
        )
        
        # 分割并投影回原始维度
        output_features = []
        for i in range(self.num_modals):
            modal_feat = attended_features[:, i, :]  # [batch_size, hidden_dim]
            output_feat = self.output_projections[i](modal_feat)  # [batch_size, original_dim]
            output_features.append(output_feat)
        
        return tuple(output_features)
