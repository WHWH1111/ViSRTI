#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VisNetV2 模型专用训练脚本

针对 VisNetV2 模型进行简化和优化的训练脚本
"""

import sys
import os
import random
import copy
import csv
import json
import shutil

# 获取当前文件的目录并将其添加到sys.path中，以确保可以进行绝对导入
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

try:
    import timeit
    import numpy as np
    import torch
    import torch.optim as optim
    from tqdm import tqdm
    import argparse

    # 导入工具函数
    from utils.plot_utils import plot_training_metrics, plot_predicted_vs_actual
    from utils.log_utils import (print_training_setup_info, print_preprocess_info, 
                                print_model_info, print_training_start_info,
                                print_training_time_estimate, print_training_header,
                                print_early_stopping_info, create_log_dir,
                                save_training_params)
    from utils.feature_utils import get_feature_config

    # 导入VisNetV2模型
    from models.visnet_v2 import VisNetV2

    # 导入训练器和测试器以及数据预处理模块
    from core import (
        Trainer, 
        Tester, 
        split_raw_data_if_needed,
        load_datasets,
        standardize_datasets_if_needed,
        preprocess_visnet_data
    )
    
    # 导入分子缓存
    from utils.molecule import MoleculeCache
except ImportError as e:
    print(f"Error importing modules: {e}")
    sys.exit(1)
    
def _ensure_model_complete(model, device, args):
    """
    确保模型的所有部分都已正确构建，特别是fusion_net
    """
    if model.output_head != 'nested_mlp':
        return

    if model.fusion_net is not None:
        model.fusion_net = model.fusion_net.to(device)
        return

    # 获取fusion_net的输入维度
    fusion_input_dim = model.get_fusion_net_input_dim()
    
    # 直接构建fusion_net而不需要前向传播
    model._build_fusion_net(fusion_input_dim)
    model._last_feature_dim = fusion_input_dim
    model.fusion_net = model.fusion_net.to(device)
    
    # 确保BatchNorm层在正确模式下
    for module in model.fusion_net.modules():
        if isinstance(module, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d)):
            if model.training:
                module.train()
            else:
                module.eval()


def _print_model_info(model, prefix=""):
    """
    打印模型关键部分的信息用于调试
    """
    print(f"{prefix} === Model Info ===")
    print(f"{prefix} Model device: {next(model.parameters()).device}")
    print(f"{prefix} fusion_net is None: {model.fusion_net is None}")
    if model.fusion_net is not None:
        print(f"{prefix} fusion_net type: {type(model.fusion_net)}")
        print(f"{prefix} fusion_net layers: {len(list(model.fusion_net.modules()))}")
        # 打印fusion_net的部分参数
        for name, param in model.fusion_net.named_parameters():
            if param.requires_grad:
                print(f"{prefix} fusion_net.{name}: shape={param.shape}, mean={param.mean().item():.6f}, std={param.std().item():.6f}")
                break  # 只打印第一个参数以减少输出
    
    print(f"{prefix} graph_processor type: {type(model.graph_processor)}")
    print(f"{prefix} physchem_processor is None: {model.physchem_processor is None}")
    if model.physchem_processor is not None:
        print(f"{prefix} physchem_processor type: {type(model.physchem_processor)}")
    
    print(f"{prefix} toxicity_processor is None: {model.toxicity_processor is None}")
    if model.toxicity_processor is not None:
        print(f"{prefix} toxicity_processor type: {type(model.toxicity_processor)}")
        
    print(f"{prefix} chromato_processor is None: {model.chromato_processor is None}")
    if model.chromato_processor is not None:
        print(f"{prefix} chromato_processor type: {type(model.chromato_processor)}")
    
    # 打印visnet的部分参数
    for name, param in model.visnet.named_parameters():
        if param.requires_grad:
            print(f"{prefix} visnet.{name}: shape={param.shape}, mean={param.mean().item():.6f}, std={param.std().item():.6f}")
            break  # 只打印第一个参数以减少输出
    print(f"{prefix} === End Model Info ===")


def get_model(args, device):
    """初始化VisNetV2模型"""
    model = VisNetV2(
        node_feature_dim=args.visnet_v2_node_feature_dim,
        physchem_feature_dim=args.visnet_v2_physchem_feature_dim,
        toxicity_feature_dim=args.visnet_v2_toxicity_feature_dim,
        chromato_feature_dim=args.visnet_v2_chromato_feature_dim,
        graph_hidden_dim=args.visnet_v2_graph_hidden_dim,
        physchem_hidden_dim=args.visnet_v2_physchem_hidden_dim,
        toxicity_hidden_dim=args.visnet_v2_toxicity_hidden_dim,
        chromato_hidden_dim=args.visnet_v2_chromato_hidden_dim,
        toxicity_intermediate_dim=args.visnet_v2_toxicity_intermediate_dim,
        fusion_hidden_dims=args.visnet_v2_fusion_hidden_dims,
        dropout_rate=args.visnet_v2_dropout_rate,
        use_attention=args.visnet_v2_use_attention,
        use_gating=args.visnet_v2_use_gating,
        output_head=args.visnet_v2_output_head,
        feature_level=args.visnet_v2_feature_level,
        physchem_feature_mask=args.visnet_v2_physchem_mask,
        toxicity_feature_mask=args.visnet_v2_toxicity_mask,
        chromato_feature_mask=args.visnet_v2_chromato_mask
    )
    
    # 添加特征级别属性以便在训练器中使用
    model.feature_level = args.visnet_v2_feature_level
    
    model.to(device)
    
    print(f"Using VisNetV2 model with parameters:")
    print(f"  node_feature_dim={args.visnet_v2_node_feature_dim}")
    print(f"  physchem_feature_dim={args.visnet_v2_physchem_feature_dim}")
    print(f"  toxicity_feature_dim={args.visnet_v2_toxicity_feature_dim}")
    print(f"  chromato_feature_dim={args.visnet_v2_chromato_feature_dim}")
    print(f"  graph_hidden_dim={args.visnet_v2_graph_hidden_dim}")
    print(f"  physchem_hidden_dim={args.visnet_v2_physchem_hidden_dim}")
    print(f"  toxicity_hidden_dim={args.visnet_v2_toxicity_hidden_dim}")
    print(f"  chromato_hidden_dim={args.visnet_v2_chromato_hidden_dim}")
    print(f"  toxicity_intermediate_dim={args.visnet_v2_toxicity_intermediate_dim}")
    print(f"  fusion_hidden_dims={args.visnet_v2_fusion_hidden_dims}")
    print(f"  dropout_rate={args.visnet_v2_dropout_rate}")
    print(f"  use_attention={args.visnet_v2_use_attention}")
    print(f"  use_gating={args.visnet_v2_use_gating}")
    print(f"  output_head={args.visnet_v2_output_head}")
    print(f"  feature_level={args.visnet_v2_feature_level}")
    print(f"  physchem_feature_mask={args.visnet_v2_physchem_mask}")
    print(f"  toxicity_feature_mask={args.visnet_v2_toxicity_mask}")
    print(f"  chromato_feature_mask={args.visnet_v2_chromato_mask}")
    
    return model


def main():
    # 添加命令行参数解析
    parser = argparse.ArgumentParser(description='Train VisNetV2 Molecular Graph Neural Network')
    
    # 通用参数
    parser.add_argument('-ds', '--debug-size', type=int, default=None, 
                        help='Number of samples to use for debugging (default: None, use all data)')
    parser.add_argument('-it', '--iteration', type=int, default=150,
                        help='Number of training iterations (default: 150)')
    parser.add_argument('--batch-train', type=int, default=256,
                        help='Training batch size (default: 256)')
    parser.add_argument('--batch-test', type=int, default=128,
                        help='Test batch size (default: 128)')
    parser.add_argument('--standardize', action='store_true',
                        help='Whether to standardize the property values')
    parser.add_argument('--standardize-features', action='store_true',
                        help='Whether to standardize additional features for VisNetV2 model')
    parser.add_argument('--dataname', type=str, default='MMF_GNN_pos',
                        choices=['MMF_GNN_pos', 'MMF_GNN_neg'],
                        help='Ionization-mode dataset to train (default: MMF_GNN_pos)')
    parser.add_argument('--split-strategy', type=str, default='scaffold',
                        choices=['scaffold', 'random'],
                        help='Train/dev/test split strategy (default: scaffold)')
    parser.add_argument('--split-seed', type=int, default=1234,
                        help='Fixed seed used for data splitting and training (default: 1234)')
    parser.add_argument('--dev-split-seed', type=int, default=None,
                        help='Inner train/development seed; defaults to 42 for random splits and --split-seed for scaffold splits')
    parser.add_argument('--force-resplit', action='store_true',
                        help='Regenerate split files even when their manifest matches')
    parser.add_argument('--data-root', type=str, default='./data/MMF-3/',
                        help='Root containing the raw CSV and generated split directory')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Exact directory for this run artifacts; defaults to a timestamped log directory')
    
    # 模型选择参数（虽然我们只支持visnet_v2，但为了兼容load_datasets函数需要保留）
    parser.add_argument('--model', type=str, default='visnet_v2', choices=['visnet_v2'],
                        help='Model type to use (default: visnet_v2)')
    
    # 优化器和调度器参数
    parser.add_argument('--learning-rate', type=float, default=1e-4,
                        help='Initial learning rate (default: 1e-4)')
    parser.add_argument('--weight-decay', type=float, default=1e-5,
                        help='Weight decay (L2 penalty) (default: 1e-5)')
    parser.add_argument('--scheduler-patience', type=int, default=10,
                        help='Scheduler patience (default: 10)')
    parser.add_argument('--scheduler-factor', type=float, default=0.5,
                        help='Scheduler factor (default: 0.5)')
    parser.add_argument('--min-lr', type=float, default=1e-6,
                        help='Minimum learning rate (default: 1e-6)')
    parser.add_argument('--early-stopping-patience', type=int, default=20,
                        help='Early stopping patience (default: 20)')
    
    # 损失函数参数
    parser.add_argument('--loss-function', type=str, default='mse', choices=['mse', 'l1'],
                        help='Loss function to use (default: mse)')
    
    # 内存优化参数
    parser.add_argument('--max-split-size-mb', type=int, default=None,
                        help='PyTorch max_split_size_mb to avoid memory fragmentation')
    
    # VisNetV2 模型参数
    parser.add_argument('--visnet-v2-node-feature-dim', type=int, default=64,
                         help='Node feature dimension for VisNetV2 model')
    parser.add_argument('--visnet-v2-physchem-feature-dim', type=int, default=4,
                         help='Input dimension of physical chemistry features for VisNetV2 model (default: 4)')
    parser.add_argument('--visnet-v2-toxicity-feature-dim', type=int, default=4,
                         help='Toxicity feature dimension for VisNetV2 model')
    parser.add_argument('--visnet-v2-chromato-feature-dim', type=int, default=2,
                         help='Chromatography feature dimension for VisNetV2 model')
    parser.add_argument('--visnet-v2-graph-hidden-dim', type=int, default=512,
                         help='Graph hidden dimension for VisNetV2 model')
    parser.add_argument('--visnet-v2-physchem-hidden-dim', type=int, default=128,
                         help='Physical chemistry hidden dimension for VisNetV2 model')
    parser.add_argument('--visnet-v2-toxicity-hidden-dim', type=int, default=128,   # INFO pos-64 / neg-128
                         help='Toxicity hidden dimension for VisNetV2 model')
    parser.add_argument('--visnet-v2-chromato-hidden-dim', type=int, default=32,
                         help='Chromatography hidden dimension for VisNetV2 model')
    parser.add_argument('--visnet-v2-toxicity-intermediate-dim', type=int, default=64,
                         help='Toxicity intermediate dimension for VisNetV2 model (default: None, use effective dim)')
    parser.add_argument('--visnet-v2-fusion-hidden-dims', type=int, nargs='+', default=[512, 256, 128],
                         help='Fusion hidden dimensions for VisNetV2 model')
    parser.add_argument('--visnet-v2-dropout-rate', type=float, default=0.0,
                         help='Dropout rate for VisNetV2 model')
    parser.add_argument('--visnet-v2-use-attention', action='store_true',
                        help='Whether to use attention mechanism in VisNetV2')
    parser.add_argument('--visnet-v2-use-gating', action='store_true',
                        help='Whether to use gating mechanism in VisNetV2')
    parser.add_argument('--visnet-v2-output-head', type=str, default='nested_mlp',
                        choices=['nested_mlp', 'original_visnet'],
                        help='Regression head used after the ViSNet backbone')
    parser.add_argument('--visnet-v2-feature-level', type=str, default='all',
                        choices=['graph', 'graph_physchem', 'graph_physchem_toxicity', 'all'],
                        help='Feature level for VisNetV2 model')
    parser.add_argument('--visnet-v2-physchem-mask', type=int, nargs='+', default=None,
                        help='Physical-chemical feature mask (1 for used, 0 for masked). '
                             'Example: "--visnet-v2-physchem-mask 1 0 1 1" to exclude the second feature (e.g. KoC)')
    parser.add_argument('--visnet-v2-toxicity-mask', type=int, nargs='+', default=None,
                        help='Toxicity feature mask (1 for used, 0 for masked). '
                             'Example: "--visnet-v2-toxicity-mask 1 0 1 1" to exclude the second feature')
    parser.add_argument('--visnet-v2-chromato-mask', type=int, nargs='+', default=None,
                        help='Chromatography feature mask (1 for used, 0 for masked). '
                             'Example: "--visnet-v2-chromato-mask 1 0" to exclude the second feature')
    parser.add_argument('--development-only', action='store_true',
                        help='Use an existing split, train and validate without loading test responses')

    args = parser.parse_args()
    if args.dev_split_seed is None:
        args.dev_split_seed = 42 if args.split_strategy == 'random' else args.split_seed
    
    # 设置PyTorch内存管理参数以避免内存碎片
    if args.max_split_size_mb is not None:
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = f'max_split_size_mb:{args.max_split_size_mb}'
        print(f"Set PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:{args.max_split_size_mb}")
    
    # 固定随机种子，并关闭与确定性执行冲突的cuDNN自动调优
    random.seed(args.split_seed)
    np.random.seed(args.split_seed)
    torch.manual_seed(args.split_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.split_seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    # 启用CUDA内存优化
    if torch.cuda.is_available():
        # 设置环境变量以减少内存碎片
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
    
    # 数据集配置
    path = os.path.abspath(args.data_root)
    dataname = args.dataname
    _visnet_train = "visnet_train_v2_" + dataname
    _visnet_test = "visnet_test_v2_" + dataname
    
    # 创建基于时间和参数的目录
    run_name = '{}_{}_{}_seed{}'.format(
        dataname, args.visnet_v2_feature_level, args.split_strategy, args.split_seed
    )
    file_paths = create_log_dir(
        path, run_name, 48, 6, 6, args.batch_train, args.learning_rate,
        args.iteration, args.debug_size, output_dir=args.output_dir
    )
    log_dir = file_paths['log_dir']
    file_MAEs = file_paths['file_MAEs']
    file_test_result = file_paths['file_test_result']
    file_predictions = file_paths['file_predictions']
    file_model = file_paths['file_model']
    loss_plot_file = file_paths['loss_plot_file']
    cp_plot_file = file_paths['cp_plot_file']
    timestamp = file_paths['timestamp']
    
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print_training_setup_info(device)
    else:
        device = torch.device('cpu')
        print_training_setup_info(device)
    
    # 检查训练集和测试集文件是否存在，如果不存在则从原始文件中拆分
    if args.development_only:
        if args.force_resplit:
            raise ValueError('Development-only runs require an existing frozen split')
        with open(os.path.join(path, dataname, 'split_manifest.json'), encoding='utf-8') as handle:
            frozen_split = json.load(handle)
        if (frozen_split['split_strategy'] != args.split_strategy
                or frozen_split['random_seed'] != args.split_seed):
            raise ValueError('Development-only split differs from the requested configuration')
        import hashlib
        with open(os.path.join(path, dataname, dataname + '_train_set.txt'), 'rb') as handle:
            train_hash = hashlib.sha256(handle.read()).hexdigest()
        if train_hash != frozen_split['train_sha256']:
            raise ValueError('Frozen training partition has changed')
        reference = os.path.join(path, dataname, 'development_partition_assignments.csv')
        args.effective_partition_reference = reference if os.path.exists(reference) else None
        if args.effective_partition_reference:
            with open(reference, encoding='utf-8-sig', newline='') as handle:
                assignments = list(csv.DictReader(handle))
            args.effective_partition_smiles = {
                subset: {row['smiles'] for row in assignments if row['subset'] == subset}
                for subset in ('train', 'development')
            }
    else:
        train_file_path, test_file_path = split_raw_data_if_needed(
            path,
            dataname,
            target_column_name='Pred_RTI_Negative_ESI' if 'neg' in dataname else 'Pred_RTI_Positive_ESI',
            split_strategy=args.split_strategy,
            random_state=args.split_seed,
            force_resplit=args.force_resplit,
        )
    
    # 根据是否为调试模式决定加载的数据量
    max_data = args.debug_size if args.debug_size is not None else None
    
    # 加载数据集
    print("Loading datasets for VisNetV2 model...")
    dataset_train, dataset_dev, dataset_test, _, additional_features_mean, additional_features_std = load_datasets(
        args, path, dataname, max_data)
    
    # 标准化数据（如果需要）
    dataset_train, dataset_dev, dataset_test, property_mean, property_std = standardize_datasets_if_needed(
        args, dataset_train, dataset_dev, dataset_test, 
        additional_features_mean=additional_features_mean,
        additional_features_std=additional_features_std)
    print("Datasets loaded.", property_mean, property_std)
    
    # VisNetV2模型需要特殊的预处理
    print("Preprocessing VisNetV2-specific data...")
    preprocess_result = preprocess_visnet_data(args, dataset_train, dataset_dev, dataset_test, dataname, _visnet_train, _visnet_test)
    
    # 根据返回值数量处理结果
    if len(preprocess_result) == 7:
        # 包含标准化信息
        train_preprocessed_data, dev_preprocessed_data, test_preprocessed_data, \
        dataset_train_filtered, dataset_dev_filtered, dataset_test_filtered, standardization_info = preprocess_result
    else:
        # 不包含标准化信息
        train_preprocessed_data, dev_preprocessed_data, test_preprocessed_data, \
        dataset_train_filtered, dataset_dev_filtered, dataset_test_filtered = preprocess_result
        standardization_info = None
    
    # 根据特征级别过滤预处理数据中的特征
    for data_dict in [train_preprocessed_data, dev_preprocessed_data, test_preprocessed_data]:
        for smiles in data_dict:
            if args.visnet_v2_feature_level == 'graph':
                # 只保留图特征，将其他特征设置为None
                data_dict[smiles]['physchem_features'] = None
                data_dict[smiles]['toxicity_features'] = None
                data_dict[smiles]['chromato_features'] = None
            elif args.visnet_v2_feature_level == 'graph_physchem':
                # 保留图特征和物化特征，将其他特征设置为None
                data_dict[smiles]['toxicity_features'] = None
                data_dict[smiles]['chromato_features'] = None
            elif args.visnet_v2_feature_level == 'graph_physchem_toxicity':
                # 保留图特征、物化特征和毒性特征，将色谱特征设置为None
                data_dict[smiles]['chromato_features'] = None
            # 如果是'all'级别，则保留所有特征
    
    # 使用过滤后的数据集
    dataset_train = dataset_train_filtered
    dataset_dev = dataset_dev_filtered
    dataset_test = dataset_test_filtered

    partition_file = os.path.join(log_dir, 'partition_assignments.csv')
    with open(partition_file, 'w', encoding='utf-8', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(['subset', 'smiles'])
        for subset, dataset in (
                ('train', dataset_train), ('development', dataset_dev),
                ('test', dataset_test)):
            for record in dataset:
                writer.writerow([subset, record[0]])
    shutil.copyfile(
        os.path.join(path, dataname, 'split_manifest.json'),
        os.path.join(log_dir, 'split_manifest.json'),
    )
    
    # 打印预处理信息
    print_preprocess_info(dataset_train, dataset_dev, dataset_test)
    
    # 初始化VisNetV2模型
    model = get_model(args, device)
    # The optimizer captures the parameters that exist at construction time.
    # Build the selected regression head first so its parameters are trained.
    _ensure_model_complete(model, device, args)
    
    # 创建训练器和测试器，使用更明确的缓存命名
    trainer_cache_name = f"{_visnet_train}_trainer"
    tester_cache_name = f"{_visnet_test}_tester"
    dev_tester_cache_name = f"{_visnet_train}_dev_tester"
    train_tester_cache_name = f"{_visnet_train}_train_tester"
    
    trainer = Trainer(model, visnet=True, batch_train=args.batch_train, device=device, loss_function=args.loss_function, MoleculeCacheName=trainer_cache_name)
    tester = Tester(model, visnet=True, batch_test=args.batch_test, device=device, name="base-tester", MoleculeCacheName=tester_cache_name)
    
    # 设置预处理数据
    trainer.set_preprocessed_data(train_preprocessed_data)
    # 为训练集创建专用测试器
    train_tester = Tester(model, visnet=True, batch_test=args.batch_test, device=device, name="train-tester", MoleculeCacheName=train_tester_cache_name)
    train_tester.set_preprocessed_data(train_preprocessed_data)
    # 创建独立的开发集测试器
    dev_tester = Tester(model, visnet=True, batch_test=args.batch_test, device=device, name="dev-tester", MoleculeCacheName=dev_tester_cache_name)
    dev_tester.set_preprocessed_data(dev_preprocessed_data)
    # 创建测试集测试器
    tester.set_preprocessed_data(test_preprocessed_data)
    
    # 设置标准化参数
    if property_mean is not None and property_std is not None:
        train_tester.set_standardization_params(property_mean, property_std)
        dev_tester.set_standardization_params(property_mean, property_std)
        tester.set_standardization_params(property_mean, property_std)
    
    # 打印模型信息
    print_model_info(model)
    
    result = 'Epoch\tTime(sec)\tLoss_train\tMAE_train\tMSE_train\tR2_train\tPCC_train\tMAE_dev\tMSE_dev\tR2_dev\tPCC_dev\tMAE_test\tMSE_test\tR2_test\tPCC_test'
    with open(file_MAEs, 'w') as f:
        f.write(result + '\n')

    # 打印训练开始信息
    print_training_start_info()
    np.random.seed(args.split_seed)
    start = timeit.default_timer()

    # 初始化最佳验证损失
    best_val_loss = float('inf')
    
    # 添加早停机制相关变量
    patience_limit = args.early_stopping_patience
    patience_counter = 0  # 计数器，记录没有改善的epoch数
    best_model_state = None
    
    # 设置优化器和学习率调度器
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=args.scheduler_patience, 
        factor=args.scheduler_factor, min_lr=args.min_lr)
    
    trainer.set_optimizer(optimizer)
    trainer.set_scheduler(scheduler)
    
    # 训练循环
    for epoch in tqdm(range(args.iteration), desc="Epochs"):
        epoch += 1
        model.train()
        try:
            loss_train = trainer.train(dataset_train, args.learning_rate)
        except RuntimeError as e:
            raise e
        
        model.eval()

        # 模型选择阶段只评估训练集和开发集；测试集在模型冻结后只评估一次。
        MAE_train, MSE_train, R2_train, PCC_train, predictions_train = train_tester.test_regressor(dataset_train)
        MAE_dev, MSE_dev, R2_dev, PCC_dev, _ = dev_tester.test_regressor(dataset_dev)
        MAE_test = MSE_test = R2_test = PCC_test = float('nan')

        time = timeit.default_timer() - start

        if epoch == 1:
            minutes = time * args.iteration / 60
            hours = int(minutes / 60)
            minutes = int(minutes - 60 * hours)
            # 打印训练时间预估信息
            print_training_time_estimate(args.iteration, time)
            # 打印训练表头
            print_training_header(result)

        results = '\t'.join(map(str, [epoch, time, loss_train, MAE_train, MSE_train, R2_train, PCC_train,
                                     MAE_dev, MSE_dev, R2_dev, PCC_dev, MAE_test, MSE_test, R2_test, PCC_test]))
        tester.save_MAEs(results, file_MAEs)
        
        # 早停机制逻辑
        if MAE_dev <= best_val_loss:
            best_val_loss = MAE_dev
            patience_counter = 0  # 重置计数器
            best_model_state = copy.deepcopy(model.state_dict())  # 冻结最佳验证集状态
            # 在保存前确保模型的所有部分都已正确构建
            # _ensure_model_complete(model, device, args)
            # 打印模型信息用于调试
            # _print_model_info(model, "🐘 Before saving best model:")
            tester.save_model(model, file_model)
            tester.save_model(model, os.path.join(log_dir, 'best.pt'))
            print(f"New best validation model saved with MAE: {MAE_dev:.6f} at epoch {epoch}")
        else:
            patience_counter += 1  # 增加计数器
            
        # 检查是否达到早停限制
        if patience_counter >= patience_limit:
            print_early_stopping_info(epoch)
            break

    # 训练结束后保存最佳模型
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        # _ensure_model_complete(model, device, args)
        # 打印模型信息用于调试
        # _print_model_info(model, "🐘 Before saving final model (with best weights):")
        tester.save_model(model, file_model)  # 确保保存最佳模型状态
    else:
        _ensure_model_complete(model, device, args)
        # 打印模型信息用于调试
        # _print_model_info(model, "🐘 Before saving final model:")
        tester.save_model(model, file_model)
            
    # 保存训练参数到JSON文件
    training_params = {
        'timestamp': timestamp,
        'model_type': 'visnet_v2',
        'batch_train': args.batch_train,
        'batch_test': args.batch_test,
        'learning_rate': args.learning_rate,
        'iteration': args.iteration,
        'path': path,
        'dataname': dataname,
        'split_strategy': args.split_strategy,
        'split_seed': args.split_seed,
        'dev_split_seed': args.dev_split_seed,
        'split_manifest': os.path.abspath(os.path.join(path, dataname, 'split_manifest.json')),
        'debug_size': args.debug_size,
        'standardize': args.standardize,
        'standardize_features': args.standardize_features,
        'property_mean': float(property_mean) if property_mean is not None else None,
        'property_std': float(property_std) if property_std is not None else None,
        'additional_features_mean': additional_features_mean.tolist() if additional_features_mean is not None and hasattr(additional_features_mean, 'tolist') else additional_features_mean,
        'additional_features_std': additional_features_std.tolist() if additional_features_std is not None and hasattr(additional_features_std, 'tolist') else additional_features_std,
        'early_stopping_patience': args.early_stopping_patience,
        'checkpoint_selection_metric': 'MAE_dev',
        'test_evaluation': 'not_loaded_development_only' if args.development_only else 'once_after_model_selection',
        'effective_partition_reference': getattr(args, 'effective_partition_reference', None),
        'scheduler_patience': args.scheduler_patience,
        'scheduler_factor': args.scheduler_factor,
        'min_lr': args.min_lr,
        'weight_decay': args.weight_decay,
        'loss_function': args.loss_function,
        'visnet_v2_node_feature_dim': args.visnet_v2_node_feature_dim,
        'visnet_v2_physchem_feature_dim': args.visnet_v2_physchem_feature_dim,
        'visnet_v2_toxicity_feature_dim': args.visnet_v2_toxicity_feature_dim,
        'visnet_v2_chromato_feature_dim': args.visnet_v2_chromato_feature_dim,
        'visnet_v2_graph_hidden_dim': args.visnet_v2_graph_hidden_dim,
        'visnet_v2_physchem_hidden_dim': args.visnet_v2_physchem_hidden_dim,
        'visnet_v2_toxicity_hidden_dim': args.visnet_v2_toxicity_hidden_dim,
        'visnet_v2_chromato_hidden_dim': args.visnet_v2_chromato_hidden_dim,
        'visnet_v2_toxicity_intermediate_dim': args.visnet_v2_toxicity_intermediate_dim,
        'visnet_v2_fusion_hidden_dims': args.visnet_v2_fusion_hidden_dims,
        'visnet_v2_use_attention': args.visnet_v2_use_attention,
        'visnet_v2_use_gating': args.visnet_v2_use_gating,
        'visnet_v2_output_head': args.visnet_v2_output_head,
        'visnet_v2_dropout_rate': args.visnet_v2_dropout_rate,
        'visnet_v2_feature_level': args.visnet_v2_feature_level,
        'visnet_v2_physchem_mask': args.visnet_v2_physchem_mask,
        'visnet_v2_toxicity_mask': args.visnet_v2_toxicity_mask,
        'visnet_v2_chromato_mask': args.visnet_v2_chromato_mask,
        'standardization_info': standardization_info  # 添加标准化信息
    }
        
    save_training_params(training_params, log_dir)
    shutil.copyfile(
        os.path.join(log_dir, 'training_params.json'),
        os.path.join(log_dir, 'experiment_config.json'),
    )
    
    # 绘制训练指标图
    try:
        plot_training_metrics(file_MAEs, loss_plot_file, 
                            json_file=os.path.join(log_dir, 'training_metrics.json'))
    except Exception as e:
        print(f"Warning: Could not plot training metrics: {e}")
    
    # 测试最佳模型
    model.eval()
    # 确保BatchNorm层处于评估状态
    for module in model.modules():
        if isinstance(module, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d)):
            module.eval()
    
    # 打印训练完成后的模型信息
    _print_model_info(model, "After training:")
    
    MAE_train, MSE_train, R2_train, PCC_train, predictions_train = train_tester.test_regressor(dataset_train)
    MAE_dev, MSE_dev, R2_dev, PCC_dev, predictions_dev = dev_tester.test_regressor(dataset_dev)
    if args.development_only:
        train_tester.save_predictions(predictions_train, file_predictions.replace('.txt', '_train.txt'))
        dev_tester.save_predictions(predictions_dev, file_predictions.replace('.txt', '_dev.txt'))
        development_metrics = {
            'partition': 'development', 'n': len(dataset_dev),
            'mae': float(MAE_dev), 'rmse': float(np.sqrt(MSE_dev)),
            'r2': float(R2_dev), 'pcc': float(PCC_dev),
            'test_responses_loaded': False,
        }
        with open(os.path.join(log_dir, 'metrics_development.json'), 'w', encoding='utf-8') as handle:
            json.dump(development_metrics, handle, indent=2)
            handle.write('\n')
        print('Development-only training completed: {}'.format(log_dir))
        return
    MAE_test, MSE_test, R2_test, PCC_test, predictions_test = tester.test_regressor(dataset_test)
    
    # 输出测试数据集的前3条记录用于对比
    print("=== First 3 records from test dataset (training evaluation) ===")
    pred_lines = predictions_test.strip().split('\n')
    for i, line in enumerate(pred_lines[:3]):
        parts = line.split('\t')
        if len(parts) >= 3:
            smiles, true_value, pred_value = parts[0], parts[1], parts[2]
            print(f"  Record {i+1}:")
            print(f"    SMILES: {smiles}")
            print(f"    True Value (Original): {true_value}")
            print(f"    Predicted Value (Original): {pred_value}")
            
            # 反标准化以显示原始值
            try:
                true_standardized = float(true_value)
                pred_standardized = float(pred_value)
                true_original = (true_standardized - property_mean) / property_std
                pred_original = (pred_standardized - property_mean) / property_std
                print(f"    True Value (Standardized): {true_original}")
                print(f"    Predicted Value (Standardized): {pred_original}")
            except ValueError:
                print("    Warning: Could not convert values for destandardization")
            
            # 如果有额外特征，显示它们（这些已经是标准化的特征）
            if smiles in test_preprocessed_data:
                data = test_preprocessed_data[smiles]
                if data.get('physchem_features') is not None:
                    print(f"    PhysChem Features (Standardized): {data['physchem_features']}")
                if data.get('toxicity_features') is not None:
                    print(f"    Toxicity Features (Standardized): {data['toxicity_features']}")
                if data.get('chromato_features') is not None:
                    print(f"    Chromato Features (Standardized): {data['chromato_features']}")
        print()

    # 打印最终测试结果
    print("=== Training Final Test Results ===")
    print(f"Train MAE: {MAE_train:.6f}, RMSE: {np.sqrt(MSE_train):.6f}, R2: {R2_train:.6f}")
    print(f"Dev MAE: {MAE_dev:.6f}, RMSE: {np.sqrt(MSE_dev):.6f}, R2: {R2_dev:.6f}")
    print(f"Test MAE: {MAE_test:.6f}, RMSE: {np.sqrt(MSE_test):.6f}, R2: {R2_test:.6f}")
    
    # 保存最终测试结果
    with open(file_test_result, 'w') as f:
        f.write(f'MAE_train: {MAE_train}\n')
        f.write(f'MAE_train_norm: {MAE_train / property_std}\n')
        f.write(f'MSE_train: {MSE_train}\n')
        f.write(f'MSE_train_norm: {MSE_train / (property_std ** 2)}\n')
        f.write(f'R2_train: {R2_train}\n')
        f.write(f'PCC_train: {PCC_train}\n')
        f.write(f'MAE_dev: {MAE_dev}\n')
        f.write(f'MAE_dev_norm: {MAE_dev / property_std}\n')
        f.write(f'MSE_dev: {MSE_dev}\n')
        f.write(f'MSE_dev_norm: {MSE_dev / (property_std ** 2)}\n')
        f.write(f'R2_dev: {R2_dev}\n')
        f.write(f'PCC_dev: {PCC_dev}\n')
        f.write(f'MAE_test: {MAE_test}\n')
        f.write(f'MAE_test_norm: {MAE_test / property_std}\n')
        f.write(f'MSE_test: {MSE_test}\n')
        f.write(f'MSE_test_norm: {MSE_test / (property_std ** 2)}\n')
        f.write(f'R2_test: {R2_test}\n')
        f.write(f'PCC_test: {PCC_test}\n')
    
    # 保存预测结果（标准化值）
    tester.save_predictions(predictions_train, file_predictions.replace('.txt', '_train.txt'))
    tester.save_predictions(predictions_dev, file_predictions.replace('.txt', '_dev.txt'))
    tester.save_predictions(predictions_test, file_predictions.replace('.txt', '_test.txt'))

    prediction_lines = [
        line for line in predictions_test.strip().splitlines() if line.strip()
    ]
    with open(os.path.join(log_dir, 'predictions_test.csv'), 'w', encoding='utf-8', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(['smiles', 'source_rti', 'predicted_rti', 'residual_source_minus_prediction'])
        for line in prediction_lines:
            smiles, source_rti, predicted_rti = line.split('\t')
            source_value = float(source_rti)
            predicted_value = float(predicted_rti)
            writer.writerow([
                smiles, source_value, predicted_value,
                source_value - predicted_value,
            ])
    metrics = {
        'n': len(prediction_lines),
        'mae': float(MAE_test),
        'rmse': float(np.sqrt(MSE_test)),
        'r2': float(R2_test),
        'pcc': float(PCC_test),
    }
    with open(os.path.join(log_dir, 'metrics.json'), 'w', encoding='utf-8') as handle:
        json.dump(metrics, handle, indent=2, ensure_ascii=False)
        handle.write('\n')
    
    # 绘制预测值vs实际值图
    try:
        plot_predicted_vs_actual(file_predictions.replace('.txt', '_train.txt'), 
                               os.path.join(log_dir, 'predicted_vs_actual_train.png'), 
                               property_mean, property_std,
                               json_file=os.path.join(log_dir, 'predicted_vs_actual_train.json'))
        plot_predicted_vs_actual(file_predictions.replace('.txt', '_dev.txt'), 
                               os.path.join(log_dir, 'predicted_vs_actual_dev.png'), 
                               property_mean, property_std,
                               json_file=os.path.join(log_dir, 'predicted_vs_actual_dev.json'))
        plot_predicted_vs_actual(file_predictions.replace('.txt', '_test.txt'), 
                               os.path.join(log_dir, 'predicted_vs_actual_test.png'), 
                               property_mean, property_std,
                               json_file=os.path.join(log_dir, 'predicted_vs_actual_test.json'))
    except Exception as e:
        print(f"Warning: Could not plot predicted vs actual: {e}")
    
    # 清理缓存资源
    del trainer.mol_cache
    del tester.mol_cache
    del train_tester.mol_cache
    del dev_tester.mol_cache
    
    print(f"Training completed. Results saved to {log_dir}")
    
    return

    # TAG 训练完成后自动进行预测以验证结果一致性（可选）
    print("\n" + "="*100)
    print("AUTOMATIC PREDICTION ON MMF_GNN_pos_test.csv AFTER TRAINING COMPLETED")
    print("="*100)
    
    # 导入预测所需模块
    try:
        from predict_visnet_v2 import load_model, load_and_process_dataset_with_features, predict_dataset
        import datetime
        import json as json_module
        
        # 准备预测参数
        class Args:
            def __init__(self, model_path, params_path, input_file, output_dir, batch_size, dataset_type=None):
                self.model_path = model_path
                self.params_path = params_path
                self.input_file = input_file
                self.output_dir = output_dir
                self.batch_size = batch_size
                self.dataset_type = dataset_type
                self.filter_column = None
                self.filter_value = None
        
        # 创建预测参数对象
        predict_args = Args(
            model_path=file_model,
            params_path=os.path.join(log_dir, 'training_params.json'),
            # input_file=os.path.join(path, f'{dataname}_test.csv'),  # INFO 使用测试数据集
            # input_file=os.path.join(path, f'{dataname}.csv'),  # INFO ori/
            input_file=os.path.join(path, 'data1028_sheet0.csv'),  # INFO target
            output_dir=os.path.join(log_dir, 'auto_test_predictions'),
            batch_size=args.batch_test,
            dataset_type='test',
        )
        
        # 检查测试数据集文件是否存在
        if not os.path.exists(predict_args.input_file):
            print(f"Warning: Test dataset file {predict_args.input_file} not found. Skipping automatic prediction.")
        else:
            # 确保输出目录存在
            os.makedirs(predict_args.output_dir, exist_ok=True)
            
            # 设置设备
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            print(f'Using device for prediction: {device}')
            
            # 加载训练参数
            with open(predict_args.params_path, 'r') as f:
                predict_params = json_module.load(f)
            
            print("Loaded training parameters for prediction:")
            for key, value in predict_params.items():
                print(f"  {key}: {value}")
            
            print("Loading model for prediction...")
            model, params = load_model(predict_args.model_path, predict_args.params_path, device)
            # 打印加载后的模型信息用于调试
            _print_model_info(model, "After loading model:")
            print("Model loaded successfully for prediction")
            
            # TEST ori-model test
            # tester.model = model
            # MAE_test, MSE_test, R2_test, PCC_test, predictions_test = tester.test_regressor(dataset_test)
            # print(f"😺 Test MAE: {MAE_test:.6f}, RMSE: {np.sqrt(MSE_test):.6f}, R2: {R2_test:.6f}")            
            # exit(0)
            
            # 处理数据集
            print(f"Processing {predict_args.dataset_type} dataset for prediction...")
            dataset = load_and_process_dataset_with_features(
                predict_args.input_file, 
                predict_params,
                predict_args.dataset_type
            )
            
            # 进行预测
            print("Making predictions on test dataset...")
            predictions = predict_dataset(model, dataset, device, predict_args.batch_size, predict_params)
            
            # 输出测试数据集的前3条记录用于对比（包含预测结果）
            print("=== First 3 records from test dataset (prediction evaluation) ===")
            for i, (data_item, pred_item) in enumerate(zip(dataset[:3], predictions[:3])):
                print(f"  Record {i+1}:")
                print(f"    SMILES: {pred_item[0]}")
                print(f"    Property Value (Original): {pred_item[2]}")
                print(f"    Predicted Value (Original): {pred_item[1]}")
                
                # 反标准化以显示原始值
                try:
                    standardized_value = pred_item[2]
                    original_value = (standardized_value - property_mean) / property_std
                    original_pred_value = (pred_item[1] - property_mean) / property_std
                    print(f"    Property Value (Standardized): {original_value}")
                    print(f"    Predicted Value (Standardized): {original_pred_value}")
                except (TypeError, ValueError):
                    print("    Warning: Could not destandardize property value")
                
                # 显示额外特征
                if data_item.get('physchem_features') is not None:
                    print(f"    PhysChem Features: {data_item['physchem_features']}")
                if data_item.get('toxicity_features') is not None:
                    print(f"    Toxicity Features: {data_item['toxicity_features']}")
                if data_item.get('chromato_features') is not None:
                    print(f"    Chromato Features: {data_item['chromato_features']}")
                print()

            print('First 5 predictions:', predictions[:5])
            
            # 保存预测结果
            timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            output_subdir = os.path.join(predict_args.output_dir, f"prediction_{timestamp}")
            os.makedirs(output_subdir, exist_ok=True)
            
            # 保存预测结果到CSV文件（进行反标准化以获得原始值）
            predictions_file = os.path.join(output_subdir, f"{predict_args.dataset_type}_{os.path.splitext(os.path.basename(predict_args.input_file))[0]}_predictions.csv")
            with open(predictions_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['SMILES', 'True_Value', 'Predicted_Value'])
                for pred in predictions:
                    # 修复：进行反标准化以获得原始值
                    smiles, standardized_pred, standardized_true = pred
                    writer.writerow([smiles, standardized_true, standardized_pred])
                    # pred_value = standardized_pred * property_std + property_mean
                    # true_value = standardized_true * property_std + property_mean
                    # writer.writerow([smiles, pred_value, true_value])
            
            print(f"Saved test predictions (destandardized) to {predictions_file}")
            
            # 计算并保存评估指标（在标准化空间中计算以保持与训练时的一致性）
            print("Calculating metrics for test dataset...")
            
            # 提取预测值和真实值（在标准化空间中）
            y_true_standardized = np.array([p[2] for p in predictions])  # Actual values (已标准化)
            y_pred_standardized = np.array([p[1] for p in predictions])  # Predicted values (已标准化)
            
            # 检查并处理NaN值
            valid_mask = np.isfinite(y_true_standardized) & np.isfinite(y_pred_standardized)
            if not np.all(valid_mask):
                print(f"Warning: Found {np.sum(~valid_mask)} NaN or infinite values in predictions. Removing them for metric calculation.")
                y_true_standardized = y_true_standardized[valid_mask]
                y_pred_standardized = y_pred_standardized[valid_mask]
            
            # 如果没有有效数据，返回空指标
            if len(y_true_standardized) == 0 or len(y_pred_standardized) == 0:
                print("Error: No valid data points for metric calculation.")
                metrics = {
                    'mae': float('nan'),
                    'rmse': float('nan'),
                    'r2': float('nan'),
                    'mre': float('nan'),
                    'valid_points': 0
                }
            else:
                # 计算各种评估指标（在标准化空间中，与训练时保持一致）
                from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
                from scipy.stats import pearsonr
                mae = mean_absolute_error(y_true_standardized, y_pred_standardized)
                mse = mean_squared_error(y_true_standardized, y_pred_standardized)
                rmse = np.sqrt(mse)
                r2 = r2_score(y_true_standardized, y_pred_standardized)
                
                # 计算皮尔逊相关系数(PCC)
                pcc, _ = pearsonr(y_true_standardized, y_pred_standardized)
                
                # 计算平均相对误差
                epsilon = 1e-8  # 避免除零错误
                mre = np.mean(np.abs((y_true_standardized - y_pred_standardized) / (np.abs(y_true_standardized) + epsilon))) * 100
                
                metrics = {
                    'mae': mae,
                    'mae_norm': mae / property_std,
                    'mse': mse,
                    'mse_norm': mse / (property_std ** 2),
                    'r2': r2,
                    'pcc': pcc,  # Pearson Correlation Coefficient
                    'mre': mre,  # Mean Relative Error (百分比)
                    'valid_points': len(y_true_standardized)
                }
            
            print("Test Metrics (in standardized space, consistent with training metrics):")
            for key, value in metrics.items():
                if isinstance(value, float):
                    print(f"  {key}: {value:.6f}")
                else:
                    print(f"  {key}: {value}")
            
            metrics_file = os.path.join(output_subdir, f"{predict_args.dataset_type}_{os.path.splitext(os.path.basename(predict_args.input_file))[0]}_metrics.json")
            with open(metrics_file, 'w') as f:
                json_module.dump(metrics, f, indent=2)
            
            print(f"Saved test metrics to {metrics_file}")
            
            print("Automatic prediction on test dataset completed successfully!")
            
    except Exception as e:
        print(f"Warning: Could not perform automatic prediction: {e}")
        import traceback
        traceback.print_exc()
    
    print("="*100)
    print("TRAINING AND AUTOMATIC PREDICTION COMPLETED")
    print("="*100)


if __name__ == "__main__":
    main()
