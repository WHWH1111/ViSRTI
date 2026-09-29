#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VisNetV2 模型专用训练脚本

针对 VisNetV2 模型进行简化和优化的训练脚本
为 5fold 所适配。
"""

import sys
import os
import random
import copy
import subprocess

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
    import json

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
        feature_level=args.visnet_v2_feature_level,
        output_head=args.visnet_v2_output_head,
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
    print(f"  feature_level={args.visnet_v2_feature_level}")
    print(f"  physchem_feature_mask={args.visnet_v2_physchem_mask}")
    print(f"  toxicity_feature_mask={args.visnet_v2_toxicity_mask}")
    print(f"  chromato_feature_mask={args.visnet_v2_chromato_mask}")
    
    return model

def load_and_preprocess_data(args, device):
    """加载和预处理数据"""
    # 固定随机种子以确保结果可重现
    np.random.seed(args.split_seed)
    torch.manual_seed(args.split_seed)
    
    # 数据集配置
    if args.cross_validation:
        # 使用交叉验证模式下的数据路径
        path = args.cv_path
        dataname = f"fold_{args.fold}"
        # 构造fold路径
        fold_path = os.path.join(path, f"{args.cv_dataname}_5")
        # 实际的数据文件名仍使用原始dataname
        _visnet_train = "visnet_train_v2_" + args.cv_dataname
        _visnet_test = _visnet_train    # INFO 好像可以直接合并起来
        target_column = 'Pred_RTI_Negative_ESI' if 'neg' in dataname.lower() else 'Pred_RTI_Positive_ESI'
    else:
        # 原始模式
        path = './data/MMF-3/'
        # dataname = 'MMF_GNN_neg'    # INFO 数据集名称
        dataname = args.cv_dataname
        _visnet_train = "visnet_train_v2_" + dataname
        _visnet_test = "visnet_test_v2_" + dataname
        target_column = 'Pred_RTI_Negative_ESI' if 'neg' in dataname.lower() else 'Pred_RTI_Positive_ESI'
    
    # 创建基于时间和参数的目录
    if args.cross_validation and args.fold is not None:
        run_name = '{}_fold_{}_{}_seed{}'.format(
            args.cv_dataname,
            args.fold,
            args.split_strategy,
            args.split_seed,
        )
        file_paths = create_log_dir(path, run_name, 48, 6, 6, args.batch_train, args.learning_rate, args.iteration, args.debug_size)
    else:
        file_paths = create_log_dir(path, dataname, 48, 6, 6, args.batch_train, args.learning_rate, args.iteration, args.debug_size)
        
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
    # 在交叉验证模式下，使用fold路径
    if args.cross_validation:
        pass
        # train_file_path = os.path.join(fold_path, f"{dataname}_train_set.txt")
        # test_file_path = os.path.join(fold_path, f"{dataname}_test_set.txt")
    else:
        train_file_path, test_file_path = split_raw_data_if_needed(
            path,
            dataname,
            target_column_name='Pred_RTI_Negative_ESI' if 'neg' in dataname else 'Pred_RTI_Positive_ESI',
            split_strategy=args.split_strategy,
            random_state=args.split_seed,
        )
    
    # 根据是否为调试模式决定加载的数据量
    max_data = args.debug_size if args.debug_size is not None else None
    
    # 加载数据集
    print("Loading datasets for VisNetV2 model...")
    dataset_train, dataset_dev, dataset_test, _, additional_features_mean, additional_features_std = load_datasets(
        args, path if not args.cross_validation else fold_path, dataname, max_data)
    
    # 标准化数据（如果需要）
    dataset_train, dataset_dev, dataset_test, property_mean, property_std = standardize_datasets_if_needed(
        args, dataset_train, dataset_dev, dataset_test, 
        additional_features_mean=additional_features_mean,
        additional_features_std=additional_features_std)
    print("Datasets loaded.", property_mean, property_std)
    
    # VisNetV2模型需要特殊的预处理
    print("Preprocessing VisNetV2-specific data...")
    feature_dataset_type = 'neg' if 'neg' in args.cv_dataname.lower() else 'pos'
    preprocess_result = preprocess_visnet_data(
        args,
        dataset_train,
        dataset_dev,
        dataset_test,
        dataname,
        _visnet_train,
        _visnet_test,
        feature_dataset_type=feature_dataset_type,
    )
    
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
    
    # 打印预处理信息
    print_preprocess_info(dataset_train, dataset_dev, dataset_test)
    
    # 返回所有需要的数据
    return {
        'dataset_train': dataset_train,
        'dataset_dev': dataset_dev,
        'dataset_test': dataset_test,
        'train_preprocessed_data': train_preprocessed_data,
        'dev_preprocessed_data': dev_preprocessed_data,
        'test_preprocessed_data': test_preprocessed_data,
        'property_mean': property_mean,
        'property_std': property_std,
        'additional_features_mean': additional_features_mean,
        'additional_features_std': additional_features_std,
        'path': path,
        'dataname': dataname,
        'log_dir': log_dir,
        'file_MAEs': file_MAEs,
        'file_test_result': file_test_result,
        'file_predictions': file_predictions,
        'file_model': file_model,
        'loss_plot_file': loss_plot_file,
        'cp_plot_file': cp_plot_file,
        'timestamp': timestamp,
        'standardization_info': standardization_info,
        '_visnet_train': _visnet_train,
        '_visnet_test': _visnet_test
    }


def train_model(model, data_dict, args, device):
    """训练模型"""
    # 解构数据字典中的关键变量
    dataset_train = data_dict['dataset_train']
    dataset_dev = data_dict['dataset_dev']
    dataset_test = data_dict['dataset_test']
    train_preprocessed_data = data_dict['train_preprocessed_data']
    dev_preprocessed_data = data_dict['dev_preprocessed_data']
    test_preprocessed_data = data_dict['test_preprocessed_data']
    property_mean = data_dict['property_mean']
    property_std = data_dict['property_std']
    log_dir = data_dict['log_dir']
    file_MAEs = data_dict['file_MAEs']
    file_test_result = data_dict['file_test_result']
    file_predictions = data_dict['file_predictions']
    file_model = data_dict['file_model']
    loss_plot_file = data_dict['loss_plot_file']
    timestamp = data_dict['timestamp']
    _visnet_train = data_dict['_visnet_train']
    _visnet_test = data_dict['_visnet_test']
    standardization_info = data_dict['standardization_info']
    
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

        # 模型选择阶段只评估训练集和开发集；外层测试折在模型冻结后只评估一次。
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
        'path': data_dict['path'],
        'dataname': data_dict['dataname'],
        'cv_dataname': args.cv_dataname if args.cross_validation else None,
        'fold': args.fold if args.cross_validation else None,
        'split_strategy': args.split_strategy,
        'split_seed': args.split_seed,
        'debug_size': args.debug_size,
        'standardize': args.standardize,
        'standardize_features': args.standardize_features,
        'property_mean': float(property_mean) if property_mean is not None else None,
        'property_std': float(property_std) if property_std is not None else None,
        'additional_features_mean': data_dict['additional_features_mean'].tolist() if data_dict['additional_features_mean'] is not None and hasattr(data_dict['additional_features_mean'], 'tolist') else data_dict['additional_features_mean'],
        'additional_features_std': data_dict['additional_features_std'].tolist() if data_dict['additional_features_std'] is not None and hasattr(data_dict['additional_features_std'], 'tolist') else data_dict['additional_features_std'],
        'early_stopping_patience': args.early_stopping_patience,
        'checkpoint_selection_metric': 'MAE_dev',
        'test_evaluation': 'once_after_model_selection',
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
        'visnet_v2_dropout_rate': args.visnet_v2_dropout_rate,
        'visnet_v2_fusion_hidden_dims': args.visnet_v2_fusion_hidden_dims,
        'visnet_v2_use_attention': args.visnet_v2_use_attention,
        'visnet_v2_use_gating': args.visnet_v2_use_gating,
        'visnet_v2_output_head': args.visnet_v2_output_head,
        'visnet_v2_feature_level': args.visnet_v2_feature_level,
        'visnet_v2_physchem_mask': args.visnet_v2_physchem_mask,
        'visnet_v2_toxicity_mask': args.visnet_v2_toxicity_mask,
        'visnet_v2_chromato_mask': args.visnet_v2_chromato_mask,
        'standardization_info': standardization_info  # 添加标准化信息
    }
        
    save_training_params(training_params, log_dir)
    
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
    
    print(f"Training completed. Results saved to {log_dir}")


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
    parser.add_argument('--split-strategy', type=str, default='scaffold',
                        choices=['scaffold', 'random'],
                        help='Train/dev split strategy (default: scaffold)')
    parser.add_argument('--split-seed', type=int, default=1234,
                        help='Fixed seed used for splitting and training (default: 1234)')

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
    parser.add_argument('--early-stopping-patience', type=int, default=30,
                        help='Early stopping patience (default: 50)')
    
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
    parser.add_argument('--visnet-v2-output-head', default='nested_mlp', choices=['nested_mlp'],
                        help='Explicitly preserve the selected nested MLP output head')
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

    # 交叉验证参数
    parser.add_argument('--cross-validation', action='store_true',
                        help='Enable cross validation mode')
    parser.add_argument('--cv-path', type=str, default='./data/MMF-3',
                        help='Base path for cross validation datasets (default: ./data/MMF-3)')
    parser.add_argument('--cv-dataname', type=str, default='MMF_GNN_pos',
                        help='Base dataset name for cross validation (default: MMF_GNN_pos)')
    parser.add_argument('--fold', type=int, choices=range(1, 6), default=None,
                        help='Specific fold to train (1-5), if not specified, train all folds sequentially')

    args = parser.parse_args()
    
    # 设置PyTorch内存管理参数以避免内存碎片
    if args.max_split_size_mb is not None:
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = f'max_split_size_mb:{args.max_split_size_mb}'
        print(f"Set PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:{args.max_split_size_mb}")
    
    # 启用CUDA内存优化
    random.seed(args.split_seed)
    np.random.seed(args.split_seed)
    torch.manual_seed(args.split_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.split_seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        # 设置环境变量以减少内存碎片
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
    
    # 处理交叉验证模式
    if args.cross_validation:
        # 如果指定了特定fold，则只训练该fold
        if args.fold is not None:
            folds_to_train = [args.fold]
        else:
            # 默认训练所有folds (1-5)
            folds_to_train = list(range(1, 6))
            
        # 保存原始工作目录
        original_cwd = os.getcwd()
        
        # 遍历执行每个fold的训练
        for fold in folds_to_train:
            print(f"\n{'='*50}")
            print(f"Training Fold {fold}")
            print(f"{'='*50}")
            
            # 切换到对应的fold目录
            fold_dir = os.path.join(args.cv_path, f"{args.cv_dataname}_5", f"fold_{fold}")
            if not os.path.exists(fold_dir):
                print(f"Error: Fold directory {fold_dir} does not exist!")
                continue
                
            print(f"Changing to fold directory: {fold_dir}")
            # os.chdir(fold_dir)
            
            try:
                # 创建新的args对象副本以避免修改原始args
                import copy
                fold_args = copy.deepcopy(args)
                fold_args.fold = fold
                
                # 加载和预处理数据
                data_dict = load_and_preprocess_data(fold_args, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
                
                # 初始化VisNetV2模型
                model = get_model(fold_args, torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
                
                # 训练模型
                train_model(model, data_dict, fold_args, torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
            finally:
                # 确保返回原始目录
                os.chdir(original_cwd)
                print(f"Returned to original directory: {original_cwd}")
    else:
        # 原始单次训练模式
        # 加载和预处理数据
        data_dict = load_and_preprocess_data(args, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        
        # 初始化VisNetV2模型
        model = get_model(args, torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        
        # 训练模型
        train_model(model, data_dict, args, torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    

def run4debug():
    """
    使用硬编码参数运行VisNetV2模型训练，方便调试
    示例用法，参数来自:
    python train_visnet_v2_5.py --cross-validation --visnet-v2-feature-level graph_physchem_toxicity --standardize --loss-function l1 --batch-train 64 --standardize-features --visnet-v2-physchem-mask 1 1 1 0
    """
    import argparse
    
    # 创建参数解析器
    parser = argparse.ArgumentParser(description='Train VisNetV2 Molecular Graph Neural Network')
    
    # 模拟命令行参数
    args_list = [
        '--cross-validation',
        '--visnet-v2-feature-level', 'graph_physchem_toxicity',
        '--standardize',
        '--loss-function', 'l1',
        '--batch-train', '64',
        '--standardize-features',
        '--visnet-v2-physchem-mask', '1', '1', '1', '0'
    ]
    
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
    parser.add_argument('--split-strategy', type=str, default='scaffold',
                        choices=['scaffold', 'random'],
                        help='Train/dev split strategy (default: scaffold)')
    parser.add_argument('--split-seed', type=int, default=1234,
                        help='Fixed seed used for splitting and training (default: 1234)')

    # 模型选择参数
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
    parser.add_argument('--early-stopping-patience', type=int, default=50,
                        help='Early stopping patience (default: 50)')
    
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
    parser.add_argument('--visnet-v2-toxicity-hidden-dim', type=int, default=128,
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
    parser.add_argument('--visnet-v2-output-head', default='nested_mlp', choices=['nested_mlp'],
                        help='Explicitly preserve the selected nested MLP output head')
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

    # 交叉验证参数
    parser.add_argument('--cross-validation', action='store_true',
                        help='Enable cross validation mode')
    parser.add_argument('--cv-path', type=str, default='./data/MMF-3',
                        help='Base path for cross validation datasets (default: ./data/MMF-3)')
    parser.add_argument('--cv-dataname', type=str, default='MMF_GNN_pos',
                        help='Base dataset name for cross validation (default: MMF_GNN_pos)')
    parser.add_argument('--fold', type=int, choices=range(1, 6), default=None,
                        help='Specific fold to train (1-5), if not specified, train all folds sequentially')

    # 解析模拟的命令行参数
    args = parser.parse_args(args_list)
    
    # 设置PyTorch内存管理参数以避免内存碎片
    if args.max_split_size_mb is not None:
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = f'max_split_size_mb:{args.max_split_size_mb}'
        print(f"Set PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:{args.max_split_size_mb}")
    
    random.seed(args.split_seed)
    np.random.seed(args.split_seed)
    torch.manual_seed(args.split_seed)

    # 启用CUDA内存优化
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.split_seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        # 设置环境变量以减少内存碎片
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
    
    # 处理交叉验证模式
    if args.cross_validation:
        # 如果指定了特定fold，则只训练该fold
        if args.fold is not None:
            folds_to_train = [args.fold]
        else:
            # 默认训练所有folds (1-5)
            folds_to_train = list(range(1, 6))
            
        # 保存原始工作目录
        original_cwd = os.getcwd()
        
        # 遍历执行每个fold的训练
        for fold in folds_to_train:
            print(f"\n{'='*50}")
            print(f"Training Fold {fold}")
            print(f"{'='*50}")
            
            # 切换到对应的fold目录
            fold_dir = os.path.join(args.cv_path, f"{args.cv_dataname}_5", f"fold_{fold}")
            if not os.path.exists(fold_dir):
                print(f"Error: Fold directory {fold_dir} does not exist!")
                continue
                
            print(f"Changing to fold directory: {fold_dir}")
            os.chdir(fold_dir)
            
            try:
                # 创建新的args对象副本以避免修改原始args
                import copy
                fold_args = copy.deepcopy(args)
                fold_args.fold = fold
                
                # 加载和预处理数据
                data_dict = load_and_preprocess_data(fold_args, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
                
                # 初始化VisNetV2模型
                model = get_model(fold_args, torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
                
                # 训练模型
                train_model(model, data_dict, fold_args, torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
            finally:
                # 确保返回原始目录
                os.chdir(original_cwd)
                print(f"Returned to original directory: {original_cwd}")
    else:
        # 原始单次训练模式
        # 加载和预处理数据
        data_dict = load_and_preprocess_data(args, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        
        # 初始化VisNetV2模型
        model = get_model(args, torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        
        # 训练模型
        train_model(model, data_dict, args, torch.device('cuda' if torch.cuda.is_available() else 'cpu'))



if __name__ == "__main__":
    # 检查是否有命令行参数传入
    if len(sys.argv) > 1:
        # 如果有命令行参数，则调用main函数处理
        main()
    else:
        # 如果没有命令行参数，则调用run4debug函数进行调试
        run4debug()
