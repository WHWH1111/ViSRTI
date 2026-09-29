#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
评估指标计算和报告生成工具函数

Created for predict.py
"""

import numpy as np
import json
import datetime


def calculate_evaluation_metrics(predictions, filename=""):
    """
    计算预测结果的评估指标
    
    Args:
        predictions (list): 包含(smiles, predicted, actual)元组的列表
        filename (str): 文件名，用于日志输出
        
    Returns:
        dict: 包含各种评估指标的字典
    """
    if len(predictions) == 0:
        return {
            "mae": float('nan'),
            "rmse": float('nan'),
            "r2": float('nan'),
            "pcc": float('nan')
        }
    
    preds = np.array([float(p[1]) for p in predictions])
    actuals = np.array([float(p[2]) for p in predictions])
    
    # 检查并处理NaN或无穷大值
    valid_mask = np.isfinite(preds) & np.isfinite(actuals)
    if not np.all(valid_mask):
        print(f"Warning: {np.sum(~valid_mask)} predictions contain NaN or Inf values, filtering them out")
        preds = preds[valid_mask]
        actuals = actuals[valid_mask]
    
    # 确保有足够的有效数据进行计算
    if len(preds) == 0 or len(actuals) == 0:
        print("Error: No valid predictions after filtering NaN/Inf values")
        return {
            "mae": float('nan'),
            "rmse": float('nan'),
            "r2": float('nan'),
            "pcc": float('nan')
        }
    
    mae = np.mean(np.abs(preds - actuals))
    mse = np.mean((preds - actuals) ** 2)
    rmse = np.sqrt(mse)
    
    # 只有当样本数大于1时才计算R²和PCC，避免警告
    if len(predictions) > 1:
        # 计算R²
        ss_res = np.sum((actuals - preds) ** 2)
        ss_tot = np.sum((actuals - np.mean(actuals)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
            
        # 计算PCC
        if np.std(preds) != 0 and np.std(actuals) != 0:
            pcc = np.corrcoef(preds, actuals)[0, 1]
        else:
            pcc = 0.0
    else:
        r2 = 0.0
        pcc = 0.0
        
    print(f"Evaluation metrics for {filename}:")
    print(f"  MAE: {mae:.4f}")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  R²: {r2:.4f}")
    print(f"  PCC: {pcc:.4f}")
    
    return {
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "pcc": pcc
    }


def calculate_accuracy_metrics(predictions, thresholds=[40, 30, 20, 10, 5]):
    """
    计算基于不同误差阈值的准确率指标
    
    Args:
        predictions (list): 包含(smiles, predicted, actual)元组的列表
        thresholds (list): 误差阈值列表
        
    Returns:
        dict: 包含各阈值下准确率的字典
    """
    if len(predictions) == 0:
        return {}
    
    preds = np.array([float(p[1]) for p in predictions])
    actuals = np.array([float(p[2]) for p in predictions])
    
    # 检查并处理NaN或无穷大值
    valid_mask = np.isfinite(preds) & np.isfinite(actuals)
    if not np.all(valid_mask):
        preds = preds[valid_mask]
        actuals = actuals[valid_mask]
    
    if len(preds) == 0 or len(actuals) == 0:
        return {}
    
    accuracy_metrics = {}
    
    # 计算相对误差百分比，避免除以零的情况
    # 使用一个很小的数来避免除以零
    epsilon = 1e-8
    relative_errors = np.abs((preds - actuals) / np.maximum(np.abs(actuals), epsilon)) * 100
    
    # 对于每个阈值，计算误差小于该阈值的预测比例
    for threshold in thresholds:
        accuracy = np.mean(relative_errors <= threshold) * 100
        accuracy_metrics[f'Accuracy_within_{threshold}%'] = accuracy
        print(f"  Accuracy within {threshold}% error: {accuracy:.2f}%")
    
    return accuracy_metrics


def generate_metrics_report(evaluation_metrics, accuracy_metrics, original_mae=None, 
                          standardized_mae=None, df=None, dataset=None, 
                          skipped_molecules_count=0, args=None, trained_dataname="",
                          filename=""):
    """
    生成完整的评估指标报告
    
    Args:
        evaluation_metrics (dict): 基本评估指标
        accuracy_metrics (dict): 准确率指标
        original_mae (float): 原始MAE（如果有）
        standardized_mae (float): 标准化MAE（如果有）
        df (pandas.DataFrame): 原始数据框
        dataset (list): 处理后的数据集
        skipped_molecules_count (int): 跳过的分子数量
        args (argparse.Namespace): 命令行参数
        trained_dataname (str): 训练数据集名称
        filename (str): 处理的文件名
        
    Returns:
        dict: 完整的评估指标报告
    """
    # 构建评估指标字典
    metrics_dict = {
        "MAE": float(evaluation_metrics["mae"]) if not np.isnan(evaluation_metrics["mae"]) else None,
        "RMSE": float(evaluation_metrics["rmse"]) if not np.isnan(evaluation_metrics["rmse"]) else None,
        "R2": float(evaluation_metrics["r2"]) if not np.isnan(evaluation_metrics["r2"]) else None,
        "PCC": float(evaluation_metrics["pcc"]) if not np.isnan(evaluation_metrics["pcc"]) else None
    }
    
    if original_mae is not None:
        metrics_dict["Original_MAE"] = float(original_mae) if not np.isnan(original_mae) else None
        
    if standardized_mae is not None:
        metrics_dict["Standardized_MAE"] = float(standardized_mae) if not np.isnan(standardized_mae) else None
        
    # 添加准确率指标到评估结果中
    metrics_dict.update({f"Accuracy_within_{k}%": v for k, v in accuracy_metrics.items()})
        
    # 构建完整的报告
    metrics_data = {
        "timestamp": datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        "evaluation_metrics": metrics_dict,
        "data_statistics": {
            "total_molecules": len(df) if df is not None else 0,
            "valid_molecules": len(dataset) if dataset is not None else 0,
            "skipped_molecules": skipped_molecules_count
        },
        "prediction_parameters": {
            "model_path": args.model_path if args is not None else "",
            "parameters_path": args.params_path if args is not None else "",
            "dataset_name": trained_dataname,
            "input_file": filename,
            "target_column": args.target_column if args is not None else "",
            "filter_column": args.filter_column if args is not None else "",
            "filter_value": args.filter_value if args is not None else "",
            "batch_size": args.batch_size if args is not None else 0
        }
    }
    
    return metrics_data


def save_metrics_report(metrics_data, output_filepath):
    """
    保存评估指标报告到JSON文件
    
    Args:
        metrics_data (dict): 评估指标数据
        output_filepath (str): 输出文件路径
    """
    with open(output_filepath, 'w') as f:
        json.dump(metrics_data, f, indent=4)
    print(f"Metrics saved to {output_filepath}")