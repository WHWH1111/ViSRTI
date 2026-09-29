#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
预测结果绘图工具函数

Created for predict.py
"""

import numpy as np
import matplotlib.pyplot as plt


def plot_error_distribution(predictions, filename, output_dir_with_timestamp):
    """
    绘制预测误差分布饼图
    
    Args:
        predictions (list): 包含(smiles, predicted, actual)元组的列表
        filename (str): 输入文件名
        output_dir_with_timestamp (str): 带时间戳的输出目录路径
        
    Returns:
        str: 生成的饼图文件路径
    """
    if len(predictions) == 0:
        return None
    
    preds = np.array([float(p[1]) for p in predictions])
    actuals = np.array([float(p[2]) for p in predictions])
    
    # 检查并处理NaN或无穷大值
    valid_mask = np.isfinite(preds) & np.isfinite(actuals)
    if not np.all(valid_mask):
        preds = preds[valid_mask]
        actuals = actuals[valid_mask]
    
    if len(preds) == 0 or len(actuals) == 0:
        return None
    
    # 计算相对误差百分比
    relative_errors = np.abs((preds - actuals) / actuals) * 100
    
    # 定义误差区间
    bins = [0, 5, 10, 20, 30, np.inf]
    labels = ['≤5%', '5-10%', '10-20%', '20-30%', '>30%']
    
    # 计算各区间的样本数量
    counts, _ = np.histogram(relative_errors, bins=bins)
    total_count = len(relative_errors)
    
    # 计算各区间的百分比
    percentages = [(count / total_count) * 100 for count in counts]
    
    # 创建饼图
    plt.figure(figsize=(10, 8))
    colors = ['#ff9999','#66b3ff','#99ff99','#ffcc99','#ff99cc']
    wedges, texts, autotexts = plt.pie(percentages, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
    
    # 添加图例
    plt.legend(wedges, [f'{label}: {count} ({percentage:.1f}%)' for label, count, percentage in zip(labels, counts, percentages)], 
              title="Error Distribution", loc="center left", bbox_to_anchor=(1, 0, 0.5, 1))
    
    plt.title(f'Prediction Error Distribution for {filename}')
    plt.axis('equal')
    
    # 保存饼图
    pie_chart_filename = filename.replace('.csv', '_error_distribution.png')
    pie_chart_filepath = f"{output_dir_with_timestamp}/{pie_chart_filename}"
    plt.tight_layout()
    plt.savefig(pie_chart_filepath, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Error distribution pie chart saved to {pie_chart_filepath}")
    
    return pie_chart_filepath