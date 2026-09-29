import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import json
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, median_absolute_error


def plot_training_metrics(file_MAEs, loss_plot_file, json_file=None):
    """
    绘制训练过程中的各种评估指标曲线
    
    Args:
        file_MAEs (str): 包含训练指标的文件路径
        loss_plot_file (str): 保存图像的文件路径
        json_file (str, optional): 保存数据为JSON格式的文件路径
    """
    loss = pd.read_table(file_MAEs)
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Plot MAE
    axes[0,0].plot(loss['MAE_train'], color='r', label='MAE of train set')
    axes[0,0].plot(loss['MAE_dev'], color='b', label='MAE of validation set')
    axes[0,0].plot(loss['MAE_test'], color='y', label='MAE of test set')
    axes[0,0].set_ylabel('MAE')
    axes[0,0].set_xlabel('Epoch')
    axes[0,0].legend()
    axes[0,0].set_title('Mean Absolute Error')
    
    # Plot MSE
    axes[0,1].plot(loss['MSE_train'], color='r', label='MSE of train set')
    axes[0,1].plot(loss['MSE_dev'], color='b', label='MSE of validation set')
    axes[0,1].plot(loss['MSE_test'], color='y', label='MSE of test set')
    axes[0,1].set_ylabel('MSE')
    axes[0,1].set_xlabel('Epoch')
    axes[0,1].legend()
    axes[0,1].set_title('Mean Squared Error')
    
    # Plot R2
    axes[1,0].plot(loss['R2_train'], color='r', label='R2 of train set')
    axes[1,0].plot(loss['R2_dev'], color='b', label='R2 of validation set')
    axes[1,0].plot(loss['R2_test'], color='y', label='R2 of test set')
    axes[1,0].set_ylabel('R2')
    axes[1,0].set_xlabel('Epoch')
    axes[1,0].legend()
    axes[1,0].set_title('R-squared')
    
    # Plot PCC
    axes[1,1].plot(loss['PCC_train'], color='r', label='PCC of train set')
    axes[1,1].plot(loss['PCC_dev'], color='b', label='PCC of validation set')
    axes[1,1].plot(loss['PCC_test'], color='y', label='PCC of test set')
    axes[1,1].set_ylabel('PCC')
    axes[1,1].set_xlabel('Epoch')
    axes[1,1].legend()
    axes[1,1].set_title('Pearson Correlation Coefficient')
    
    plt.tight_layout()
    plt.savefig(loss_plot_file, dpi=300)
    plt.close()
    
    # 如果指定了json_file，则将数据保存为JSON格式
    if json_file:
        # 将DataFrame转换为字典格式
        data_dict = loss.to_dict(orient='list')
        # 保存为JSON文件
        with open(json_file, 'w') as f:
            json.dump(data_dict, f, indent=2)


def rmse(y_true, y_pred):
    """
    计算均方根误差
    
    Args:
        y_true: 真实值
        y_pred: 预测值
        
    Returns:
        float: RMSE值
    """
    return np.sqrt(mean_squared_error(y_true, y_pred))


def plot_predicted_vs_actual(file_test_result, cp_plot_file, property_mean=None, property_std=None, json_file=None):
    """
    绘制预测值vs实际值散点图
    
    Args:
        file_test_result (str): 测试结果文件路径
        cp_plot_file (str): 保存图像的文件路径
        property_mean (float, optional): 标准化均值
        property_std (float, optional): 标准化标准差
        json_file (str, optional): 保存数据为JSON格式的文件路径
    """
    res = pd.read_table(file_test_result)
    
    # 如果没有有效数据（只有表头），跳过评估指标计算和绘图
    if len(res) == 0:
        print("No valid predictions found. Skipping final evaluation and plotting.")
        return False
    
    # 标准化处理（如果提供了标准化参数） # UPDATE 目前版本 Tester 中会进行反标准化。
    if property_mean is not None and property_std is not None:
        res['Correct'] = (res['Correct'] - property_mean) / property_std
        res['Predict'] = (res['Predict'] - property_mean) / property_std
    
    # 获取数据范围用于调整图表
    min_val = min(res['Correct'].min(), res['Predict'].min())
    max_val = max(res['Correct'].max(), res['Predict'].max())
    range_val = max_val - min_val
    min_plot = min_val - 0.1 * range_val
    max_plot = max_val + 0.1 * range_val
    
    # 只有当样本数大于1时才计算R2和PCC，避免警告
    if len(res) > 1:
        r2 = r2_score(res['Correct'], res['Predict'])
        pcc = np.corrcoef(res['Correct'], res['Predict'])[0, 1]
    else:
        r2 = 0.0
        pcc = 0.0
    
    mae = mean_absolute_error(res['Correct'], res['Predict'])
    medae = median_absolute_error(res['Correct'], res['Predict'])
    mse = mean_squared_error(res['Correct'], res['Predict'])
    rmse_val = rmse(res['Correct'], res['Predict'])
    
    # 计算MRE指标时需要避免除以0的情况
    if len(res) > 0:
        # 检查是否有0值以避免除以0
        nonzero_mask = res['Correct'] != 0
        if np.sum(nonzero_mask) > 0:
            mean_re = np.mean(np.abs(res['Correct'][nonzero_mask] - res['Predict'][nonzero_mask]) / res['Correct'][nonzero_mask])
            median_re = np.median(np.abs(res['Correct'][nonzero_mask] - res['Predict'][nonzero_mask]) / res['Correct'][nonzero_mask])
        else:
            mean_re = 0.0
            median_re = 0.0
    else:
        mean_re = 0.0
        median_re = 0.0
    
    plt.plot(res['Correct'], res['Predict'], '.', color='blue')
    plt.plot([min_plot, max_plot], [min_plot, max_plot], color='red')
    plt.ylabel('Predicted RT')
    plt.xlabel('Experimental RT')
    
    # 根据数据范围调整文本位置
    text_x1 = min_plot + 0.05 * (max_plot - min_plot)
    text_x2 = min_plot + 0.5 * (max_plot - min_plot)
    text_y1 = max_plot - 0.05 * (max_plot - min_plot)
    text_y2 = max_plot - 0.15 * (max_plot - min_plot)
    text_y3 = max_plot - 0.25 * (max_plot - min_plot)
    text_y4 = max_plot - 0.35 * (max_plot - min_plot)
    text_y5 = max_plot - 0.45 * (max_plot - min_plot)
    text_y6 = max_plot - 0.55 * (max_plot - min_plot)
    
    plt.text(text_x1, text_y1, 'R2='+str(round(r2, 4)), fontsize=12)
    plt.text(text_x2, text_y1, 'MAE='+str(round(mae, 4)), fontsize=12)
    plt.text(text_x1, text_y2, 'MedAE='+str(round(medae, 4)), fontsize=12)
    plt.text(text_x2, text_y2, 'MSE='+str(round(mse, 4)), fontsize=12)
    plt.text(text_x1, text_y3, 'RMSE='+str(round(rmse_val, 4)), fontsize=12)
    plt.text(text_x2, text_y3, 'PCC='+str(round(pcc, 4)), fontsize=12)
    plt.text(text_x1, text_y4, 'MRE='+str(round(mean_re, 4)), fontsize=12)
    plt.text(text_x2, text_y4, 'MedRE='+str(round(median_re, 4)), fontsize=12)
    
    # 设置坐标轴范围
    plt.xlim(min_plot, max_plot)
    plt.ylim(min_plot, max_plot)
    
    plt.savefig(cp_plot_file, dpi=300)
    plt.close()
    
    # 如果指定了json_file，则将数据保存为JSON格式
    if json_file:
        # 创建包含所有相关信息的字典
        plot_data = {
            'data': {
                'correct': res['Correct'].tolist(),
                'predict': res['Predict'].tolist()
            },
            'metrics': {
                'r2': r2,
                'mae': mae,
                'medae': medae,
                'mse': mse,
                'rmse': rmse_val,
                'pcc': pcc,
                'mean_re': mean_re,
                'median_re': median_re
            },
            'plot_params': {
                'min_val': min_val,
                'max_val': max_val,
                'min_plot': min_plot,
                'max_plot': max_plot
            }
        }
        
        # 保存为JSON文件
        with open(json_file, 'w') as f:
            json.dump(plot_data, f, indent=2)
    
    return True