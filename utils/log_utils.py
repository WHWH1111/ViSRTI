import os
import torch
from datetime import datetime
import numpy as np

import pickle


def standardize_dataset(dataset):
    """Standardize the property values in the dataset and return mean and std"""
    properties = [data[-1] for data in dataset]
    properties = np.array(properties)
    
    mean = np.mean(properties)
    std = np.std(properties)
    
    # 标准化数据集中的属性值
    standardized_dataset = []
    for data in dataset:
        if len(data) == 5:  # 原始格式
            smiles, fingerprints, adjacency, molecular_size, property_value = data
            standardized_property = (property_value - mean) / std
            standardized_dataset.append((smiles, fingerprints, adjacency, molecular_size, standardized_property))
        else:  # 扩展格式 (6个元素)
            smiles, fingerprints, adjacency, molecular_size, additional_features, property_value = data
            standardized_property = (property_value - mean) / std
            standardized_dataset.append((smiles, fingerprints, adjacency, molecular_size, additional_features, standardized_property))
    
    return standardized_dataset, mean, std


def print_training_setup_info(device):
    """
    打印训练设备信息
    
    Args:
        device: 训练设备 (CPU/GPU)
    """
    if device.type == 'cuda':
        print('The code uses a GPU!')
    else:
        print('The code uses a CPU...')


def print_dataset_split_info(train_file_path, test_file_path, raw_data_path):
    """
    打印数据集拆分信息
    
    Args:
        train_file_path (str): 训练集文件路径
        test_file_path (str): 测试集文件路径
        raw_data_path (str): 原始数据文件路径
    """
    print("Train or test dataset files not found. Splitting from raw CSV file...")
    print(f"Raw data file {raw_data_path} not found!")


def print_preprocess_info(dataset_train, dataset_dev, dataset_test):
    """
    打印数据预处理完成信息
    
    Args:
        dataset_train: 训练数据集
        dataset_dev: 验证数据集
        dataset_test: 测试数据集
    """
    print('-'*100)
    print('The preprocess has finished!')
    print('# of training data samples:', len(dataset_train))
    print('# of development data samples:', len(dataset_dev))
    print('# of test data samples:', len(dataset_test))
    print('-'*100)
    print('Creating a model.')


def print_model_info(model):
    """
    打印模型信息
    
    Args:
        model: GNN模型
    """
    print('# of model parameters:',
          sum([np.prod(p.size()) for p in model.parameters()]))
    print('-'*100)


def print_cuda_memory_info():
    """打印CUDA内存信息"""
    if torch.cuda.is_available():
        device = torch.cuda.current_device()
        memory_allocated = torch.cuda.memory_allocated(device) / 1024**3
        memory_reserved = torch.cuda.memory_reserved(device) / 1024**3
        print(f"CUDA memory allocated: {memory_allocated:.2f} GB")
        print(f"CUDA memory reserved: {memory_reserved:.2f} GB")


def print_training_start_info():
    """打印训练开始信息"""
    print("----------------------------------------------------------------------------------------------------")
    print("Start training.")
    print("The result is saved in the output directory every epoch!")
    # 添加CUDA内存信息输出
    print_cuda_memory_info()


def print_training_time_estimate(iteration, time):
    """
    打印训练时间预估信息
    
    Args:
        iteration (int): 迭代次数
        time (float): 已用时间
    """
    minutes = time * iteration / 60
    hours = int(minutes / 60)
    minutes = int(minutes - 60 * hours)
    print('The training will finish in about',
          hours, 'hours', minutes, 'minutes.')


def print_training_header(result_header):
    """
    打印训练结果表头
    
    Args:
        result_header (str): 结果表头
    """
    print('-'*100)
    print(result_header)


def print_early_stopping_info(epoch):
    """
    打印早停信息
    
    Args:
        epoch (int): 当前轮次
    """
    print(f"Early stopping triggered after {epoch} epochs.")


def save_training_params(params_dict, log_dir):
    """
    保存训练参数到日志目录
    
    Args:
        params_dict (dict): 参数字典
        log_dir (str): 日志目录路径
    """
    # 保存为文本格式
    params_file = os.path.join(log_dir, 'training_params.txt')
    with open(params_file, 'w') as f:
        f.write("Training Parameters:\n")
        for key, value in params_dict.items():
            f.write(f"{key}: {value}\n")
    
    # 同时保存为JSON格式，便于程序读取
    import json
    params_json_file = os.path.join(log_dir, 'training_params.json')
    with open(params_json_file, 'w') as f:
        json.dump(params_dict, f, indent=4, default=str)  # 添加default=str以处理numpy类型


def create_log_dir(
        path, dataname, dim, layer_hidden, layer_output, batch_train, lr,
        iteration, debug_size=None, output_dir=None):
    """
    创建日志目录
    
    Args:
        path (str): 数据路径
        dataname (str): 数据集名称
        dim (int): 维度
        layer_hidden (int): 隐藏层数
        layer_output (int): 输出层数
        batch_train (int): 训练批次大小
        lr (float): 学习率
        iteration (int): 迭代次数
        debug_size (int): 调试模式下的数据量
        
    Returns:
        dict: 包含所有日志文件路径和timestamp的字典
    """
    # 创建基于时间和参数的目录
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    params_str = f"dim{dim}_layerH{layer_hidden}_layerO{layer_output}_batch{batch_train}_lr{lr}_iter{iteration}"
    if debug_size is not None:
        params_str += f"_debug{debug_size}"
    # log_dir = f'{path}/train_{timestamp}_{params_str}'
    log_dir = (
        os.path.abspath(output_dir)
        if output_dir is not None
        else f'./log/train_{timestamp}_{dataname}_{params_str}'
    )
    # 确保日志目录存在
    os.makedirs(os.path.dirname(log_dir), exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    
    # 定义保存路径
    file_paths = {
        'log_dir': log_dir,
        'file_MAEs': os.path.join(log_dir, 'MAEs.txt'),
        'file_test_result': os.path.join(log_dir, 'test_prediction.txt'),
        'file_predictions': os.path.join(log_dir, 'train_prediction.txt'),
        'file_model': os.path.join(log_dir, 'model.pt'),
        'loss_plot_file': os.path.join(log_dir, 'loss.png'),
        'cp_plot_file': os.path.join(log_dir, 'c-p.png'),
        'timestamp': timestamp
    }
    
    return file_paths


def split_dataset(dataset, ratio):
    """Split a dataset into two subsets according to the ratio."""
    n = int(ratio * len(dataset))
    dataset_train = dataset[:n]
    dataset_test = dataset[n:]
    return dataset_train, dataset_test


def dump_dictionary(dictionary, filename):
    """Dump dictionary to file."""
    with open(filename, 'wb') as f:
        pickle.dump(dict(dictionary), f)
