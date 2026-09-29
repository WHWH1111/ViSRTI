#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
特征配置工具函数和常量定义

用于统一管理模型训练和预测中的特征配置
"""

from collections import OrderedDict

# 定义默认特征配置
DEFAULT_FEATURE_CONFIG = {
    'basic': [
        'Monoiso_Mass',  # 单同位素质量
        'M+H+',          # M+H⁺质量
        'M-H-'           # M-H⁻质量
    ],
    'physchem': [
        'Monoiso_Mass', 'average_mass', 'M+H+', 'M-H-',
        'logKow_EPISuite', 'Exp_logKow_EPISuite', 'alogp_ChemSpider', 'xlogp_ChemSpider',
        'Koc_min_experimental (L/kg)', 'Koc_max_experimental (L/kg)', 
        'Koc_min_predicted (L/kg)', 'Koc_max_predicted (L/kg)'
    ],
    'toxicity': [
        'Tetrahymena_pyriformis_toxicity', 'Daphnia_toxicity', 
        'Algae_toxicity', 'Pimephales_promelas_toxicity'
    ],
    'chromatography': [
        'Prob. +ESI', 'Prob. -ESI'
    ]
}

# 定义所有特征的完整配置
ALL_FEATURES_CONFIG = OrderedDict([
    ('monoiso_mass', 'Monoiso_Mass'),           # 单同位素质量
    ('average_mass', 'average_mass'),           # 平均质量
    ('m_plus_h', 'M+H+'),                       # M+H⁺质量
    ('m_minus_h', 'M-H-'),                      # M-H⁻质量
    ('logKow_EPISuite', 'logKow_EPISuite'),
    ('Exp_logKow_EPISuite', 'Exp_logKow_EPISuite'),
    ('alogp_ChemSpider', 'alogp_ChemSpider'),
    ('xlogp_ChemSpider', 'xlogp_ChemSpider'),
    ('Koc_min_experimental', 'Koc_min_experimental (L/kg)'),
    ('Koc_max_experimental', 'Koc_max_experimental (L/kg)'),
    ('Koc_min_predicted', 'Koc_min_predicted (L/kg)'),
    ('Koc_max_predicted', 'Koc_max_predicted (L/kg)'),
    ('Tetrahymena_toxicity', 'Tetrahymena_pyriformis_toxicity'),
    ('Daphnia_toxicity', 'Daphnia_toxicity'),
    ('Algae_toxicity', 'Algae_toxicity'),
    ('Pimephales_toxicity', 'Pimephales_promelas_toxicity'),
    ('Prob_positive_ESI', 'Prob. +ESI'),
    ('Prob_negative_ESI', 'Prob. -ESI')
])


def get_feature_config(config_name):
    """
    根据配置名称获取特征配置
    
    Args:
        config_name (str): 配置名称 ('basic', 'physchem', 'toxicity', 'chromatography', 'all')
                        也支持逗号分隔的多个配置名称，如 'basic,toxicity'
        
    Returns:
        list: 特征列表
    """
    # 处理逗号分隔的多个配置
    if isinstance(config_name, str) and ',' in config_name:
        config_names = [name.strip() for name in config_name.split(',')]
        combined_features = []
        for name in config_names:
            if name == 'all':
                # 如果包含'all'，直接返回所有特征
                return get_feature_config('all')
            else:
                features = DEFAULT_FEATURE_CONFIG.get(name, [])
                # 添加不重复的特征
                for feature in features:
                    if feature not in combined_features:
                        combined_features.append(feature)
        return combined_features
    elif config_name == 'all':
        # 返回所有特征
        return [
            'Monoiso_Mass', 'average_mass', 'M+H+', 'M-H-',
            'logKow_EPISuite', 'Exp_logKow_EPISuite', 'alogp_ChemSpider', 'xlogp_ChemSpider',
            'Koc_min_experimental (L/kg)', 'Koc_max_experimental (L/kg)', 
            'Koc_min_predicted (L/kg)', 'Koc_max_predicted (L/kg)',
            'Tetrahymena_pyriformis_toxicity', 'Daphnia_toxicity', 
            'Algae_toxicity', 'Pimephales_promelas_toxicity',
            'Prob. +ESI', 'Prob. -ESI'
        ]
    else:
        # 返回指定的特征配置
        return DEFAULT_FEATURE_CONFIG.get(config_name, [])


def get_feature_display_name(feature_key):
    """
    根据特征键名获取显示名称
    
    Args:
        feature_key (str): 特征键名
        
    Returns:
        str: 特征显示名称
    """
    return ALL_FEATURES_CONFIG.get(feature_key, feature_key)
