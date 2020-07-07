# -*- coding: utf-8 -*-
"""
Created on 2020/1/21 下午3:01

@Project -> File: pollution-forecast-offline-training-version-2 -> __init__.py

@Author: luolei

@Email: dreisteine262@163.com

@Describe: 初始化
"""

import sys

sys.path.append('../')

from mod.config.config_loader import config_loader

proj_dir, proj_cmap = config_loader.proj_dir, config_loader.proj_cmap

# 项目变量配置.
environ_config = config_loader.environ_config
model_config = config_loader.model_config
test_params = config_loader.test_params

# ============ 通用函数 ============

# ============ 环境变量 ============

# ============ 模型参数 ============
VALUE_TYPES_AVAILABLE = ['continuous', 'discrete']
METHODS_AVAILABLE = {
	'continuous': ['isometric', 'equifreq', 'quasi_chi2'],
	'discrete': ['label']
}

# ============ 测试参数 ============



