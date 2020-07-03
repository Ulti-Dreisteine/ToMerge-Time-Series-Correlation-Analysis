# -*- coding: utf-8 -*-
"""
Created on 2020/7/3 5:11 下午

@File: __init__.py

@Department: AI Lab, Rockontrol, Chengdu

@Author: luolei

@Email: dreisteine262@163.com

@Describe:
"""

import logging

logging.basicConfig(level = logging.INFO)

import sys, os

sys.path.append('../')

from mod.config.config_loader import config_loader

proj_dir, proj_cmap = config_loader.proj_dir, config_loader.proj_cmap

# 项目变量配置.
environ_config = config_loader.environ_config
model_config = config_loader.model_config
test_params = config_loader.test_params

# ============ 环境变量 ============

# ============ 模型参数 ============

# ============ 测试参数 ============





