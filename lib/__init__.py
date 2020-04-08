# -*- coding: utf-8 -*-
"""
Created on 2020/4/8 21:56

@Project Name: time-series-analysis

@File: __init__.py

@Author: luolei

@Email: dreisteine262@163.com

@Describe: 
"""

import logging

logging.basicConfig(level = logging.INFO)

import sys

sys.path.append('../')

from mod.config.config_loader import config

proj_dir, proj_cmap = config.proj_dir, config.proj_cmap