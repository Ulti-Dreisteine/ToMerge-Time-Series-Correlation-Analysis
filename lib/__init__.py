# -*- coding: utf-8 -*-
"""
Created on 2020/7/9 12:28 下午

@File: __init__.py.py

@Department: AI Lab, Rockontrol, Chengdu

@Author: luolei

@Email: dreisteine262@163.com

@Describe: 项目初始化脚本
"""

from collections import defaultdict
import pandas as pd
import numpy as np
import sys, os

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
def load_test_data(label: str):
	data = None
	if label == 'pollution':
		data = pd.read_csv(os.path.join(proj_dir, 'data/raw/pollutant_conc_and_weather.csv'))
	elif label == 'patient':
		data = pd.read_excel(os.path.join(proj_dir, 'data/raw/patient_info.xlsx'))
	else:
		pass
	return data


def _convert_series_label2number(x: list or np.ndarray):
	"""将离散值序列中的类别转为数值"""
	x = np.array(x).flatten()
	_map = defaultdict(int)
	_labels = list(np.unique(x))
	for i in range(len(_labels)):
		_label = _labels[i]
		_map[_label] = i
	
	for k, v in _map.items():
		x[x == k] = int(v)
	
	x = np.array(x, dtype = np.float16)
	return x


def convert_series_values(x: list or np.ndarray, x_type: str):
	"""单序列值转换, 连续值转为np.float64, 离散值转换为np.float16"""
	_d_types = {'continuous': np.float64, 'discrete': np.float16}
	try:
		x = np.array(x, dtype = _d_types[x_type]).flatten()
		return x
	except:
		if x_type == 'discrete':
			try:
				x = _convert_series_label2number(x)
				return x
			except:
				raise RuntimeError('Cannot convert x into numpy.ndarray with numerical values np.float64 or np.float16')
		else:
			raise RuntimeError('Cannot convert x into numpy.ndarray with numerical values np.float64 or np.float16')



