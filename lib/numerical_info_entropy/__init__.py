# -*- coding: utf-8 -*-
"""
Created on 2020/7/8 1:45 下午

@File: __init__.py.py

@Department: AI Lab, Rockontrol, Chengdu

@Author: luolei

@Email: dreisteine262@163.com

@Describe:
"""

import logging

logging.basicConfig(level = logging.INFO)

from collections import defaultdict
import numpy as np
import sys, os

sys.path.append('../..')


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





