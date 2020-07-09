# -*- coding: utf-8 -*-
"""
Created on 2020/7/8 1:45 下午

@File: univariate.py

@Department: AI Lab, Rockontrol, Chengdu

@Author: luolei

@Email: dreisteine262@163.com

@Describe: 单变量信息熵数值计算
"""

import logging

logging.basicConfig(level = logging.INFO)

from collections import defaultdict
import numpy as np
import sys, os

sys.path.append('../..')

from lib import VALUE_TYPES_AVAILABLE
from lib.numerical_info_entropy import convert_series_values
from lib.unsupervised_data_binning.series_binning import SeriesBinning

eps = 1e-12


class UnivarInfoEntropy(object):
	"""单变量信息熵"""
	
	def __init__(self, x: list or np.ndarray, x_type: str):
		"""
		初始化
		Note:
			* 载入序列如果为连续值, 则统一为np.float64值类型序列; 若为离散值, 则尝试将离散值转为np.float16数值序列;
			
		:param x: 待分箱一维序列
		:param x_type: 序列值类型, {'continuous', 'label'}
		"""
		# 检查参数x_type, 必须在VALUE_TYPES_AVAILABLE里选择.
		if x_type not in VALUE_TYPES_AVAILABLE:
			raise ValueError('Invalid x_type {}'.format(x_type))
		
		# 检查序列数值, 连续值和离散值输入序列必须转为数值形式.
		# 连续值为np.float64, 离散值为np.float32.
		# _d_types = {'continuous': np.float64, 'discrete': np.float16}
		# try:
		# 	x = np.array(x, dtype = _d_types[x_type]).flatten()
		# except:
		# 	if x_type == 'discrete':
		# 		try:
		# 			x = convert_series_label2number(x)
		# 			# x = np.array(x).flatten()
		# 			# _map = defaultdict(int)
		# 			# _labels = list(np.unique(x))
		# 			# for i in range(len(_labels)):
		# 			# 	_label = _labels[i]
		# 			# 	_map[_label] = i
		# 			#
		# 			# for k, v in _map.items():
		# 			# 	x[x == k] = int(v)
		# 			# x = np.array(x, dtype = _d_types[x_type])
		# 		except:
		# 			raise RuntimeError('Cannot convert x into numpy.ndarray with numerical values np.float64 or np.float16')
		# 	else:
		# 		raise RuntimeError('Cannot convert x into numpy.ndarray with numerical values np.float64 or np.float16')
		x = convert_series_values(x, x_type)
		
		self.x = x[~np.isnan(x)]  # 这一步默认删除了数据中的nan值
		self.x_type = x_type
		self.N = len(self.x)  # 样本量
		
		# 加入微量噪音.
		if self.x_type == 'continuous':
			self.x += eps * np.random.random(self.N)
			
	@staticmethod
	def _get_probas(freq_ns: list) -> np.ndarray:
		"""根据频率计算概率"""
		f = np.array(freq_ns, dtype = np.float64)
		probas = f / (np.sum(f) + eps)
		return probas
	
	def do_series_binning(self, **params):
		"""
		进行序列分箱, 这里统一采用isometric等距分箱形式
		:param bins: 等距分箱个数
		"""
		binning = SeriesBinning(self.x, self.x_type)
		
		if self.x_type == 'continuous':
			self.freq_ns, self.labels = binning.isometric_binning(bins = params['bins'])
		else:
			self.freq_ns, self.labels = binning.label_binning()
			
		self.probas = self._get_probas(self.freq_ns)                # type: np.ndarray
		
		# 对于连续变量, 计算各区间长度备用.
		if self.x_type == 'continuous':
			_labels = [binning.normal_bounds[0]] + self.labels
			_interv_len_lst = [_labels[i + 1] - _labels[i] for i in range(len(self.labels))]
			self.norm_interv_len_arr = np.array(
				[p / np.sum(_interv_len_lst) for p in _interv_len_lst],
				dtype = np.float64
			)                                                       # 区间长度归一化
			
	def cal_H(self):
		"""计算信息熵"""
		probas_ = self.probas[self.probas > 0]
		H = -np.dot(probas_, np.log2(probas_))
		return H
	
	def cal_H_c(self):
		if self.x_type == 'continuous':
			_idxes = np.where(self.probas > 0)[0]                   # 获得概率大于0的索引
			_probas = self.probas[_idxes]
			_norm_interv_len_arr = self.norm_interv_len_arr[_idxes]
			H_c = -np.dot(_probas, np.log2(_probas / _norm_interv_len_arr))
			return H_c
		else:
			return self.cal_H()
	
	def cal_H_delta(self):
		if self.x_type == 'continuous':
			_delta = np.mean(self.norm_interv_len_arr)
			H_delta = - np.log2(_delta)
			return H_delta
		else:
			return 0.0
		

if __name__ == '__main__':
	# ============ 载入数据和参数 ============
	import pandas as pd
	from lib import proj_dir
	from lib import load_test_data
	
	data = load_test_data(label = 'patient')
	
	# ============ 测试类 ============
	x_col = 'CAPRINI_SCORE'
	x = list(data[x_col])
	x_type = 'continuous'
	
	self = UnivarInfoEntropy(x, x_type)
	
	# bins = 50
	# self.do_series_binning(bins = bins)
	# H = self.cal_H()
	# H_c = self.cal_H_c()
	# H_delta = self.cal_H_delta()
	
	# ============ 测试连续变量分箱参数设置 ============
	import matplotlib.pyplot as plt
	bins_list = list(range(1, 100, 1))
	results = defaultdict(list)
	for bins in bins_list:
		self.do_series_binning(bins = bins)

		# 总体离散熵.
		H = self.cal_H()
		H_c = self.cal_H_c()
		H_delta = self.cal_H_delta()

		results['H'].append(H)
		results['H_c'].append(H_c)
		results['H_delta'].append(H_delta)
		results['-log(Delta)'].append(-np.log2(self.norm_interv_len_arr[0] + 1e-12))
		results['H_c_inf'].append(
			-np.log2(
				bins / self.N + 1e-12
			)
		)

	plt.figure(figsize = [6, 6])
	# plt.plot(results['H'], label = 'H')
	plt.plot(results['H_c'], label = 'H_c')
	# plt.plot(results['H_delta'], label = 'H_delta')
	# plt.plot(results['-log(Delta)'], label = '-log(Delta)')
	# plt.plot(results['H_c_inf'], label = 'H_c_inf')
	plt.legend(loc = 'upper right', fontsize = 6.0)
	plt.xticks(fontsize = 6.0)
	plt.yticks(fontsize = 6.0)
	plt.xlabel('bins', fontsize = 8.0)
	plt.ylabel('entropy', fontsize = 8.0)
	plt.show()
	