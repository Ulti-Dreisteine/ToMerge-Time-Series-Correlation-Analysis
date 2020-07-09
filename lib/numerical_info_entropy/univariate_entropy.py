# -*- coding: utf-8 -*-
"""
Created on 2020/7/9 1:43 下午

@File: univariate_entropy.py

@Department: AI Lab, Rockontrol, Chengdu

@Author: luolei

@Email: dreisteine262@163.com

@Describe: 单变量熵计算
"""

import logging

logging.basicConfig(level = logging.INFO)

import numpy as np
import sys, os

sys.path.append('../..')

from lib import VALUE_TYPES_AVAILABLE
from lib import convert_series_values
from lib.unsupervised_data_binning.univariate_binning import UnivarBinning

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
		
		# 序列值转换.
		x = convert_series_values(x, x_type)
		
		# 默认删除数据中的nan值.
		self.x = x[~np.isnan(x)]
		self.x_type = x_type
		self.N = len(self.x)            # 样本量
		
		# 加入微量噪音.
		if self.x_type == 'continuous':
			self.x += eps * np.random.random(self.N)
	
	@staticmethod
	def _get_probas(freq_ns: list) -> np.ndarray:
		"""根据频率计算概率"""
		f = np.array(freq_ns, dtype = np.float64)
		probas = f / (np.sum(f) + eps)
		return probas
	
	def do_univar_binning(self, **params):
		"""
		进行序列分箱, 这里统一采用isometric等距分箱形式
		:param bins: 等距分箱个数
		"""
		_binning = UnivarBinning(self.x, self.x_type)
		
		# 分箱.
		if self.x_type == 'continuous':
			self.freq_ns, self.labels = _binning.isometric_binning(bins = params['bins'])
		else:
			self.freq_ns, self.labels = _binning.label_binning()
		
		self.probas = self._get_probas(self.freq_ns)  # type: np.ndarray
		
		# 计算各分箱区间长度备用.
		if self.x_type == 'continuous':
			_labels = [_binning.binning_bounds[0]] + self.labels
			_interv_len_lst = [_labels[i + 1] - _labels[i] for i in range(len(self.labels))]
			self.norm_interv_len_arr = np.array(
				[p / np.sum(_interv_len_lst) for p in _interv_len_lst],
				dtype = np.float64
			)  # 区间长度归一化
		else:
			self.norm_interv_len_arr = np.array(
				[1.0 for _ in range(len(self.freq_ns))],
				dtype = np.float64
			)
			
	def cal_H(self):
		"""计算信息熵"""
		_probas = self.probas[self.probas > 0]
		H = -np.dot(_probas, np.log2(_probas))
		return H
	
	def cal_H_c(self):
		_idxes = np.where(self.probas > 0)[0]                   # 获得概率大于0的索引
		_probas = self.probas[_idxes]
		_norm_interv_len_arr = self.norm_interv_len_arr[_idxes]
		H_c = -np.dot(_probas, np.log2(_probas / _norm_interv_len_arr))
		return H_c
	
	def cal_H_delta(self):
		_delta = np.mean(self.norm_interv_len_arr)              # 这个式子只适用于isometric分箱
		H_delta = - np.log2(_delta)
		return H_delta
			
			
if __name__ == '__main__':
	# ============ 载入测试数据和参数 ============
	from collections import defaultdict
	from lib import load_test_data
	
	data = load_test_data(label = 'patient')
	
	# ============ 测试连续值分箱 ============
	col = 'K'
	x_type = 'continuous'
	x = np.array(data[col])
	
	bins = 50
	self = UnivarInfoEntropy(x, x_type)
	self.do_univar_binning(bins = bins)
	
	# ============ 测试连续变量分箱参数设置 ============
	import matplotlib.pyplot as plt
	
	bins_list = list(range(1, 500, 1))
	results = defaultdict(list)
	for bins in bins_list:
		self.do_univar_binning(bins = bins)
		
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
	plt.plot(results['H'], label = 'H')
	plt.plot(results['H_c'], label = 'H_c')
	plt.plot(results['H_delta'], label = 'H_delta')
	plt.plot(results['-log(Delta)'], label = '-log(Delta)')
	plt.plot(results['H_c_inf'], label = 'H_c_inf')
	plt.legend(loc = 'upper right', fontsize = 6.0)
	plt.xticks(fontsize = 6.0)
	plt.yticks(fontsize = 6.0)
	plt.xlabel('bins', fontsize = 8.0)
	plt.ylabel('entropy', fontsize = 8.0)
	plt.show()
	
	



