# -*- coding: utf-8 -*-
"""
Created on 2020/7/9 2:00 下午

@File: multivariate_entropy.py

@Department: AI Lab, Rockontrol, Chengdu

@Author: luolei

@Email: dreisteine262@163.com

@Describe:
"""

import logging

logging.basicConfig(level = logging.INFO)

import numpy as np
import sys, os

sys.path.append('../..')

from lib import VALUE_TYPES_AVAILABLE
from lib import convert_series_values, drop_nans_and_add_noise
from lib.unsupervised_data_binning.multivariate_binning import MultivarBinning

eps = 1e-12


def _check_var_types(var_types: list):
	try:
		for type_ in var_types:
			if type_ not in VALUE_TYPES_AVAILABLE:
				raise ValueError('Invalid value type {}'.format(type_))
	except Exception as e:
		raise ValueError(e)
	
	
def _check_len_equal(x, y):
	try:
		assert len(x) == len(y)
	except:
		raise ValueError('Length x != length y')


class PairJointEntropy(object):
	"""二元变量联合信息熵"""
	
	def __init__(self, x: list or np.ndarray, y: list or np.ndarray, var_types: list):
		"""
		初始化
		:param x: 序列x
		:param y: 序列y
		:param var_types: 各序列值类型, 先后对应x和y
		:param bins: 各序列分箱数, 如果序列值为离散值则不采用该参数
		"""
		_check_var_types(var_types)
		_check_len_equal(x, y)
		
		x = convert_series_values(x, var_types[0])
		y = convert_series_values(y, var_types[1])
		
		# 去掉异常值并对连续值加入噪声.
		_arr = np.vstack((x, y)).T
		_arr = drop_nans_and_add_noise(_arr, var_types)
		
		self.x, self.y = _arr[:, 0], _arr[:, 1]
		self.x_type, self.y_type = var_types
		self.N, self.D = _arr.shape
		self.arr = _arr
		self.var_types = var_types
	
	@staticmethod
	def _get_probas(hist: np.ndarray) -> np.ndarray:
		"""根据频率计算概率"""
		f = hist.astype(np.float64)
		probas = f / (np.sum(f) + eps)
		return probas
	
	# def do_joint_binning(self, **params):
	# 	"""
	# 	进行序列联合分箱, 这里统一采用isometric等距分箱或label类别分箱形式
	# 	"""
	# 	self.bins_list = params['bins_list']
	# 	_arr = np.vstack((self.x, self.y)).T
	# 	joint_binning = MultivarBinning(_arr, self.var_types)
	#
	# 	methods_lst, params_lst = [], []
	# 	var_types = [self.x_type, self.y_type]
	# 	for i in range(2):
	# 		if var_types[i] == 'continuous':
	# 			methods_lst.append('isometric')
	# 			params_lst.append({'bins': self.bins_list[i]})
	# 		else:
	# 			methods_lst.append('label')
	# 			params_lst.append({})
	#
	# 	self.hist, _ = joint_binning.do_joint_binning(methods_lst, params_lst)  # TODO: 此处有bug
	# 	self.probas = self._get_probas(self.hist)  # type: np.ndarray
	
	# def cal_H(self):
	# 	probas_ = self.probas.flatten()
	# 	probas_ = probas_[probas_ > 0]
	# 	H = -np.dot(probas_, np.log2(probas_))
	# 	return H
	#
	# def cal_H_c(self):
	# 	_delta_lst = [
	# 		1 / self.bins_list[0] if self.x_type == 'continuous' else 1.0,
	# 		1 / self.bins_list[1] if self.y_type == 'continuous' else 1.0,
	# 	]
	#
	# 	probas_ = self.probas.flatten()
	# 	proba_dens_ = probas_ / (_delta_lst[0] * _delta_lst[1])
	#
	# 	_idxes = np.where(probas_ > 0)[0]  # 获得概率大于0的索引
	# 	probas_ = probas_[_idxes]
	# 	proba_dens_ = proba_dens_[_idxes]
	#
	# 	H_c = -np.dot(probas_, np.log2(proba_dens_))
	# 	return H_c
	#
	# def cal_H_delta(self):
	# 	_delta_lst = [
	# 		1 / self.bins_list[0] if self.x_type == 'continuous' else 1.0,
	# 		1 / self.bins_list[1] if self.y_type == 'continuous' else 1.0,
	# 	]
	# 	H_delta = -np.log2(_delta_lst[0]) - np.log2(_delta_lst[1])
	# 	return H_delta
	
	def do_joint_binning(self, **params):
		"""进行序列联合分箱, 这里统一采用isometric等距分箱或label类别分箱形式"""
		self.bins_list = params['bins_list']
		joint_binning = MultivarBinning(self.arr, self.var_types)

		# 分箱.
		binning_methods_, binning_params_ = [], []
		for i in range(self.D):
			if self.var_types[i] == 'continuous':
				binning_methods_.append('isometric')
				binning_params_.append({'bins': params['bins_list'][i]})
			else:
				binning_methods_.append('label')
				binning_params_.append({})

		self.hist, _ = joint_binning.do_joint_binning(binning_methods_, binning_params_)
		self.probas = self._get_probas(self.hist)   # type: np.ndarray

		# 计算各分箱区间长度备用.
		self.norm_interv_len_arr_list = []
		for i in range(self.D):
			if self.var_types[i] == 'continuous':
				_labels = [joint_binning.binning_bounds[i][0]] + joint_binning.edges[i]
				_interv_len_lst = [_labels[i + 1] - _labels[i] for i in range(len(joint_binning.edges[i]))]
				self.norm_interv_len_arr_list.append(
					np.array(
						[p / np.sum(_interv_len_lst) for p in _interv_len_lst],
						dtype = np.float64
					)
				)  # 区间长度归一化
			else:
				self.norm_interv_len_arr_list.append(
					(
						np.array(
							[1.0 for _ in range(self.hist.shape[i])],
							dtype = np.float64
						)
					)
				)
		
	def cal_H(self):
		probas_ = self.probas.flatten()
		probas_ = probas_[probas_ > 0]
		H = -np.dot(probas_, np.log2(probas_))
		return H

	def cal_H_c(self):
		_delta_lst = [np.mean(p) for p in self.norm_interv_len_arr_list]

		_probas = self.probas.flatten()
		_probas_dens = _probas / (_delta_lst[0] * _delta_lst[1])    # 仅二维

		_idxes = np.where(_probas > 0)[0]                           # 获得概率大于0的索引
		_probas = _probas[_idxes]
		_probas_dens = _probas_dens[_idxes]

		H_c = -np.dot(_probas, np.log2(_probas_dens))
		return H_c

	def cal_H_delta(self):
		_delta_lst = [np.mean(p) for p in self.norm_interv_len_arr_list]
		H_delta = 0.0
		for i in range(self.D):
			H_delta += -np.log2(_delta_lst[i])
		return H_delta
		

if __name__ == '__main__':
	# ============ 载入测试数据和参数 ============
	from collections import defaultdict
	import matplotlib.pyplot as plt
	from lib import load_test_data

	data = load_test_data(label = 'patient')

	# ============ 准备参数 ============
	x_col = 'CK'
	y_col = 'VTE'
	x_type, y_type = 'continuous', 'discrete'

	x, y = list(data[x_col]), list(data[y_col])
	var_types = [x_type, y_type]
	bins_list = [50, 50]

	# ============ 测试类 ============
	self = PairJointEntropy(x, y, var_types)
	self.do_joint_binning(bins_list = bins_list)
	
	# ============ 连续测试 ============
	bins_list = list(range(1, 500, 1))
	results = defaultdict(list)
	for bins in bins_list:
		self.do_joint_binning(bins_list = [bins, bins])

		# 总体离散熵.
		H = self.cal_H()
		H_c = self.cal_H_c()
		H_delta = self.cal_H_delta()

		results['H'].append(H)
		results['H_c'].append(H_c)
		results['H_delta'].append(H_delta)

	plt.figure(figsize = [6, 6])
	plt.plot(results['H'], label = 'H')
	plt.plot(results['H_c'], label = 'H_c')
	plt.plot(results['H_delta'], label = 'H_delta')
	plt.legend(loc = 'upper right', fontsize = 6.0)
	plt.xticks(fontsize = 6.0)
	plt.yticks(fontsize = 6.0)
	plt.xlabel('bins', fontsize = 8.0)
	plt.ylabel('entropy', fontsize = 8.0)
	plt.show()

	



