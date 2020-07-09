# -*- coding: utf-8 -*-
"""
Created on 2020/7/7 2:57 下午

@File: joint_binning.py

@Department: AI Lab, Rockontrol, Chengdu

@Author: luolei

@Email: dreisteine262@163.com

@Describe: 联合分箱
"""

import logging

logging.basicConfig(level = logging.INFO)

import sys, os

sys.path.append('../..')

import logging

logging.basicConfig(level = logging.INFO)

import warnings
import numpy as np
import sys

sys.path.append('../..')

from lib import VALUE_TYPES_AVAILABLE, METHODS_AVAILABLE
from lib.unsupervised_data_binning.series_binning import SeriesBinning

eps = 1e-12


def _check_value_type_and_method(value_type, method):
	if value_type == 'continuous':
		assert method in METHODS_AVAILABLE[value_type]
	elif value_type == 'discrete':
		assert method in METHODS_AVAILABLE[value_type]
	else:
		raise ValueError('ERROR: unknown value_type {}'.format(value_type))


def _check_data_and_params(data, value_types, methods, params):
	# 检查数据格式.
	try:
		N, D = data.shape
		assert (N > 0) & (D > 1)
	except:
		raise ValueError('ERROR: data_shape = {}'.format(data.shape))
	
	# 检查值类型.
	try:
		for t in value_types:
			assert t in VALUE_TYPES_AVAILABLE
	except:
		raise ValueError('ERROR: value_types = {}'.format(value_types))
	
	# 检查分箱方法.
	try:
		all_methods = []
		for key in METHODS_AVAILABLE.keys():
			m = METHODS_AVAILABLE[key]
			if type(m) == str:
				all_methods.append(m)
			elif type(m) == list:
				all_methods += m
			else:
				raise ValueError('ERROR: invalid type(m) == {}'.format(type(m)))
		
		for method in methods:
			assert method in all_methods
	except:
		raise ValueError('ERROR: methods = {}'.format(methods))
	
	# 检查数据维数.
	try:
		assert len(value_types) == data.shape[1]
		assert len(methods) == data.shape[1]
		assert len(params) == data.shape[1]
	except:
		raise ValueError('data dims = {}, value_types length = {}, methods length = {}'.format(
			data.shape[1], len(value_types), len(methods)))
	
	# 检查value_type与method匹配关系.
	try:
		for i in range(data.shape[1]):
			value_type = value_types[i]
			method = methods[i]
			_check_value_type_and_method(value_type, method)
	except:
		warnings.warn("WARNING: value_types and methods don't match")


class JointBinning(object):
	"""多变量联合分箱"""
	
	def __init__(self, arr: np.array, value_types: list):
		"""
		初始化
		:param data: np.array, shape = (N, D), N为数据样本数, D为样本维数
		:param value_types: list of strs, 各维数上的数据类型
		"""
		self.arr = arr
		self.N, self.D = self.arr.shape
		self.value_types = value_types
	
	# @time_cost
	def joint_binning(self, methods: list, params: list) -> (np.ndarray, list):
		"""
		联合分箱
		:param methods: list of strs, 各维数采用分箱方式
		:param params: list of dicts, 各维数对应分箱计算参数
		:return:
			hist: np.ndarray, 高维分箱后箱子中的频数
			edges: list of lists, 各维度上的标签

		Example:
		------------------------------------------------------------
		from mod.data_binning import gen_two_dim_samples
		value_types = ['discrete', 'continuous']
		methods = ['label', 'quasi_chi2']
		params = [{}, {'init_bins': 150, 'final_bins': 50}]
		data = gen_two_dim_samples(samples_len = 2000000, value_types = value_types)

		joint_binning_ = JointBinning(data, value_types, methods = methods, params = params)
		hist, edges = joint_binning_.joint_binning()
		------------------------------------------------------------
		"""
		_check_data_and_params(self.arr, self.value_types, methods, params)
		
		# 各维度序列边际分箱.
		edges = []
		for d in range(self.D):
			binning_ = SeriesBinning(self.arr[:, d], x_type = self.value_types[d])
			_, e_ = binning_.series_binning(method = methods[d], **params[d])
			edges.append(e_)
		
		# 在各个维度上将数据值向label进行插入, 返回插入位置.
		insert_locs_ = np.zeros_like(self.arr, dtype = int)     # TODO: 这里的arr值需要限制在序列插值范围内
		for d in range(self.D):
			insert_locs_[:, d] = np.searchsorted(edges[d], self.arr[:, d], side = 'left')
		
		# 将高维坐标映射到一维坐标上, 然后统计各一维坐标上的频率.
		edges_len_ = list(np.max(insert_locs_, axis = 0) + 1)
		ravel_locs_ = np.ravel_multi_index(insert_locs_.T, dims = edges_len_)
		hist = np.bincount(ravel_locs_, minlength = np.array(edges_len_).prod())
		
		# reshape转换形状.
		hist = hist.reshape(edges_len_)
		
		return hist, edges


if __name__ == '__main__':
	# ============ 载入数据和参数 ============
	import matplotlib.pyplot as plt
	from collections import defaultdict
	from lib import load_test_data
	from lib.numerical_info_entropy.univariate import UnivarInfoEntropy
	from lib.numerical_info_entropy.mutivariate import PairJointEntropy
	
	data = load_test_data(label = 'patient')
	
	# ============ 数据和参数准备 ============
	continuous_bins = {
		'AGE': 40, 'P': 50, 'MBP': 50, 'SHOCK_INDEX': 50, 'BMI': 50, 'RBC': 50,
		'HGB': 50, 'PLT': 50, 'WBC': 50, 'ALB': 50, 'CRE': 50, 'UA': 50, 'AST': 50,
		'ALT': 50, 'GLU': 50, 'TG': 50, 'CHO': 50, 'CA': 100, 'MG': 40, 'LDL': 50,
		'NA': 50, 'K': 50, 'CL': 50, 'GFR': 50, 'PT': 50, 'FIB': 50, 'DD': 50, 'CK': 50,
		'CAPRINI_SCORE': 10,
	}
	continuous_cols = list(continuous_bins.keys())
	
	x_col = 'CRE'
	y_col = 'VTE'
	x, y = list(data[x_col]), list(data[y_col])
	var_types = [
		'continuous' if x_col in continuous_cols else 'discrete',
		'continuous' if y_col in continuous_cols else 'discrete',
	]
	
	results = defaultdict(list)
	bins_ = 1
	
	# 边际熵.
	univar_entropy_x = UnivarInfoEntropy(x, var_types[0])
	univar_entropy_y = UnivarInfoEntropy(y, var_types[1])
	
	univar_entropy_x.do_series_binning(bins = bins_)
	univar_entropy_y.do_series_binning(bins = None)
	
	H_c_x = univar_entropy_x.cal_H_c()
	H_c_y = univar_entropy_y.cal_H_c()
	
	bins = [bins_, None]
	pair_joint_entropy = PairJointEntropy(x, y, var_types, bins)
	
	methods, params = [], []
	var_types = [pair_joint_entropy.x_type, pair_joint_entropy.y_type]
	for i in range(2):
		if var_types[i] == 'continuous':
			methods.append('isometric')
			params.append({'bins': pair_joint_entropy.bins[i]})
		else:
			methods.append('label')
			params.append({})
	
	# ============ 测试类 ============
	arr = np.vstack((pair_joint_entropy.x, pair_joint_entropy.y)).T
	value_types = [pair_joint_entropy.x_type, pair_joint_entropy.y_type]
	
	self = JointBinning(arr, value_types)



