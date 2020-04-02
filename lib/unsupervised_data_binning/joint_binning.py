# -*- coding: utf-8 -*-
"""
Created on 2020/3/8 13:27

@Project -> File: algorithm-tools -> joint_binning.py

@Author: luolei

@Email: dreisteine262@163.com

@Describe: 联合分箱
"""

import logging

logging.basicConfig(level = logging.INFO)

from lake.decorator import time_cost
import warnings
import numpy as np
import sys

sys.path.append('../..')

from lib.unsupervised_data_binning import VALUE_TYPES_AVAILABLE, METHODS_AVAILABLE
from lib.unsupervised_data_binning.series_binning import SeriesBinning


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
	"""联合分箱"""
	
	def __init__(self, data: np.array, value_types: list, methods: list, params: list):
		"""
		初始化
		:param data: np.array, shape = (N, D), N为数据样本数, D为样本维数
		:param value_types: list of strs, 各维数上的数据类型
		:param methods: list of strs, 各维数采用分箱方式
		:param params: list of dicts, 各维数对应分箱计算参数
		"""
		_check_data_and_params(data, value_types, methods, params)
		
		self.data = data
		self.N, self.D = self.data.shape
		self.value_types = value_types
		self.methods = methods
		self.params = params
	
	@time_cost
	def joint_binning(self) -> (np.ndarray, list):
		"""
		联合分箱
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
		# 各维度序列边际分箱.
		edges = []
		# edges_len_ = []
		for d in range(self.D):
			binning_ = SeriesBinning(self.data[:, d], x_type = self.value_types[d])
			_, e_ = binning_.series_binning(method = self.methods[d], params = self.params[d])
			edges.append(e_)
		
		# 在各个维度上将数据值向label进行插入, 返回插入位置.
		insert_locs_ = np.zeros_like(self.data, dtype = int)
		for d in range(self.D):
			insert_locs_[:, d] = np.searchsorted(edges[d], self.data[:, d], side = 'left')
		
		# 将高维坐标映射到一维坐标上, 然后统计各一维坐标上的频率.
		edges_len_ = list(np.max(insert_locs_, axis = 0) + 1)
		ravel_locs_ = np.ravel_multi_index(insert_locs_.T, dims = edges_len_)
		hist = np.bincount(ravel_locs_, minlength = np.array(edges_len_).prod())
		
		# reshape转换形状.
		hist = hist.reshape(edges_len_)
		
		return hist, edges
