# -*- coding: utf-8 -*-
"""
Created on 2020/4/7 13:34

@Project -> File: gujiao-power-plant-optimization -> mutual_info_entropy.py

@Author: luolei

@Email: dreisteine262@163.com

@Describe: 互信息熵计算
"""

import logging

logging.basicConfig(level = logging.INFO)

from lake.decorator import time_cost
import numpy as np
import copy
import sys

sys.path.append('../..')

from lib.unsupervised_data_binning import VALUE_TYPES_AVAILABLE, METHODS_AVAILABLE
from lib.unsupervised_data_binning.series_binning import SeriesBinning
from lib.unsupervised_data_binning.joint_binning import JointBinning

eps = 1e-6


def _check_value_types_and_methods(value_types: list, methods: list):
	# 参数长度相同.
	assert len(value_types) == len(methods)
	
	# 所选定method必须与value_type相匹配.
	for i in range(len(value_types)):
		value_type = value_types[i]
		method = methods[i]
		try:
			assert method in METHODS_AVAILABLE[value_type]
		except:
			raise ValueError('ERROR: method {} does not match value_type {}'.format(method, value_type))


def _get_probability_from_freqs(freq_ns: np.ndarray):
	freq_sum = np.sum(freq_ns)
	probs = freq_ns.copy() / freq_sum
	return probs


def _get_entropy_from_freqs(freq_ns_arr: np.ndarray):
	"""
	通过频率分布求解熵值
	"""
	probs = _get_probability_from_freqs(freq_ns_arr.copy())
	log_probs = np.log(probs + eps)
	entropy = - np.dot(probs.flatten(), log_probs.flatten())
	return entropy


class PairwiseMutualInfoEntropy(object):
	"""
	成对互信息熵检验
	"""

	def __init__(self, x: list or np.ndarray, y: list or np.ndarray, value_types: list):
		"""
		初始化
		:param x: np.ndarray, 一维数组
		:param y: np.ndarray, 一维数组
		:param value_types: list of strs, like ['discrete', 'continuous'], x和y的值类型
		"""
		x = np.array(x).flatten()
		y = np.array(y).flatten()

		# 参数检查.
		if len(x) != len(y):
			raise ValueError('Series x and y are not in the same length')

		if len(value_types) != 2:
			raise ValueError

		for value_type in value_types:
			if value_type not in VALUE_TYPES_AVAILABLE:
				raise ValueError('Value type {} is not in VALUE_TYPES_AVAILABLE = {}'.format(value_type, VALUE_TYPES_AVAILABLE))

		self.x = x
		self.y = y
		self.D = 2
		self.N = len(self.x)
		self.value_types = value_types

		self._add_noise()

	def _add_noise(self):
		"""
		数据中加入微量噪声
		"""
		if self.value_types[0] == 'continuous':
			self.x += 1e-12 * np.random.random(self.N)
		if self.value_types[1] == 'continuous':
			self.y += 1e-12 * np.random.random(self.N)
	
	def _cal_marginal_entropy(self, methods: list, params: list):
		"""
		计算边际分布熵
		"""
		series_binning_x_ = SeriesBinning(self.x, x_type = self.value_types[0])
		freq_ns_x_, edges_x_ = series_binning_x_.series_binning(method = methods[0], params = params[0])
		series_binning_y_ = SeriesBinning(self.y, x_type = self.value_types[1])
		freq_ns_y_, edges_y_ = series_binning_y_.series_binning(method = methods[1], params = params[1])
		
		univar_entropy_x_ = _get_entropy_from_freqs(freq_ns_x_)
		univar_entropy_y_ = _get_entropy_from_freqs(freq_ns_y_)
		edges_ = [edges_x_, edges_y_]
		
		return univar_entropy_x_, univar_entropy_y_, edges_
	
	@time_cost
	def cal_mie(self, methods: list, params: list) -> float:
		"""
		互信息熵计算
		:param methods: list of strs, x和y各维度上的分箱方法
		:param params: list of dicts, x和y各维度上的分箱方法对应参数
		"""
		_check_value_types_and_methods(self.value_types, methods)
		
		# 各维度边际熵.
		univar_entropy_x_, univar_entropy_y_, _ = self._cal_marginal_entropy(methods, params)
		
		# 联合分布熵.
		data_ = np.vstack((self.x, self.y)).T
		joint_binning_ = JointBinning(data_, self.value_types)
		H, _ = joint_binning_.joint_binning(methods = methods, params = params)
		joint_entropy_ = _get_entropy_from_freqs(H)
		
		# 互信息熵.
		mie = univar_entropy_x_ + univar_entropy_y_ - joint_entropy_
		
		return mie
	
	@time_cost
	def cal_time_delayed_mie(self, methods: list, params: list, lags: list) -> dict:
		"""
		含时滞的互信息熵计算
		:param methods: list of strs, x和y各维度上的分箱方法
		:param params: list of dicts, x和y各维度上的分箱方法对应参数
		:param lags: list of ints, 时滞值列表
		"""
		_check_value_types_and_methods(self.value_types, methods)
		
		# 各维度边际熵.
		univar_entropy_x_, univar_entropy_y_, edges_ = self._cal_marginal_entropy(methods, params)
		
		# 计算各时滞上的联合熵.
		td_mie_dict = {}
		for lag in lags:
			# 序列平移.
			lag_remain = np.abs(lag) % len(self.x)  # 整除后的余数
			x_td = copy.deepcopy(self.x)
			y_td = copy.deepcopy(self.y)
			
			if lag_remain == 0:
				pass
			else:
				if lag > 0:
					y_td = np.hstack((y_td[lag_remain:], y_td[:lag_remain]))
				else:
					x_td = np.hstack((x_td[lag_remain:], x_td[:lag_remain]))
			
			data_ = np.vstack((x_td, y_td)).T
			
			# 在各个维度上将数据值向label进行插入, 返回插入位置.
			insert_locs_ = np.zeros_like(data_, dtype = int)
			for d in range(self.D):
				insert_locs_[:, d] = np.searchsorted(edges_[d], data_[:, d], side = 'left')
			
			# 将高维坐标映射到一维坐标上, 然后统计各一维坐标上的频率.
			edges_len_ = list(np.max(insert_locs_, axis = 0) + 1)
			ravel_locs_ = np.ravel_multi_index(insert_locs_.T, dims = edges_len_)
			hist_ = np.bincount(ravel_locs_, minlength = np.array(edges_len_).prod())
			
			# reshape转换形状.
			hist_ = hist_.reshape(edges_len_)
			
			# 计算联合分布熵和互信息熵.
			joint_entropy_ = _get_entropy_from_freqs(hist_)
			mutual_info_entropy_ = univar_entropy_x_ + univar_entropy_y_ - joint_entropy_
			td_mie_dict[lag] = mutual_info_entropy_
			
		return td_mie_dict
	
	
class ManyForOneMutualInfoEntropy(object):
	"""
	多对一互信息熵检验
	"""
	
	def __init__(self, x: np.ndarray, y: np.ndarray, value_types: list):
		"""
		初始化
		:param x: np.ndarray, 高维x序列
		:param y: np.ndarray, 一维y序列
		:param value_types: list, x和y的值类型, like ['continuous', 'continuous', 'discrete', ...]
		"""
		y = np.array(y).flatten()
		
		if x.shape[0] != len(y):
			raise ValueError('Input x and y are not in the same length')
		
		if type(value_types) != list:
			raise ValueError('value_types is not a list')
		else:
			try:
				assert len(value_types) == 1 + x.shape[1]
				for p in value_types:
					assert p in VALUE_TYPES_AVAILABLE
			except Exception:
				raise ValueError('Invalid value_types "{}"'.format(value_types))
		
		self.value_types = value_types
		self.N = len(y)
		self.x = x.reshape(self.N, -1)
		self.y = y
		self.D = 1 + self.x.shape[1]
		
		self._add_noise()
		
	def _add_noise(self):
		"""
		数据中加入微量噪声
		"""
		for i in range(self.D - 1):
			if self.value_types[i] == 'continuous':
				self.x[:, i] += 1e-12 * np.random.random(self.N)
		if self.value_types[-1] == 'continuous':
			self.y += 1e-12 * np.random.random(self.N)
	
	def _cal_marginal_entropy(self, methods: list, params: list):
		"""
		计算边际分布熵
		"""
		joint_binning_x_ = JointBinning(self.x, self.value_types[: -1])
		freq_ns_x_, edges_x_ = joint_binning_x_.joint_binning(methods = methods[: -1], params = params[: -1])
		series_binning_y_ = SeriesBinning(self.y, x_type = self.value_types[-1])
		freq_ns_y_, edges_y_ = series_binning_y_.series_binning(method = methods[-1], params = params[-1])
		
		univar_entropy_x_ = _get_entropy_from_freqs(freq_ns_x_)
		univar_entropy_y_ = _get_entropy_from_freqs(freq_ns_y_)
		edges_ = edges_x_ + [edges_y_]
		
		return univar_entropy_x_, univar_entropy_y_, edges_
			
	@time_cost
	def cal_mie(self, methods: list, params: list) -> list:
		"""
		互信息熵计算
		:param methods: list of strs, x和y各维度上的分箱方法
		:param params: list of dicts, x和y各维度上的分箱方法对应参数
		"""
		_check_value_types_and_methods(self.value_types, methods)
		
		# 各维度边际熵.
		univar_entropy_x_, univar_entropy_y_, _ = self._cal_marginal_entropy(methods, params)
		
		# 联合分布熵.
		data_ = np.hstack((self.x, self.y.reshape(-1, 1)))
		joint_binning_ = JointBinning(data_, self.value_types)
		H, _ = joint_binning_.joint_binning(methods = methods, params = params)
		joint_entropy_ = _get_entropy_from_freqs(H)
		
		# 互信息熵.
		mutual_info_entropy = univar_entropy_x_ + univar_entropy_y_ - joint_entropy_
		
		return mutual_info_entropy
	
	@time_cost
	def cal_time_delayed_mie(self, methods: list, params: list, lags: list) -> dict:
		"""
		含时滞的互信息熵计算
		:param methods: list of strs, x和y各维度上的分箱方法
		:param params: list of dicts, x和y各维度上的分箱方法对应参数
		:param lags: list of ints, 时滞值列表
		"""
		_check_value_types_and_methods(self.value_types, methods)
		
		# 各维度边际熵.
		univar_entropy_x_, univar_entropy_y_, edges_ = self._cal_marginal_entropy(methods, params)
		
		# 计算时滞联合分布熵.
		td_mie_dict = {}
		for lag in lags:
			# 序列平移.
			lag_remain = np.abs(lag) % len(self.x)  # 整除后的余数
			x_td = copy.deepcopy(self.x)
			y_td = copy.deepcopy(self.y)
			
			if lag_remain == 0:
				pass
			else:
				if lag > 0:
					y_td = np.hstack((y_td[lag_remain:], y_td[:lag_remain]))
				else:
					x_td = np.vstack((x_td[lag_remain:], x_td[:lag_remain]))
			
			data_ = np.hstack((x_td, y_td.reshape(-1, 1)))
			
			# 在各个维度上将数据值向label进行插入, 返回插入位置.
			insert_locs_ = np.zeros_like(data_, dtype = int)
			for d in range(self.D):
				insert_locs_[:, d] = np.searchsorted(edges_[d], data_[:, d], side = 'left')
			
			# 将高维坐标映射到一维坐标上, 然后统计各一维坐标上的频率.
			edges_len_ = list(np.max(insert_locs_, axis = 0) + 1)
			ravel_locs_ = np.ravel_multi_index(insert_locs_.T, dims = edges_len_)
			hist_ = np.bincount(ravel_locs_, minlength = np.array(edges_len_).prod())
			
			# reshape转换形状.
			hist_ = hist_.reshape(edges_len_)
			
			# 计算联合分布熵和互信息熵.
			joint_entropy_ = _get_entropy_from_freqs(hist_)
			mutual_info_entropy_ = univar_entropy_x_ + univar_entropy_y_ - joint_entropy_
			td_mie_dict[lag] = mutual_info_entropy_
		
		return td_mie_dict
	
	



