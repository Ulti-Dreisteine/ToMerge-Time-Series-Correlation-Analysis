# -*- coding: utf-8 -*-
"""
Created on 2020/7/8 2:59 下午

@File: mutivariate.py

@Department: AI Lab, Rockontrol, Chengdu

@Author: luolei

@Email: dreisteine262@163.com

@Describe: 多元互信息熵
"""

import logging

logging.basicConfig(level = logging.INFO)

import pandas as pd
import numpy as np
import sys, os

sys.path.append('../..')

from lib import VALUE_TYPES_AVAILABLE
from lib.numerical_info_entropy import convert_series_values
from lib.unsupervised_data_binning.joint_binning import JointBinning

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


def _deter_dtype(x_type: str):
	if x_type == 'continuous':
		return np.float64
	else:
		return np.float16


def _load_data_arr_and_process(x, y, var_types):
	"""在数组中去掉异常值, 并对连续值加入噪声"""
	arr = np.vstack(
		(
			np.array(x, dtype = _deter_dtype(var_types[0])),
			np.array(y, dtype = _deter_dtype(var_types[1]))
		)
	).T
	_d = pd.DataFrame(arr).dropna(axis = 0, how = 'any')
	arr = np.array(_d)
	
	for i in range(2):
		if var_types[i] == 'continuous':
			arr[:, i] += eps * np.random.random(len(arr))
	return arr


class PairJointEntropy(object):
	"""二元成对信息熵检验"""
	
	def __init__(self, x: list or np.ndarray, y: list or np.ndarray, var_types: list, bins: list):
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
		
		self.x_type, self.y_type = var_types
		_arr = _load_data_arr_and_process(x, y, var_types)      # 去掉异常值并对连续值加入噪声
		self.N = _arr.shape[0]                                  # 样本量
		self.x, self.y = _arr[:, 0], _arr[:, 1]
		self.bins = bins                                        # 分箱数
		
		if self.x_type == 'discrete':
			self.x = self.x.astype(np.float16)
		
		if self.y_type == 'discrete':
			self.y = self.y.astype(np.float16)
	
	@staticmethod
	def _get_probas(hist: np.ndarray) -> np.ndarray:
		"""根据频率计算概率"""
		f = hist.astype(np.float64)
		probas = f / (np.sum(f) + eps)
		return probas
			
	def do_joint_binning(self):
		"""
		进行序列联合分箱, 这里统一采用isometric等距分箱或label类别分箱形式
		"""
		_arr = np.vstack((self.x, self.y)).T
		joint_binning = JointBinning(_arr, [self.x_type, self.y_type])
		
		methods_lst, params_lst = [], []
		var_types = [self.x_type, self.y_type]
		for i in range(2):
			if var_types[i] == 'continuous':
				methods_lst.append('isometric')
				params_lst.append({'bins': self.bins[i]})
			else:
				methods_lst.append('label')
				params_lst.append({})
		
		self.hist, _ = joint_binning.joint_binning(methods_lst, params_lst)  # TODO: 此处有bug
		
		self.probas = self._get_probas(self.hist)  # type: np.ndarray
		
	def cal_H(self):
		probas_ = self.probas.flatten()
		probas_ = probas_[probas_ > 0]
		H = -np.dot(probas_, np.log2(probas_))
		return H
		
	def cal_H_c(self):
		_delta_lst = [
			1 / self.bins[0] if self.x_type == 'continuous' else 1.0,
			1 / self.bins[1] if self.y_type == 'continuous' else 1.0,
		]
		
		probas_ = self.probas.flatten()
		proba_dens_ = probas_ / (_delta_lst[0] * _delta_lst[1])
		
		_idxes = np.where(probas_ > 0)[0]  # 获得概率大于0的索引
		probas_ = probas_[_idxes]
		proba_dens_ = proba_dens_[_idxes]
		
		H_c = -np.dot(probas_, np.log2(proba_dens_))
		return H_c
	
	def cal_H_delta(self):
		_delta_lst = [
			1 / self.bins[0] if self.x_type == 'continuous' else 1.0,
			1 / self.bins[1] if self.y_type == 'continuous' else 1.0,
		]
		H_delta = -np.log2(_delta_lst[0]) - np.log2(_delta_lst[1])
		return H_delta
		
		


if __name__ == '__main__':
	# ============ 载入数据和参数 ============
	import matplotlib.pyplot as plt
	from collections import defaultdict
	from lib import load_test_data
	from lib.numerical_info_entropy.univariate import UnivarInfoEntropy
	
	data = load_test_data(label = 'pollution')
	
	# ============ 测试计算熵 ============
	# continuous_bins = {
	# 	'AGE': 40, 'P': 50, 'MBP': 50, 'SHOCK_INDEX': 50, 'BMI': 50, 'RBC': 50,
	# 	'HGB': 50, 'PLT': 50, 'WBC': 50, 'ALB': 50, 'CRE': 50, 'UA': 50, 'AST': 50,
	# 	'ALT': 50, 'GLU': 50, 'TG': 50, 'CHO': 50, 'CA': 100, 'MG': 40, 'LDL': 50,
	# 	'NA': 50, 'K': 50, 'CL': 50, 'GFR': 50, 'PT': 50, 'FIB': 50, 'DD': 50, 'CK': 50,
	# 	'CAPRINI_SCORE': 10,
	# }
	# continuous_cols = list(continuous_bins.keys())
	#
	# x_col = 'CRE'
	# y_col = 'VTE'
	# x, y = list(data[x_col]), list(data[y_col])
	# var_types = [
	# 	'continuous' if x_col in continuous_cols else 'discrete',
	# 	'continuous' if y_col in continuous_cols else 'discrete',
	# ]
	#
	# results = defaultdict(list)
	# for bins_ in range(1, 500, 1):
	# 	# 边际熵.
	# 	univar_entropy_x = UnivarInfoEntropy(x, var_types[0])
	# 	univar_entropy_y = UnivarInfoEntropy(y, var_types[1])
	#
	# 	univar_entropy_x.do_series_binning(bins = bins_)
	# 	univar_entropy_y.do_series_binning(bins = None)
	#
	# 	H_c_x = univar_entropy_x.cal_H_c()
	# 	H_c_y = univar_entropy_y.cal_H_c()
	#
	# 	# 计算联合熵和互信息熵.
	# 	bins = [bins_, None]
	# 	self = PairJointEntropy(x, y, var_types, bins)
	# 	self.do_joint_binning()
	# 	H_c_xy = self.cal_H_c()
	#
	# 	results['H_c_x'].append(H_c_x)
	# 	results['H_c_y'].append(H_c_y)
	# 	results['H_c_xy'].append(H_c_xy)
	# 	results['H_mutual'].append(H_c_x + H_c_y - H_c_xy)
	#
	# 	# break
	#
	# plt.plot(results['H_c_x'], label = 'H_c_x')
	# plt.plot(results['H_c_y'], label = 'H_c_y')
	# plt.plot(results['H_c_xy'], label = 'H_c_xy')
	# plt.plot(results['H_mutual'], label = 'H_mutual')
	# plt.legend(loc = 'upper right', fontsize = 6.0)
	
	# ============ 测试污染物浓度数据 ============
	continuous_bins = {
		'pm25': 110, 'pm10': 120, 'so2': 90, 'no2': 110, 'o3': 200, 'co': 200
	}
	continuous_cols = list(continuous_bins.keys())

	for x_col in continuous_cols[:]:
		y_col = 'pm25'
		x, y = list(data[x_col]), list(data[y_col])
		var_types = [
			'continuous' if x_col in continuous_cols else 'discrete',
			'continuous' if y_col in continuous_cols else 'discrete',
		]

		mutual_info_entropy_results = []
		for bins_ in range(1, 1000, 1):
			# 边际熵.
			univar_entropy_x = UnivarInfoEntropy(x, var_types[0])
			univar_entropy_y = UnivarInfoEntropy(y, var_types[1])

			univar_entropy_x.do_series_binning(bins = bins_)
			univar_entropy_y.do_series_binning(bins = bins_)

			H_c_x = univar_entropy_x.cal_H_c()
			H_c_y = univar_entropy_y.cal_H_c()

		# 计算联合熵和互信息熵.
			bins = [
				bins_ if x_col in continuous_cols else None,
				bins_ if y_col in continuous_cols else None,
			]

			self = PairJointEntropy(x, y, var_types, bins)
			self.do_joint_binning()
			H_c = self.cal_H_c()

			mutual_info_entropy = H_c_x + H_c_y - H_c
			mutual_info_entropy_results.append(mutual_info_entropy)

		plt.plot(mutual_info_entropy_results, linewidth = 0.3, label = x_col)
		plt.legend(loc = 'upper right', fontsize = 4.0)
		plt.show()
		plt.pause(0.1)
	
	# ============ 测试医疗数据 ============
	# continuous_bins = {
	# 	'AGE': 40, 'P': 50, 'MBP': 50, 'SHOCK_INDEX': 50, 'BMI': 50, 'RBC': 50,
	# 	'HGB': 50, 'PLT': 50, 'WBC': 50, 'ALB': 50, 'CRE': 50, 'UA': 50, 'AST': 50,
	# 	'ALT': 50, 'GLU': 50, 'TG': 50, 'CHO': 50, 'CA': 100, 'MG': 40, 'LDL': 50,
	# 	'NA': 50, 'K': 50, 'CL': 50, 'GFR': 50, 'PT': 50, 'FIB': 50, 'DD': 50, 'CK': 50,
	# 	'CAPRINI_SCORE': 10,
	# }
	# continuous_cols = list(continuous_bins.keys())

	# # 曲线图.
	# for x_col in continuous_cols:
	# 	y_col = 'VTE'
	# 	x, y = list(data[x_col]), list(data[y_col])
	# 	var_types = [
	# 		'continuous' if x_col in continuous_cols else 'discrete',
	# 		'continuous' if y_col in continuous_cols else 'discrete',
	# 	]
	#
	# 	mutual_info_entropy_results = []
	# 	for bins_ in range(1, 500, 1):
	#
	# 		# 边际熵.
	# 		univar_entropy_x = UnivarInfoEntropy(x, var_types[0])
	# 		univar_entropy_y = UnivarInfoEntropy(y, var_types[1])
	#
	# 		univar_entropy_x.do_series_binning(bins = bins_)
	# 		univar_entropy_y.do_series_binning(bins = None)
	#
	# 		H_c_x = univar_entropy_x.cal_H_c()
	# 		H_c_y = univar_entropy_y.cal_H_c()
	#
	# 		# 计算联合熵和互信息熵.
	#
	# 	# for bins_ in range(1, 500, 1):
	#
	# 		bins = [
	# 			bins_ if x_col in continuous_cols else None,
	# 			bins_ if y_col in continuous_cols else None,
	# 		]
	#
	# 		self = PairJointEntropy(x, y, var_types, bins)
	# 		self.do_joint_binning()
	# 		H_c = self.cal_H_c()
	#
	# 		mutual_info_entropy = H_c_x + H_c_y - H_c
	# 		mutual_info_entropy_results.append(mutual_info_entropy)
	#
	# 	plt.plot(mutual_info_entropy_results[: 120], linewidth = 0.3, label = x_col)
	# 	plt.legend(loc = 'upper right', fontsize = 4.0)
	# 	plt.show()
	# 	plt.pause(0.1)
	
	# 柱状图.
	# joint_bins = {
	# 	'AGE': 80, 'P': 80, 'MBP': 80, 'SHOCK_INDEX': 80, 'BMI': 80, 'RBC': 80,
	# 	'HGB': 80, 'PLT': 80, 'WBC': 80, 'ALB': 80, 'CRE': 80, 'UA': 80, 'AST': 80,
	# 	'ALT': 80, 'GLU': 80, 'TG': 80, 'CHO': 80, 'CA': 80, 'MG': 80, 'LDL': 80,
	# 	'NA': 80, 'K': 80, 'CL': 80, 'GFR': 80, 'PT': 80, 'FIB': 80, 'DD': 80, 'CK': 80,
	# 	'CAPRINI_SCORE': 13,
	# }
	# x_cols = [p for p in data.columns if p not in ['INPATIENT_ID', 'VTE']]
	# mie_results = {}
	# for x_col in x_cols:
	# 	y_col = 'VTE'
	# 	var_types = [
	# 		'continuous' if x_col in continuous_cols else 'discrete',
	# 		'continuous' if y_col in continuous_cols else 'discrete',
	# 	]
	# 	bins = [
	# 		joint_bins[x_col] if x_col in continuous_cols else None,
	# 		joint_bins[y_col] if y_col in continuous_cols else None,
	# 	]
	# 	x, y = list(data[x_col]), list(data[y_col])
	# 	pair_joint_entropy = PairJointEntropy(x, y, var_types, bins)
	# 	pair_joint_entropy.do_joint_binning()
	# 	H_c_xy = pair_joint_entropy.cal_H_c()
	#
	# 	univar_entropy_x = UnivarInfoEntropy(x, var_types[0])
	# 	univar_entropy_y = UnivarInfoEntropy(y, var_types[1])
	# 	univar_entropy_x.do_series_binning(bins = bins[0])
	# 	univar_entropy_y.do_series_binning(bins = bins[1])
	#
	# 	H_c_x = univar_entropy_x.cal_H_c()
	# 	H_c_y = univar_entropy_y.cal_H_c()
	#
	# 	mie_results[x_col] = H_c_x + H_c_y - H_c_xy
	#
	# sorted_lst = sorted(mie_results.items(), key = lambda x: x[1], reverse = True)
	# mie_results = dict(zip([p[0] for p in sorted_lst], [p[1] for p in sorted_lst]))
	#
	# plt.figure(figsize = [16, 6])
	# plt.bar(
	# 	mie_results.keys(),
	# 	mie_results.values(),
	# )
	# plt.xticks(rotation = 90, fontsize = 6)
	# plt.yticks(fontsize = 6)
	# plt.ylabel('mutual info entropy value', fontsize = 6)
	# plt.tight_layout()
	
