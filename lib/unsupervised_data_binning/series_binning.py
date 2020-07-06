# -*- coding: utf-8 -*-
"""
Created on 2020/7/3 5:38 下午

@File: series_binning.py

@Department: AI Lab, Rockontrol, Chengdu

@Author: luolei

@Email: dreisteine262@163.com

@Describe: 单序列分箱算法
"""

import logging
from typing import List, Any

logging.basicConfig(level = logging.INFO)

from lake.decorator import time_cost
import pandas as pd
import numpy as np
import warnings
import sys, os

sys.path.append('../..')

from lib.unsupervised_data_binning import VALUE_TYPES_AVAILABLE, METHODS_AVAILABLE


class SeriesBinning(object):
	"""单序列数据分箱量化"""
	
	def __init__(self, x: list or np.ndarray, x_type: str in VALUE_TYPES_AVAILABLE):
		"""
		:param x: 待分箱序列
		:param x_type: 序列值类型, 必须在VALUE_TYPES_AVAILABLE中选择
		"""
		if x_type not in VALUE_TYPES_AVAILABLE:
			raise ValueError('Invalid x_type {}'.format(x_type))
		
		self.x = np.array(x).flatten()
		self.x = self.x[~np.isnan(self.x)]      # 这一步默认删除了数据中的nan值
		self.x_type = x_type
	
	def _cal_stat_characters(self):
		try:
			assert self.x_type == 'continuous'
		except:
			warnings.warn('WARNING: stat characters may not be accurate for x_type = {}'.format(self.x_type))
		
		_mean = np.mean(self.x)
		_std = np.std(self.x)
		_q1, _q2, _q3 = np.percentile(self.x, (25, 50, 75), interpolation = 'midpoint')
		_iqr = abs(_q3 - _q1)
		
		stat_params = {
			'mean': _mean,
			'std': _std,
			'percentiles': {
				'q1': _q1,  # 下25%位
				'q2': _q2,  # 中位数
				'q3': _q3,  # 上25%位
				'iqr': _iqr
			}
		}
		return stat_params
	
	@property
	def stat_characters(self):
		"""序列数据统计学特征"""
		return self._cal_stat_characters()
	
	def _check_binning_match(self, current_method: str, suit_x_type: str, suit_method: str):
		"""检查分箱方法与待分箱值类型是否匹配"""
		try:
			assert self.x_type == suit_x_type
		except Exception as e:
			print(e)
			warnings.warn(
				'x_type is not "{}" for self.{}, try switch to self.{}'.format(
					self.x_type, current_method, suit_method)
			)
	
	@time_cost
	def isometric_binning(self, bins: int) -> (list, list):
		"""
		连续数据等距分箱
		:param bins: 分箱个数
		"""
		self._check_binning_match('isometric_binning', 'continuous', 'label_binning')
		
		_percentiles = self.stat_characters['percentiles']
		_q3, _q1, _iqr = _percentiles['q3'], _percentiles['q1'], _percentiles['iqr']
		_binning_range = [
			max(np.min(self.x), _q1 - 1.5 * _iqr),
			min(np.max(self.x), _q3 + 1.5 * _iqr)
		]
		
		# 分箱.
		freq_ns, _intervals = np.histogram(self.x, bins, range = _binning_range)
		labels = _intervals[1:]                                         # **以每个分箱区间的右边界为label
		
		# 转为list类型.
		freq_ns = list(freq_ns)
		labels = list(labels.astype(np.float32))                        # TODO: 此处的数值精度是否能够足够用于区分, 是否需要将数据进行归一化处理
		
		return freq_ns, labels
	
	@time_cost
	def equifreq_binning(self, equi_freq_n: int) -> (list, list):
		"""
		等频分箱
		:param equi_freq_n: 分箱箱子的样本数(上限)
		"""
		self._check_binning_match('equifreq_binning', 'continuous', 'label_binning')
		x = list(np.sort(self.x))
		
		freq_ns, labels = [], []
		while True:
			if len(x) <= equi_freq_n:  # 将该箱与上一个箱合并
				freq_ns[-1] += len(x)
				labels[-1] = x[-1]
				break
			else:
				freq_ns.append(equi_freq_n)
				labels.append(x[equi_freq_n - 1])
				x = x[equi_freq_n:]
				continue
		
		return freq_ns, labels
		
	@time_cost
	def quasi_chi2_binning(self, init_bins: int, final_bins: int, merge_freq_thres: float = None) -> (list, list):
		"""
		连续数据拟卡方分箱
		:param init_bins: 初始分箱数
		:param final_bins: 最终分箱数下限
		:param merge_freq_thres: 合并分箱的密度判据阈值
		"""
		self._check_binning_match('quasi_chi2_binning', 'continuous', 'label_binning')
		
		if merge_freq_thres is None:
			merge_freq_thres = len(self.x) / init_bins / 10             # 默认分箱密度阈值
			
		# 初始化.
		init_freq_ns, init_labels = self.isometric_binning(init_bins)
		densities = init_freq_ns                                        # 这里使用箱频率密度表示概率分布意义上的密度
		init_box_lens = [1] * init_bins
		
		# 根据相邻箱密度差异判断是否合并箱.
		bins = init_bins
		freq_ns = init_freq_ns
		labels = init_labels
		box_lens = init_box_lens
		
		while True:
			do_merge = 0
			
			# 在一次循环中优先合并具有最高相似度的箱.
			similar_ = {}
			for i in range(bins - 1):
				j = i + 1
				density_i, density_j = densities[i], densities[j]
				s = abs(density_i - density_j)                          # 密度相似度，
				
				if s <= merge_freq_thres:
					similar_[i] = s
					do_merge = 1
				else:
					continue
			
			if (do_merge == 0) | (bins == final_bins):
				break
			else:
				similar_ = sorted(similar_.items(), key = lambda x: x[1], reverse = False)  # 升序排列
				i = list(similar_[0])[0]
				j = i + 1
				
				# 执行i和j箱合并, j合并到i箱
				freq_ns[i] += freq_ns[j]
				box_lens[i] += box_lens[j]
				densities[i] = freq_ns[i] / box_lens[i]                 # 使用i、j箱混合后的密度
				labels[i] = labels[j]
				
				freq_ns = freq_ns[: j] + freq_ns[j + 1:]
				densities = densities[: j] + densities[j + 1:]
				labels = labels[: j] + labels[j + 1:]
				box_lens = box_lens[: j] + box_lens[j + 1:]
				
				bins -= 1
		
		return freq_ns, labels
	
	@time_cost
	def label_binning(self) -> (list, list):
		"""根据离散数据自身标签值进行分箱"""
		self._check_binning_match('label_binning', 'discrete', 'isometric_binning/quasi_chi2_binning/equifreq_binning')
		
		labels = sorted(list(set(self.x)))  # **按照值从小到大排序
		
		# 统计freq_ns.
		_df = pd.DataFrame(self.x, columns = ['label'])
		_df['index'] = _df.index
		_freq_counts = _df.groupby('label').count()
		_freq_counts = _freq_counts.to_dict()['index']
		
		freq_ns = []
		for i in range(len(labels)):
			freq_ns.append(_freq_counts[labels[i]])
		
		return freq_ns, labels
	
	def series_binning(self, method: str, **params):
		"""序列分箱算法"""
		freq_ns, labels = None, None
		if method in METHODS_AVAILABLE['continuous'] + METHODS_AVAILABLE['discrete']:
			if method == 'isometric':
				freq_ns, labels = self.isometric_binning(params['bins'])
			elif method == 'equifreq':
				freq_ns, labels = self.equifreq_binning(params['equi_freq_n'])
			elif method == 'quasi_chi2':
				freq_ns, labels = self.quasi_chi2_binning(params['init_bins'], params['final_bins'])
			elif method == 'label':
				freq_ns, labels = self.label_binning()
			return freq_ns, labels
		else:
			raise ValueError('Invalid method {}'.format(method))
		
		
if __name__ == '__main__':
	# ============ 载入测试数据和参数 ============
	import matplotlib.pyplot as plt
	from collections import defaultdict
	from lib import proj_dir
	
	data = pd.read_excel(os.path.join(proj_dir, 'data/raw/patient_info.xlsx'))
	
	# ============ 测试分箱 ============
	col = 'K'
	x = np.array(data[col])
	x_type = 'continuous'
	self = SeriesBinning(x, x_type)
	
	test_results = defaultdict(dict)
	test_params = {
		'isometric': {'bins': 30},
		'equifreq': {'equi_freq_n': 90},
		'quasi_chi2': {'init_bins': 150, 'final_bins': 30}
	}
	
	# 测试各分箱函数.
	for method in ['isometric', 'equifreq', 'quasi_chi2']:
		if method == 'isometric':
			freq_ns, labels = self.isometric_binning(**test_params[method])
		elif method == 'equifreq':
			freq_ns, labels = self.equifreq_binning(**test_params[method])
		elif method == 'quasi_chi2':
			freq_ns, labels = self.quasi_chi2_binning(**test_params[method])
		test_results[method] = {'freq_ns': freq_ns, 'labels': labels}
	
	# 测试通用分箱函数.
	test_results['by_general_func'] = {}
	for method in ['isometric', 'equifreq', 'quasi_chi2']:
		freq_ns, labels = self.series_binning(method, **test_params[method])
		test_results['by_general_func'][method] = {'freq_ns': freq_ns, 'labels': labels}
	
	
