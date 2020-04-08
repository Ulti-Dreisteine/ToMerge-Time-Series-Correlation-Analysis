# -*- coding: utf-8 -*-
"""
Created on 2020/3/8 10:44

@Project -> File: algorithm-tools -> data_binning_new.py

@Author: luolei

@Email: dreisteine262@163.com

@Describe: 数据分箱
"""

import logging

logging.basicConfig(level = logging.INFO)

# from lake.decorator import time_cost
import pandas as pd
import numpy as np
import warnings
import sys

sys.path.append('../..')

from lib.unsupervised_data_binning import VALUE_TYPES_AVAILABLE


class SeriesBinning(object):
	"""
	一维序列分箱
	"""
	
	def __init__(self, x: list or np.array, x_type: str):
		"""
		初始化
		:param x: list or array like, 待分箱序列
		:param x_type: str in ['continuous', 'discrete']
		"""
		if x_type not in VALUE_TYPES_AVAILABLE:
			raise ValueError('Param x_type {} not in value_types_availabel = {}.'.format(x_type, VALUE_TYPES_AVAILABLE))
		
		self.x = np.array(x).flatten()  # flatten处理
		self.x_type = x_type
	
	def _get_stat_params(self) -> dict:
		"""
		计算1维数据x的统计学参数
		"""
		# 计算均值方差.
		mean = np.mean(self.x)
		std = np.std(self.x)
		
		# 计算四分位数.
		q1, q2, q3 = np.percentile(self.x, (25, 50, 75), interpolation = 'midpoint')
		iqr = abs(q3 - q1)
		
		# 汇总结果.
		stat_params = {
			'mean': mean,
			'std': std,
			'percentiles': {
				'q1': q1,  # 下25%位
				'q2': q2,  # 中位数
				'q3': q3,  # 上25%位
				'iqr': iqr
			}
		}
		return stat_params
	
	@property
	def stat_params(self):
		return self._get_stat_params()
	
	# @time_cost
	def isometric_binning(self, bins: int) -> (list, list):
		"""
		等距分箱，适用于对类似高斯型数据进行分箱
		:param bins: int > 0, 分箱个数
		:return:
			freq_ns: list of ints, 箱子中的频数
			labels: list of strs or ints, 各箱的标签
		
		Example:
		------------------------------------------------------------
		from mod.data_binning import gen_series_samples
		x_1 = gen_series_samples(sample_len = 200000, value_type = 'discrete')
		self = SeriesBinning(x_1, x_type = 'discrete')
		freq_ns, labels = self.isometric_binning(bins = 50)
		------------------------------------------------------------
		"""
		try:
			assert self.x_type != 'discrete'
		except:
			warnings.warn('Invalid x_type: "discrete", self.isometric_binning is better for x_type = "continuous", '
						  'please switch to self.label_binning')
		
		# 计算数据整体的上下四分位点以及iqr距离.
		# 按照上下四分位点外扩1.5倍iqr距离获得分箱外围边界以及每个箱子长度
		percentiles = self.stat_params['percentiles']
		q3, q1, iqr = percentiles['q3'], percentiles['q1'], percentiles['iqr']
		binning_range = [
			max(np.min(self.x), q1 - 1.5 * iqr),
			min(np.max(self.x), q3 + 1.5 * iqr)
		]
		
		# 分箱.
		freq_ns, intervals = np.histogram(self.x, bins, range = binning_range)
		labels = intervals[1:]  # **以每个分箱区间右侧为label
		
		# 转为list类型.
		freq_ns = list(freq_ns)
		labels = list(labels)
		
		return freq_ns, labels
	
	# @time_cost
	def quasi_chi2_binning(self, init_bins: int, final_bins: int, merge_freq_thres: float = None) -> (list, list):
		"""
		拟卡方分箱
		:param init_bins: int, 初始分箱数
		:param final_bins: int, 最终分箱数下限
		:param merge_freq_thres: float, 合并分箱的密度判据阈值
		:return:
			freq_ns: list of ints, 箱子中的频数
			labels: list of strs or ints, 各箱的标签
		
		Example:
		------------------------------------------------------------
		init_bins = 100
		final_bins = 50
		freq_ns, labels = self.quasi_chi2_binning(init_bins, final_bins)
		------------------------------------------------------------
		"""
		try:
			assert self.x_type != 'discrete'
		except:
			warnings.warn('Invalid x_type: "discrete", self.quasi_chi2_binning is better for x_type = "continuous", '
						  'please switch to self.label_binning')
		
		if merge_freq_thres is None:
			merge_freq_thres = len(self.x) / init_bins / 10
		
		# 第一次分箱.
		init_freq_ns, init_labels = self.isometric_binning(init_bins)
		densities = init_freq_ns  # 这里使用箱频率密度表示概率分布意义上的密度
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
				s = abs(density_i - density_j)  # 密度相似度，
				
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
				densities[i] = freq_ns[i] / box_lens[i]  # 使用i、j箱混合后的密度
				labels[i] = labels[j]
				
				freq_ns = freq_ns[: j] + freq_ns[j + 1:]
				densities = densities[: j] + densities[j + 1:]
				labels = labels[: j] + labels[j + 1:]
				box_lens = box_lens[: j] + box_lens[j + 1:]
				
				bins -= 1
		
		return freq_ns, labels
	
	# @time_cost
	def label_binning(self) -> (list, list):
		"""
		根据离散标签进行分箱
		:return:
			freq_ns: list of ints, 箱子中的频数
			labels: list of strs or ints, 各箱的标签
		
		Example:
		------------------------------------------------------------
		freq_ns, labels = self.label_binning()
		------------------------------------------------------------
		"""
		try:
			assert self.x_type != 'continuous'
		except:
			warnings.warn('Invalid x_type: "continuous", self.label_binning is better for x_type = "discrete", '
						  'please switch to self.label_binning or self.quasi_chi2_binning')
		
		# 设定label.
		labels = sorted(list(set(self.x)))  # **按照值从小到大排序
		
		# 统计freq_ns.
		d_ = pd.DataFrame(self.x, columns = ['label'])
		d_['index'] = d_.index
		freq_counts_ = d_.groupby('label').count()
		freq_counts_ = freq_counts_.to_dict()['index']
		
		freq_ns = []
		for i in range(len(labels)):
			freq_ns.append(freq_counts_[labels[i]])
		
		return freq_ns, labels
	
	# @time_cost
	def series_binning(self, method: str, params: dict = None):
		"""
		序列分箱
		:return:
			freq_ns: list of ints, 箱子中的频数
			labels: list of strs or ints, 各箱的标签
			
		Example:
		------------------------------------------------------------
		x_1 = gen_series_samples(sample_len = 200000, value_type = 'continuous')
		self = SeriesBinning(x_1, x_type = 'continuous')
		freq_ns, labels = self.series_binning(method = 'isometric', params = {'bins': 150})
		------------------------------------------------------------
		"""
		if method == 'isometric':
			freq_ns, labels = self.isometric_binning(params['bins'])
		elif method == 'quasi_chi2':
			freq_ns, labels = self.quasi_chi2_binning(params['init_bins'], params['final_bins'])
		elif method == 'label':
			freq_ns, labels = self.label_binning()
		else:
			raise ValueError('Invalid method {}'.format(method))
		return freq_ns, labels
