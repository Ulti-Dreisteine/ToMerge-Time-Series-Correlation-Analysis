# -*- coding: utf-8 -*-
"""
Created on 2020/7/10 5:02 下午

@File: mutual_info_entropy_pollution.py

@Department: AI Lab, Rockontrol, Chengdu

@Author: luolei

@Email: dreisteine262@163.com

@Describe:
"""

import logging

logging.basicConfig(level = logging.INFO)

from collections import defaultdict
import numpy as np
import copy
import sys, os

sys.path.append('../..')

from lib.numerical_info_entropy.univariate_entropy import UnivarInfoEntropy
from lib.numerical_info_entropy.multivariate_entropy import PairJointEntropy

if __name__ == '__main__':
	# ============ 载入测试数据和参数 ============
	from collections import defaultdict
	import matplotlib.pyplot as plt
	from lib import load_test_data
	
	data = load_test_data(label = 'pollution')
	
	# ============ 参数设定 ============
	continuous_cols = [
		'pm25', 'pm10', 'so2', 'co', 'no2', 'o3', 'ws', 'temp', 'sd'
	]
	
	# ============ 测试 ============
	# TODO: 优化计算效率.
	y_col = 'pm25'
	bins = 2
	
	x_cols = [p for p in data.columns if p != 'time']
	td_results = defaultdict(dict)
	plt.figure(figsize = [8.0, 12.0])
	for x_col in x_cols:
		print('x_col = {}'.format(x_col))
		td_results[x_col] = {}
		x_type = 'continuous' if x_col in continuous_cols else 'discrete'
		y_type = 'continuous' if y_col in continuous_cols else 'discrete'
		x, y = list(data[x_col]), list(data[y_col])
		var_types = [x_type, y_type]
		
		univar_entropy_x = UnivarInfoEntropy(x, x_type)
		univar_entropy_y = UnivarInfoEntropy(y, y_type)
		univar_entropy_x.do_univar_binning(bins = bins)
		univar_entropy_y.do_univar_binning(bins = bins)
		
		lags = list(np.arange(-1000, 1000 + 1, 1))
		
		for lag in lags:
			# 序列平移.
			lag_remain = np.abs(lag) % len(x)  # 整除后的余数
			x_td = copy.deepcopy(x)
			y_td = copy.deepcopy(y)
			
			if lag_remain == 0:
				pass
			else:
				if lag > 0:
					y_td = np.hstack((y_td[lag_remain:], y_td[:lag_remain]))
				else:
					x_td = np.hstack((x_td[lag_remain:], x_td[:lag_remain]))
			
			pair_joint_entropy_xy = PairJointEntropy(x_td, y_td, var_types)
			pair_joint_entropy_xy.do_joint_binning(bins_list = [bins, bins])
			
			H_mutual = univar_entropy_x.cal_H_c() + univar_entropy_y.cal_H_c() - pair_joint_entropy_xy.cal_H_c()
			td_results[x_col][lag] = H_mutual
	
	# plt.figure(figsize = [8.0, 12.0])
	# for i in range(len(x_cols)):
		i = x_cols.index(x_col)
		plt.subplot(len(x_cols) // 3 + 1, 3, i + 1)
		plt.plot(
			list(td_results[x_cols[i]].keys()),
			list(td_results[x_cols[i]].values()),
			linewidth = 0.3
		)
		plt.legend([x_cols[i]], loc = 'upper right', fontsize = 6.0)
		plt.xticks(fontsize = 8.0)
		plt.yticks(fontsize = 8.0)
		plt.tight_layout()
		plt.show()
		plt.pause(0.1)
		
		
	



