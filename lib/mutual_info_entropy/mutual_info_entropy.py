# -*- coding: utf-8 -*-
"""
Created on 2020/7/9 2:36 下午

@File: mutual_info_entropy.py

@Department: AI Lab, Rockontrol, Chengdu

@Author: luolei

@Email: dreisteine262@163.com

@Describe: 互信息熵
"""

import logging

logging.basicConfig(level = logging.INFO)

import sys, os

sys.path.append('../..')

from lib.numerical_info_entropy.univariate_entropy import UnivarInfoEntropy
from lib.numerical_info_entropy.multivariate_entropy import PairJointEntropy
from mod.data_process.normalize_and_denoise import savitzky_golay

# TODO: 应该用分箱0点处斜率表示相关性, 避免样本量限制.

if __name__ == '__main__':
	# ============ 载入测试数据和参数 ============
	from collections import defaultdict
	import matplotlib.pyplot as plt
	from lib import load_test_data
	
	data = load_test_data(label = 'patient')
	
	# ============ 准备参数 ============
	continuous_bins = {
		'AGE': 40, 'P': 50, 'ISS': 10, 'HEIGHT': 10, 'WEIGHT': 10, 'MBP': 50, 'SHOCK_INDEX': 50, 'BMI': 50, 'RBC': 50,
		'HGB': 50, 'PLT': 50, 'WBC': 50, 'ALB': 50, 'CRE': 50, 'UA': 50, 'AST': 50,
		'ALT': 50, 'GLU': 50, 'TG': 50, 'CHO': 50, 'CA': 100, 'MG': 40, 'LDL': 50,
		'NA': 50, 'K': 50, 'CL': 50, 'GFR': 50, 'PT': 50, 'FIB': 50, 'DD': 50, 'CK': 50,
		'CAPRINI_SCORE': 10,
	}
	continuous_cols = list(continuous_bins.keys())
	
	# 连续测试.
	plt.figure(figsize = [6, 8])
	# for x_col in [p for p in data.columns if p not in ['INPATIENT_ID', 'VTE']]:
	for x_col in continuous_cols:
		y_col = 'VTE'
		x_type = 'continuous' if x_col in continuous_cols else 'discrete'
		y_type = 'discrete'

		x, y = list(data[x_col]), list(data[y_col])
		var_types = [x_type, y_type]

		bins_values = list(range(1, 1000, 1))
		results = defaultdict(list)
		for bins in bins_values:
			univar_entropy_x = UnivarInfoEntropy(x, x_type)
			univar_entropy_y = UnivarInfoEntropy(y, y_type)
			pair_joint_entropy_xy = PairJointEntropy(x, y, var_types)

			univar_entropy_x.do_univar_binning(bins = bins)
			univar_entropy_y.do_univar_binning(bins = bins)
			pair_joint_entropy_xy.do_joint_binning(bins_list = [bins, bins])

			H_mutual = univar_entropy_x.cal_H_c() + univar_entropy_y.cal_H_c() - pair_joint_entropy_xy.cal_H_c()
			# H_mutual = univar_entropy_x.cal_H() + univar_entropy_y.cal_H() - pair_joint_entropy_xy.cal_H()
			results['H_mutual'].append(H_mutual)

		results['H_mutual'] = savitzky_golay(results['H_mutual'], window_size = 51, order = 1)
		plt.plot(results['H_mutual'], label = x_col, linewidth = 0.3)
		plt.legend(loc = 'upper right', fontsize = 6.0)
		plt.xticks(fontsize = 6.0)
		plt.yticks(fontsize = 6.0)
		plt.xlabel('bins', fontsize = 8.0)
		plt.ylabel('entropy', fontsize = 8.0)
		plt.show()
		plt.pause(0.1)
	
	# 柱状图.
	# bins_optim = 10
	# mie_results = {}
	# for x_col in [p for p in data.columns if p not in ['INPATIENT_ID', 'VTE']]:
	# # for x_col in continuous_cols:
	# 	y_col = 'VTE'
	# 	var_types = [
	# 		'continuous' if x_col in continuous_cols else 'discrete',
	# 		'continuous' if y_col in continuous_cols else 'discrete',
	# 	]
	# 	bins_list = [
	# 		bins_optim if x_col in continuous_cols else None,
	# 		bins_optim if y_col in continuous_cols else None,
	# 	]
	# 	x, y = list(data[x_col]), list(data[y_col])
	# 	pair_joint_entropy_xy = PairJointEntropy(x, y, var_types)
	# 	pair_joint_entropy_xy.do_joint_binning(bins_list = bins_list)
	# 	H_c_xy = pair_joint_entropy_xy.cal_H_c()
	#
	# 	univar_entropy_x = UnivarInfoEntropy(x, var_types[0])
	# 	univar_entropy_y = UnivarInfoEntropy(y, var_types[1])
	# 	univar_entropy_x.do_univar_binning(bins = bins_optim)
	# 	univar_entropy_y.do_univar_binning(bins = bins_optim)
	#
	# 	H_c_x = univar_entropy_x.cal_H_c()
	# 	H_c_y = univar_entropy_y.cal_H_c()
	#
	# 	mie_results[x_col] = H_c_x + H_c_y - H_c_xy
	#
	# sorted_lst = sorted(mie_results.items(), key = lambda x: x[1], reverse = True)
	# mie_results = dict(zip([p[0] for p in sorted_lst], [p[1] for p in sorted_lst]))
	#
	# plt.figure(figsize = [8, 6])
	# plt.bar(
	# 	mie_results.keys(),
	# 	mie_results.values(),
	# )
	# plt.xticks(rotation = 90, fontsize = 6)
	# plt.yticks(fontsize = 6)
	# plt.ylabel('mutual info entropy value', fontsize = 6)
	# plt.tight_layout()



