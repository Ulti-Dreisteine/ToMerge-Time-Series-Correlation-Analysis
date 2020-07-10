# -*- coding: utf-8 -*-
"""
Created on 2020/7/10 4:03 下午

@File: mutual_info_entropy_patient.py

@Department: AI Lab, Rockontrol, Chengdu

@Author: luolei

@Email: dreisteine262@163.com

@Describe: 互信息熵测试
"""

import logging

logging.basicConfig(level = logging.INFO)

import sys, os

sys.path.append('../..')

from lib.numerical_info_entropy.univariate_entropy import UnivarInfoEntropy
from lib.numerical_info_entropy.multivariate_entropy import PairJointEntropy

if __name__ == '__main__':
	# ============ 载入测试数据和参数 ============
	from collections import defaultdict
	import matplotlib.pyplot as plt
	from lib import load_test_data
	
	data = load_test_data(label = 'patient')
	
	# ============ 参数设置 ============
	continuous_cols = [
		'AGE', 'P', 'ISS', 'HEIGHT', 'WEIGHT', 'MBP', 'SHOCK_INDEX', 'BMI', 'RBC',
		'HGB', 'PLT', 'WBC', 'ALB', 'CRE', 'UA', 'AST', 'ALT', 'GLU', 'TG', 'CHO', 'CA', 'MG', 'LDL',
		'NA', 'K', 'CL', 'GFR', 'PT', 'FIB', 'DD', 'CK', 'CAPRINI_SCORE',
	]
	
	# ============ 测试 ============
	y_col = 'VTE'
	bins = 2
	results = {}
	for x_col in [p for p in data.columns if p not in ['INPATIENT_ID', 'VTE']]:
		x_type = 'continuous' if x_col in continuous_cols else 'discrete'
		y_type = 'discrete'
		
		x, y = list(data[x_col]), list(data[y_col])
		var_types = [x_type, y_type]
		
		univar_entropy_x = UnivarInfoEntropy(x, x_type)
		univar_entropy_y = UnivarInfoEntropy(y, y_type)
		pair_joint_entropy_xy = PairJointEntropy(x, y, var_types)
		
		univar_entropy_x.do_univar_binning(bins = bins)
		univar_entropy_y.do_univar_binning(bins = bins)
		pair_joint_entropy_xy.do_joint_binning(bins_list = [bins, bins])
		
		H_mutual = univar_entropy_x.cal_H_c() + univar_entropy_y.cal_H_c() - pair_joint_entropy_xy.cal_H_c()
		results[x_col] = H_mutual if H_mutual > 0.0 else 0.0
	
	sorted_lst = sorted(results.items(), key = lambda x: x[1], reverse = True)
	results = dict(zip([p[0] for p in sorted_lst], [p[1] for p in sorted_lst]))
	plt.figure(figsize = [8, 6])
	plt.bar(
		results.keys(),
		results.values(),
	)
	plt.xticks(rotation = 90, fontsize = 6)
	plt.yticks(fontsize = 6)
	plt.ylabel('mutual info entropy value', fontsize = 6)
	plt.tight_layout()



