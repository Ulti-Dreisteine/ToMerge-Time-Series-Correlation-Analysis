# -*- coding: utf-8 -*-
"""
Created on 2020/4/8 21:51

@Project Name: time-series-analysis

@File: test_mutual_info_entropy.py

@Author: luolei

@Email: dreisteine262@163.com

@Describe: 
"""

import logging

logging.basicConfig(level = logging.INFO)

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import sys, os

sys.path.append('../')

from lib import proj_dir
from lib.mutual_info_entropy.mutual_info_entropy import PairwiseMutualInfoEntropy, ManyForOneMutualInfoEntropy


if __name__ == '__main__':
	# %% 载入测试数据.
	data = pd.read_csv(os.path.join(proj_dir, 'data/data.csv'))
	discrete_fields = ['weather', 'wd', 'month', 'weekday', 'clock_num']
	
	# %% 测试成对检验.
	y_field = 'pm25'
	lags = list(np.arange(-500, 500 + 5, 5))
	plt.figure('td_mie', figsize = [8, 6])
	plt.suptitle('Time-Delayed Mutual Info Entropy Test for {}'.format(y_field), fontsize = 10, fontweight = 'bold')
	for i in range(1, len(data.columns)):
		x_field = data.columns[i]
		if x_field not in discrete_fields:
			value_types = ['continuous', 'continuous']
			methods = ['quasi_chi2', 'quasi_chi2']
			params = [{'init_bins': 250, 'final_bins': 100}, {'init_bins': 250, 'final_bins': 100}]
		else:
			value_types = ['discrete', 'continuous']
			methods = ['label', 'quasi_chi2']
			params = [{}, {'init_bins': 250, 'final_bins': 100}]
		
		x = np.array(data[x_field])
		y = np.array(data[y_field])
		pmie = PairwiseMutualInfoEntropy(x, y, value_types)
		
		mie = pmie.cal_mie(methods, params)
		print('Mutual Info Entropy "{}" -> "{}" at lag = 0: {:.6f}'.format(x_field, y_field, mie))
		
		td_mie = pmie.cal_time_delayed_mie(methods, params, lags)
		
		plt.subplot((len(data.columns) - 1) // 3, 3, i)
		plt.plot(list(td_mie.keys()), list(td_mie.values()))
		plt.legend([x_field], fontsize = 6, loc = 'lower left')
		plt.xticks(fontsize = 8)
		plt.yticks(fontsize = 8)
		plt.xlabel('time lag', fontsize = 8)
		plt.ylabel('mie', fontsize = 8)
		plt.xlim([-500, 500])
		plt.tight_layout()
		plt.subplots_adjust(top = 0.94)
	plt.savefig(os.path.join(proj_dir, 'img/pairwise_mie_test.png'), dpi = 450)
	
	
	