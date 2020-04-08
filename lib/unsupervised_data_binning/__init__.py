# -*- coding: utf-8 -*-
"""
Created on 2020/2/11 13:39

@Project -> File: ruima_galvanization_optimization -> __init__.py

@Author: luolei

@Email: dreisteine262@163.com

@Describe: 初始化
"""

import numpy as np

VALUE_TYPES_AVAILABLE = ['continuous', 'discrete']
METHODS_AVAILABLE = {
	'continuous': ['isometric', 'quasi_chi2'],
	'discrete': ['label']
}


def gen_series_samples(sample_len, value_type):
	if value_type == 'continuous':
		x = np.hstack((np.random.normal(0, 1.0, sample_len // 2), np.random.normal(20, 1.0, sample_len // 2)))
		return x
	elif value_type == 'discrete':
		import random
		labels = [0, 0.5, 1]
		x = []
		for i in range(sample_len):
			x += random.sample(labels, k = 1)
		x = np.array(x)
		return x
	else:
		raise ValueError('value_type {} not in ["continuous", "discrete"].'.format(value_type))
	

def gen_two_dim_samples(samples_len, value_types):
	samples = None
	for dim in range(2):
		x = gen_series_samples(samples_len, value_types[dim])
		if dim == 0:
			samples = x.reshape(-1, 1)
		else:
			samples = np.hstack((samples, x.reshape(-1, 1)))
	return samples







