# -*- coding: utf-8 -*-
"""
Created on 2020/7/3 5:35 下午

@File: __init__.py

@Department: AI Lab, Rockontrol, Chengdu

@Author: luolei

@Email: dreisteine262@163.com

@Describe:
"""

VALUE_TYPES_AVAILABLE = ['continuous', 'discrete']

METHODS_AVAILABLE = {
	'continuous': ['isometric', 'equifreq', 'quasi_chi2'],
	'discrete': ['label']
}

__all__ = ['VALUE_TYPES_AVAILABLE', 'METHODS_AVAILABLE']



