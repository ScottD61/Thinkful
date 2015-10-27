# -*- coding: utf-8 -*-
"""
Created on Mon Oct 26 14:29:41 2015

@author: scdavis6
"""

"""Import packages"""
import scipy.stats as stats
import matplotlib.pyplot as plt


"""Create dataset"""
x = [1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 3, 4, 4, 4, 4, 5, 6, 6, 6, 7, 7, 7, 7, 7, 7, 7, 7, 8, 8, 9, 9]

"""Boxplot"""
plt.boxplot(x)
plt.show()

"""Histogram"""
plt.hist(x, histtype='bar')
plt.show()

"""Q-Q plot"""
plt.figure()
graph = stats.probplot(x, dist="norm", plot=plt)
plt.show()