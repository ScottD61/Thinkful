# -*- coding: utf-8 -*-
"""
Created on Tue Oct 27 14:45:35 2015

@author: scdavis6
"""

"""Load dataset"""
import matplotlib.pyplot as plt
import pandas as pd
import scipy.stats as stats

loansData = pd.read_csv('https://spark-public.s3.amazonaws.com/dataanalysis/loansData.csv')

"""Remove null values"""
loansData.dropna(inplace = True)

"""Descriptive statistics for loan amounts"""
"""Generate boxplot for loan amounts"""
loansData.boxplot(column = 'Amount.Funded.By.Investors')
plt.show()

"""Generate histogram for loan amounts"""
loansData.hist(column='Amount.Funded.By.Investors')
plt.show()

"""Generate QQ-plot for loan amounts"""
plt.figure()
graph = stats.probplot(loansData['Amount.Funded.By.Investors'], dist="norm", plot=plt)
plt.show()

"""Descriptive statistics for amount requested"""
"""Generate boxplot for amount requested"""
loansData.boxplot(column = 'Amount.Requested')
plt.show()

"""Generate histogram for amount requested"""
loansData.hist(column='Amount.Requested')
plt.show()

"""Generate QQ-plot for amount requested"""
plt.figure()
graph = stats.probplot(loansData['Amount.Requested'], dist="norm", plot=plt)
plt.show()


#Compare and contrast
#Amount requested and amount funded by investors has the same median 
#Boxplot shows a higher quantity in Q3 for amount requested
#Both histograms and Q-Q plots show a positive skew