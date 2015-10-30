# -*- coding: utf-8 -*-
"""
Created on Tue Oct 27 15:58:23 2015

@author: scdavis6
"""
#Import packages
import matplotlib.pyplot as plt
from scipy import stats
import pandas as pd
import collections

#Load dataset
loansData = pd.read_csv('https://spark-public.s3.amazonaws.com/dataanalysis/loansData.csv')
#Drop NA values
loansData.dropna(inplace=True)
#Counts of observations for each number of credit lines
freq = collections.Counter(loansData['Open.CREDIT.Lines'])
#Plot histogram
plt.figure()
plt.bar(freq.keys(), freq.values(), width=1)
plt.show()
#Perform chi-squared test
chi, p = stats.chisquare(freq.values())