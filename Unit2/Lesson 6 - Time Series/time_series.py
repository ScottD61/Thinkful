# -*- coding: utf-8 -*-
"""
Created on Sat Nov  7 20:23:11 2015

@author: scdavis6
"""
#Import packages
import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt

#Data is LoanStats3b.csv from link: https://www.lendingclub.com/info/download-data.action

#Load dataset
loandata = pd.read_csv('/Users/scdavis6/Thinkful/Unit2/Lesson 6 - Time Series/LoanStats3b.csv', 
                       header = 1, low_memory = False)
#Inspect data
print(loandata.head())

#Convert column issue date from string to datetime in pandas
loandata['issue_d_format'] = pd.to_datetime(loandata['issue_d']) 
dfts = loandata.set_index('issue_d_format') 
year_month_summary = dfts.groupby(lambda x : x.year * 100 + x.month).count()
loan_count_summary = year_month_summary['issue_d']

#Plot 
plt.plot(loan_count_summary)
#Not stationary because increases with time

#Stationarize data
#Take first order discrete difference
loan_count_summary_diff = loan_count_summary.diff()
#Fill NA values
loan_count_summary_diff = loan_count_summary_diff.fillna(0)
#Plot data
plt.plot(loan_count_summary_diff)
#Still not stationary because negative values
#Since lowest negative value -316, moved up all points 316 points up
loan_count_summary_diff = loan_count_summary_diff + 316
#Plot data
plt.plot(loan_count_summary_diff)
#Standardize data between 0 and 1 
loan_count_summary_diff = loan_count_summary_diff/max(loan_count_summary_diff)
#Plot data
plt.plot(loan_count_summary_diff)

#Plot ACF and PACF
sm.graphics.tsa.plot_acf(loan_count_summary_diff)
sm.graphics.tsa.plot_pacf(loan_count_summary_diff)

#Most likely mixed ARMA processes