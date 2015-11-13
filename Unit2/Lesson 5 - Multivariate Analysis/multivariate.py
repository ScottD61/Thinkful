# -*- coding: utf-8 -*-
"""
Created on Fri Nov  6 14:42:01 2015

@author: scdavis6
"""
#Import packages
import pandas as pd
import statsmodels.api as sm
import numpy as np

#Load dataset
#loandata = pd.read_csv('https://spark-public.s3.amazonaws.com/dataanalysis/loansData.csv')
loandata = pd.read_csv('/home/sroy/Desktop/Thinkful/Students/Scott/loansData_clean1.csv')

#Data pre-processing
#View columns
print(loandata.head())
#Get number of levels for home ownership
print(set(loandata['Home.Ownership']))

#Set str in Home ownership to number of levels
#Subset home ownership column from dataframe
house_ownership = loandata['Home.Ownership']
house_ownership = [4 if x == 'OWN' else 3 if x == 'MORTGAGE' else 2 if x == 'RENT' else 1 if x == 'OTHER' else 0 for x in house_ownership]
#Check result
print(house_ownership)
#Replace home ownership in dataframe with new variable - house ownership
loandata['Home.Ownership'] = house_ownership

#Linear regression model
#Putting the ordinal and numeric variables in same M-matrix
#Hypothesis gives ordinal values a rank and is comparable to numeric
#Independent variables
#Replaces NaN values with 0 using numpy

intrate = loandata['Interest.Rate']
intrate[np.isnan(intrate)] = 0
loanamt = loandata['Amount.Requested']
loanamt[np.isnan(loanamt)] = 0
fico = loandata['FICO.Score']
fico[np.isnan(fico)] = 0
monthly_income = loandata['Monthly.Income']
monthly_income[np.isnan(monthly_income)] = 0

x1 = np.matrix(fico).transpose()
x2 = np.matrix(loanamt).transpose()
x3 = np.matrix(monthly_income).transpose()
x4 = np.matrix(house_ownership).transpose()

#Dependent variable
y = np.matrix(loandata['Interest.Rate']).transpose()
#Add constant to independent variables
x = np.column_stack([x1, x2 ,x3 , x4])


x = sm.add_constant(x)

#Run linear regression
est = sm.OLS(y,x).fit()
#View results
est.summary()

#                            OLS Regression Results                            
#==============================================================================
#Dep. Variable:                      y   R-squared:                       0.900
#Model:                            OLS   Adj. R-squared:                  0.900
#Method:                 Least Squares   F-statistic:                     7473.
#Date:                Thu, 12 Nov 2015   Prob (F-statistic):               0.00
#Time:                        16:12:48   Log-Likelihood:                 4294.2
#No. Observations:                2500   AIC:                            -8582.
#Df Residuals:                    2497   BIC:                            -8565.
#Df Model:                           3                                         
#Covariance Type:            nonrobust                                         
#==============================================================================
#                 coef    std err          t      P>|t|      [95.0% Conf. Int.]
#------------------------------------------------------------------------------
#x1             0.0002   2.57e-06     58.813      0.000         0.000     0.000
#x2          2.316e-06   1.21e-07     19.127      0.000      2.08e-06  2.55e-06
#x3         -1.156e-06   2.39e-07     -4.838      0.000     -1.62e-06 -6.88e-07
#const               0          0        nan        nan             0         0
#==============================================================================
#Omnibus:                       85.167   Durbin-Watson:                   1.957
#Prob(Omnibus):                  0.000   Jarque-Bera (JB):               39.222
#Skew:                           0.009   Prob(JB):                     3.04e-09
#Kurtosis:                       2.387   Cond. No.                          inf
#==============================================================================
#
#Warnings:
#[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
#[2] The smallest eigenvalue is      0. This might indicate that there are
#strong multicollinearity problems or that the design matrix is singular
