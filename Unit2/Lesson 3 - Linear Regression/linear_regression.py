# -*- coding: utf-8 -*-
"""
Created on Wed Oct 28 13:18:26 2015

@author: scdavis6
"""
#Load packages
import pandas as pd
import numpy as np
import statsmodels.api as sm


#Load dataset
loansData = pd.read_csv('https://spark-public.s3.amazonaws.com/dataanalysis/loansData.csv')

#Observe dataset
print(loansData['Interest.Rate'][0:5])
print(loansData['Loan.Length'][0:5])
print(loansData['FICO.Range'][0:5])

#Data preparation
#Remove '%' symbols in interest rate
loansData['Interest.Rate'] = loansData['Interest.Rate'].map(lambda x: round(float(x.rstrip('%')) / 100, 4))
#Inspect results
print(loansData['Interest.Rate'][0:5])

#Remove month string in loan length
loansData['Loan.Length'] = loansData['Loan.Length'].map(lambda x: int(x.rstrip(' months')))
#Inspect results
print(loansData['Loan.Length'][0:5])

#Convert FICO range to a score
#Replace range with start and end numbers in list
loansData['FICO.Range'] = loansData['FICO.Range'].map(lambda x: x.split('-'))
#Replace start and end numbers in list with start number - problem not converting to a string
#Create variable FICO.Score as first item in list to integer
loansData['FICO.Score'] = loansData['FICO.Range'].map(lambda x: int(x[0]))

#Conversion did work
print(loansData['FICO.Score'][0:5])

#81174    735-739
#99592    715-719
#80059    690-694
#15825    695-699
#33182    695-699
#Name: FICO.Range, dtype: object

#Export cleaned dataset as .csv file
loansData.to_csv('loansData_clean.csv', header = True, index = False)

#Subset interest rate, amount requested, and FICO score columns
intrate = loansData['Interest.Rate']
loanamt = loansData['Amount.Requested']
fico = loansData['FICO.Score']

#Label independent and dependent variables
y = np.matrix(intrate).transpose()
x1 = np.matrix(fico).transpose()
x2 = np.matrix(loanamt).transpose()
#Formula for Least Squares v = (M^t M)^-1(M^t)y
#Create M matrix
x = np.column_stack([x1,x2])
X = sm.add_constant(x)
#Multiply together
model = sm.OLS(y,X)
f = model.fit()
#Output results
f.summary()


#                            OLS Regression Results                            
#==============================================================================
#Dep. Variable:                      y   R-squared:                       0.657
#Model:                            OLS   Adj. R-squared:                  0.656
#Method:                 Least Squares   F-statistic:                     2388.
#Date:                Thu, 29 Oct 2015   Prob (F-statistic):               0.00
#Time:                        20:31:44   Log-Likelihood:                 5727.6
#No. Observations:                2500   AIC:                        -1.145e+04
#Df Residuals:                    2497   BIC:                        -1.143e+04
#Df Model:                           2                                         
#Covariance Type:            nonrobust                                         
#==============================================================================
#                 coef    std err          t      P>|t|      [95.0% Conf. Int.]
#------------------------------------------------------------------------------
#const          0.7288      0.010     73.734      0.000         0.709     0.748
#x1            -0.0009    1.4e-05    -63.022      0.000        -0.001    -0.001
#x2          2.107e-06    6.3e-08     33.443      0.000      1.98e-06  2.23e-06
#==============================================================================
#Omnibus:                       69.496   Durbin-Watson:                   1.979
#Prob(Omnibus):                  0.000   Jarque-Bera (JB):               77.811
#Skew:                           0.379   Prob(JB):                     1.27e-17
#Kurtosis:                       3.414   Cond. No.                     2.96e+05
#=============================================================================

#Answers:
#P-value under 0.05
#R^2 is 0.66