# -*- coding: utf-8 -*-
"""
Created on Fri Oct 30 15:27:06 2015

@author: scdavis6
"""
#Load packages
import pandas as pd
import statsmodels.api as sm
import math as mt
import matplotlib.pyplot as plt

#Load dataset
cleanloansdata = pd.read_csv('/Users/scdavis6/Thinkful/Unit2/Lesson 4 - Logistic Regression/loansData_clean.csv')

#print(cleanloansdata[0:5])

#Dependent variable 
#Add column named IR_TF 
#labeled 1 if interest rate <12%, 0 if >12%
ir = cleanloansdata['Interest.Rate']
ir = [1 if x < .12 else 0 for x in ir]
cleanloansdata['IR_TF'] = ir

#Create variables for model
#Add intercept of 1
intercept = [1] * len(cleanloansdata)
cleanloansdata['Intercept'] = intercept
#Independent variables
x = cleanloansdata[['Intercept', 'FICO.Score', 'Amount.Requested']]
#Dependent variable
y = cleanloansdata['IR_TF']
#Create model
logit = sm.Logit(y, x).fit()
#Get coefficients
coeff = logit.params
print(coeff)
#Logistic function
def logistic_function(FicoScore, LoanAmount, coeff):
    prob = 1 / (1 + mt.exp(coeff[0] + coeff[1] * FicoScore + coeff[2] * LoanAmount))
    if prob > 0.7:
        p = 1
    else:
        p = 0
    return prob, p       
#FICO score at 720 and loan amount at $10000    
prob = logistic_function(720, 10000, coeff)[0]
print(prob)
#0.253621411048485
#p is below 0.70 - low chance of getting the loan
decision = logistic_function(720, 10000, coeff)[1] 


#Plotting
Fico = range(550, 950, 10)
p_plus = []
p_minus = []
p = []
for j in Fico:
    p_plus.append(1 / (1 + mt.exp(coeff[0] + coeff[1] * j + coeff[2] * 10000)))
    p_minus.append(1 / (1 + mt.exp(-coeff[0] - coeff[1] * j - coeff[2] * 10000)))
    p.append(logistic_function(j, 10000, coeff)[1])

plt.plot(Fico, p_plus, label = 'p(x) = 1/(1 + exp(b + mx))', color = 'blue')
plt.hold(True)
plt.plot(Fico, p_minus, label = 'p(x) = 1/(1 + exp(-b - mx))', color = 'green')    
plt.hold(True)
plt.plot(Fico, p, 'ro', label = 'Decision for 10000 USD')
plt.legend(loc='upper right')
plt.xlabel('Fico Score')
plt.ylabel('Probability and decision, yes = 1, no = 0')
   