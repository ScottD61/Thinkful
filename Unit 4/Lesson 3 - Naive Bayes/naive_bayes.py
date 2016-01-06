# -*- coding: utf-8 -*-
"""
Created on Sun Dec 13 17:55:02 2015

@author: scdavis6
"""
#Import modules
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB


#Load dataset
dataframe = pd.read_csv('ideal_weight.csv')

#Data preparation
#Remove '' from column names - don't use 
#dataframe.columns = dataframe.columns.str.replace(" ' ", '')
#roy's - works
dataframe.columns = dataframe.columns.str.replace("'", "")
#Remove '' from sex column 
dataframe['sex'] = dataframe['sex'].str.replace("'", "")

#Convert sex to a categorical variable
dataframe['sex'].dtype
dataframe['sex'] = dataframe['sex'].astype('category')

#Plot distributions of actual weight and ideal weight
#Subset data
x = dataframe['actual']
y = dataframe['ideal']
#Create visualization - two variable same plot
plt.hist(x, alpha = 0.5, label = 'Actual')
plt.hist(y, alpha = 0.5, label = 'Ideal')
plt.legend(loc = 'upper right')
plt.show()

#Plot distributions of difference in weight
#Subset data
z = dataframe['diff']
#Create visualization
plt.hist(z)
plt.show()

#Distribution of men to women
#Count factors in categorical variable
dataframe['sex'].value_counts()
#'Female'    119
#'Male'       63
#More females in dataset

#Classification
#Subset Data
X = dataframe[['actual', 'ideal', 'diff']]
Y = dataframe['sex']
#Create classificaiton object
clf = GaussianNB()
#Fit classification model
clf.fit(X, Y)
#Predict new values of y
pred_nb = clf.predict(X)

#Number of mislabled points in entire dataset
print("Number of mislabeled points out of a total %d points : %d"
       % (dataframe.shape[0],(Y!= pred_nb).sum()))
#Number of mislabeled points out of a total 182 points : 14
       
#Prediction
#Actual wight on 145, ideal of 160, and diff of -15
#Create dictionary of desires values       
d = {'actual': 145, 'ideal': 160, 'diff': -15}
df = pd.DataFrame(data = d, index = [1])
#Predict
pred = clf.predict(df)
print(pred)
#['Male']

#Actual weight of 160, ideal of 145, and diff of 15
#Create dictionary of desires values
d1 = {'actual': 160, 'ideal': 145, 'diff': 15}
df1 = pd.DataFrame(data = d1, index = [1])
#Predict
pred1 = clf.predict(df1)
print(pred1)
#['Male']