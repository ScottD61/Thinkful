# -*- coding: utf-8 -*-
"""
Created on Mon Jan  4 10:06:04 2016

@author: scdavis6
"""
#Import modules
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pandas.tools.plotting import scatter_matrix
from sklearn.cross_validation import cross_val_score
from sklearn.preprocessing import StandardScaler
import sklearn.linear_model as ln
from sklearn.metrics import roc_auc_score
from unbalanced_dataset import UnderSampler, NearMiss, CondensedNearestNeighbour, OneSidedSelection,\
NeighbourhoodCleaningRule, TomekLinks, ClusterCentroids, OverSampler, SMOTE,\
SMOTETomek, SMOTEENN, EasyEnsemble, BalanceCascade

#Import data
german_credit = pd.read_csv('German.csv')

#Explore dataframe
#Dimension of dataset
german_credit.shape
#(1000, 21)
#Column datatypes
german_credit.dtypes
#Number of NaN values in each column
german_credit.notnull().sum()
#None

#Data exploration
#Summary statistics
german_credit.describe()
#Scatterplot matrix - what's the right size?
#Subset data - numeric variables
#df = dataframe[['Duration', 'Credit amount', 'Installment rate', 'Present resident since',
#                'Age', 'Number existing credits', 'Number of people liable']]
#Scatterplot matrix
#scatter_matrix(df, alpha = 0.3, diagonal = 'dke')

#Standardize  first!!!
#matrix
german_credit.corr()

#Histograms - check skews
#Duration
plt.hist(german_credit['Duration'])
plt.xlabel('Duration')
plt.ylabel('Frequency')
plt.title('Histogram of Duration in Months')
plt.show()
#Credit amount
plt.hist(german_credit['Credit amount'])
plt.xlabel('Credit amount')
plt.ylabel('Frequency')
plt.title('Histogram of Credit Amount')
plt.show()
#Installment rate
plt.hist(german_credit['Installment rate'])
plt.xlabel('Installment rate')
plt.ylabel('Frequency')
plt.title('Histogram of Installment Rate in % of Disposable Income')
plt.show()
#Years of present residence - fix 
plt.hist(german_credit['Present resident since'])
plt.xlabel('Present resident since')
plt.ylabel('Frequency')
plt.title('Histogram of Present Residence Since')
plt.show()
#Age
plt.hist(german_credit['Age'])
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.title('Histogram of Age')
plt.show()
#Number of existing credits - fix 
plt.hist(german_credit['Number existing credits'])
plt.xlabel('Number existing credits')
plt.ylabel('Frequency')
plt.title('Histogram Of Number Of Existing Credits At This Bank ')
plt.show()
#Number of people liable
plt.hist(german_credit['Number of people liable'])
plt.xlabel('Number of people liable')
plt.ylabel('Frequency')
plt.title('Histogram of Number of people being liable to provide maintenance for ')
plt.show()

#Counting factors in categorical variables 
german_credit['Status checking'].value_counts()

#Checking balance of dependent variable
german_credit['Classification'].value_counts()
#1    700
#2    300

#Outlier detection - do boxplots
#Boxplots
plt.boxplot(german_credit['Credit amount']) 

#Check logistic regression ???
plt.scatter(german_credit['Credit amount'], german_credit['Classification'])

#Correlation between numeric
#Anova between categorical and numeric?
#Chi-squared test between all unordered categorical?

#Data cleaning
#Change data type of classification
german_credit['Classification'] = german_credit['Classification'].astype('category')

#Change classification label - HELP!!!!
german_credit['Classification'] = german_credit['Classification'].str.replace('1', '0')
#ERROR: AttributeError: Can only use .str accessor with string values, which use np.object_ dtype in pandas



#Change categorical variables to dummy
dummy_var = pd.get_dummies(german_credit[['Status checking', 'Credit history', 'Purpose', 
                              'Savings account/bonds', 'Present employment', 
                              'Personal status/sex', 'Debtors/guarantors', 
                              'Property', 'Other installment plans', 'Housing', 
                              'Job', 'Telephone', 'Foreign worker']])
#Get dimensions of new dataframe                              
dummy_var.shape   
#(1000, 54)   
#Drop old categorical variables from german credit dataframe
credit_new = german_credit.drop(['Status checking', 'Credit history', 'Purpose', 
                              'Savings account/bonds', 'Present employment', 
                              'Personal status/sex', 'Debtors/guarantors', 
                              'Property', 'Other installment plans', 'Housing', 
                              'Job', 'Telephone', 'Foreign worker'], axis = 1)
credit_new.shape                              
#(1000, 8)                              
#Join dataframes together
german_new_credit = dummy_var.join(credit_new)  
german_new_credit.shape        
#(1000, 62)
                       

#Model building
#Subset dataframe into indepedent and dependent variables
X = german_new_credit.drop('Classification', axis = 1)
Y = german_new_credit['Classification']

#Logistic regression 
#Get c parameter for regularization 
#Create logistic regression object
logreg = ln.LogisticRegression()
#Convert dataframe to matrix 
X = X.as_matrix()
Y = Y.as_matrix()

#Fit the logistic regression 
logreg.fit(X, Y)

#Model testing
#Test for accuracy of test set
score = cross_val_score(logreg, X, Y, scoring = 'accuracy', cv = 10)
np.mean(score)
#0.748
#Test for recall of test set
recall_score = cross_val_score(logreg, X, Y, scoring = 'recall', cv = 10)
np.mean(recall_score)
#0.869
#Test for precision of test set
precision_score = cross_val_score(logreg, X, Y, scoring = 'precision', cv = 10)
np.mean(precision_score)
#0.791

#Test for AUC
auc_score = cross_val_score(logreg, X, Y, scoring = 'roc_auc', cv = 10)
np.mean(auc_score)


#Model building pt. 2
#Standardize numeric variables
#Re-import data and subset numeric variables to standardize in matrix
#Import data
german_credit = pd.read_csv('German.csv')
#Data type changes
#Change data type of classification
german_credit['Classification'] = german_credit['Classification'].astype('category')
#Convert integer to float
#Subset numeric data
num_credit = german_credit[['Duration', 'Credit amount', 
                            'Installment rate', 'Present resident since',
                            'Age', 'Number existing credits', 'Number of people liable']]
#Apply function to change datatype
num_credit_st = num_credit.astype('float')                            
#Standardization object and fit to data
stan = StandardScaler().fit(num_credit_st)
#Transform dataset
stan_data = stan.transform(num_credit_st)
#Subset categorical data
cat_credit = german_credit[['Status checking', 'Credit history', 'Purpose', 
                              'Savings account/bonds', 'Present employment', 
                              'Personal status/sex', 'Debtors/guarantors', 
                              'Property', 'Other installment plans', 'Housing', 
                              'Job', 'Telephone', 'Foreign worker']]
#Change categorical variables to dummy
dummy_var = pd.get_dummies(cat_credit)  
#Join dataframes together 
german_new_credit = dummy_var.join(num_credit) 

#Correlation matrix of standardized numeric variables
#Subset data
num_st = german_new_credit[['Duration', 'Credit amount', 
                            'Installment rate', 'Present resident since',
                            'Age', 'Number existing credits', 'Number of people liable']]
num_st.corr()

                            
#Subset dataframe into indepedent and dependent variables
X_st = german_new_credit
Y_st = german_credit['Classification'] 

#Logistic regression 
#Get c parameter for regularization 
#parameters = {'C': [0.01, 0.05, 0.10, 0.25, 0.5, 0.75, 1]}

#Create logistic regression object
logreg_st = ln.LogisticRegression()
#Convert dataframe to matrix 
X_stm = X_st.as_matrix()
Y_stm = Y_st.as_matrix()

#Fit the logistic regression 
logreg_st.fit(X_stm, Y_stm)

#Model testing pt. 2 - no changes b/c need a different value for regularization!!!!!!!!!
#Test for accuracy of test set
score_st = cross_val_score(logreg_st, X_stm, Y_stm, scoring = 'accuracy', cv = 10)
np.mean(score_st)
#0.748
#Test for recall of test set
recall_score_st = cross_val_score(logreg_st, X_stm, Y_stm, scoring = 'recall', cv = 10)
np.mean(recall_score_st)
#0.869
#Test for precision of test set
precision_score_st = cross_val_score(logreg_st, X_stm, Y_stm, scoring = 'precision', cv = 10)
np.mean(precision_score_st)
#0.791


#Model building pt.3 
#Remove outliers by variable transformation or deleting observations
    #Do scatterplots and boxplots
    #See if standardizing did anything - only report the ones that made a difference
#Feature engineering
    #Remove skewness of variables
    #Create new features

#Scatterplots



#To do later
#More complex models 
#Things to vary:
    #Training set size
        #Compare different classification methods with learning curves
    #Parameter values
        #Grid search

#Grid search for multiple parameters for several classifiers
parameters = [
    {'weights': ['uniform', 'distance'], 'n_neighbors': [5, 10, 15, 20, 25, 30, 35]}, ###KNN
    {'kernel': ('linear', 'rbf'), 'C':[1, 10]}, ###SVM
    {'n_estimators': [100, 200, 250, 300, 350, 400, 450, 500, 550]}, ###random forest?? idk if will work
    #Logistic regression
    
    