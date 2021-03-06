# -*- coding: utf-8 -*-
"""
Created on Mon Jan  4 10:06:04 2016

@author: scdavis6
"""
#Import modules
import pandas as pd
import matplotlib.pyplot as plt
import seaborn
import numpy as np
from sklearn.cross_validation import cross_val_score
from sklearn.preprocessing import StandardScaler
import sklearn.linear_model as ln
from unbalanced_dataset import SMOTE

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
plt.title('Histogram Of Number Of Existing Credits At This Bank')
plt.show()
#Number of people liable
plt.hist(german_credit['Number of people liable'])
plt.xlabel('Number of people liable')
plt.ylabel('Frequency')
plt.title('Histogram of Number of people being liable to provide maintenance for')
plt.show()



#Counting factors in categorical variables 
#Status checking
german_credit['Status checking'].value_counts()
#Credit history
german_credit['Credit history'].value_counts()
#Purpose
german_credit['Purpose'].value_counts()
#Savings account/bonds
german_credit['Savings account/bonds'].value_counts()
#Present employment
german_credit['Present employment'].value_counts()
#Personal status/sex
german_credit['Personal status/sex'].value_counts()
#Debtors/guarantors
german_credit['Debtors/guarantors'].value_counts()
#Property
german_credit['Property'].value_counts()
#Other installment plans
german_credit['Other installment plans'].value_counts()
#Housing
german_credit['Housing'].value_counts()
#Job
german_credit['Job'].value_counts()
#Telephone
german_credit['Telephone'].value_counts()
#Foreign worker
german_credit['Foreign worker'].value_counts()

#Checking balance of dependent variable
german_credit['Classification'].value_counts()
#1    700
#2    300

#Data cleaning
#Change data type of classification
#german_credit['Classification'] = german_credit['Classification'].astype('category')

#Change classification label from (1,2) to (0,1)
german_credit['Classification'] = german_credit['Classification'].map(lambda x: x-1) 

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
Y = german_credit['Classification']

#Logistic regression 
#Create logistic regression object
logreg = ln.LogisticRegression()
#Convert dataframe to matrix 
X_mat = X.as_matrix()
Y_mat = Y.as_matrix()

#Fit the logistic regression 
logreg.fit(X_mat, Y_mat)

#Identify size of training data with learning curves - USE POST FROM ULTRAVIOLET ANALYTICS

#Model testing
#Accuracy of test set
score = cross_val_score(logreg, X_mat, Y_mat, scoring = 'accuracy', cv = 10)
sum_score = np.mean(score)
#0.748
#Recall of test set
recall_score = cross_val_score(logreg, X_mat, Y_mat, scoring = 'recall', cv = 10)
sum_recall_score = np.mean(recall_score)
#0.463 
#Precision of test set
precision_score = cross_val_score(logreg, X_mat, Y_mat, scoring = 'precision', cv = 10)
sum_precision_score = np.mean(precision_score)
#0.617 
#AUC
auc_score = cross_val_score(logreg, X_mat, Y_mat, scoring = 'roc_auc', cv = 10)
sum_auc_score = np.mean(auc_score)
#0.791



#Model building pt. 2
#Standardize numeric variables
#Re-import data and subset numeric variables to standardize in matrix
#Import data
german_credit = pd.read_csv('German.csv')
#Data type changes
#Change data type of classification
#german_credit['Classification'] = german_credit['Classification'].astype('category')
#Convert class labels
german_credit['Classification'] = german_credit['Classification'].map(lambda x: x-1) 

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
#Convert array to  dataframe
#Get strings of numeric column name_
col_names = ['Duration', 'Credit amount', 'Installment rate', 'Present resident since',
             'Age', 'Number existing credits', 'Number of people liable']
new_stan = pd.DataFrame(stan_data, columns = col_names)                            


#Subset categorical data
cat_credit = german_credit[['Status checking', 'Credit history', 'Purpose', 
                              'Savings account/bonds', 'Present employment', 
                              'Personal status/sex', 'Debtors/guarantors', 
                              'Property', 'Other installment plans', 'Housing', 
                              'Job', 'Telephone', 'Foreign worker']]
#Change categorical variables to dummy
dummy_var = pd.get_dummies(cat_credit)  
#Join dataframes together
german_new_credit = dummy_var.join(new_stan) 

#Correlation matrix of standardized numeric variables
#Subset data
num_st = german_new_credit[['Duration', 'Credit amount', 
                            'Installment rate', 'Present resident since',
                            'Age', 'Number existing credits', 'Number of people liable']]
#Check correlation
num_st.corr()
                      
#Subset dataframe into indepedent and dependent variables
X_st = german_new_credit
Y_st = german_credit['Classification'] 

#Create logistic regression object
logreg_st = ln.LogisticRegression()
#Convert dataframe to matrix 
X_stm = X_st.as_matrix()
Y_stm = Y_st.as_matrix()

#Fit the logistic regression 
logreg_st.fit(X_stm, Y_stm)

#Test for accuracy of test set
score_st = cross_val_score(logreg_st, X_stm, Y_stm, scoring = 'accuracy', cv = 10)
sum_score_st = np.mean(score_st)
#0.75
#Test for recall of test set
recall_score_st = cross_val_score(logreg_st, X_stm, Y_stm, scoring = 'recall', cv = 10)
sum_recall_score_st = np.mean(recall_score_st)
#0.467
#Test for precision of test set
precision_score_st = cross_val_score(logreg_st, X_stm, Y_stm, scoring = 'precision', cv = 10)
sum_precision_score_st = np.mean(precision_score_st)
#0.62
#Test for AUC
auc_score_st = cross_val_score(logreg_st, X_stm, Y_stm, scoring = 'roc_auc', cv = 10)
sum_auc_score_st = np.mean(auc_score_st)
#0.79

#Specificity must of made up for a lower recall rate to get a higher accuracy
#recall for fraud problem would be good  - objective
#Decisions around how I build my model






#Model building pt.3 
#Fixed unbalanced dataset w/o standardization
#Import data
german_credit = pd.read_csv('German.csv')
#Data type changes
german_credit['Classification'] = german_credit['Classification'].map(lambda x: x-1) 
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
Y = german_credit['Classification']

#Apply SMOTE
#Get ratio
#ratio = float(np.count_nonzero(Y == 0)) / float(np.count_nonzero(Y == 1))
#Set ratio manually b/c it did not work 
#Ratio output
#print(ratio)
#2.33
#Set verbose as false to show less information
verbose = False
#Create SMOTE object
#smote = SMOTE(ratio = ratio, verbose = False, kind = 'regular') #Don't use

#Another way - leave this way
smote = SMOTE(ratio = 1.335, verbose = False, kind = 'regular')

#Fit data and transform
X_mod = X.as_matrix()
Y_mod = np.array(Y)
#Create new dataset
smox, smoy = smote.fit_transform(X_mod, Y_mod) 

#Check ratio of good and bad creditors
#Convert matrix to dataframe
y_data = pd.DataFrame(smoy, columns = ['classification'])
#check work
y_data['classification'].value_counts()


#New visualizations
#Convert matrix to dataframe to plot numeric columns
#Create list of column names
col_names = ['Status checking_A11', 'Status checking_A12', 'Status checking_A13',
       'Status checking_A14', 'Credit history_A30', 'Credit history_A31',
       'Credit history_A32', 'Credit history_A33', 'Credit history_A34',
       'Purpose_A40', 'Purpose_A41', 'Purpose_A410', 'Purpose_A42',
       'Purpose_A43', 'Purpose_A44', 'Purpose_A45', 'Purpose_A46',
       'Purpose_A48', 'Purpose_A49', 'Savings account/bonds_A61',
       'Savings account/bonds_A62', 'Savings account/bonds_A63',
       'Savings account/bonds_A64', 'Savings account/bonds_A65',
       'Present employment_A71', 'Present employment_A72',
       'Present employment_A73', 'Present employment_A74',
       'Present employment_A75', 'Personal status/sex_A91',
       'Personal status/sex_A92', 'Personal status/sex_A93',
       'Personal status/sex_A94', 'Debtors/guarantors_A101',
       'Debtors/guarantors_A102', 'Debtors/guarantors_A103', 'Property_A121',
       'Property_A122', 'Property_A123', 'Property_A124',
       'Other installment plans_A141', 'Other installment plans_A142',
       'Other installment plans_A143', 'Housing_A151', 'Housing_A152',
       'Housing_A153', 'Job_A171', 'Job_A172', 'Job_A173', 'Job_A174',
       'Telephone_A191', 'Telephone_A192', 'Foreign worker_A201',
       'Foreign worker_A202', 'Duration', 'Credit amount', 'Installment rate',
       'Present resident since', 'Age', 'Number existing credits',
       'Number of people liable']
#Convert matrix to dataframe       
old_data = pd.DataFrame(smox, columns = col_names )  


#Histograms - check skews
#Duration
plt.hist(old_data['Duration'])
plt.xlabel('Duration')
plt.ylabel('Frequency')
plt.title('Histogram of Duration in Months after Oversampling')
plt.show()
#Credit amount
plt.hist(old_data['Credit amount'])
plt.xlabel('Credit amount')
plt.ylabel('Frequency')
plt.title('Histogram of Credit Amount after Oversampling')
plt.show()
#Installment rate
plt.hist(old_data['Installment rate'])
plt.xlabel('Installment rate')
plt.ylabel('Frequency')
plt.title('Histogram of Installment Rate in % of Disposable Income after Oversampling')
plt.show()
#Years of present residence - fix 
plt.hist(old_data['Present resident since'])
plt.xlabel('Present resident since')
plt.ylabel('Frequency')
plt.title('Histogram of Present Residence Since after Oversampling')
plt.show()
#Age
plt.hist(old_data['Age'])
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.title('Histogram of Age after Oversampling')
plt.show()
#Number of existing credits - fix 
plt.hist(old_data['Number existing credits'])
plt.xlabel('Number existing credits')
plt.ylabel('Frequency')
plt.title('Histogram Of Number Of Existing Credits At This Bank after Oversampling')
plt.show()
#Number of people liable
plt.hist(old_data['Number of people liable'])
plt.xlabel('Number of people liable')
plt.ylabel('Frequency')
plt.title('Histogram of Number of people being liable to provide maintenance for after Oversampling')
plt.show()

#Create logistic regression object
logreg_ov = ln.LogisticRegression()
#Model building
#Fit the logistic regression 
logreg_ov.fit(smox, smoy)

#Model testing
#Test for accuracy of test set
score_ov = cross_val_score(logreg_ov, smox, smoy, scoring = 'accuracy', cv = 10)
sum_score_ov = np.mean(score_ov)
#0.765
#Test for recall of test set
recall_score_ov = cross_val_score(logreg_ov, smox, smoy, scoring = 'recall', cv = 10)
sum_recall_score_ov = np.mean(recall_score_ov)
#0.799
#Test for precision of test set
precision_score_ov = cross_val_score(logreg_ov, smox, smoy, scoring = 'precision', cv = 10)
sum_precision_score_ov = np.mean(precision_score_ov)
#0.751
#Test for AUC
auc_score_ov = cross_val_score(logreg_ov, smox, smoy, scoring = 'roc_auc', cv = 10)
sum_auc_score_ov = np.mean(auc_score_ov)
#0.823 



#Model building pt. 4
#Dimensionality reduction using PCA
from sklearn.decomposition import PCA
from sklearn import grid_search

#Use gridsearch for number of principal components
#Choose parameters for PCA
parameters = [{'n_components': [30, 35, 40, 45, 46, 47, 48, 49, 50, 55, 60], 'whiten': ['True', 'False']}]
#Create PCA object
pca = PCA()
#Gridsearch function for optimal number of principal components
gr = grid_search.GridSearchCV(pca, parameters)
#Fit function to data
gr.fit(smox, smoy)
#Optimal number of principal components
gr.best_params_
#{'n_components': 48, 'whiten': 'True'}
#Use 48 principal components with normalization

#Create PCA object  - 49
pca1 = PCA(n_components = 48, whiten = True) #Tune PCA
#Fit
pca1.fit(smox)
#Transform data
smox2 = pca1.transform(smox)

#Classification
#Create logistic regression object
logreg_p = ln.LogisticRegression()
#Model building
#Fit the logistic regression 
logreg_p.fit(smox2, smoy)

#Model testing
#Test for accuracy of test set
score_p = cross_val_score(logreg_p, smox2, smoy, scoring = 'accuracy', cv = 10)
sum_score_p = np.mean(score_p)
#0.749
#Test for recall of test set
recall_score_p = cross_val_score(logreg_p, smox2, smoy, scoring = 'recall', cv = 10)
sum_recall_score_p = np.mean(recall_score_p)
#0.77
#Test for precision of test set
precision_score_p = cross_val_score(logreg_p, smox2, smoy, scoring = 'precision', cv = 10)
sum_precision_score_p = np.mean(precision_score_p)
#0.7398
#Test for AUC
auc_score_p = cross_val_score(logreg_p, smox2, smoy, scoring = 'roc_auc', cv = 10)
sum_auc_score_p = np.mean(auc_score_p)
#0.822

#Result: slightly reduced results


#MISTAKE - GRIDSEARCH ONLY APPLIES TO ACCURACY W/ CLASSIFICATION
#AND R2 W/ REGRESSION 


#Compareing results of four logitistic regression models                          
                                   
#Create pandas dataframe from dictionary of objects
#Create dictionary
#Accuracy, precision, recall, and AUC for all four models
Measure = {
    'Accuracy': [sum_score, sum_score_st, sum_score_ov, sum_score_p],
    'Precision': [sum_precision_score, sum_precision_score_st, sum_precision_score_ov,
sum_precision_score_p],
    'Recall': [sum_recall_score, sum_recall_score_st, sum_recall_score_ov,
sum_recall_score_p],
    'AUC': [sum_auc_score, sum_auc_score_st, sum_auc_score_ov, sum_auc_score_p],
    'Model': ['Standard', 'Standardized', 'SMOTE', 'PCA']
}                           
 
new = pd.DataFrame(dict)  

new=pd.DataFrame(Measure)     
#Plot multiple barplots
new.plot(kind = 'bar', x = 'Model', figsize = (9, 11))
