# -*- coding: utf-8 -*-
"""
Created on Wed Oct 21 16:31:28 2015

@author: scdavis6
"""
import pandas as pd
from scipy import stats

"""Create dataset as a list"""

data = '''Region,Alcohol,Tobacco
North,6.47,4.03
Yorkshire,6.13,3.76
Northeast,6.19,3.77
East Midlands,4.89,3.34
West Midlands,5.63,3.47
East Anglia,4.52,2.92
Southeast,5.89,3.20
Southwest,4.79,2.71
Wales,5.27,3.53
Scotland,6.08,4.51
Northern Ireland,4.02,4.56'''

"""Split string on the newlines"""
data = data.split('\n')

"""Split each item in list on commas"""
data = [i.split(',') for i in data]

"""Convert list to pandas dataframe"""
column_names = data[0] 
data_rows = data[1::] 
df = pd.DataFrame(data_rows, columns = column_names)

"""Convert alcohol and tobacco columns to float"""
df['Alcohol'] = df['Alcohol'].astype(float)
df['Tobacco'] = df['Tobacco'].astype(float)

"""Create results for alcohol"""
s = df['Alcohol'].std()
v = df['Alcohol'].var()
r = max(df['Alcohol']) - min(df['Alcohol'])
me = df['Alcohol'].mean() 
med = df['Alcohol'].median()
mo = stats.mode(df['Alcohol'])

"""Print results for alcohol"""
"""Create dictionary for alcohol"""
table1 = {'standard deviation': s, 'variance': v, 'range': r, 
            'mean': me, 'median': med, 'mode': mo}
for operation, res in table1.items():
    print('The {} of alcohol is {}.'.format(operation, res))
    
#Printed results for alcohol
#The standard deviation of alcohol is 0.7977627808725191.
#The mode of alcohol is (array([ 4.02]), array([ 1.])).
#The mean of alcohol is 5.443636363636363.
#The variance of alcohol is 0.6364254545454548.
#The median of alcohol is 5.63.
#The range of alcohol is 2.45    
    
"""Create results for tobacco"""
s1 = df['Tobacco'].std()
v1 = df['Tobacco'].var()
r1 = max(df['Tobacco']) - min(df['Tobacco'])
me1 = df['Tobacco'].mean() 
med1 = df['Tobacco'].median()
mo1 = stats.mode(df['Tobacco'])

"""Print results for tobacco"""
"""Create dictionary for tobacco"""
table2 = {'standard deviation': s1, 'variance': v1, 'range': r1, 
            'mean': me1, 'median': med1, 'mode': mo1}
for operation1, res1 in table2.items():
    print('The {} of tobacco is {}.'.format(operation1, res1))

#Printed results for tobacco
#The standard deviation of tobacco is 0.5907083575135564.
#The mode of tobacco is (array([ 2.71]), array([ 1.])).
#The mean of tobacco is 3.6181818181818186.
#The variance of tobacco is 0.34893636363636354.
#The median of tobacco is 3.53.
#The range of tobacco is 1.8499999999999996
