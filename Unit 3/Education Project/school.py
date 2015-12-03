# -*- coding: utf-8 -*-
"""
Created on Sun Nov 22 13:10:14 2015

@author: scdavis6
"""
#Import modules
from bs4 import BeautifulSoup
import requests
import pandas as pd
import numpy as np

#Import page
url = "http://web.archive.org/web/20110514112442/http://unstats.un.org/unsd/demographic/products/socind/education.htm"

r = requests.get(url)

#Pass result to BeautifulSoup object 
soup = BeautifulSoup(r.content)

#Extract specific table by filtering through tables
for row in soup('table'):
    print(row)
#Choose 7th table
table7 = soup('table')[6]
#Length of 7th table    
len(soup('table')[6])

       
len(soup('table')[6].findAll('tr', {'class': 'tcont'}))

#Describe table from beautifulsoup
title = soup.title.text
print(title)
#Find tr elements (rows) from table 7 - idk if useful
rows = table7.find_all('tr')
print(rows)

#Find td elements - idk if useful
rows = table7.find_all('td')
print(rows)

#My r = his raw_table
#My soup = his soup_data
#Roy's part - 
soup_tag = soup('table')[6].tr.td
soup_table = soup_tag('table')[1].tr.td.div
raw_table = soup_table('table')[0]

#Put into pandas dataframe
col_name = []
for child in raw_table('tr'): 
    if child.get('class', ['Null'])[0] == 'lheader': 
        for td in child.find_all('td'): 
            if td.get_text() != '': 
                col_name.append(td.get_text())
        break
country_table = pd.DataFrame(columns = col_name)

edu_row_num = 0
for child in raw_table('tr'):
    row_curr = []
    if child.get('class', ['Null']) == 'tcont':
        row_curr.append(child.find('td').get_text())
        for td in child.find_all('td')[1:]: 
            if td.get('align'): 
                row_curr.append(td.get_text())
    else: 
        row_curr.append(child.find('td').get_text())
        for td in child.find_all('td')[1:]:
            if td.get('align'): 
                row_curr.append(td.get_text())
    if len(row_curr) == len(col_name): 
        country_table.loc[edu_row_num] = row_curr
        edu_row_num += 1
country_table[['Total', 'Men', 'Women']] = country_table[['Total', 'Men', 'Women']].convert_objects(convert_numeric = True)
 
#Load dataset
dataframe = pd.read_csv('ny.gdp.mktp.cd_Indicator_en_csv_v2/ny.gdp.mktp.cd_Indicator_en_csv_v2.csv')
#Get columns of dataset
dataframe.columns
country_table.columns
 
#Common countries
list1 = list(set(country_table['Country or area'].tolist()))
list2 = list(set(dataframe['Country Name'].tolist()))
list_common_countries = list(set(list1) & set(list2)) 
#Combine datasets
gdp = []
school_life_T = []
school_life_M = []
school_life_W = []
for j in list_common_countries:
    df1 = country_table[country_table['Country or area']==j]
    df2 = dataframe[dataframe['Country Name']==j]
    if (df2[''+ df1['Year'].irow(0)].irow(0) != ''):
        school_life_T.append(int(df1['Total'].irow(0)))
        school_life_M.append(int(df1['Men'].irow(0)))
        school_life_W.append(int(df1['Women'].irow(0)))
        gdp.append(np.log(df2[''+ df1['Year'].irow(0)].irow(0)))
    
df_final = pd.DataFrame({'Total': school_life_T, 'Men':school_life_M, 'Women': school_life_W, 'gdp': gdp})    
    
print(df_final.corr())

#            Men     Total     Women       gdp
#Men    1.000000  0.967884  0.935418  0.498361
#Total  0.967884  1.000000  0.975588  0.464729
#Women  0.935418  0.975588  1.000000  0.461950
#gdp    0.498361  0.464729  0.461950  1.000000
