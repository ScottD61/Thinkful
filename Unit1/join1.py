# -*- coding: utf-8 -*-
"""
Created on Mon Oct 19 17:05:52 2015

@author: scdavis6
"""
import sqlite3 as lite
import pandas as pd

con = lite.connect('/Users/scdavis6/getting_started.db')

join_query = "SELECT name, state, year, warm_month, cold_month FROM cities INNER JOIN weather ON name = city;"

joined = pd.read_sql(join_query, con)

joined_july = joined[joined['warm_month'] == 'July']

joined_july.columns

z = zip(joined_july['name'], joined_july['state'])

together = joined_july.apply(lambda x:'%s, %s' % (x['name'],x['state']),axis=1)
print("cities that are warmest in July are:", ', '.join(together.tolist()))

#cities that are warmest in July are: New York City, NY, Boston, MA, Chicago, IL, 
#Dallas, TX, Seattle, WA, Portland, O