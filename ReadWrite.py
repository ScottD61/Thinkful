# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
#Import module
import csv
#Read file
with open('lecz-urban-rural-population-land-area-estimates-v2-csv/lecz-urban-rural-population-land-area-estimates_continent-90m.csv', ) as csvfile:
    pop = csv.reader(csvfile, delimiter = ',')
    header = next(pop)
    next(pop, None)
    for row in pop:
        print(', '.join(row))



#Open file

