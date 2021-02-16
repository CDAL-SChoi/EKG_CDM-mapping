import numpy as np
import pandas as pd
import os
import csv
import xml.etree.cElementTree as ET
from collections import Counter
from ast import literal_eval
import time

from ECG2CDM import EKG_rule
from openpyxl import load_workbook

start = time.time()

print("Rule Learning Start...")

Combined_rule = EKG_rule()
Combined_rule.Data_Load('dictionary_data/GE_mapping.xlsx')
Combined_rule.additional_Data_Load('dictionary_data/Philips_mapping.xlsx')
Combined_rule.additional_Data_Load('dictionary_data/Kohden_mapping.xlsx')

print("Learning End.")

data = pd.read_csv('test.csv')
input_vendor = data['Vendor']
input_source = data['Statement']

#mapping = input_source[data['Vendor']!='GE']
mapping = input_source
mapping.reset_index(drop=True, inplace=True)

total = mapping.shape[0]
print(total)

Statements = []

for i in range(0, total):
    Statements = Statements + literal_eval(mapping[i])

print ("extracting statements done, ", time.time()-start)
Statements = list(set(Statements))

print(len(Statements))

print("unique entity End.", time.time()-start)


## output 
with open('EKG_mapping_result.csv', 'w', newline='') as out:
    csv_out=csv.writer(out)
    for i in range(len(Statements)):
        id, name, score = Combined_rule.Get_similar_simscore(statement=[Statements[i]])
        if score < 0.84:  # Threshold
            id, name = Combined_rule.Check_if_any(statement=[Statements[i]])
            score=0
        csv_out.writerow([Statements[i], name[0], id[0], score])
        if i % 3000 == 0: print(i, ' done, time:', time.time()-start)

print("done, ", time.time()-start)
