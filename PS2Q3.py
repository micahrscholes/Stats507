#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 10 11:51:47 2021

@author: micahscholes
"""

import pandas as pd


nhanes_2011 = pd.read_sas("/Users/micahscholes/Downloads/DEMO_G.XPT")
nhanes_2011 = nhanes_2011.assign(cohort=2011)
nhanes_2013 = pd.read_sas("/Users/micahscholes/Downloads/DEMO_H.XPT")
nhanes_2013 = nhanes_2013.assign(cohort=2013)
nhanes_2015 = pd.read_sas("/Users/micahscholes/Downloads/DEMO_I.XPT")
nhanes_2015 = nhanes_2015.assign(cohort=2015)
nhanes_2017 = pd.read_sas("/Users/micahscholes/Downloads/DEMO_J.XPT")
nhanes_2017 = nhanes_2017.assign(cohort=2017)
nhanes_all = pd.concat([nhanes_2011, nhanes_2013, nhanes_2015, nhanes_2017])
nhanes_reduced = nhanes_all[["SEQN", "RIDAGEYR", "RIDRETH3", "DMDEDUC2", 
                             "DMDMARTL", "RIDSTATR", "SDMVPSU", "SDMVSTRA", 
                             "WTMEC2YR", "WTINT2YR", "cohort"]]
nhanes_reduced = nhanes_reduced.rename(
    columns={"SEQN":"id", "RIDAGEYR":"age", "RIDRETH3":"race_and_ethnicity", 
             "DMDEDUC2":"education", "DMDMARTL":"marital_status", 
             "RIDSTATR":"exam_status", "SDMVPSU":"masked_variance_pseudo-psu", 
             "SDMVSTRA":"masked_variance_pseudo-stratum", 
             "WTMEC2YR":"mec_exam_weight", "WTINT2YR":"interview_weight"})
nhanes_reduced.age = nhanes_reduced.age.astype(int)
nhanes_reduced.id = nhanes_reduced.id.astype(int)
nas=["race_and_ethnicity", "education", "marital_status", "exam_status",
     "masked_variance_pseudo-psu", "masked_variance_pseudo-stratum"]
nhanes_reduced[nas] = nhanes_reduced[nas].fillna(-1)
nhanes_reduced[nas] = nhanes_reduced[nas].astype(int)
nhanes_reduced.to_pickle("./nhanes.pkl")
print(nhanes_reduced.shape[0])

dental_2011 = pd.read_sas("/Users/micahscholes/Downloads/OHXDEN_G.XPT")
dental_2011 = dental_2011.assign(cohort=2011)
dental_2013 = pd.read_sas("/Users/micahscholes/Downloads/OHXDEN_H.XPT")
dental_2013 = dental_2013.assign(cohort=2013)
dental_2015 = pd.read_sas("/Users/micahscholes/Downloads/OHXDEN_I.XPT")
dental_2015 = dental_2015.assign(cohort=2015)
dental_2017 = pd.read_sas("/Users/micahscholes/Downloads/OHXDEN_J.XPT")
dental_2017 = dental_2017.assign(cohort=2017)
dental_all = pd.concat([dental_2011, dental_2013, dental_2015, dental_2017])
tooth_count = []
for i in range(32):
    num=str(i+1)
    if i<9:
        tooth_count.append("OHX0"+num+"TC")
    
    else:
        tooth_count.append("OHX"+num+"TC")
cavities_count = []
for i in range(2,32):
    num = str(i)
    if i<=9:
        cavities_count.append("OHX0"+num+"CTC")
    elif i==16 or i==17:
        continue
    else:
        cavities_count.append("OHX"+num+"CTC")
dental_reduced = dental_all[["SEQN", "OHDDESTS"]+tooth_count+cavities_count
                            +["cohort"]]

for col in cavities_count:
    dental_reduced[col] = dental_reduced[col].str.decode('utf-8')
dental_reduced = dental_reduced.rename(columns={"SEQN":"id",
                                                "OHDDESTS":"dentician_status"})
dental_reduced[tooth_count]=dental_reduced[tooth_count].fillna(-1)
dental_reduced[tooth_count]=dental_reduced[tooth_count].astype(int)
i = 1
for col in tooth_count:
    
    dental_reduced=dental_reduced.rename(columns={col:"tooth_"+str(i)})
    i = i+1

i = 2
for col in cavities_count:
    
    dental_reduced=dental_reduced.rename(columns={col:"cavity_"+str(i)})
    i = i+1
    if i==16:
        i = 18

dental_reduced.id = dental_reduced.id.astype(int)
dental_reduced.dentician_status = dental_reduced.dentician_status.astype(int)


dental_reduced.to_pickle("./dental.pkl")
print(dental_reduced.shape[0])