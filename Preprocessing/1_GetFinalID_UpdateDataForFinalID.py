#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 27 15:25:45 2021

@author: lucasliu
"""
import glob
import re
import pandas as pd
import numpy as np

from Preprocessing_Ultilites import load_time_info, load_outcome_info, load_demo


def diagnosis_before_or_at(pt_diagnoise_df,duration_start_date):
    #First check if the patient has this diagnosis at any time
    if pt_diagnoise_df.shape[0] > 0: #if have this diagnosis
        #check if any diagnosis happened before or at the duration of anlayiss
        pt_diagnosis_before_or_at = pt_diagnoise_df[pt_diagnoise_df['DIAGNOSIS_ENCNTR_ADMT_DT'] <= duration_start_date]
        if pt_diagnosis_before_or_at.shape[0] > 0 : #if has diagnosis before or at
            diag_flag = 1  
        else:
            diag_flag = 0
    else: #no diagnosis at any time
        diag_flag = 0 
    
    return diag_flag

#####################################################################################
# data dir
#####################################################################################
#User input
analysis_duration = "onRRT"

#input dir
raw_data_dir = "/Users/lucasliu/Desktop/DrChen_Projects/All_AKI_Projects/UK_Data/Extracted_data/"
intermediate_dir = "/Users/lucasliu/Desktop/DrChen_Projects/All_AKI_Projects/AKID_Project2/Intermediate_Data/practice_data_092721/"

#output dir 
outdir = intermediate_dir + analysis_duration +  "_analysis/" + 'patient_level_data/'


#####################################################################################
#1. Load all outcome and ESRD and kidney transplant info
#####################################################################################
all_outcome_df = load_outcome_info(raw_data_dir) #1320


#####################################################################################
#2.Load all time info
#####################################################################################
all_timeinfo_df = load_time_info(raw_data_dir)  #1472

#only include pt has on RRT (CRRT or HD) 
cond1 = all_timeinfo_df['RRT_Duration'] != "0days"     
cond2 = ~ pd.isnull(all_timeinfo_df['RRT_Duration'])   
all_timeinfo_df = all_timeinfo_df[cond1 & cond2] #1399

#only include pt in ICU  
cond1 = all_timeinfo_df['ICU_Duration'] != "0days"  #0 duration of duration
cond2 = ~ pd.isnull(all_timeinfo_df['ICU_Duration'])   #NA duration of duration 
all_timeinfo_df = all_timeinfo_df[cond1 & cond2] #1351


#####################################################################################
#3.Load all demographic info
#####################################################################################
all_demo_df = load_demo(raw_data_dir) #37697

#only include pt >= 18
all_demo_df = all_demo_df[all_demo_df['AGE'] >= 18]


#####################################################################################
#3. Load diagnosis info
#####################################################################################
diabetes_df = pd.read_csv(raw_data_dir + "ALL_Diagnosis/RAW_Diabetes.csv", parse_dates=['DIAGNOSIS_ENCNTR_ADMT_DT'])
hypertension_df = pd.read_csv(raw_data_dir + "ALL_Diagnosis/RAW_Hypertension.csv", parse_dates=['DIAGNOSIS_ENCNTR_ADMT_DT'])

#####################################################################################
#4. From pt level raw feature (Extracted from KGDAL, exterme value excluded)
#4.1. get PT ID has feautre and 
#4.2. get PT ID has feature in range (>=72 hours and < 2000 hours of feature data in analysis_duration)
#####################################################################################
raw_feature_dir = intermediate_dir + analysis_duration + "_analysis/patient_level_data/" + analysis_duration + "_raw/"
raw_feature_files = glob.glob(raw_feature_dir + "*.csv")
IDs_has_feature_inrange = []    #ID has feature availiable >= 72Hours and < 2000 hours
for file in raw_feature_files:
    #get curr id 
    curr_id = re.search(analysis_duration + "_raw/(.*)" + analysis_duration  + "_raw", file).group(1)
    
    #get curr df
    curr_df = pd.read_csv(file , parse_dates= ['Record_Time'])
    
    if curr_df.shape[0] != 0: #if pts feature file is not empty
        curr_data_start = min(curr_df['Record_Time'])
        curr_data_end   = max(curr_df['Record_Time'])
        curr_hours = curr_data_end - curr_data_start
        if (curr_hours >= pd.Timedelta("72hours") and curr_hours < pd.Timedelta("2000hours")):
            IDs_has_feature_inrange.append(curr_id)

#####################################################################################
#5. Inclusion ID: pt has time, has demo, has outcome, has feature data in range
#####################################################################################
ID_hasTime = set(all_timeinfo_df['ENCNTR_ID']) #only RRT and ICU pts
ID_hasDemo = set(all_demo_df['ENCNTR_ID'])
ID_hasOutcome = set(all_outcome_df['ENCNTR_ID'])
IDs_has_feature_inrange = set(IDs_has_feature_inrange)

Inclusion_IDs = set.intersection(ID_hasTime, ID_hasDemo, ID_hasOutcome,IDs_has_feature_inrange) #644 INHOSP, #625onRRT


#####################################################################################
#5. Exclusions
#5.1 ESRD
#5.2 Kidney Transplant
#####################################################################################
ESRD_IDs         = list(all_outcome_df.loc[all_outcome_df['ESRD_status'] == 1,"ENCNTR_ID"])
Kidney_Trans_IDs = list(all_outcome_df.loc[all_outcome_df['Kidney_Transplant'] == 1,"ENCNTR_ID"])
All_exclusion_IDs = set(ESRD_IDs + Kidney_Trans_IDs)

#####################################################################################
#6. Final IDs
#####################################################################################
final_IDs = [pt for pt in Inclusion_IDs if pt not in All_exclusion_IDs] #748 in HOSP, 597onRRT



#####################################################################################
#7. Output outcome, time, demo, and final ID for fianl IDs for analysis duration 
#####################################################################################
final_outcome_df = all_outcome_df[all_outcome_df['ENCNTR_ID'].isin(final_IDs)]    #748,597
final_outcome_df['Hospital_Death'].value_counts()  #0: 390, 1: 358
final_time_df    = all_timeinfo_df[all_timeinfo_df['ENCNTR_ID'].isin(final_IDs)]  #748,597
final_demo_df   = all_demo_df[all_demo_df['ENCNTR_ID'].isin(final_IDs)]           #748,597
final_IDs_df    = pd.DataFrame({'ENCNTR_ID':final_IDs})



final_outcome_df.to_csv(outdir + "outcome_finalID_" + analysis_duration + ".csv", index = False)
final_time_df.to_csv(outdir + "time_finalID_" + analysis_duration + ".csv",index=False)
final_demo_df.to_csv(outdir + "demo_finalID_" + analysis_duration + ".csv",index=False)
final_IDs_df.to_csv(outdir + "finalID_" + analysis_duration + ".csv", index = False)

#####################################################################################
#7. Get patient level diangosis info (Before or at duration start) 
#####################################################################################
IDs = []
db  = []
ht  = []

for pt in final_IDs:
    #get duration info 
    curr_time_df = all_timeinfo_df[all_timeinfo_df['ENCNTR_ID'] == pt]
    
    
    if analysis_duration == "onRRT":
        curr_start = curr_time_df['RRT_START_DATE'].item()
    elif analysis_duration == "inICU":    
        curr_start = curr_time_df['ICU_ADMIT'].item()
    elif analysis_duration == "inHOSP":
        curr_start = curr_time_df['HOSP_ADMT_DT'].item()
        
    #get diabete info
    curr_db = diabetes_df[diabetes_df['ENCNTR_ID'] == pt]
    curr_db_flag = diagnosis_before_or_at(curr_db, curr_start)

    #get hypertension info
    curr_ht = hypertension_df[hypertension_df['ENCNTR_ID'] == pt]
    curr_ht_flag = diagnosis_before_or_at(curr_ht, curr_start)
    
    IDs.append(pt)
    db.append(curr_db_flag)
    ht.append(curr_ht_flag)

final_diag_df = pd.DataFrame({'ENCNTR_ID':IDs,
                              'DB_BeforeOrAt':db,
                              'HT_BeforeOrAt':ht})
final_diag_df.to_csv(outdir + "diagnosis_finalID_" + analysis_duration + ".csv", index = False)

#####################################################################################
#8. train and test IDs and validation ID
#####################################################################################
test_IDs_df = final_IDs_df.sample(frac = 0.2, replace=False , random_state=1)
#Rest for train and validation
rest_IDs_df = final_IDs_df[ ~ final_IDs_df['ENCNTR_ID'].isin(test_IDs_df['ENCNTR_ID'])]

train_IDs_df = rest_IDs_df.sample(frac = 0.95, replace=False , random_state=1)
validation_IDs_df = rest_IDs_df[ ~ rest_IDs_df['ENCNTR_ID'].isin(train_IDs_df['ENCNTR_ID'])]


train_IDs_df.to_csv(outdir + "trainID_" + analysis_duration + ".csv",index=False)   
test_IDs_df.to_csv(outdir + "testID_" + analysis_duration + ".csv", index = False)
validation_IDs_df.to_csv(outdir + "ValidationID_" + analysis_duration + ".csv",index=False)

