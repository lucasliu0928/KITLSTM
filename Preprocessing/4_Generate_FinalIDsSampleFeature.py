#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 28 20:48:22 2021

@author: lucasliu
"""


import pandas as pd
import glob
import numpy as np
from sklearn.impute import SimpleImputer
import os

from Preprocessing_Ultilites import compute_time_to_last_seen_value, compute_ontology_value
#####################################################################################
# data dir
#####################################################################################
#User input
analysis_duration = "onRRT"
features = ['SBP','DBP','Scr','Bicarbonate','Hematocrit','Potassium', 'Billirubin', 'Sodium', 'Temp', 'WBC', 'HR', 'RR']

#proj dir
CURR_DIR = os.path.dirname(os.path.abspath("../"))
proj_dir = CURR_DIR + "/Intermediate_Data/practice_data_092721/" + analysis_duration +  "_analysis/"

#input dir 
indir1 =  proj_dir + "patient_level_data/"
indir2 =  proj_dir + 'sample_level_data/'

#output dir 
outdir = proj_dir + 'sample_level_data/'

#####################################################################################
#1. Load demo data
#NOTE: This will be added later
#####################################################################################
pt_demo_df = pd.read_csv(indir1 +  "demo_finalID_" + analysis_duration + ".csv")

#####################################################################################
#2. Load diabetes and hypertension
#####################################################################################
pt_diagnosis_df = pd.read_csv(indir1 +  "diagnosis_finalID_" + analysis_duration + ".csv")


#####################################################################################
#2. Load final ID outcome and durations
#####################################################################################
sp_outcome_df = pd.read_csv(indir2 +  "FinalIDs_spOutcome.csv", parse_dates=['Start', 'End'])


#####################################################################################
#3. patient level dynamic feature file dir 
#####################################################################################
raw_feature_dir =  indir1 + analysis_duration + "_raw/"
raw_feature_files = glob.glob(raw_feature_dir + "*.csv")

####################################################################################
#4.1 generete dynamic feature for each sample in range
####################################################################################
n_samples = sp_outcome_df.shape[0] #22430,17896

for i in range(n_samples):
    if i % 1000 == 0 :
        print (i)
    pt_id    = sp_outcome_df.iloc[i]['ENCNTR_ID']
    sp_id    = sp_outcome_df.iloc[i]['SAMPLE_ID']
    start_d  = sp_outcome_df.iloc[i]['Start']
    end_d    = sp_outcome_df.iloc[i]['End']    
    
    #get dynmaic feature in hosp or onRRT
    file = raw_feature_dir + pt_id + analysis_duration + "_raw.csv"
    p_df = pd.read_csv(file , parse_dates= ['Record_Time'],index_col=0)


    #select features
    selected_p_df = p_df.loc[:,features]
    
    #select time range for curr sample
    selected_p_df = selected_p_df[(selected_p_df.index >= start_d) & (selected_p_df.index <= end_d)]
    
    #sort by time
    selected_p_df.sort_index(inplace = True)

    # #remove leading all NAs
    # first_idx = selected_p_df.first_valid_index()
    # last_idx = selected_p_df.last_valid_index()
    # selected_p_df = selected_p_df.loc[first_idx:last_idx]
    
    #output
    selected_p_df.to_csv(outdir + "raw_features/" + sp_id + '_' + analysis_duration + '.csv')

###########################################################################################################
# 2.For each sample feature file, each time step, compute the time difference from time last seen a value to current time 
#updates : if there is a value at current time, then delta t = 0
###########################################################################################################
sample_Ids = sp_outcome_df['SAMPLE_ID']
ct = 0
for sp_id in sample_Ids:
    if ct % 500 == 0: 
        print(ct)
    ct +=1
    #file name
    file = sp_id + '_' + analysis_duration + '.csv'
    #Read file
    curr_data = pd.read_csv(outdir + 'raw_features/' + file, index_col=0)
    
    list_of_delta_time_df = []
    for ft in features:
        curr_delta_time_df = compute_time_to_last_seen_value(curr_data,ft)
        list_of_delta_time_df.append(curr_delta_time_df)

    comb_df = pd.concat(list_of_delta_time_df,axis=1)
    
    comb_df.to_csv(outdir + 'Delta_time_feature/' + sp_id + '_' + analysis_duration + '.csv')
    
#####################################################################################
#3. For each feature, create its binary version indicates >high, <low, or abnormal (<low or >high)
# this feature can be used to determine ontology attention later
# Terms in HPO: 
# SBP: Elevated systolic blood pressure, Decreased systolic blood pressure. 
# DBP :  "Elevated diastolic blood pressure", Decreased diastolic blood pressure"
# sCr: Decreased serum creatinine 
# Bicarbonate: "Abnormal serum bicarbonate concentration"
# Hematocrit: "Increased hematocrit", "Reduced hematocrit"
# Potassium: "Elevated serum potassium levels", "Low blood potassium levels"
# Billirubin: "High blood bilirubin levels",  "Decreased circulation of bilirubin in the blood circulation"
# Sodum :  "High blood sodium levels", "Low blood sodium levels"
# Temp "fewer", "Abnormally low body temperature"
# WBC:  "High white blood count", "Low white blood cell count"
# HR:   "Abnormal heart rate variability"
# RR:  Abnormal pattern of respiration 


#####################################################################################
# sample_Ids = sp_outcome_df['SAMPLE_ID']
# ct = 0
# for sp_id in sample_Ids:
#     if ct % 500 == 0: 
#         print(ct)
#     ct +=1
    
#     #file name
#     file = sp_id + '_' + analysis_duration + '.csv'
#     #Read file
#     curr_data = pd.read_csv(outdir + 'raw_features/' + file, index_col=0)
    
#     binary_feature_SBP    = compute_ontology_value(curr_data,'SBP',90,130)
#     binary_feature_DBP    = compute_ontology_value(curr_data,'DBP',60,90)
#     bianry_feature_SCR    = compute_ontology_value(curr_data,'Scr',0.6,1.1)
#     bianry_feature_Bicar  = compute_ontology_value(curr_data,'Bicarbonate',22, 29)
#     bianry_feature_Hemat  = compute_ontology_value(curr_data,'Hematocrit' ,34, 45)
#     bianry_feature_Pot    = compute_ontology_value(curr_data,'Potassium', 3.6, 4.9)
#     bianry_feature_billi  = compute_ontology_value(curr_data,'Billirubin',0.2 ,1.1)
#     bianry_feature_sodium = compute_ontology_value(curr_data,'Sodium',135 , 145)
#     bianry_feature_temp   = compute_ontology_value(curr_data,'Temp',  36.1, 37.2)
#     bianry_feature_wbc    = compute_ontology_value(curr_data,'WBC',3.7 ,10.3)
#     bianry_feature_hr     = compute_ontology_value(curr_data,'HR', 60, 100)
#     binary_feature_RR     = compute_ontology_value(curr_data,'RR' ,12,25)

#     comb_df = pd.concat([binary_feature_SBP,
#                          binary_feature_DBP, 
#                          bianry_feature_SCR,
#                          bianry_feature_Bicar,
#                          bianry_feature_Hemat,
#                          bianry_feature_Pot,
#                          bianry_feature_billi,
#                          bianry_feature_sodium,
#                          bianry_feature_temp,
#                          bianry_feature_wbc,
#                          bianry_feature_hr,
#                          binary_feature_RR],axis=1)
    
#     comb_df.to_csv(outdir + 'Binary_feature/' + sp_id + '_' + analysis_duration + '.csv')


#####################################################################################
#4. Generate interpolated dynamic feature data (For models need interpolated data, e.g, LSTM)
#####################################################################################
# sample_Ids = sp_outcome_df['SAMPLE_ID']
# ct = 0
# for sp_id in sample_Ids:
#     if ct % 500 == 0: 
#         print(ct)
#     ct +=1
#     #file name
#     file = sp_id + '_' + analysis_duration + '.csv'

#     #Read file
#     curr_data = pd.read_csv(outdir + 'raw_features/' + file, index_col=0)
    
#     #Interplotation
#     curr_data_inter = curr_data.interpolate()
#     curr_data_inter = curr_data_inter.fillna(method="ffill")         #fill end NA with last non-NA 
#     curr_data_inter = curr_data_inter.fillna(method="bfill")         #fill begining NA with first non-NA 


#     curr_data_inter.to_csv(outdir + 'raw_features_interpolated/' + sp_id + '_' + analysis_duration + '.csv')

#####################################################################################
#5. Generate Copy forward dynamic feature data (copy the last seen value to the current time step if NA in current time step)
#####################################################################################
# sample_Ids = sp_outcome_df['SAMPLE_ID']
# ct = 0
# for sp_id in sample_Ids:
#     if ct % 500 == 0: 
#         print(ct)
#     ct +=1
#     #file name
#     file = sp_id + '_' + analysis_duration + '.csv'

#     #Read file
#     curr_data = pd.read_csv(outdir + 'raw_features/' + file, index_col=0)
    
#     #Copy forward
#     curr_data_cf = curr_data.fillna(method="ffill")                  #propagate last valid observation forward to next
#     curr_data_cf = curr_data_cf.fillna(method="bfill")         #fill begining NA with first non-NA 

#     curr_data_cf.to_csv(outdir + 'raw_features_copyforward/' + sp_id + '_' + analysis_duration + '.csv')
    
    
#####################################################################################
#5. Generate AVG/MAX/MIN of dynamic feature data (For models need static verion of feature, e.g XGBOOST)
#####################################################################################
# sample_Ids = sp_outcome_df['SAMPLE_ID']

# AVG_MAX_MIN_Data_List = []
# ct = 0
# for sp_id in sample_Ids:
#     if ct % 500 == 0: 
#         print(ct)
#     ct +=1
#     #file name
#     file = sp_id + '_' + analysis_duration + '.csv'

#     #Read file
#     curr_data = pd.read_csv(outdir + 'raw_features/' + file, index_col=0)
    
#     #Compute AVG/MAX/MIN cross all time points for each sample

#     curr_mean = pd.DataFrame(curr_data.mean(axis = 0, skipna=True)).transpose()
#     curr_mean = curr_mean.add_prefix('AVG_')
    
#     curr_max  = pd.DataFrame(curr_data.max(axis = 0, skipna=True)).transpose()
#     curr_max = curr_max.add_prefix('MAX_')
    
#     curr_min  = pd.DataFrame(curr_data.min(axis = 0, skipna=True)).transpose()
#     curr_min = curr_min.add_prefix('MIN_')
    
#     curr_comb = pd.concat([curr_mean,curr_max,curr_min], axis = 1)
#     AVG_MAX_MIN_Data_List.append(curr_comb)

# #Combine all samples data into one df
# All_AVG_MAX_MIN_Data = pd.concat(AVG_MAX_MIN_Data_List, axis = 0)
# All_AVG_MAX_MIN_Data.index = sample_Ids

# #Mean imputation for samples that has NA values each feature (It is possible sample might has no values for one feature, but has values for other features)
# imp_mean = SimpleImputer(missing_values=np.nan, strategy='mean')
# imp_mean.fit(All_AVG_MAX_MIN_Data)
# All_AVG_MAX_MIN_Data_imputed = pd.DataFrame(imp_mean.transform(All_AVG_MAX_MIN_Data))
# All_AVG_MAX_MIN_Data_imputed.index = All_AVG_MAX_MIN_Data.index
# All_AVG_MAX_MIN_Data_imputed.columns = All_AVG_MAX_MIN_Data.columns

# All_AVG_MAX_MIN_Data_imputed.to_csv(outdir + 'AVG_MAX_MIN_Feature/' 'All_AVGMAXMIN_' + analysis_duration + '_Imputed.csv')
    

# #####################################################################################
# #6 generete static feature for each sample 
# #####################################################################################
# n_samples = sp_outcome_df.shape[0] #17896

# sample_IDs = []
# age = []
# gender = []
# race = []
# admit_wt = []
# bmi =[]
# charlson = []

# for i in range(n_samples):
#     if i % 1000 == 0 :
#         print (i)
#     pt_id    = sp_outcome_df.iloc[i]['ENCNTR_ID']
#     sp_id    = sp_outcome_df.iloc[i]['SAMPLE_ID']
 
    
#     #get static feature 
#     #get demo
#     curr_demo = pt_demo_df[pt_demo_df['ENCNTR_ID'] == pt_id]
#     curr_age = curr_demo['AGE'].item()
#     curr_gender = curr_demo['GENDR_CD_DES'].item()
#     curr_race = curr_demo['RACE_CD_DES'].item()
#     curr_wt = curr_demo['WT_KG'].item()
#     curr_bmi = curr_demo['BMI'].item()
#     curr_charlson = curr_demo['CHARLSON_INDEX'].item()

#     sample_IDs.append(sp_id)
#     age.append(curr_age)
#     gender.append(curr_gender)
#     race.append(curr_race)
#     admit_wt.append(curr_wt)
#     bmi.append(curr_bmi)
#     charlson.append(curr_charlson)

# sp_static_feature_df = pd.DataFrame({'SAMPLE_ID':sample_IDs,
#                                       'AGE':age,
#                                       'GENDER':gender,
#                                       'RACE':race,
#                                       'ADMISSION_WT': admit_wt,
#                                       'BMI':bmi,
#                                       'CHARLSON': charlson})
# #output
# sp_static_feature_df.to_csv(outdir + "static_features/"  'samples_static_features_' + analysis_duration + '.csv')


#####################################################################################
#7 generete dignosis for each sample 
# Diagnosis are the same for samples from one pt, before or at the start of duration (e.g, onRRT)
#####################################################################################
n_samples = sp_outcome_df.shape[0] #17896

sample_IDs = []
db = []
ht = []

for i in range(n_samples):
    if i % 1000 == 0 :
        print (i)
    pt_id    = sp_outcome_df.iloc[i]['ENCNTR_ID']
    sp_id    = sp_outcome_df.iloc[i]['SAMPLE_ID']

    #Get diagnosis
    curr_diag_df = pt_diagnosis_df[pt_diagnosis_df['ENCNTR_ID'] == pt_id]
    curr_db_flag = curr_diag_df['DB_BeforeOrAt'].item()
    curr_ht_flag = curr_diag_df['HT_BeforeOrAt'].item()
        
    sample_IDs.append(sp_id)
    db.append(curr_db_flag)
    ht.append(curr_ht_flag)

sp_diag_feature_df = pd.DataFrame({'SAMPLE_ID':sample_IDs,
                                   'DB_BeforeOrAt':db,
                                   'HT_BeforeOrAt':ht})
#output
sp_diag_feature_df.to_csv(outdir + "static_features/"  'samples_diagnosis_' + analysis_duration + '.csv')


###########################################################################################################
# 7.For each sample feature file, each time step, compute the time elapsed from the first time step
#Call it TimeSinceFirstMeasurement
###########################################################################################################
sample_Ids = sp_outcome_df['SAMPLE_ID']
ct = 0
for sp_id in sample_Ids:
    if ct % 500 == 0: 
        print(ct)
    ct +=1
    #file name
    file = sp_id + '_' + analysis_duration + '.csv'
    #Read file
    curr_data = pd.read_csv(outdir + 'raw_features/' + file, index_col=0)
    
    #Convert index to datetime
    curr_data.index = pd.to_datetime(curr_data.index) 

    #sort by time
    curr_data.sort_index(inplace = True)
    
    #Initial a last seen time
    initial_time = curr_data.index[0]  #first value
    
    time_since_list = []
    for time in curr_data.index:
        time_elapsed = time - initial_time
        time_elapsed_inMin = time_elapsed.total_seconds()//60 #Unit: Minites
        time_since_list.append(time_elapsed_inMin)
        
    #Store it in a df
    time_elapsed_df = pd.DataFrame({'TimeSince_1stM_inMin': time_since_list})
    time_elapsed_df.index = curr_data.index
    
    time_elapsed_df.to_csv(outdir + 'Delta_time_Since1stM_feature/' + sp_id + '_' + analysis_duration + '.csv')
    
###########################################################################################################
#8.For each sample feature file, each time step, compute the time elapsed from the last time step
#Call it TimeSinceLastVisit
###########################################################################################################
sample_Ids = sp_outcome_df['SAMPLE_ID']
ct = 0
for sp_id in sample_Ids:
    if ct % 500 == 0: 
        print(ct)
    ct +=1
    #file name
    file = sp_id + '_' + analysis_duration + '.csv'
    #Read file
    curr_data = pd.read_csv(outdir + 'raw_features/' + file, index_col=0)
    
    #Convert index to datetime
    curr_data.index = pd.to_datetime(curr_data.index) 

    #sort by time
    curr_data.sort_index(inplace = True)
    
    #Initial a last visit time
    last_t = curr_data.index[0]  #first value
    
    time_since_last_t = []
    for time in curr_data.index:
        time_elapsed = time - last_t
        time_elapsed_inMin = time_elapsed.total_seconds()//60 #Unit: Minites
        time_since_last_t.append(time_elapsed_inMin)
        #update last_t: 
        last_t = time
        
    #Store it in a df
    time_elapsed_df = pd.DataFrame({'TimeSince_LastVisit': time_since_last_t})
    time_elapsed_df.index = curr_data.index
    
    time_elapsed_df.to_csv(outdir + 'Delta_time_Since_LastVisit/' + sp_id + '_' + analysis_duration + '.csv')
    