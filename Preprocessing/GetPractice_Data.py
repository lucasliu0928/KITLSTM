#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  5 22:10:46 2021

@author: lucasliu
"""

import pandas as pd
import glob
import numpy as np
import re
from Preprocessing_Ultilites import load_time_info, load_outcome_info,compute_time_to_last_seen_value,compute_ontology_value


#####################################################################################
# data dir
#####################################################################################
raw_feature_dir = "/Users/lucasliu/Desktop/DrChen_Projects/All_AKI_Projects/AKID_Project/Intermediate_data_Dynamic/2021_Data/0320_21_data/Original_Sample_Data/InHOSP/inHOSP_raw/"
raw_feature_files = glob.glob(raw_feature_dir + "*.csv")
outcome_dir = "/Users/lucasliu/Desktop/DrChen_Projects/All_AKI_Projects/Data/Extracted_data/"

out_dir = "/Users/lucasliu/Desktop/DrChen_Projects/All_AKI_Projects/AKID_Project2/Intermediate_Data/practice_data_092021/"


#####################################################################################
# 2.Load time info and outcome
#####################################################################################
#Load time info
AKID_timeinfo_df = load_time_info(outcome_dir) 

#remove pts has no CRRT nor HD
remove_idxes = []
for i in range(0,AKID_timeinfo_df.shape[0]):
    AKID_timeinfo_df.loc[i, "ENCNTR_ID"]
    curr_CRRT_start = AKID_timeinfo_df.loc[i, "CRRT_START_DATE"]
    curr_HD_start   = AKID_timeinfo_df.loc[i, "HD_START_DATE"]
    if (pd.isnull(curr_CRRT_start) and  pd.isnull(curr_HD_start)) :
        remove_idxes.append(i)
        
AKID_timeinfo_df = AKID_timeinfo_df.drop(remove_idxes) #1399 

#Load outcome info
all_outcome_df = load_outcome_info(outcome_dir) #1320



    


#####################################################################################
#Update outcome by Remove ESRD before admission:
#####################################################################################
ESRD_IDs = all_outcome_df.loc[all_outcome_df['ESRD_status'] == 1,"ENCNTR_ID"]
all_outcome_df = all_outcome_df.loc[~all_outcome_df['ENCNTR_ID'].isin(ESRD_IDs),]
all_outcome_df.shape[0] #1278

#####################################################################################
#Update outcome by Remove Kidney Transplant
#####################################################################################
Kidney_Trans_IDs = all_outcome_df.loc[all_outcome_df['Kidney_Transplant'] == 1,"ENCNTR_ID"]
all_outcome_df = all_outcome_df.loc[~all_outcome_df['ENCNTR_ID'].isin(Kidney_Trans_IDs),]
all_outcome_df['Hospital_Death'].value_counts()
all_outcome_df.shape #1273

#####################################################################################
#Update outcome by keeping outcome only for AKID pts who has time info
#####################################################################################
all_outcome_df = all_outcome_df[all_outcome_df['ENCNTR_ID'].isin(AKID_timeinfo_df['ENCNTR_ID'])] 
all_outcome_df.shape #1260

#####################################################################################
#get pts Id who has feature files
#####################################################################################
IDs_has_feature = []
for file in raw_feature_files:
    #get curr id 
    curr_id = re.search('inHOSP_raw/(.*)inHOSP_raw', file).group(1)
    IDs_has_feature.append(curr_id)

#####################################################################################
#Update outcome for Id who has feature
#####################################################################################
all_outcome_df = all_outcome_df[all_outcome_df['ENCNTR_ID'].isin(IDs_has_feature)]
all_outcome_df.shape #863

#####################################################################################
#Output outcome
#####################################################################################
all_outcome_df.to_csv(out_dir + 'Outcome.csv')


#####################################################################################
#Anlaysis ID and features
#####################################################################################
analysis_ID = all_outcome_df['ENCNTR_ID'].tolist()
features = ['SBP','DBP','Scr','Bicarbonate','Hematocrit','Potassium', 'Billirubin', 'Sodium', 'Temp', 'WBC', 'HR', 'RR']

#####################################################################################
# 1.Get 3 Features for entire HOSP for 933 AKID patients
#####################################################################################

for pt in analysis_ID:
    #file name
    curr_file_name = pt + "inHOSP_raw.csv"
    #Read file
    curr_data = pd.read_csv(raw_feature_dir + curr_file_name, index_col=0)
    #select featres
    curr_selected_data = curr_data.loc[:,features]
    
    #sort by time
    curr_selected_data.sort_index(inplace = True)

    #remove leading all NAs
    first_idx = curr_selected_data.first_valid_index()
    last_idx = curr_selected_data.last_valid_index()
    curr_selected_data = curr_selected_data.loc[first_idx:last_idx]
    
    #output
    curr_selected_data.to_csv(out_dir + 'raw_features/' + pt + '_inHOSP.csv')
    
    
###########################################################################################################
# 2.For each pt feature file, each time step, compute the time difference from time last seen a value to current time 
###########################################################################################################
for pt in analysis_ID:
    #file name
    curr_file_name = pt + "_inHOSP.csv"
    #Read file
    curr_data = pd.read_csv(out_dir + 'raw_features/' + curr_file_name, index_col=0)
    
    list_of_delta_time_df = []
    for ft in features:
        curr_delta_time_df = compute_time_to_last_seen_value(curr_data,ft)
        list_of_delta_time_df.append(curr_delta_time_df)

    comb_df = pd.concat(list_of_delta_time_df,axis=1)
    
    comb_df.to_csv(out_dir + 'Delta_time_feature/' + pt + '_inHOSP.csv')


    
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
for pt in analysis_ID:
    #file name
    curr_file_name = pt + "_inHOSP.csv"
    #Read file
    curr_data = pd.read_csv(out_dir + 'raw_features/' + curr_file_name, index_col=0)
 
    binary_feature_SBP    = compute_ontology_value(curr_data,'SBP',90,130)
    binary_feature_DBP    = compute_ontology_value(curr_data,'DBP',60,90)
    bianry_feature_SCR    = compute_ontology_value(curr_data,'Scr',0.6,1.1)
    bianry_feature_Bicar  = compute_ontology_value(curr_data,'Bicarbonate',22, 29)
    bianry_feature_Hemat  = compute_ontology_value(curr_data,'Hematocrit' ,34, 45)
    bianry_feature_Pot    = compute_ontology_value(curr_data,'Potassium', 3.6, 4.9)
    bianry_feature_billi  = compute_ontology_value(curr_data,'Billirubin',0.2 ,1.1)
    bianry_feature_sodium = compute_ontology_value(curr_data,'Sodium',135 , 145)
    bianry_feature_temp   = compute_ontology_value(curr_data,'Temp',  36.1, 37.2)
    bianry_feature_wbc    = compute_ontology_value(curr_data,'WBC',3.7 ,10.3)
    bianry_feature_hr     = compute_ontology_value(curr_data,'HR', 60, 100)
    binary_feature_RR     = compute_ontology_value(curr_data,'RR' ,12,25)

    comb_df = pd.concat([binary_feature_SBP,
                         binary_feature_DBP, 
                         bianry_feature_SCR,
                         bianry_feature_Bicar,
                         bianry_feature_Hemat,
                         bianry_feature_Pot,
                         bianry_feature_billi,
                         bianry_feature_sodium,
                         bianry_feature_temp,
                         bianry_feature_wbc,
                         bianry_feature_hr,
                         binary_feature_RR],axis=1)
    
    comb_df.to_csv(out_dir + 'Binary_feature/' + pt + '_inHOSP.csv')


