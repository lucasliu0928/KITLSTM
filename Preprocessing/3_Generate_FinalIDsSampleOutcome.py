#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 28 18:49:36 2021

@author: lucasliu
"""


import pandas as pd
import numpy as np

from Preprocessing_Ultilites import compute_sample_outcome2
#####################################################################################
# data dir
#####################################################################################
#User input
analysis_duration = "onRRT"

#proj dir
proj_dir = "/Users/lucasliu/Desktop/DrChen_Projects/All_AKI_Projects/AKID_Project2/Intermediate_Data/practice_data_092721/" + analysis_duration +  "_analysis/"

#input dir 
indir =  proj_dir + "patient_level_data/"
indir2 = proj_dir + 'sample_level_data/'

#output dir 
outdir = proj_dir + 'sample_level_data/'

#####################################################################################
#1. Load final ID
#####################################################################################
final_IDs_df = pd.read_csv(indir +  "finalID_" + analysis_duration + ".csv")
final_IDs = list(final_IDs_df['ENCNTR_ID']) 

#####################################################################################
#2. Load final ID time info and outcome info
#####################################################################################
final_IDs_outcome_df = pd.read_csv(indir +  "outcome_finalID_" + analysis_duration + ".csv")
final_IDs_time_df    = pd.read_csv(indir +  "time_finalID_" + analysis_duration + ".csv", parse_dates = ['HOSP_ADMT_DT','HOPS_DISCHRG_DT'])

#####################################################################################
# 3. Load fianl Id sample durations
#####################################################################################
final_IDs_SP_durations_df = pd.read_csv(indir2 +  "FinalIDs_spDuration.csv", parse_dates = ['Start', 'End'])

# #####################################################################################
# #4. Generate outcome for sample
# #4.1 Death in the following 24 hours (Death date >= END date and death date <= END date + 24 h)
# #4.2 Death in 24 - 48 hours (Death date >= END date + 24 and death date <= END date + 48 h)
# #4.3 Death in 48 - 72 hours (Death date >= END date + 48 and death date <= END date + 72 h)
# #4.4 ...
# #4.5 ...
# ############################################################################################
# sp_outcome_df = final_IDs_SP_durations_df.copy()
# n_samples = sp_outcome_df.shape[0]

# sp_outcome_in24 = []
# sp_outcome_24to48 = []
# sp_outcome_48to72 = []
# sp_outcome_72to96 = []
# sp_outcome_96to120 = []
# sp_outcome_after120 = []
# pt_outcome_list = []
# death_date_list = []

# for i in range(n_samples):
#     pt_id    = sp_outcome_df.iloc[i]['ENCNTR_ID']
#     start_d  = sp_outcome_df.iloc[i]['Start']
#     end_d    = sp_outcome_df.iloc[i]['End']    
     
#     #get pt hosp disc date    
#     hosp_stop_date = final_IDs_time_df.loc[final_IDs_time_df['ENCNTR_ID'] == pt_id, 'HOPS_DISCHRG_DT'].tolist()[0]
#     #get pt outcome
#     pt_outcome  = final_IDs_outcome_df.loc[final_IDs_outcome_df['ENCNTR_ID'] == pt_id, 'Hospital_Death'].tolist()[0]
#     pt_outcome_list.append(pt_outcome)
#     if pt_outcome == 0 :
#         death_date = np.nan
#         death_date_list.append(death_date)
#         sp_outcome_in24.append(0)
#         sp_outcome_24to48.append(0)
#         sp_outcome_48to72.append(0)
#         sp_outcome_72to96.append(0)
#         sp_outcome_96to120.append(0)
#         sp_outcome_after120.append(0)
        
#     elif pt_outcome == 1:
#         death_date = hosp_stop_date
#         death_date_list.append(death_date)
#         sp_outcome_in24.append(compute_sample_outcome(death_date,end_d,"0H","24H"))
#         sp_outcome_24to48.append(compute_sample_outcome(death_date,end_d,"24H","48H"))
#         sp_outcome_48to72.append(compute_sample_outcome(death_date,end_d,"48H","72H"))
#         sp_outcome_72to96.append(compute_sample_outcome(death_date,end_d,"72H","96H"))
#         sp_outcome_96to120.append(compute_sample_outcome(death_date,end_d,"96H","120H"))
#         sp_outcome_after120.append(compute_sample_outcome(death_date,end_d,"120H","5000H")) #use large H for infinit end

# sp_outcome_df["sample_death_in24h"] = sp_outcome_in24
# sp_outcome_df["sample_death_between24hto48h"] = sp_outcome_24to48
# sp_outcome_df["sample_death_between48hto72h"] = sp_outcome_48to72
# sp_outcome_df["sample_death_between72hto96h"] = sp_outcome_72to96
# sp_outcome_df["sample_death_between96hto120h"] = sp_outcome_96to120
# sp_outcome_df["sample_death_after120h"] = sp_outcome_after120
# sp_outcome_df["patient_hospital_death"] = pt_outcome_list
# sp_outcome_df["decease_date"] = death_date_list

# sp_outcome_df.to_csv(outdir + "FinalIDs_spOutcome.csv", index=False)

#####################################################################################
#4. Generate outcome for sample
#4.1 Death in the following 24 hours (Death date >= END date and death date <= END date + 24 h)
#4.2 Death in the following 48 hours (Death date >= END date and death date <= END date + 48 h)
#4.3 Death in the following 72 hours (Death date >= END date and death date <= END date + 72 h)
#4.4 ...
#4.5 ...
############################################################################################
sp_outcome_df = final_IDs_SP_durations_df.copy()
n_samples = sp_outcome_df.shape[0]

sp_outcome_in24 = []
sp_outcome_in48 = []
sp_outcome_in72 = []
sp_outcome_in96 = []
sp_outcome_in120 = []
pt_outcome_list = []
death_date_list = []

for i in range(n_samples):
    pt_id    = sp_outcome_df.iloc[i]['ENCNTR_ID']
    start_d  = sp_outcome_df.iloc[i]['Start']
    end_d    = sp_outcome_df.iloc[i]['End']    
     
    #get pt hosp disc date    
    hosp_stop_date = final_IDs_time_df.loc[final_IDs_time_df['ENCNTR_ID'] == pt_id, 'HOPS_DISCHRG_DT'].tolist()[0]
    #get pt outcome
    pt_outcome  = final_IDs_outcome_df.loc[final_IDs_outcome_df['ENCNTR_ID'] == pt_id, 'Hospital_Death'].tolist()[0]
    pt_outcome_list.append(pt_outcome)
    if pt_outcome == 0 :
        death_date = np.nan
        death_date_list.append(death_date)
        sp_outcome_in24.append(0)
        sp_outcome_in48.append(0)
        sp_outcome_in72.append(0)
        sp_outcome_in96.append(0)
        sp_outcome_in120.append(0)
        
    elif pt_outcome == 1:
        death_date = hosp_stop_date
        death_date_list.append(death_date)
        sp_outcome_in24.append(compute_sample_outcome2(death_date,end_d,"24H"))
        sp_outcome_in48.append(compute_sample_outcome2(death_date,end_d,"48H"))
        sp_outcome_in72.append(compute_sample_outcome2(death_date,end_d,"72H"))
        sp_outcome_in96.append(compute_sample_outcome2(death_date,end_d,"96H"))
        sp_outcome_in120.append(compute_sample_outcome2(death_date,end_d,"120H"))

sp_outcome_df["sample_death_in24h"] = sp_outcome_in24
sp_outcome_df["sample_death_in48h"] = sp_outcome_in48
sp_outcome_df["sample_death_in72h"] = sp_outcome_in72
sp_outcome_df["sample_death_in96h"] = sp_outcome_in96
sp_outcome_df["sample_death_in120h"] = sp_outcome_in120
sp_outcome_df["patient_hospital_death"] = pt_outcome_list
sp_outcome_df["decease_date"] = death_date_list

sp_outcome_df.to_csv(outdir + "FinalIDs_spOutcome.csv", index=False)

############################################################################################
#Report count
############################################################################################

sp_outcome_df["sample_death_in24h"].value_counts()    # 1608
sp_outcome_df["sample_death_in48h"].value_counts() # 2662
sp_outcome_df["sample_death_in72h"].value_counts()  #3372
sp_outcome_df["sample_death_in96h"].value_counts()  # 3923
sp_outcome_df["sample_death_in120h"].value_counts() #4374
sp_outcome_df["patient_hospital_death"].value_counts() #7246
