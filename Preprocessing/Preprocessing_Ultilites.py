#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 27 15:31:55 2021

@author: lucasliu
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#Load time info
def load_time_info(data_dir):    
    timeinfo_df = pd.read_excel(data_dir + "Time_Info/ALL_Time_info_df_corrected_0711.xlsx") 

    for j in range(1,timeinfo_df.shape[1]):
        timeinfo_df.iloc[:,j] = pd.to_datetime(timeinfo_df.iloc[:,j])
    
    #Add Hosp duration
    timeinfo_df['HOSP_Duration'] =  timeinfo_df['HOPS_DISCHRG_DT'] - timeinfo_df['HOSP_ADMT_DT'] 
    timeinfo_df['ICU_Duration']  =  timeinfo_df['ICU_DISCHRG'] - timeinfo_df['ICU_ADMIT'] 
    
    #Add RRT start and RRT end for whole RRT duration, start = min(CRRT, HD), end = max(CRRT, HD)
    timeinfo_df['RRT_START_DATE'] =  timeinfo_df[['CRRT_START_DATE','HD_START_DATE']].min(axis = 1)
    timeinfo_df['RRT_STOP_DATE']  =  timeinfo_df[['CRRT_STOP_DATE','HD_STOP_DATE']].max(axis = 1)
    timeinfo_df['RRT_Duration']   =  timeinfo_df['RRT_STOP_DATE'] - timeinfo_df['RRT_START_DATE'] 


    return timeinfo_df


def load_outcome_info(data_dir):
    #prospective outcome
    prosp_outcome_df = pd.read_excel(data_dir + "Final_prosp_chart_review111pts.xlsx")
    prosp_outcome_df = prosp_outcome_df.loc[:,['ENCNTR_ID','Hospital_Death','ESRD_Status_PriorDC','Kidney_Transplant']] 
    prosp_outcome_df.columns = ['ENCNTR_ID','Hospital_Death','ESRD_status','Kidney_Transplant']

    #retro outcome
    retro_outcome_df = pd.read_excel(data_dir + "Retrospective chart review_final _MJedits (1).xlsx",sheet_name= 1)
    retro_outcome_df = retro_outcome_df.loc[:,['ENCNTR_ID','Hospital_Death ','ESRD_status','Kidney_Transplant']] 
    retro_outcome_df.columns = ['ENCNTR_ID','Hospital_Death','ESRD_status','Kidney_Transplant']
    
    #combine
    all_outcome_df = pd.concat([prosp_outcome_df,retro_outcome_df],axis=0)
    all_outcome_df['ENCNTR_ID'] = 'X'+ all_outcome_df['ENCNTR_ID'].astype(str)
    all_outcome_df = all_outcome_df[~all_outcome_df['ENCNTR_ID'].duplicated(keep='first')]
    
    #recode NA outcome = 0, denotes alive
    all_outcome_df.loc[np.isnan(all_outcome_df['Hospital_Death']) == True,'Hospital_Death'] = 0
    #recode 2 outcome to 1, denotes death
    all_outcome_df.loc[all_outcome_df['Hospital_Death'] == 2,'Hospital_Death'] = 1
    
    print(all_outcome_df['Hospital_Death'].value_counts().to_dict())
    return all_outcome_df

def load_demo(data_dir):
    demo_df = pd.read_excel(data_dir + "BASE/ALL_BASE_DuplicatesRemoved.xlsx", sheet_name=0)
    
    #recode
    demo_df.loc[demo_df['RACE_CD_DES'] == 'WHITE','RACE_CD_DES'] = 0
    demo_df.loc[demo_df['RACE_CD_DES'] == 'BLACK/AFR AMERI','RACE_CD_DES'] = 1
    demo_df.loc[demo_df['RACE_CD_DES'] == 'HAWAIIAN/PACISL','RACE_CD_DES'] = 2
    demo_df.loc[demo_df['RACE_CD_DES'] == 'SPANISH AMRICAN','RACE_CD_DES'] = 2
    demo_df.loc[demo_df['RACE_CD_DES'] == 'ASIAN','RACE_CD_DES'] = 2
    demo_df.loc[demo_df['RACE_CD_DES'] == 'UNREPORT','RACE_CD_DES'] = 2
    demo_df.loc[demo_df['RACE_CD_DES'] == 'AM INDIAN/ALASK','RACE_CD_DES']= 2
    demo_df.loc[demo_df['RACE_CD_DES'] == 'MIDDLE EASTERN','RACE_CD_DES']= 2
    demo_df.loc[demo_df['RACE_CD_DES'] == 'BI-RACIAL','RACE_CD_DES']= 2
    demo_df.loc[demo_df['RACE_CD_DES'] == 'REFUSE','RACE_CD_DES']= 2
    demo_df.loc[demo_df['RACE_CD_DES'] == 'OTHER PACIF ISL','RACE_CD_DES']= 2
    demo_df.loc[demo_df['RACE_CD_DES'].isnull(),'RACE_CD_DES']= 2
    
    demo_df.loc[demo_df['GENDR_CD_DES'] == 'FEMALE','GENDR_CD_DES'] = 0
    demo_df.loc[demo_df['GENDR_CD_DES'] == 'MALE','GENDR_CD_DES'] = 1
    demo_df.loc[demo_df['GENDR_CD_DES'] == 'UNKNOWN','GENDR_CD_DES'] = np.nan
    
    demo_df.dropna(inplace = True, subset = ['RACE_CD_DES','GENDR_CD_DES','ENCNTR_ID']) #remove rows that has any NA
    
    return demo_df

def load_diagnosis(data_dir):
    diag_df = pd.read_csv(data_dir + "BASE/ALL_BASE_DuplicatesRemoved.xlsx", sheet_name=0)

#####Functions for generate more samples

#This function generate possible end time for each start time point for sp_duration long
def generate_endtime (start_times,sp_duration):
    endtimes = []
    for s in start_times:
        end_t = s + pd.Timedelta(sp_duration)
        endtimes.append(end_t)
        
    sp_time_df = pd.DataFrame({'Start':    start_times, 
                               'End' :     endtimes,
                               'Duration': sp_duration})
    return sp_time_df

#This function Generate all possible start and end segment 
def generate_all_possible_segment (feature_data, start_date, end_date, possible_duration):
    #1.Start hours for segment is every 1 hour from the start date
    start_hours = pd.date_range(start = start_date, end = end_date, freq = "1H") #Hour list
    
    #2.End hours for segemnt is each start hour + desired durations , and end hour can not be >end_date
    sg_list = []
    for dura in possible_duration:
        curr_sgs = generate_endtime(start_hours, dura)
        sg_list.append(curr_sgs)
    #comcatete all segment for all possible duratiion
    all_possible_sg_df = pd.concat(sg_list,axis = 0,ignore_index=True)

    #3. remove end time beyond max end 
    all_possible_sg_df = all_possible_sg_df[all_possible_sg_df['End'] <= end_date]
    
    #4.For each possible start and end , check if pt actually has faeture data (it is possible, pt has none of the feature in this range)
    n_timepoints = []
    for j in range(all_possible_sg_df.shape[0]):
        check_start =  all_possible_sg_df.iloc[j]['Start']
        check_end    = all_possible_sg_df.iloc[j]['End']   

        sp_df = feature_data[(feature_data['Record_Time'] >= check_start) & (feature_data['Record_Time'] <= check_end)]
        n_timepoints.append(sp_df.shape[0])
    all_possible_sg_df['n_timepoints_hasMeasurement'] = n_timepoints
    #5.remove segment that has time points < 10
    all_possible_sg_df = all_possible_sg_df[all_possible_sg_df['n_timepoints_hasMeasurement'] >= 10]

    return all_possible_sg_df

#This function determines if sample died between end + timewin1 and end + timewin2
def compute_sample_outcome (decease_date, end_date, timewin1,  timewin2):
    cond1 = decease_date >=  end_date + pd.Timedelta(timewin1)
    cond2 = decease_date <  end_date + pd.Timedelta(timewin2)
    if cond1 & cond2 : 
        outcome = 1
    else:
        outcome = 0 
    return outcome

#This function determines if sample died between end and end + timewin1
def compute_sample_outcome2 (decease_date, end_date, timewin1):
    cond1 = decease_date >=  end_date 
    cond2 = decease_date <  end_date + pd.Timedelta(timewin1)
    if cond1 & cond2 : 
        outcome = 1
    else:
        outcome = 0 
    return outcome

####### Plot pt features
def plot_individual_feature_and_pt_func(pt_feature_df):
    #idex must be the time, columns be the value
    plt.figure();
    pt_feature_df.plot(linewidth=2);
    
    
def compute_time_to_last_seen_value (input_df,feature):
    #input_df = curr_data
    #feature = ft
    
    #Get feature colun
    feature_value_df = input_df[feature]
    
    #Convert index to datetime
    feature_value_df.index = pd.to_datetime(feature_value_df.index) 
    
    #sort by time
    feature_value_df.sort_index(inplace = True)
    
    #Initial a last seen time
    value_1st = feature_value_df.iloc[0]  #first value
     
    if pd.isna(value_1st) == False:     #The initial time = 1st time step if there is a value at the first time step
       last_seen_time = feature_value_df.index[0]
    else:
       last_seen_time = feature_value_df.index[0] + pd.Timedelta(days=1000) #initial last seen a value time as the future time, so the the delta time would be negative
    
    #number of time points 
    n_tp = feature_value_df.shape[0]
    
    time_diff = []
    for t in range(n_tp):
        curr_time = feature_value_df.index[t]
        curr_value = feature_value_df.iloc[t]
        
        #If current value is not NA, update last_seen_time
        if pd.isna(curr_value) == False:
           last_seen_time = curr_time

           
        #Get time past since last seen a value
        delta_time = curr_time - last_seen_time
        delta_time_inMin = delta_time.total_seconds()//60 #Unit: Minites
        
        if (delta_time_inMin < 0 ): #all negateive values will be treated as -1
            delta_time_inMin = -1
            
        time_diff.append(delta_time_inMin)
 
    d = {'Record_Time':feature_value_df.index, 'Value': feature_value_df.tolist() ,'TimeSinceLastSeenAValue':time_diff}
    Time_Diff_df = pd.DataFrame(d)
    Time_Diff_df.columns = ['Record_Time',feature, 'TimeSinceLastSeen'+ feature ]
    return Time_Diff_df



def compute_ontology_value (input_df,feature,low_th,high_th):
    # input_df = curr_data
    # feature = 'SBP'
    # low_th = 90
    # high_th = 130
    
    #create feature name
    feature_name_high = 'HIGH_' + feature
    feature_name_low  = 'LOW_' + feature
    feature_name_comb = 'Abnormal_' + feature

    #Get feature colun
    feature_value_df = pd.DataFrame(input_df[feature])

    #For onotology corresponding high
    if pd.isnull(high_th) ==  False:
       #Add new feature column
       feature_value_df[feature_name_high] = np.nan
       #Location qualifies
       high_loc = feature_value_df[feature] > high_th
       feature_value_df[feature_name_high][high_loc] = 1
       feature_value_df[feature_name_high][~high_loc] = 0
    
    #For onotology corresponding low
    if pd.isnull(low_th) ==  False:
       #Add new feature column
       feature_value_df[feature_name_low] = np.nan
       #Location qualifies
       low_loc = feature_value_df[feature] < low_th
       feature_value_df[feature_name_low][low_loc] = 1
       feature_value_df[feature_name_low][~low_loc] = 0
       
    #Abnormal Combination of >high and <low 
    if pd.isnull(high_th) ==  False and pd.isnull(low_th) ==  False:
       #Add new feature column
       feature_value_df[feature_name_comb] = np.nan
       #Location qualifies
       abnormal_loc = (feature_value_df[feature] > high_th) | (feature_value_df[feature] < low_th)
       feature_value_df[feature_name_comb][abnormal_loc] = 1
       feature_value_df[feature_name_comb][~abnormal_loc] = 0
       
    ontology_feature_df = feature_value_df.iloc[:,[1,2,3]]
    return ontology_feature_df