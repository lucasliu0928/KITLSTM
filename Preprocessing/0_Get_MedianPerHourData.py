#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  6 14:21:33 2022

@author: lucasliu
"""


import glob
import re
import pandas as pd
import numpy as np

from Preprocessing_Ultilites import load_time_info, load_outcome_info, load_demo

#####################################################################################
# data dir
#####################################################################################
#input dir
raw_data_dir = "/Users/lucasliu/Desktop/DrChen_Projects/All_AKI_Projects/UK_Data/Extracted_data/"
intermediate_dir = "/Users/lucasliu/Desktop/DrChen_Projects/All_AKI_Projects/AKID_Project2/Intermediate_Data/practice_data_220106/"

#output dir 
outdir = intermediate_dir + 'patient_level_data/medianPerHour_data/'

#####################################################################################
#2.Load all time info
#####################################################################################
all_timeinfo_df = load_time_info(raw_data_dir)  

#####################################################################################
#4. From pt level raw feature (Extracted from KGDAL, exterme value excluded)
# Get median-per-hour data from duration start - duration end
#####################################################################################
analysis_duration = "onRRT"
raw_feature_dir = intermediate_dir + "patient_level_data/raw_data/" + analysis_duration + "_raw/"
raw_feature_files = glob.glob(raw_feature_dir + "*.csv")
for file in raw_feature_files:
    file = raw_feature_files[0]
    #get curr id 
    curr_id = re.search(analysis_duration + "_raw/(.*)" + analysis_duration  + "_raw", file).group(1)
    
    #get curr df
    curr_df = pd.read_csv(file , parse_dates= ['Record_Time'])
    
    #get analysis duration start and end time
    curr_time_df = all_timeinfo_df[all_timeinfo_df['ENCNTR_ID'] == curr_id]
    curr_rrt_start = curr_time_df['RRT_START_DATE'].iloc[0]
    curr_rrt_end   = curr_time_df['RRT_STOP_DATE'].iloc[0]
    
    
    duration_start = pd.date_range(start= curr_rrt_start, end = curr_rrt_end, freq="H")

    curr_hourly_df = pd.DataFrame({"Hour_Start": duration_start,
                                   "SBP":None,
                                   ""})
    for i in curr_hourly_df.shape[0]:
        i = 0
        curr_hour_start = curr_hourly_df.iloc[0]['Hour_Start']
        curr_hour_end   = curr_hour_start + pd.Timedelta(hours = 1)
        curr_inhour_data     = curr_df[(curr_df['Record_Time'] >= curr_hour_start) & 
                                  (curr_df['Record_Time'] <  curr_hour_end)]
        curr_median_values   = curr_inhour_data.median(axis = 0, skipna = True)
