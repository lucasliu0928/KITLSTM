#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 26 20:38:37 2021

@author: lucasliu
"""
import pandas as pd
import glob
import re

from Preprocessing_Ultilites import generate_all_possible_segment


#####################################################################################
# data dir
#####################################################################################
#User input
analysis_duration = "onRRT"
durations = ["48H","72H","96H"]  #selected duration
n_segments = 30                  #selected number of segments to generate

#input dir
intermediate_dir = "/Users/lucasliu/Desktop/DrChen_Projects/All_AKI_Projects/AKID_Project2/Intermediate_Data/practice_data_092721/"

#output dir 
outdir = intermediate_dir + analysis_duration +  "_analysis/" + 'sample_level_data/'

#####################################################################################
#1. Load final ID
#####################################################################################
final_IDs_df = pd.read_csv(intermediate_dir + analysis_duration +  "_analysis/patient_level_data/" + "finalID_" + analysis_duration + ".csv")
final_IDs = list(final_IDs_df['ENCNTR_ID']) 


#####################################################################################
#1. Get sample durations for each pts
#####################################################################################
raw_feature_dir = intermediate_dir + analysis_duration + "_analysis/patient_level_data/" + analysis_duration + "_raw/"
raw_feature_files = glob.glob(raw_feature_dir + "*.csv")

All_Sample_duration_list = [] 
for file in raw_feature_files:
    pt_id = re.search(analysis_duration + "_raw/(.*)" + analysis_duration  + "_raw", file).group(1)
    if pt_id in final_IDs: 

        #get patient df
        p_df = pd.read_csv(file , parse_dates= ['Record_Time'])
        p_start = min(p_df['Record_Time'])
        p_end   = max(p_df['Record_Time'])
    
        #generate all possible start and end segment (all end hours must <= p_end), and make sure has data in range
        all_seg_df = generate_all_possible_segment(p_df,p_start,p_end,durations) 
        
        #Random select 30 segment without replacement
        n_possible_segment = all_seg_df.shape[0]
        
        if n_possible_segment > 30:
            selected_seg_df = all_seg_df.sample(n= n_segments,replace = False, random_state=1)
        else : #if less than n_segments, then take all the possible segments
            selected_seg_df = all_seg_df
        
        #reorder by start hour
        selected_seg_df.sort_values(by=['Start'], inplace=True, ignore_index=True)
        
        #Add Pt ID
        selected_seg_df['ENCNTR_ID'] = pt_id

        #Add sample ID = 'ENCNTR_ID' + sample index (1,2,3...)
        final_n_seg = selected_seg_df.shape[0]
        
        sample_ids = [pt_id + '_SP' + str(x) for x in range(1, final_n_seg + 1)]
        selected_seg_df['SAMPLE_ID'] = sample_ids

        #reorder column names
        selected_seg_df = selected_seg_df[['ENCNTR_ID','SAMPLE_ID','Start', 'End', 'Duration','n_timepoints_hasMeasurement']]
    
        All_Sample_duration_list.append(selected_seg_df)
        
All_Sample_duration_df = pd.concat(All_Sample_duration_list, axis=0)
All_Sample_duration_df.to_csv(outdir + "FinalIDs_spDuration.csv", index= False)

