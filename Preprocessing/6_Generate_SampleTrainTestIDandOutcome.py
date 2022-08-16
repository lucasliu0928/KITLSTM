#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 29 21:45:26 2021

@author: lucasliu
"""

import math
import pandas as pd
import numpy as np


#####################################################################################
# data dir
#####################################################################################
#User input
analysis_duration = "onRRT"
proj_dir = "/Users/lucasliu/Desktop/DrChen_Projects/All_AKI_Projects/AKID_Project2/Intermediate_Data/practice_data_092721/"
data_dir1 = proj_dir + analysis_duration + '_analysis/patient_level_data/'
data_dir2 = proj_dir + analysis_duration + '_analysis/sample_level_data/'



out_dir = proj_dir + analysis_duration + '_analysis//sample_level_data/'


#####################################################################################
#1. Load train and test Pt IDs and validation IDs
#####################################################################################
train_pt_ID_df = pd.read_csv(data_dir1 + "trainID_" + analysis_duration  + ".csv", index_col=0) #568
test_pt_ID_df = pd.read_csv(data_dir1 + "testID_" + analysis_duration  + ".csv", index_col=0)  #150
validation_pt_ID_df = pd.read_csv(data_dir1 + "ValidationID_" + analysis_duration  + ".csv", index_col=0) #30
train_pt_Ids = list(train_pt_ID_df.index)
test_pt_Ids = list(test_pt_ID_df.index)
validation_pt_Ids = list(validation_pt_ID_df.index)
#####################################################################################
#2.Remove obvious negtive samples by PCA most contributed two features
#####################################################################################
sp_duration_df = pd.read_csv(data_dir2 + "FinalIDs_spOutcome.csv") #all ID


sp_obsneg_df = pd.read_csv(data_dir2 + "Z_PCA_data/OBV_NEG_SAMPLE_IDs_AndData.csv") #obs neg ID
sp_duration_df_on_exc = sp_duration_df[~sp_duration_df['SAMPLE_ID'].isin(sp_obsneg_df['Unnamed: 0'])]


#####################################################################################
#Report count
#####################################################################################
#Before
neg_df = sp_duration_df[sp_duration_df['sample_death_in24h'] == 0]
pos_df = sp_duration_df[sp_duration_df['sample_death_in24h'] == 1] 
len(set(sp_duration_df['SAMPLE_ID'])) #total number of samples
len(set(neg_df['SAMPLE_ID'])) #number neg of samples
len(set(pos_df['SAMPLE_ID'])) #number pos of samples

len(set(sp_duration_df['ENCNTR_ID'])) #total number of patient
len(set(neg_df['ENCNTR_ID'])) #number neg of patient
len(set(pos_df['ENCNTR_ID'])) #number pos of patient


#After
neg_df = sp_duration_df_on_exc[sp_duration_df_on_exc['sample_death_in24h'] == 0]
pos_df = sp_duration_df_on_exc[sp_duration_df_on_exc['sample_death_in24h'] == 1] 
len(set(sp_duration_df_on_exc['SAMPLE_ID'])) #total number of samples
len(set(neg_df['SAMPLE_ID'])) #number neg of samples
len(set(pos_df['SAMPLE_ID'])) #number pos of samples

len(set(sp_duration_df_on_exc['ENCNTR_ID'])) #total number of patient
len(set(neg_df['ENCNTR_ID'])) #number neg of patient
len(set(pos_df['ENCNTR_ID'])) #number pos of patient

#AFter patient level
neg_df = sp_duration_df_on_exc[sp_duration_df_on_exc['patient_hospital_death'] == 0]
pos_df = sp_duration_df_on_exc[sp_duration_df_on_exc['patient_hospital_death'] == 1] 
len(set(neg_df['ENCNTR_ID'])) #number neg of patient
len(set(pos_df['ENCNTR_ID'])) #number pos of patient

#####################################################################################
#3.Get train and test sample IDs without exclusion
#####################################################################################
train_sp_df1 = sp_duration_df[sp_duration_df['ENCNTR_ID'].isin(train_pt_Ids)]
test_sp_df1  = sp_duration_df[sp_duration_df['ENCNTR_ID'].isin(test_pt_Ids)]
validation_sp_df1  = sp_duration_df[sp_duration_df['ENCNTR_ID'].isin(validation_pt_Ids)]

train_IDs1 = train_sp_df1['SAMPLE_ID']
test_IDs1  = test_sp_df1['SAMPLE_ID']
validation_Ids1 = validation_sp_df1['SAMPLE_ID']
train_IDs1.to_csv(out_dir + "Train_SampleIDs_withoutExc.csv", index = False)
test_IDs1.to_csv(out_dir + "Test_SampleIDs_withoutExc.csv",index = False)
validation_Ids1.to_csv(out_dir + "Validation_SampleIDs_withoutExc.csv",index = False)

#####################################################################################
#3.Get train and test sample IDs after exclusion
#####################################################################################
train_sp_df = sp_duration_df_on_exc[sp_duration_df_on_exc['ENCNTR_ID'].isin(train_pt_Ids)]
test_sp_df  = sp_duration_df_on_exc[sp_duration_df_on_exc['ENCNTR_ID'].isin(test_pt_Ids)]
validation_sp_df  = sp_duration_df_on_exc[sp_duration_df_on_exc['ENCNTR_ID'].isin(validation_pt_Ids)]

train_IDs = train_sp_df['SAMPLE_ID']
test_IDs  = test_sp_df['SAMPLE_ID']
validation_Ids = validation_sp_df['SAMPLE_ID']
train_IDs.to_csv(out_dir + "Train_SampleIDs_afterExc.csv", index = False)
test_IDs.to_csv(out_dir + "Test_SampleIDs_afterExc.csv",index = False)
validation_Ids.to_csv(out_dir + "Validation_SampleIDs_afterExc.csv",index = False)

#####################################################################################
#4.Report stats for train and test and validation
#####################################################################################
#train:
neg_df = train_sp_df[train_sp_df['sample_death_in24h'] == 0]
pos_df = train_sp_df[train_sp_df['sample_death_in24h'] == 1] 
len(set(train_sp_df['SAMPLE_ID'])) #total number of samples
len(set(neg_df['SAMPLE_ID'])) #number neg of samples
len(set(pos_df['SAMPLE_ID'])) #number pos of samples

len(set(train_sp_df['ENCNTR_ID'])) #total number of patient
len(set(neg_df['ENCNTR_ID'])) #number neg of patient
len(set(pos_df['ENCNTR_ID'])) #number pos of patient

#validation:
neg_df = validation_sp_df[validation_sp_df['sample_death_in24h'] == 0]
pos_df = validation_sp_df[validation_sp_df['sample_death_in24h'] == 1] 
len(set(validation_sp_df['SAMPLE_ID'])) #total number of samples
len(set(neg_df['SAMPLE_ID'])) #number neg of samples
len(set(pos_df['SAMPLE_ID'])) #number pos of samples

len(set(validation_sp_df['ENCNTR_ID'])) #total number of patient
len(set(neg_df['ENCNTR_ID'])) #number neg of patient
len(set(pos_df['ENCNTR_ID'])) #number pos of patient


#test:
neg_df = test_sp_df[test_sp_df['sample_death_in24h'] == 0]
pos_df = test_sp_df[test_sp_df['sample_death_in24h'] == 1] 
len(set(test_sp_df['SAMPLE_ID'])) #total number of samples
len(set(neg_df['SAMPLE_ID'])) #number neg of samples
len(set(pos_df['SAMPLE_ID'])) #number pos of samples

len(set(test_sp_df['ENCNTR_ID'])) #total number of patient
len(set(neg_df['ENCNTR_ID'])) #number neg of patient
len(set(pos_df['ENCNTR_ID'])) #number pos of patient
