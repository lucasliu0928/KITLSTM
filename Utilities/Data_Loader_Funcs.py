#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 21 02:21:39 2021

@author: lucasliu
"""
import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler,MinMaxScaler   


def Load_allPts_valuedata(analysis_ID,data_dir,feature_names,duration_type):
    
    data_list = []
    
    for pt in analysis_ID:
        #file name
        file_name = pt + "_" + duration_type + ".csv"
        #Read file
        df = pd.read_csv(data_dir + file_name, index_col=0)
        df = df[feature_names]

        #convert to numpy
        df = df.to_numpy()
        #convert to tensor and add a dmension
        df = torch.tensor(df,dtype= torch.float64).unsqueeze(0)
        
        #only collect data for 7 days
    
        data_list.append(df)
        
    return data_list

# def Load_allPts_valuedata_interpolated(analysis_ID,data_dir):
#     data_list = []
    
#     for pt in analysis_ID:
#         #file name
#         file_name = pt + "_inHOSP.csv"
#         #Read file
#         df = pd.read_csv(data_dir + file_name, index_col=0)
#         df = df.reset_index(drop = True)
#         df = df.interpolate()
       
#         df = df.fillna(method="ffill")         #fill end NA with last non-NA 
#         df = df.fillna(method="bfill")         #fill begining NA with first non-NA 

        
#         #convert to numpy
#         df = df.to_numpy()
#         #convert to tensor and add a dmension
#         df = torch.tensor(df,dtype= torch.float64).unsqueeze(0)
        
#         #only collect data for 7 days
    
#         data_list.append(df)
        
#     return data_list


def Load_allPts_deltatimedata(analysis_ID,data_dir,delta_feature_names, duration_type):
    
    data_list = []
    
    for pt in analysis_ID:
        #file name
        file_name = pt + "_" + duration_type + ".csv"
        #Read file
        df = pd.read_csv(data_dir + file_name, index_col=0)
        df = df[delta_feature_names]
        #convert to numpy
        df = df.to_numpy()
        #convert to tensor and add a dmension
        df = torch.tensor(df,dtype= torch.float64).unsqueeze(0)
        
        #only collect data for 7 days
    
        data_list.append(df)
        
    return data_list


def Load_allPts_binarydata(analysis_ID,data_dir, duration_type):
    
    data_list = []
    
    for pt in analysis_ID:
        #file name
        file_name = pt + "_" + duration_type + ".csv"
        #Read file
        df = pd.read_csv(data_dir + file_name, index_col=0)
        df = df[['HIGH_SBP','LOW_SBP',
                 'HIGH_DBP','LOW_DBP',
                 'LOW_Scr',
                 'Abnormal_Bicarbonate',
                 'HIGH_Hematocrit',	'LOW_Hematocrit',
                 'HIGH_Potassium', 'LOW_Potassium',
                 'HIGH_Billirubin',	'LOW_Billirubin',
                 'HIGH_Sodium' ,	'LOW_Sodium',
                 'HIGH_Temp',	'LOW_Temp',
                 'HIGH_WBC',	'LOW_WBC',
                 'Abnormal_HR',             
                 'Abnormal_RR']]


        #convert to numpy
        df = df.to_numpy()
        #convert to tensor and add a dmension
        df = torch.tensor(df,dtype= torch.float64).unsqueeze(0)
        
        #only collect data for 7 days
    
        data_list.append(df)
        
    return data_list

def Load_allPts_staticdata(analysis_ID,data_dir, duration_type):
    all_static_df = pd.read_csv(data_dir + "samples_static_features_" + duration_type + ".csv", index_col=0)

    #doing the following to make  sure the data is in the order of analysis_ID
    data_list = []
    for pt in analysis_ID:
        
        #Get curr static df for pt
        df = all_static_df[all_static_df['SAMPLE_ID'] == pt]
        
        #select f
        df = df[['AGE','GENDER','RACE','ADMISSION_WT','BMI','CHARLSON']]


        #convert to numpy
        df = df.to_numpy()
        #convert to tensor and add a dmension
        df = torch.tensor(df,dtype= torch.float64).unsqueeze(0)
        
        #only collect data for 7 days
    
        data_list.append(df)
    
    final_df = torch.cat(data_list, axis = 0)
    final_df = final_df.squeeze(1) #remove one reduant dim
    return final_df

def Load_ontologydata(data_dir):
    #Read file
    ent_emb = pd.read_csv(data_dir + "ent_embeddings.csv", index_col=0)
    target_emb = ent_emb.iloc[0,:]
    other_emb =  ent_emb.iloc[1:,:]
    
    target_emb = target_emb.to_numpy()
    target_emb = target_emb.reshape(target_emb.shape[0],1)
   

    other_emb = other_emb.to_numpy()
    other_emb = np.transpose(other_emb) #rows: embedding index, col: ontology (high_SBP,LowSBP,lowScr,Ab_RR)


    
    rel_emb = pd.read_csv(data_dir + "rel_embeddings.csv", index_col=0)
    rel_emb = rel_emb.iloc[:,1:] #drop fisrt n_ca col
    rel_emb = rel_emb.to_numpy() 
    rel_emb = np.transpose(rel_emb) #rows: embedding index, col: ontology (#AKItohighSBP, AKItolowSBP,AKItoScr,AKItoAbRR)
 
    
    #scale/normalize to center
    scalers = MinMaxScaler()    
    target_emb = scalers.fit_transform(target_emb) 
    scalers = MinMaxScaler()    
    other_emb = scalers.fit_transform(other_emb) 
    scalers = MinMaxScaler()    
    rel_emb = scalers.fit_transform(rel_emb) 

    target_emb =  torch.FloatTensor(target_emb)
    other_emb =  torch.FloatTensor(other_emb)
    rel_emb =  torch.FloatTensor(rel_emb)
    
    return target_emb,other_emb,rel_emb


def Load_allPts_deltatime_since1stM_data(analysis_ID,data_dir, duration_type):
    data_list = []
    
    for pt in analysis_ID:
        #file name
        file_name = pt + "_" + duration_type + ".csv"
        #Read file
        df = pd.read_csv(data_dir + file_name, index_col=0)

        #convert to numpy
        df = df.to_numpy()
        #convert to tensor and add a dmension
        df = torch.tensor(df,dtype= torch.float64).unsqueeze(0)
        
        #only collect data for 7 days
    
        data_list.append(df)
        
    return data_list

def Load_allPts_deltatime_sinceLastVisit(analysis_ID,data_dir, duration_type):
    data_list = []
    
    for pt in analysis_ID:
        #file name
        file_name = pt + "_" + duration_type + ".csv"
        #Read file
        df = pd.read_csv(data_dir + file_name, index_col=0)

        #convert to numpy
        df = df.to_numpy()
        #convert to tensor and add a dmension
        df = torch.tensor(df,dtype= torch.float64).unsqueeze(0)
        
        #only collect data for 7 days
    
        data_list.append(df)
        
    return data_list

def compute_nTimesteps(data_input):
    #first compute the max number of rows (timesteps)
    all_n_time_steps = []
    for df in data_input:
        n_timesteps = df.shape[1]
        all_n_time_steps.append(n_timesteps)
    return all_n_time_steps

def make_consist_timestep_deltaTdata(data_input,mean_timestep):
    new_data_input= []
    for df in data_input:
        nrows = df.shape[1]
        ncols = df.shape[2]
        last_row = df[0,nrows-1,:]
        if (nrows < mean_timestep): #if < ts, append with the last row
        
          extra_row = torch.Tensor(1, (mean_timestep - nrows), ncols)
          extra_row[:] = last_row
          new_df = torch.cat([df, extra_row],axis=1)

        else:
          new_df = df[:,:mean_timestep,:] #if > ts, only keep nrows
        
        new_data_input.append(new_df)
    return new_data_input

def make_consist_timestep_valueFdata(data_input,mean_timestep):
    
    new_data_input= []
    for df in data_input:
        nrows = df.shape[1]
        ncols = df.shape[2]
        if (nrows < mean_timestep): #if < ts, append np.nan
        
          extra_row = torch.Tensor(1, (mean_timestep - nrows), ncols)
          extra_row[:] = np.nan
          new_df = torch.cat([df, extra_row],axis=1)

        else:
          new_df = df[:,:mean_timestep,:] #if > ts, only keep nrows
        
        new_data_input.append(new_df)
    return new_data_input

#for 3d feature data
def scaler_func(in_data):
    scalers = {}
    for i in range(in_data.shape[2]): #for each feature
        scalers[i] = MinMaxScaler()    
        in_data[:, :, i] = scalers[i].fit_transform(in_data[:, :, i]) 
    return in_data

#for 3d delta data
def scaler_func_deltaT(in_data):
    scalers = {}
    for i in range(in_data.shape[0]): #for each pt
        scalers[i] = MinMaxScaler()   
        in_data[i,:,:] = scalers[i].fit_transform(in_data[i,:,:])            
    return in_data

#for 2D data
def scaler_func2(in_data):
    scaler = MinMaxScaler()    
    in_data = scaler.fit_transform(in_data)
    # scalers = {}
    # for i in range(in_data.shape[1]):
    #     scalers[i] = MinMaxScaler()    
    #     in_data[:, i] =  np.squeeze(scalers[i].fit_transform(in_data[:, i].reshape(-1,1)).reshape(-1,1)) 
    return in_data