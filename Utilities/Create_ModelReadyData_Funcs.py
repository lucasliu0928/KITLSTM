#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 29 22:38:31 2021

@author: lucasliu
"""
import torch
from torch.utils.data import Dataset
import random 
import numpy as np

#For Vanilla LSTM
class model_data_for_1X(Dataset):
    def __init__(self, X1_data, y):
        self.x = X1_data
        self.y = y
        
        #convert to float
        self.x  = torch.FloatTensor(self.x)
        self.y = torch.FloatTensor(self.y)
        
        self.len = self.x.shape[0]
    def __getitem__(self, index):
        return self.x[index], self.y[index]
    
    def __len__(self):
        return self.len

#For KG-TimeAware LSTM
#two X
class model_data_for_2X(Dataset):
    def __init__(self, X1_data, X2_data, y):
        self.x1 = X1_data
        self.x2 = X2_data
        self.y = y
        
        #convert to float
        self.x1  = torch.FloatTensor(self.x1)
        self.x2  = torch.FloatTensor(self.x2)
        self.y = torch.FloatTensor(self.y)
        
        self.len = self.x1.shape[0]
        
    def __getitem__(self, index):
        return self.x1[index], self.x2[index], self.y[index]
    
    def __len__(self):
        return self.len
        
#three X
class model_data_for_3X(Dataset):
    def __init__(self, X1_data, X2_data, X3_data, y):
        self.x1 = X1_data
        self.x2 = X2_data
        self.x3 = X3_data
        self.y = y
        
        #convert to float
        self.x1  = torch.FloatTensor(self.x1)
        self.x2  = torch.FloatTensor(self.x2)
        self.x3  = torch.FloatTensor(self.x3)
        self.y = torch.FloatTensor(self.y)
        
        self.len = self.x1.shape[0]
        
    def __getitem__(self, index):
        return self.x1[index], self.x2[index],self.x3[index], self.y[index]
    
    def __len__(self):
        return self.len
    
#four X
class model_data_for_4X(Dataset):
    def __init__(self, X1_data, X2_data, X3_data, X4_data, y):
        self.x1 = X1_data
        self.x2 = X2_data
        self.x3 = X3_data
        self.x4 = X4_data
        self.y = y
        
        #convert to float
        self.x1  = torch.FloatTensor(self.x1)
        self.x2  = torch.FloatTensor(self.x2)
        self.x3  = torch.FloatTensor(self.x3)
        self.x4  = torch.FloatTensor(self.x4)
        self.y = torch.FloatTensor(self.y)
        
        self.len = self.x1.shape[0]
        
    def __getitem__(self, index):
        return self.x1[index], self.x2[index],self.x3[index], self.x4[index] ,self.y[index]
    
    def __len__(self):
        return self.len
    
    
#Five X
class model_data_for_5X(Dataset):
    def __init__(self, X1_data, X2_data, X3_data, X4_data, X5_data, y):
        self.x1 = X1_data
        self.x2 = X2_data
        self.x3 = X3_data
        self.x4 = X4_data
        self.x5 = X5_data
        self.y = y
        
        #convert to float
        self.x1  = torch.FloatTensor(self.x1)
        self.x2  = torch.FloatTensor(self.x2)
        self.x3  = torch.FloatTensor(self.x3)
        self.x4  = torch.FloatTensor(self.x4)
        self.x5  = torch.FloatTensor(self.x5)
        self.y = torch.FloatTensor(self.y)
        
        self.len = self.x1.shape[0]
        
    def __getitem__(self, index):
        return self.x1[index], self.x2[index],self.x3[index], self.x4[index] , self.x5[index], self.y[index]
    
    def __len__(self):
        return self.len
    
    
def create_balanced_batch(X1, X4, y, num_X, batch_size, X2 = None,X3 = None,X5 = None):
    #this function generate balacned batch samples
    # X2 and X3 is optional 
    #for each batch, sample batch_size/2 pos samples from all pos samples
    #                sample batch_size/2 neg samples from all neg samples left, every time remove the already select neg samples.
    # Num of batches = num of all neg samples / 32
    # batch_size = 32 
    # X1 = valid_X1
    # X2 = valid_X2
    # X3 = valid_X3
    # X4 = valid_X4
    # y  = valid_y
    # num_X = 4
    
    all_1_idxes = list(np.where(y == 1)[0]) #379
    all_0_idxes = list(np.where(y == 0)[0]) #4121
    
    batch_list = []
    ct = 0
    while all_0_idxes: #if not empty
        random.seed(ct) #use ct make sure random sample different pos
        ct += 1
        #random sample from pos samples without replacement
        sp_1_idxes = random.sample(all_1_idxes, k = int(batch_size/2)) 

        #random sample from neg samples without replacement
        if len(all_0_idxes) > int(batch_size/2):
            sp_0_idxes = random.sample(all_0_idxes, k = int(batch_size/2))
        else: 
            sp_0_idxes = random.sample(all_0_idxes, k = len(all_0_idxes))
            
        #Current batch indexes
        curr_sp_idxes = sp_1_idxes + sp_0_idxes
        
        #udpated xs and y
        curr_y  = y[curr_sp_idxes,:]
        curr_x1 = X1[curr_sp_idxes,:,:]
        curr_x4 = X4[curr_sp_idxes,:]

        if num_X == 5: ##value f, delta t, binary, static, delta t2
            curr_x2 = X2[curr_sp_idxes,:,:]
            curr_x3 = X3[curr_sp_idxes,:,:] 
            curr_x5 = X5[curr_sp_idxes,:,:] 
            curr_batch = model_data_for_5X(curr_x1,curr_x2,curr_x3,curr_x4,curr_x5,curr_y)        
        if num_X == 4: ##value f, delta t, binary, delta t2
            curr_x2 = X2[curr_sp_idxes,:,:]
            curr_x5 = X5[curr_sp_idxes,:,:] 
            curr_batch = model_data_for_4X(curr_x1,curr_x2,curr_x4,curr_x5,curr_y)
        elif num_X == 3: #value f, delta t, static
            curr_x2 = X2[curr_sp_idxes,:,:] 
            curr_batch = model_data_for_3X(curr_x1,curr_x2,curr_x4,curr_y)
        elif num_X == 2: #value f , static
            curr_batch = model_data_for_2X(curr_x1,curr_x4,curr_y)
            
        batch_list.append(curr_batch)

        #update X1, X2, X3, x4 and y by exclude the index of curr neg samples
        y  = np.delete(y, sp_0_idxes, axis = 0)
        X1 = np.delete(X1, sp_0_idxes, axis = 0)
        X4 = np.delete(X4, sp_0_idxes, axis = 0)
  
        if num_X == 5: 
            X2 = np.delete(X2, sp_0_idxes, axis = 0)
            X3 = np.delete(X3, sp_0_idxes, axis = 0)
            X5 = np.delete(X5, sp_0_idxes, axis = 0)
        if num_X == 4: 
            X2 = np.delete(X2, sp_0_idxes, axis = 0)
            X5 = np.delete(X5, sp_0_idxes, axis = 0)
        elif num_X == 3: 
            X2 = np.delete(X2, sp_0_idxes, axis = 0)
        elif num_X == 2: 
            pass     
        
        #update 1 and 0 index
        all_1_idxes = list(np.where(y == 1)[0]) #379
        all_0_idxes = list(np.where(y == 0)[0]) #4089,4057

    return batch_list


 
