#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 17 11:30:52 2021

@author: lucasliu
"""


#This script train KITLSTM model 
r'''
NOTE: Pytorch version 1.7.1 to  reproduce results
'''
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import os
import h5py

from torch.utils.data import DataLoader
import torch.optim as optim

from Model.KITLSTM import KITLSTM_M
from Utilities.Create_ModelReadyData_Funcs import model_data_for_5X, create_balanced_batch
from Utilities.Performance_Funcs import plot_LOSS
from Utilities.Training_Util import BCE_WithRegularization_EmbDist

##For GPU
device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
print(device)

    


#####################################################################################
#Data Dir
#####################################################################################
torch.manual_seed(1)
CURR_DIR = os.path.dirname(os.path.abspath("./"))

#Input dir
analysis_duration = "onRRT"
prediction_w      = "in24h"
proj_dir = CURR_DIR + "/Intermediate_Data/practice_data_092721/" + analysis_duration + "_analysis/"
data_dir = proj_dir + 'Model_Ready_data/NATAL_' + prediction_w + "/"
out_dir = CURR_DIR + "/output/120921/" + analysis_duration + "_" + prediction_w + "/KG_TIMEAWARE_LSTM3_V2/"




####################################################
#Load data
####################################################
#Train
hf_train = h5py.File(data_dir + 'train_data.h5', 'r')
train_X1  = np.array(hf_train.get('train_X1')) #value_feature
train_X2  = np.array(hf_train.get('train_X2')) #deltaT_feature
train_X3  = np.array(hf_train.get('train_X3')) #binary_feature
train_X4  = np.array(hf_train.get('train_X4')) #static_f_data
train_X6  = np.array(hf_train.get('train_X6')) #deltaT3_feature
train_y  =  np.array(hf_train.get('train_y'))


#Test
hf_test = h5py.File(data_dir + 'test_data.h5', 'r')
test_X1  = np.array(hf_test.get('test_X1'))
test_X2  = np.array(hf_test.get('test_X2'))
test_X3  = np.array(hf_test.get('test_X3'))
test_X4  = np.array(hf_test.get('test_X4'))
test_X6  = np.array(hf_test.get('test_X6'))
test_y  = np.array(hf_test.get('test_y'))
test_IDs_df =  pd.read_csv(data_dir + "test_Ids.csv")
test_IDs  = test_IDs_df['SAMPLE_ID']

#Valid
hf_valid = h5py.File(data_dir + 'valid_data.h5', 'r')
valid_X1  = np.array(hf_valid.get('valid_X1'))
valid_X2  = np.array(hf_valid.get('valid_X2'))
valid_X3  = np.array(hf_valid.get('valid_X3'))
valid_X4  = np.array(hf_valid.get('valid_X4'))
valid_X6  = np.array(hf_valid.get('valid_X6'))
valid_y  = np.array(hf_valid.get('valid_y'))


#ontology
hf_onto = h5py.File(data_dir + 'Ontology_Embeddings.h5', 'r')
other_emb  = torch.FloatTensor(hf_onto.get('other_emb'))
target_emb  = torch.FloatTensor(hf_onto.get('target_emb'))
rel_emb  = torch.FloatTensor(hf_onto.get('rel_emb'))

#Compute emb distance from each concept to target
emb_dist1 = (other_emb - target_emb)**2
emb_dist2 = torch.sqrt(torch.sum(emb_dist1,dim=0))
#norm softmax
m = nn.Softmax(dim=0)
emb_dist_normed = m(emb_dist2)
emb_dist_normed = emb_dist_normed.unsqueeze(dim = 0)

####################################################
#KG time aware LSTM model data
####################################################
train_data = model_data_for_5X(train_X1,train_X2,train_X3,train_X4,train_X6,train_y) #all train data
test_data  = model_data_for_5X(test_X1,test_X2,test_X3,test_X4,test_X6,test_y)  #all test data
valid_data  = model_data_for_5X(valid_X1,valid_X2,valid_X3,valid_X4,valid_X6,valid_y)  #all valid data

####################################################
#Train model
####################################################
drop_out_flag = True
drop_out_rate = 0.2
EPOCHS = 100
BATCH_SIZE = 258 
LEARNING_RATE = 0.01
N_FEATURE = 12
D_HIDDEN = 8
N_ONTOLOGY = 20
D_TransE = 200
reg_lambbda = 0.01
N_STATIC = 6
class_weight = [1,8] 


#Non balanced batch
train_loader = DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)


#Construct model
model = KITLSTM_M(N_FEATURE,D_HIDDEN,N_STATIC, N_ONTOLOGY,D_TransE,drop_out_rate)
model.to(device)

#Optimizer
optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.9)

#train:
train_loss = []
valid_loss = []
for epoch in range(EPOCHS):
    for x1,x2,x3,x4,x5,y in train_loader:
        #zero the parameter gradients
        optimizer.zero_grad()
        #forward 
        yhat,learned_dist,_ = model(x1,x2,x3,x4,x5,emb_dist_normed,other_emb,target_emb,rel_emb, drop_out_flag = drop_out_flag)
        loss = BCE_WithRegularization_EmbDist(yhat, y,learned_dist,emb_dist_normed, reg_lambbda, 'None',  model, class_weight) #loss with regularization
        #backward
        loss.backward()           
        #optimize
        optimizer.step()
        

    #compute loss after one epoch for train and test
    #For all Training data
    pred, learned_dist,_ = model(train_data.x1,train_data.x2,train_data.x3,train_data.x4,train_data.x5,emb_dist_normed, other_emb,target_emb,rel_emb,drop_out_flag= False)
    curr_trainloss = BCE_WithRegularization_EmbDist(pred, train_data.y,learned_dist,emb_dist_normed, reg_lambbda,'None' ,model, [1,1]) #loss with regularization

    train_loss.append(round(curr_trainloss.item(),6))
    
    #For all validation data
    with torch.no_grad():
        pred,learned_dist,_ = model(valid_data.x1,valid_data.x2,valid_data.x3,valid_data.x4,valid_data.x5,emb_dist_normed, other_emb,target_emb,rel_emb,drop_out_flag= False)
        curr_validloss = BCE_WithRegularization_EmbDist(pred, valid_data.y,learned_dist,emb_dist_normed,reg_lambbda,'None' ,model, [1,1]) #loss with regularization
        valid_loss.append(round(curr_validloss.item(),6))
    
    print("Epoch",epoch ,":","Train LOSS:", train_loss[epoch], "Validation LOSS:", valid_loss[epoch])
    #Save model parameters
    curr_model_name = "model" + str(epoch)
    torch.save(model.state_dict(), out_dir + "saved_model/" + curr_model_name)

#Plot LOSS
plot_LOSS(train_loss,valid_loss, out_dir)

#output losses
loss_df = pd.DataFrame({'EPOCH': list(range(EPOCHS)),'train_loss': train_loss, 'valid_loss': valid_loss})
loss_df.to_csv(out_dir + "losses.csv")

####################################################
#Testing model
####################################################
minloss_model_index = int(loss_df[loss_df['valid_loss'] == min(loss_df['valid_loss'])].iloc[0]['EPOCH']) #if multiple, choose the first one

#Instaitate model
minmodel = KITLSTM_M(N_FEATURE,D_HIDDEN,N_STATIC, N_ONTOLOGY,D_TransE,drop_out_rate)

#Load model with min loss
minmodel.load_state_dict(torch.load(out_dir + "/saved_model/" + "model" + str(minloss_model_index)))

#Final prediction for all test data
with torch.no_grad():
    pred_prob,_,atentions = minmodel(test_data.x1,test_data.x2,test_data.x3, test_data.x4,test_data.x5, emb_dist_normed, other_emb,target_emb,rel_emb, drop_out_flag= False)
    pred_prob_minModel = torch.flatten(pred_prob)
    pred_class_minModel = torch.flatten(torch.round(pred_prob_minModel))

pred_df = pd.DataFrame({"Test_IDs": test_IDs, 
                        "Y_True": torch.flatten(test_data.y).tolist(), 
                        "Pred_Prob_MinModel":  pred_prob_minModel.tolist(),
                        "Pred_Class_MinModel": pred_class_minModel.tolist()})
pred_df.to_csv(out_dir + "pred_df.csv")

##Output attention scores
atentions_df = atentions.numpy()
att_s = h5py.File(out_dir + 'att_scores.h5', 'w')
att_s.create_dataset('att_scores', data=atentions_df) #[N_ontology,ts,n_pts]
att_s.close()
