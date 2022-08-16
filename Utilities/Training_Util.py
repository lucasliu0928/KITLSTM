#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  7 16:15:49 2021

@author: lucasliu
"""


import torch 


def BCE_WithRegularization(output, target, lambda_coef, reg_type, model, class_weight,reduction = 'mean'):
    
    #Compute loss for each sample
    loss = - ( target * torch.log(output) + (1-target)*torch.log(1-output))
    
    #Weight loss for each class
    pos_idex = torch.where(target == 1)[0] #index of pos
    neg_idex = torch.where(target == 0)[0] #index of neg
    
    loss[neg_idex] =  loss[neg_idex]*class_weight[0] #neg class weight * correponding loss
    loss[pos_idex] =  loss[pos_idex]*class_weight[1]

    
    if reduction == 'mean':
        loss = loss.mean()
    #Regularization
    l1_regularization = 0
    l2_regularization = 0
    for param in model.parameters():
        l1_regularization += param.abs().sum()
        l2_regularization += param.square().sum()
    if reg_type == "L1":
        loss = loss + lambda_coef*l1_regularization     
    elif reg_type == "L2":
        loss = loss + lambda_coef*l2_regularization
    else:
        loss = loss
       
    return loss

#Regularization with concept embedding dist
def BCE_WithRegularization_EmbDist(output, target, l_dist, actual_dist, lambda_coef, reg_type, model, class_weight,reduction = 'mean'):
    
    #Embed distant regualrization term
    actual_dist = torch.transpose(actual_dist,0,1)
    emb_reg = torch.sum((l_dist - actual_dist)**2)
    #print(l_dist)
    #print(emb_reg)
    #Compute loss for each sample
    loss = - ( target * torch.log(output) + (1-target)*torch.log(1-output))
    
    #Weight loss for each class
    pos_idex = torch.where(target == 1)[0] #index of pos
    neg_idex = torch.where(target == 0)[0] #index of neg
    
    loss[neg_idex] =  loss[neg_idex]*class_weight[0] #neg class weight * correponding loss
    loss[pos_idex] =  loss[pos_idex]*class_weight[1]

    
    if reduction == 'mean':
        loss = loss.mean()
    #Regularization
    l1_regularization = 0
    l2_regularization = 0
    for param in model.parameters():
        l1_regularization += param.abs().sum()
        l2_regularization += param.square().sum()
    if reg_type == "L1":
        loss = loss + lambda_coef*l1_regularization + emb_reg     
    elif reg_type == "L2":
        loss = loss + lambda_coef*l2_regularization + emb_reg
    else:
        loss = loss + emb_reg
       
    return loss

#Regularization with concept embed + rel embed supposed to close to target embed
def BCE_WithRegularization_EmbDist2(output, target, l_dist, lambda_coef, reg_type, model, class_weight,reduction = 'mean'):
    
    #Embed distant regualrization term
    emb_reg = torch.sum(l_dist)


    #Compute loss for each sample
    loss = - ( target * torch.log(output) + (1-target)*torch.log(1-output))
    
    #Weight loss for each class
    pos_idex = torch.where(target == 1)[0] #index of pos
    neg_idex = torch.where(target == 0)[0] #index of neg
    
    loss[neg_idex] =  loss[neg_idex]*class_weight[0] #neg class weight * correponding loss
    loss[pos_idex] =  loss[pos_idex]*class_weight[1]

    
    if reduction == 'mean':
        loss = loss.mean()
    #Regularization
    l1_regularization = 0
    l2_regularization = 0
    for param in model.parameters():
        l1_regularization += param.abs().sum()
        l2_regularization += param.square().sum()
    if reg_type == "L1":
        loss = loss + lambda_coef*l1_regularization + emb_reg     
    elif reg_type == "L2":
        loss = loss + lambda_coef*l2_regularization + emb_reg
    else:
        loss = loss + emb_reg
       
    return loss


from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

def removeTrain_bytSNE(in_data,in_label, plot_flg = False):
    #in_data = train_X1
    #in_label = train_y

    
    n_feature = in_data.shape[2]
    
    indexes_toremove_list = []
    #indexes_tokeep_list = []
    for i in range(n_feature):
        data= pd.DataFrame(in_data[:,:,i])
        tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
        tsne_results = tsne.fit_transform(data)
        
        data['y'] = in_label
        data['tsne-2d-one'] = tsne_results[:,0]
        data['tsne-2d-two'] = tsne_results[:,1]
        
        if (plot_flg == True):
            plt.figure(figsize=(8,8))
            sns.scatterplot(
                x="tsne-2d-one", y="tsne-2d-two",
                hue="y",
                style = "y",
                #palette=sns.color_palette("hls", 2),
                data=data,
                legend="full",
                alpha=0.5
            )
        
            plt.hist(data['tsne-2d-one'])
            plt.hist(data['tsne-2d-two'])
        
        th1 = np.quantile(data['tsne-2d-one'],q = [0.20,0.80])
        th2 = np.quantile(data['tsne-2d-two'],q = [0.20,0.80])
        
        #indexes to remove (obv negtives)
        toremove_data =  data[(data['tsne-2d-one'] < th1[0]) | 
                              (data['tsne-2d-one'] > th1[1]) | 
                              (data['tsne-2d-two'] < th2[0]) | 
                              (data['tsne-2d-two'] > th2[1])]
        toremove_data = toremove_data[toremove_data['y'] == 0] #negtives

        tomoreve_indexes = set(list(toremove_data.index))
        indexes_toremove_list.append(tomoreve_indexes)
        #left_data = np.delete(in_data,list(tomoreve_indexes),0)
        
        #data left
        # left_data = data[(data['tsne-2d-one'] >= th1[0]) & 
        #                  (data['tsne-2d-one'] <= th1[1]) &
        #                  (data['tsne-2d-two'] >= th2[0]) &
        #                  (data['tsne-2d-two'] <= th2[1])]
        # left_indxes = set(list(left_data.index))
        # indexes_tokeep_list.append(left_indxes)


        # plt.figure(figsize=(8,8))
        # sns.scatterplot(
        #     x="tsne-2d-one", y="tsne-2d-two",
        #     hue="y",
        #     style = "y",
        #     #palette=sns.color_palette("hls", 2),
        #     data=left_data,
        #     legend="full",
        #     alpha=0.5
        # )
        

    #union of indexes to keep from different features
    #final_indxes_tokeep = list(set.union(*indexes_tokeep_list))

    #intersection of indexs to remove from different features
    final_indexes_toremove = list(set.intersection(*indexes_toremove_list))
    return final_indexes_toremove