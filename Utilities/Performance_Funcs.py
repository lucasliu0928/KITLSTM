#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 29 23:49:28 2021

@author: lucasliu
"""

import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix,fbeta_score
from sklearn import metrics
import pandas as pd
import numpy as np

def plot_LOSS (train_loss, test_loss, outdir):
    plt.figure(figsize=(10,5))
    plt.title("Training and Validation Loss")
    plt.plot(test_loss,label="Validation")
    plt.plot(train_loss,label="Train")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(outdir + 'LOSS.png')
    
    
def compute_performance(y_true,y_pred_prob,y_pred_class,cohort_name):
    confusion_matrix(y_true, y_pred_class) #CM
    fpr, tpr, thresholds = metrics.roc_curve(y_true, y_pred_prob, pos_label=1)
    
    AUC = round(metrics.auc(fpr, tpr),2)
    ACC = round(accuracy_score(y_true, y_pred_class),2)
    F1 = round(f1_score(y_true, y_pred_class),2)
    F2 = round(fbeta_score(y_true, y_pred_class,beta = 2),2)
    F3 = round(fbeta_score(y_true, y_pred_class,beta = 3),2)
    Recall = round(recall_score(y_true, y_pred_class),2)
    Precision = round(precision_score(y_true, y_pred_class),2)
    perf_tb = pd.DataFrame({"AUC": AUC, 
                            "ACC": ACC,
                            "F1": F1,
                            "F2": F2,
                            "F3": F3,
                            "Recall": Recall,
                            "Precision":Precision},index = [cohort_name])
    
    return perf_tb


#Performance
def compute_formance_for_listofsubset(subset_pred_list, subset_names,pred_prob_col, pred_class_col):
    
    perf_tb_list = []
    for i in range(len(subset_pred_list)):
        sb_pred_df = subset_pred_list[i]
        
        if (sb_pred_df.shape[0] > 0):
            y_pred_prob = sb_pred_df[pred_prob_col]
            y_pred_class= sb_pred_df[pred_class_col]
            y_true = sb_pred_df['Y_True']
            curr_perf = compute_performance(y_true,y_pred_prob,y_pred_class,subset_names[i])
            
            
            #If current AUC is NAN, meaning eaither only 1s or 0s, 
            #recode other perforamnce as NAN, so the subpopulation will be excluded
            curr_auc = curr_perf['AUC'].item()
            
            if ( pd.isna(curr_auc) == True):
                curr_perf[['AUC', 'ACC', 'F1','F2', 'F3','Recall', 'Precision']] = np.nan
                
            #total number of samples and patients
            n_samples = sb_pred_df.shape[0] #samples
            n_pts   =   len(set(sb_pred_df['ENCNTR_ID']))     #patients
                    
            
            #number of actual POS samples and pateints 
            ytrue1_df = sb_pred_df[sb_pred_df['Y_True'] == 1]
            if (ytrue1_df.shape[0] != 0):
                n_ytrue1_sp  = ytrue1_df.shape[0]               #samples
                n_ytrue1_pt  = len(set(ytrue1_df['ENCNTR_ID'])) #patients
            else: 
                n_ytrue1_sp  = 0            
                n_ytrue1_pt  = 0
                
    
            #number of actual NEG samples and pateints 
            ytrue0_df = sb_pred_df[sb_pred_df['Y_True'] == 0]
            if (ytrue0_df.shape[0] != 0):
                n_ytrue0_sp  = ytrue0_df.shape[0]               #samples
                n_ytrue0_pt  = len(set(ytrue0_df['ENCNTR_ID'])) #patients
            else: 
                n_ytrue0_sp  = 0              
                n_ytrue0_pt  = 0
            
            #ratio NEG:POS samples and pateints
            if (n_ytrue1_sp == 0) :
                ratio_sp = "All 0s"
                ratio_pt = "All 0s"
            elif (n_ytrue0_sp == 0):
                ratio_sp = "All 1s"
                ratio_pt = "All 1s"
            else:
                ratio_sp = round(n_ytrue0_sp/n_ytrue1_sp,1)
                ratio_pt = round(n_ytrue0_pt/n_ytrue1_pt,1)
    
            #Add to perf table
            #combine numbers samples and pts
            curr_perf['N_Samples(Patient)']  = str(n_samples) + " (" + str(n_pts) + ")"
            curr_perf['N_POS_Samples(Patient)']  = str(n_ytrue1_sp) + " (" + str(n_ytrue1_pt) + ")"
            curr_perf['N_NEG_Samples(Patient)']  = str(n_ytrue0_sp) + " (" + str(n_ytrue0_pt) + ")"
            curr_perf['Ratio_NEGtoPOS_Samples(Patient)']  = str(ratio_sp) + " (" + str(ratio_pt) + ")"
        else:
            curr_perf = pd.DataFrame({'AUC': [np.nan], 
                                      'ACC': [np.nan], 
                                      'F1': [np.nan],
                                      'F2': [np.nan],
                                      'F3': [np.nan],
                                      'Recall': [np.nan],
                                      'Precision' : [np.nan],
                                      'N_Samples(Patient)': "0 (0)",
                                      'N_POS_Samples(Patient)': "0 (0)",
                                      'N_NEG_Samples(Patient)': "0 (0)", 
                                      'Ratio_NEGtoPOS_Samples(Patient)': [np.nan]},
                                       index = [subset_names[i]] )


        
        perf_tb_list.append(curr_perf)
    
    #combine all performance
    all_perf_tb = pd.concat(perf_tb_list)
    
    #Compute mean and std for performanc across different subset
    avg_perf = round(pd.DataFrame(all_perf_tb.mean()).transpose(),2) #NA skiped
    avg_perf.index = ["AVG"]
    std_perf = round(pd.DataFrame(all_perf_tb.std()).transpose(),2) #NA skiped
    std_perf.index = ["STD"]

    Final_all_perf_tb = pd.concat([all_perf_tb,avg_perf,std_perf], axis= 0) 
   
    #store AVG and STD in one entry
    AVG_STD = []
    for i in range(Final_all_perf_tb.shape[1]):
        curr_avg = Final_all_perf_tb.loc['AVG'][i]
        curr_std = Final_all_perf_tb.loc['STD'][i]
        AVG_STD.append(str(curr_avg) + '(' + str(curr_std) + ')')

    Final_all_perf_tb.loc[Final_all_perf_tb.shape[0]] = AVG_STD
    Final_all_perf_tb = Final_all_perf_tb.rename(index={Final_all_perf_tb.shape[0] -1: 'AVG_STD'})

    return Final_all_perf_tb