#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 17 22:04:14 2022

@author: lucasliu
@info: the script defines subpopulation on training, validation and testing data using clustering
       using patient level data
"""

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from  scipy.cluster.hierarchy import dendrogram,linkage
from scipy.cluster import hierarchy
from sklearn.manifold import TSNE
import pylab
# set the font globally
plt.rcParams.update({'font.family':'sans-serif'})
# set the font name for a font family
plt.rcParams.update({'font.sans-serif':'Arial'}) #Arial
plt.rcParams['font.sans-serif']

def dengengrams_atLevels (linkage_z,line_threshold, level_n,out_dir):
    #Control number of clusters in the plot + add horizontal line.
    plt.figure(figsize=(10, 7))
    plt.title("Subpopulation Dendograms")
    deng = dendrogram(linkage_z, color_threshold= line_threshold,above_threshold_color='steelblue',
                      truncate_mode = "level", p = level_n)
    ax = plt.gca()
    plt.axhline(y= line_threshold, c='darkred', lw=2, linestyle='dashed')
    #ax.set_xticks([])  #hide x labels
    plt.savefig(out_dir + 'dend_L'+ str(level_n) + '.png',dpi = 300)
    
#####################################################################################
#Dir
#####################################################################################
analysis_duration = "onRRT"
outcome_col = "sample_death_in24h"
proj_dir    = "/Users/lucasliu/Desktop/DrChen_Projects/All_AKI_Projects/AKID_Project2/Intermediate_Data/practice_data_092721/"
indir1      = proj_dir + analysis_duration + '_analysis/sample_level_data/'
indir2      = proj_dir + analysis_duration + '_analysis/patient_level_data/'
outdir      = indir2 + 'Clustered_Subpopulation/'


#####################################################################################
#Load training, validation and testing IDs
#####################################################################################
test_Id_df = pd.read_csv(indir1 + "Test_SampleIDs_afterExc.csv")
test_Id_df['Cohort'] = 'T'
valid_Id_df = pd.read_csv(indir1 + "Validation_SampleIDs_afterExc.csv")
valid_Id_df['Cohort'] = 'S'
train_Id_df = pd.read_csv(indir1 + "Train_SampleIDs_afterExc.csv")
train_Id_df['Cohort'] = 'S'
#Combine all IDs in to one df
all_Id_df  = pd.concat([test_Id_df,valid_Id_df,train_Id_df])
#add pt ID
all_Id_df['ENCNTR_ID'] = all_Id_df['SAMPLE_ID'].str.split("_", expand = True)[0]
#only keep the unique pt IDs df
unique_pt_Id_df = all_Id_df.drop_duplicates(subset=['ENCNTR_ID'])
unique_pt_Id_df = unique_pt_Id_df.drop(columns=['SAMPLE_ID'])

#####################################################################################
#Load Patient level demo info
#####################################################################################
demo_df  = pd.read_csv(indir2 + "demo_finalID_"+ analysis_duration + ".csv")

#####################################################################################
#Load Patient level diagnosis info
#####################################################################################
diagnosis_df = pd.read_csv(indir2 + "diagnosis_finalID_" + analysis_duration + ".csv")

#####################################################################################
#Load Patient level outcome (hospital death) info
#####################################################################################
outcome_df = pd.read_csv(indir2 + "outcome_finalID_" + analysis_duration + ".csv")

#####################################################################################
#Combine demo, diagnosis and outcome
#####################################################################################
comb_df1 = demo_df.merge(diagnosis_df, how = 'inner', on ='ENCNTR_ID')
comb_df2  = comb_df1.merge(outcome_df, how = 'inner', on ='ENCNTR_ID')

#####################################################################################
#Only keep comb df for final Ids
#####################################################################################
comb_df  = unique_pt_Id_df.merge(comb_df2, how = 'inner', on ='ENCNTR_ID')

#####################################################################################
##Preprocess
#####################################################################################
selected_fearues = ["AGE","GENDR_CD_DES","RACE_CD_DES","WT_KG","BMI","CHARLSON_INDEX","DB_BeforeOrAt","HT_BeforeOrAt"]
comb_df_selected_fs = comb_df.copy()
comb_df_selected_fs = comb_df_selected_fs[selected_fearues]

#Mean imputation
for f in selected_fearues:
    mean_v = comb_df_selected_fs[f].mean()
    comb_df_selected_fs[f].fillna(value=mean_v, inplace=True)
    
#Min-Max norm
for f in selected_fearues:
    max_v = comb_df_selected_fs[f].max()
    min_v = comb_df_selected_fs[f].min()
    normed_v = (comb_df_selected_fs[f] - min_v) / (max_v-min_v)
    comb_df_selected_fs[f]= normed_v


#####################################################################################
#Cluster dengengrams
#####################################################################################
#Compute distances
Z = linkage(comb_df_selected_fs, 'ward')

#Get Dendrogram for all levels
plt.figure(figsize=(10, 7))
plt.title("")

hierarchy.set_link_color_palette(['steelblue']) # Set the colour of the cluster
deng = dendrogram(Z, #truncate_mode = "level",p = 5,
                  color_threshold = 20,
                  labels= list(comb_df["Cohort"]))
ax = plt.gca()
ax.set_xticks([])  #hide x labels
hierarchy.set_link_color_palette(None)  # reset to default after use
plt.tick_params(axis='y', labelsize=20)
plt.ylabel('Height', fontsize=20)

#label_colors = {'T': 'steelblue', 'S': 'seagreen'} # Assignment of colors to labels (Source or Target)
# xlbls = ax.get_xmajorticklabels()
# for lbl in xlbls:
#     lbl.set_color(label_colors[lbl.get_text()])
plt.savefig(outdir + 'dend.png',dpi = 300,bbox_inches='tight')

#####################################################################################
#Clutersring at differnet level 
#####################################################################################
Z = linkage(comb_df_selected_fs, 'ward')

#Clustering
cluster_n = 2
level_n   = 1
ths       = 12.5
cluster_level1 = AgglomerativeClustering(n_clusters=cluster_n, affinity='euclidean', linkage='ward')
cluster_labels_level1 = cluster_level1.fit_predict(comb_df_selected_fs)
dengengrams_atLevels(Z,ths,level_n,outdir)



cluster_n = 4
level_n   = 2
ths       = 10
cluster_level2 = AgglomerativeClustering(n_clusters=cluster_n, affinity='euclidean', linkage='ward')
cluster_labels_level2 = cluster_level2.fit_predict(comb_df_selected_fs)
dengengrams_atLevels(Z,ths,level_n,outdir)


cluster_n = 8
level_n   = 3
ths       = 5
cluster_level3 = AgglomerativeClustering(n_clusters=cluster_n, affinity='euclidean', linkage='ward')
cluster_labels_level3 = cluster_level3.fit_predict(comb_df_selected_fs)
dengengrams_atLevels(Z,ths,level_n,outdir)


cluster_n = 16
level_n   = 4
ths       = 2.2
cluster_level4 = AgglomerativeClustering(n_clusters=cluster_n, affinity='euclidean', linkage='ward')
cluster_labels_level4 = cluster_level4.fit_predict(comb_df_selected_fs)
dengengrams_atLevels(Z,ths,level_n,outdir)

cluster_n = 32
level_n   = 5
ths       = 1
cluster_level5 = AgglomerativeClustering(n_clusters=cluster_n, affinity='euclidean', linkage='ward')
cluster_labels_level5 = cluster_level5.fit_predict(comb_df_selected_fs)
dengengrams_atLevels(Z,ths,level_n,outdir)

#####################################################################################
#TSNE 3D
#####################################################################################
tsne_results = TSNE(n_components=3, random_state=0).fit_transform(comb_df_selected_fs)


#####################################################################################
#Combined results
#####################################################################################
results_df = pd.DataFrame({"ENCNTR_ID": comb_df["ENCNTR_ID"], 
                         "Cluster_L1": cluster_labels_level1,
                         "Cluster_L2": cluster_labels_level2,
                         "Cluster_L3": cluster_labels_level3,
                         "Cluster_L4": cluster_labels_level4,
                         "Cluster_L5": cluster_labels_level5,
                         "tSNE0": tsne_results[:,0],
                         "tSNE1": tsne_results[:,1],
                         "tSNE2": tsne_results[:,2]})
results_df = results_df.merge(comb_df, on = "ENCNTR_ID")
results_df.to_csv(outdir + "Clustered_AllPTs.csv")


#####################################################################################
#REport stats
#####################################################################################
L1_res_df1 = results_df[results_df['Cluster_L1'] == 0]
L1_res_df1['Cohort'].value_counts()
L1_res_df2 = results_df[results_df['Cluster_L1'] == 1]
L1_res_df2['Cohort'].value_counts()

results_df['Cluster_L1'].value_counts()
results_df['Cluster_L2'].value_counts()
results_df['Cluster_L3'].value_counts()
results_df['Cluster_L4'].value_counts()
results_df['Cluster_L5'].value_counts()

#####################################################################################
#Plot 3d
#####################################################################################
x = results_df['tSNE0']
y = results_df['tSNE1']
z = results_df['tSNE2']
#colors = {'Train':'red', 'Validation':'green', 'Test':'blue'}
#c = results_df['Cohort'].map(colors)

levels =[1,2,3,4,5]
for l in levels:
    c = results_df['Cluster_L' + str(l)]
    
    fig = pylab.figure(figsize=(6,5))
    ax = fig.add_subplot(111, projection = '3d')
    sc = ax.scatter(x,y,z, c= c)
    legend1 = ax.legend(*sc.legend_elements(),
                        loc="upper center", title="Subpopulation",
                        frameon = True, shadow = False,
                        facecolor = "white",
                        ncol=4,
                        fontsize = 'large',
                        title_fontsize = 'large')
    ax.add_artist(legend1)
    ax.set_xlabel('t-SNE-0', fontsize=12)
    ax.set_ylabel('t-SNE-1', fontsize=12)
    ax.set_zlabel('t-SNE-2', fontsize=12)
    # ax.set_xticks([])
    # ax.set_yticks([])
    # ax.set_zticks([])

    
    plt.savefig(outdir + 'tSNE_L'+ str(l) +'.png',dpi = 300,bbox_inches='tight')

