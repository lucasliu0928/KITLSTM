library(Rtsne)
library("FactoMineR")
library("factoextra")
norm_minmax <- function(x){
  (x- min(x,na.rm = T)) /(max(x,na.rm = T)-min(x,na.rm = T))
}

################################################################################
#Data dir
################################################################################
proj_dir  <- "/Users/lucasliu/Desktop/DrChen_Projects/All_AKI_Projects/AKID_Project2/Intermediate_Data/practice_data_092721/onRRT_analysis/sample_level_data/"

#data dir
data_dir1  <- paste0(proj_dir, "AVG_MAX_MIN_Feature/")
data_dir2  <- paste0(proj_dir, "static_features/")

outdir <- paste0(proj_dir,"/Z_PCA_data/")

######################################################################################################## 
#Load labels
######################################################################################################## 
label_df <- read.csv(paste0(proj_dir,"FinalIDs_spOutcome.csv"),stringsAsFactors = F)

######################################################################################################## 
#1. Load all stationary feature data
######################################################################################################## 
avg_max_min_df <- read.csv(paste0(data_dir1,"All_AVGMAXMIN_onRRT_Imputed.csv"),stringsAsFactors = F)
static_df      <- read.csv(paste0(data_dir2,"samples_static_features_onRRT.csv"),stringsAsFactors = F)
#Inpute static with median
for (j in 3:ncol(static_df)){
  curr_col <- static_df[,j]
  na_indxes <- which(is.na(curr_col)==T)
  if (length(na_indxes) > 0){
    static_df[na_indxes,j] <- median(static_df[,j],na.rm = T)
  }

}

#MAtch ID order
match_order <- match(avg_max_min_df[,"SAMPLE_ID"],static_df[,"SAMPLE_ID"])

comb_df <- cbind(static_df[match_order,-1],avg_max_min_df[,-1])
rownames(comb_df) <- comb_df$SAMPLE_ID
comb_df <- comb_df[,-1]

match_order2 <- match(label_df[,"SAMPLE_ID"],rownames(comb_df))
comb_df_withLabel <- cbind(comb_df,label_df[,c("sample_death_in24h")])
colnames(comb_df_withLabel)[ncol(comb_df_withLabel)] <- "sample_death_in24h"

#########################################################
#Input df, remove duplicated rows
#########################################################
#remove duplicated rows for tSNE and PCA
comb_df_withLabel <- comb_df_withLabel[!duplicated(comb_df_withLabel[,-ncol(comb_df_withLabel)]),]
input_df <- comb_df_withLabel[,-ncol(comb_df_withLabel)]


#Run PCA
res.pca <- PCA(input_df, graph = FALSE)
eig.val <- get_eigenvalue(res.pca)
#write.csv(eig.val,paste0(outdir,"PCA_Eigenvalues.csv"))


#Perc of explained Variation
p <- fviz_eig(res.pca, ncp = 10, addlabels = TRUE, ylim = c(0, max(eig.val[,2] + eig.val[,2]/5)))
p <- p +  ggtitle("PCA Explained Variation") 

png(paste0(outdir,"PCA_Explained_Var.png"),res = 150,width = 1800,height = 1200)
print(p)
dev.off()

#Get varaible contribution
var <- get_pca_var(res.pca)
var_contribution <- as.data.frame(var$contrib)
write.csv(var_contribution,paste0(outdir,"PCA_Variable_Contribution.csv"))


#plot
p <- fviz_pca_ind(res.pca,
                  geom.ind = "point", # show points only (nbut not "text")
                  col.ind = as.factor(comb_df_withLabel$sample_death_in24h), # color by groups
                  palette = c("#00AFBB", "#E7B800"),
                  addEllipses = TRUE, # Concentration ellipses
                  legend.title = "Groups")

png(paste0(outdir,"PCA_2DPlot.png"),res = 150,width = 1800,height = 1200)
print(p)
dev.off()


####################################################################################################
# Run tsne
####################################################################################################
# set.seed(42)
# tsne_out <- Rtsne(input_df,pca=TRUE,perplexity=30) # Run TSNE
# 
# tsne_out_df <- data.frame(Y = tsne_out$Y,
#                           Class_label = comb_df_withLabel$sample_death_in24h,
#                           ID = rownames(input_df))
# 
# p <- ggplot(tsne_out_df, aes(x=Y.1, y=Y.2,color = Class_label)) +
#   geom_point(size = 5) +
#   theme_bw() + theme(panel.border = element_blank(), panel.grid.major = element_blank(),
#                      panel.grid.minor = element_blank(), axis.line = element_line(colour = "black")) +
#   theme(legend.position="none",legend.title = element_blank(),legend.text=element_text(size=20)) +
#   theme(axis.text = element_text(size = 20),axis.title=element_text(size=20,face="bold")) +
#   guides(color = guide_legend(nrow = 3, byrow = TRUE)) +
#   scale_x_continuous(name ="Tsne Dim1",limits = c(min(tsne_out_df$Y.1),max(tsne_out_df$Y.1))) +
#   scale_y_continuous(name ="Tsne Dim2",limits = c(min(tsne_out_df$Y.2),max(tsne_out_df$Y.2)))
# 
# png(paste0(outdir,"tSNE_2DPlot.png"),res = 150,width = 1800,height = 1200)
# print(p)
# dev.off()


####################################################################################################
#Add weighted sum scores for each sample, based on the contribution of top 7 features on Dim 1
####################################################################################################
top7_features_contribution_df  <- var_contribution[order(var_contribution[,"Dim.1"],decreasing = T)[1:7],]
top7_features                  <- rownames(top7_features_contribution_df)
top7_features_contributions    <- top7_features_contribution_df[,"Dim.1"]

#Normalized feature values for each sample
normed_data <- as.data.frame(lapply(comb_df_withLabel[,top7_features], norm_minmax))
colnames(normed_data) <- paste0("Normed_",colnames(normed_data))
comb_df_withLabel[,colnames(normed_data)] <- normed_data

#Compute weightes sum 
comb_df_withLabel[,"WeightedSumScore_Dim1Top7Fs"] <- NA
for (i in 1:nrow(comb_df_withLabel)){
  if (i %% 1000 == 0){print(i)}
  curr_pt_vals <- comb_df_withLabel[i,paste0("Normed_",top7_features)]
  
  curr_wss     <- sum(curr_pt_vals*top7_features_contributions) #weighted sum
  comb_df_withLabel[i,"WeightedSumScore_Dim1Top7Fs"] <- curr_wss
}

wss_df <- comb_df_withLabel[, c(top7_features, 
                               paste0("Normed_",top7_features),
                              "WeightedSumScore_Dim1Top7Fs",
                              "sample_death_in24h")]
#write.csv(wss_df,paste0(outdir,"WSS_Scores.csv"))

####################################################################################################
#Boxplot most contributed feature 
####################################################################################################
outcome_col <- "sample_death_in24h"
feature_col1 <- "MIN_Scr"
feature_col2 <- "AVG_Bicarbonate"
feature_col3 <- "WeightedSumScore_Dim1Top7Fs"

comb_df_withLabel[,outcome_col] <- as.factor(comb_df_withLabel[,outcome_col])


top_fs <- c(feature_col1,feature_col2,feature_col3)
for (i in 1:length(top_fs)){
  feature_col <- top_fs[i]
  p<-ggplot(comb_df_withLabel, aes_string(x=outcome_col, y=feature_col, color=outcome_col)) + 
    #geom_violin() +
    geom_boxplot() +
    stat_summary(fun=mean, geom="point", shape=23, size=2) +
    theme(axis.text=element_text(size=10),
          axis.title=element_text(size=10,face="bold"))+
    scale_color_manual(values=c("darkgreen", "darkred"))
  png(paste0(outdir,"Box_Plot/",feature_col,".png"),res = 150,width = 500,height = 500)
  print(p)
  dev.off()
}


#######################################################################################################
#5. Violin Plot
#######################################################################################################
for (i in 1:length(top_fs)){
  feature_col <- top_fs[i]
  p<-ggplot(comb_df_withLabel, aes_string(x=outcome_col, y=feature_col, color=outcome_col)) + 
    geom_violin() +
    #geom_boxplot() +
    stat_summary(fun=mean, geom="point", shape=23, size=2) +
    theme(axis.text=element_text(size=10),
          axis.title=element_text(size=10,face="bold"))+
    scale_color_manual(values=c("darkgreen", "darkred"))
  png(paste0(outdir,"Violin_Plot/",feature_col,".png"),res = 150,width = 500,height = 500)
  print(p)
  dev.off()
}


#######################################################################################################
#6. histogram Plot
#######################################################################################################
colnames(comb_df_withLabel)[which(colnames(comb_df_withLabel)=="sample_death_in24h")] <- "Death in Next 24h"
outcome_col <- "Death in Next 24h"
for (i in 1:length(top_fs)){
  if (i %% 10 == 0){print(i)}
  feature_col <- top_fs[i]
  p<-ggplot(comb_df_withLabel, aes_string(x=feature_col, color="`Death in Next 24h`")) +
    geom_histogram(fill="white",bins = 50) +
    theme(axis.text=element_text(size=20),
          axis.title=element_text(size=20,face="bold"))+
    scale_color_manual(values=c("darkgreen", "darkred"))+
    theme_bw() + theme(panel.border = element_blank(), panel.grid.major = element_blank(),
                         panel.grid.minor = element_blank(), axis.line = element_line(colour = "black")) + 
    theme(legend.position="top")+
    xlab("Weighted Sum of Top 7 Contributed Features") + 
    ylab("Count") +
    theme(axis.text = element_text(size = 14),
          axis.title = element_text(size = 14)) +
    geom_vline(xintercept = 20, linetype="dashed", 
                 color = "darkblue", size=1.5) + 
    annotate(geom = 'text', label = 'Exclusion', x = 20, y = 800, hjust = -0.1,color = "darkblue")
  p
  
  png(paste0(outdir,"Histogram/",feature_col,".png"),res = 150,width = 800,height = 500)
  print(p)
  dev.off()
}


####################################################################################################
##Get sample IDs of obvious negtives samples on WeightedSumScore_Dim1Top7Fs
####################################################################################################
wss_obv_neg_df <- comb_df_withLabel[which(comb_df_withLabel[,"WeightedSumScore_Dim1Top7Fs"] > 20),]  
table(wss_obv_neg_df$sample_death_in24h) #4411   152, ratio:29:1
#write.csv(wss_obv_neg_df,paste0(outdir,"OBV_NEG_SAMPLE_IDs_AndData.csv"))

####################################################################################################
#Check distribution of exclued samples feature values
####################################################################################################
summary(wss_obv_neg_df[,top7_features])

####################################################################################################
#Distribution of exclued samples
####################################################################################################
features <- top7_features
for (i in 1:length(features)){
  if (i %% 10 == 0){print(i)}
  feature_col <- features[i]
  p<-ggplot(comb_df_withLabel, aes_string(x=feature_col, color=outcome_col)) +
    geom_histogram(fill="white",bins = 50) +
    theme(axis.text=element_text(size=10),
          axis.title=element_text(size=10,face="bold"))+
    scale_color_manual(values=c("darkgreen", "darkred"))
  
  
  png(paste0(outdir,"Histogram/Excluded_Samples/",feature_col,".png"),res = 150,width = 800,height = 800)
  print(p)
  dev.off()
}

# ####################################################################################################
# ##Get sample IDs of obvious negtives samples on MIN_Scr and AVG_Bicarbonate
# ####################################################################################################
# min_scrGT3_df <- comb_df_withLabel[which(comb_df_withLabel[,"MIN_Scr"]>3),]  #when min_scr > 3,  NEG:4263,POS:186
# obv_neg_IDs1  <- rownames(min_scrGT3_df)[which(min_scrGT3_df[,"sample_death_in24h"]==0)]
# 
# avg_bicar_GT26_df <- comb_df_withLabel[which(comb_df_withLabel[,"AVG_Bicarbonate"]>26),]  ##when avg_bica > 26,  NEG:3988,POS:254
# obv_neg_IDs2  <- rownames(avg_bicar_GT26_df)[which(avg_bicar_GT26_df[,"sample_death_in24h"]==0)]
# 
# final_obv_neg_IDs_df <- data.frame(unique(c(obv_neg_IDs1,obv_neg_IDs2))) #7661
# colnames(final_obv_neg_IDs_df) <- "OBV_NEG_SampleIDs"
# 
# write.csv(final_obv_neg_IDs_df,paste0(outdir,"OBV_NEG_SAMPLE_IDs.csv"))
# 
# 
