"""
 This script computes the correlation between the real AUDRC and the predicted AUDRC for each drug on an individual basis.
 In the case of multiple models due to k-fold cross-validation, an average correlation is derived.
 It also computes the density plot of all models and its metrics. 
"""

import argparse
import sys
import torch
import torch.nn as nn
import torch.utils.data as du
import pandas as pd
import pickle
from matplotlib import pyplot as plt
import statistics
import wandb
import os 

parser = argparse.ArgumentParser(description='SparseGO metrics')

parser.add_argument('-input_folder', help='Directory containing the input data folders.', type=str, default="../data/toy_example_DrugCell/")
parser.add_argument('-output_folder', help='Directory containing the folders that have the resulting models', type=str, default="../results/toy_example_DrugCell/")
parser.add_argument('-samples_folders', help='Folders to analyze', type=str, default=["samples1"]) # CHANGE THIS HERE
parser.add_argument('-txt_type', help='If txts of data have a particular extra name', type=str, default="") # CHANGE THIS HERE
parser.add_argument('-model_name', help='Model to use to compute individual drug predictions', type=str, default="best_model_p.pt")
parser.add_argument('-predictions_name', help='Which results to use for the density plot', type=str, default="ModelPearson_test_predictions.txt")
parser.add_argument('-labels_name', help='Which results to use for the density plot', type=str, default="drugcell_test.txt")
parser.add_argument('-ontology_name', help='Which results to use for the density plot', type=str, default="drugcell_ont.txt")
parser.add_argument('-genomics_name', help='Which results to use for the density plot', type=str, default="cell2mutation.txt")
parser.add_argument('-cuda', help='Cuda ID', type=str, default=0)
parser.add_argument('-project', help='W&B project name', type=str, default="Test")

opt = parser.parse_args()

sys.path.append("./util.py")

import util
from util import *

def predict(predict_data, model, batch_size, cell_features, drug_features, CUDA_ID):

    predict_feature, predict_label = predict_data

    predict_label_gpu = predict_label.to(device, non_blocking=True).detach()

    model.eval()

    test_loader = du.DataLoader(du.TensorDataset(predict_feature,predict_label), batch_size=batch_size, shuffle=False)

    #Test
    test_predict = torch.zeros(0, 1, device=device)
    with torch.no_grad():
        for i, (inputdata, labels) in enumerate(test_loader):
            # Convert torch tensor to Variable
            features = build_input_vector(inputdata, cell_features, drug_features)
            features = features.to(device)

            # make prediction for test data
            out = model(features)

            test_predict = torch.cat([test_predict, out])

    test_corr = pearson_corr(test_predict, predict_label_gpu)
    test_corr_spearman = spearman_corr(test_predict.cpu().detach().numpy(), predict_label_gpu.cpu())

    return test_corr, test_corr_spearman

def load_select_data(file_name, cell2id, drug2id,selected_drug): # only select samples of chosen drug 
    feature = []
    label = []

    with open(file_name, 'r') as fi: # ,encoding='UTF-16'
        for line in fi:
            tokens = line.strip().split('\t')
            if tokens[1] == selected_drug:
                feature.append([cell2id[tokens[0]], drug2id[tokens[1]]])
                label.append([float(tokens[2])])
    return feature, label

def get_compound_names(file_name):
    compounds = []

    with open(file_name, 'r') as fi:
        for line in fi:
            tokens = line.strip().split('\t')
            compounds.append([tokens[1],tokens[2]])

    return compounds

# Resulting figures are uploaded to w&b
wandb.init(project=opt.project, entity="katynasada", name="final metrics")

# Calculate both Pearson and Spearman correlation coefficients for each drug across the different models
for folder in opt.samples_folders:
    inputdir=opt.input_folder+folder+"/" # CHANGE
    resultsdir=opt.output_folder+folder+"/" # CHANGE
    
    txt_type=opt.txt_type
    
    drug2fingerprint=inputdir+"drug2fingerprint"+txt_type+".txt"
    test=inputdir+opt.labels_name
    drug2id=inputdir+"drug2ind"+txt_type+".txt"
    genotype=inputdir+opt.genomics_name
    onto=inputdir+opt.ontology_name
    cell2id=inputdir+"cell2ind"+txt_type+".txt"
    gene2id=inputdir+"gene2ind"+txt_type+".txt"
    
    cell_features = np.genfromtxt(genotype, delimiter=',')
    drug_features = np.genfromtxt(drug2fingerprint, delimiter=',')
    
    cell2id_mapping = load_mapping(cell2id)
    drug2id_mapping = load_mapping(drug2id)
    gene2id_mapping = load_mapping(gene2id)
    
    num_cells = len(cell2id_mapping)
    num_drugs = len(drug2id_mapping)
    
    names = get_compound_names(inputdir+"compound_names"+txt_type+".txt")
    names.pop(0)
        
    load=resultsdir+opt.model_name
    device = torch.device(f"cuda:{opt.cuda}" if torch.cuda.is_available() else "cpu")
    model = torch.load(load, map_location=device)
    batchsize =10000
    
    # If the pre-trained model was wrapped with DataParallel, extract the underlying model
    if isinstance(model, torch.nn.DataParallel):
        model = model.module
    model = model.to(device)
    
    predictions = {}
    predictions_spearman = {}
    for selected_drug_data in names:
        selected_drug = names[633][0]
        features, labels = load_select_data(test, cell2id_mapping, drug2id_mapping,selected_drug)
        predict_data = (torch.Tensor(features), torch.FloatTensor(labels))
        pearson, spearman=predict(predict_data,model, batchsize, cell_features, drug_features, opt.cuda)
        predictions[selected_drug_data[1]]=pearson.item()
        predictions_spearman[selected_drug_data[1]]=spearman.item()
    
    predictions_dataframe = pd.DataFrame(predictions_spearman.items(),columns=['Name', folder+' Spearman'])
    predictions_dataframe_sort = predictions_dataframe.sort_values(by=[folder+' Spearman'], ascending=False)
    
    # Save results for each model
    with open(resultsdir+"drug_predictions_dataframe_sort_test_spearman.pkl", 'wb') as file:
        pickle.dump(predictions_dataframe_sort, file)
    
    predictions_dataframe = pd.DataFrame(predictions.items(),columns=['Name', folder+' Pearson'])
    predictions_dataframe_sort = predictions_dataframe.sort_values(by=[folder+' Pearson'], ascending=False)
        
    # Save results for each model
    with open(resultsdir+"drug_predictions_dataframe_sort_test_pearson.pkl", 'wb') as file:
        pickle.dump(predictions_dataframe_sort, file)   
  

# Compute the average correlation for each drug

# PEARSON ----------------------------------------------------
# Import all the results 
list_correlation = {sample: None for sample in opt.samples_folders}
for folder in opt.samples_folders:
    with open(opt.output_folder+folder+"/"+"drug_predictions_dataframe_sort_test_pearson.pkl", 'rb') as dictionary_file:
        list_correlation[folder] = pickle.load(dictionary_file)        
  
# Join results   
# Start with an empty dataframe
all_df = pd.DataFrame()

# Loop through the list of dataframes and merge them
for key in list_correlation:
    df = list_correlation[key]
    if all_df.empty:
        all_df = df
    else:
        all_df = pd.merge(all_df, df, on=['Name'])
all_df = all_df.set_index("Name")

# Get the mean and std
all_df_mean = all_df.mean(axis=1).reset_index(name='Mean')
all_df_std = all_df.std(axis=1).reset_index(name='Std')

mean_pearson = all_df_mean["Mean"].mean()
print("Average pearson correlations of all drugs: ",mean_pearson)
wandb.log({"Average pearson correlations of all drugs:": mean_pearson})

all_df_values = pd.merge(all_df_mean,all_df_std,on=['Name'])
all_df_values.columns = ["Name","Mean","Std"]
all_df_values =all_df_values.sort_values(by=["Mean"], ascending=False)

all_df_values.to_csv(opt.output_folder+'pearson_means.txt', sep='\t', index=False)

artifact = wandb.Artifact("means_pearson_drug",type="drug means")
artifact.add_file(opt.output_folder+'pearson_means.txt')
wandb.log_artifact(artifact)

# Waterfall Plot Pearson
plt.rcParams['figure.figsize'] = (12, 9)
drugs = all_df_values["Name"]
rhos = all_df_values["Mean"]
error = all_df_values["Std"]

percentage = round((sum(rhos>0.5)/len(rhos))*100)

fig, ax = plt.subplots()
colors = ['#C9C9C9' if (x < 0.5) else '#B4D04F' for x in rhos ]
ax.bar(
    x=drugs,
    height=rhos,
    edgecolor=colors,
    color=colors,
    linewidth=1

)
plt.xticks([])
plt.yticks(fontsize=28)

# First, let's remove the top, right and left spines (figure borders)
# which really aren't necessary for a bar chart.
# Also, make the bottom spine gray instead of black.
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['bottom'].set_visible(False)
#ax.spines['bottom'].set_color('#DDDDDD')

# Second, remove the ticks as well.
ax.tick_params(bottom=False, left=False)

# Third, add a horizontal grid (but keep the vertical grid hidden).
ax.set_axisbelow(False)
ax.yaxis.grid(False)
ax.xaxis.grid(False)

# Add labels and a title. Note the use of `labelpad` and `pad` to add some
# extra space between the text and the tick labels.
ax.set_xlabel('Drugs', labelpad=-30, color='#333333',fontsize=50)
ax.set_ylabel('Pearson correlation', labelpad=15, color='#333333',fontsize=50)
ax.set_title('', color='#333333', weight='bold')

colors2 = {'High confidence drugs (r>0.5)':'#A4C61A'}  
labels = list(colors2.keys())
handles = [plt.Rectangle((0,0),1,1, color=colors2[label]) for label in labels]
#plt.legend(handles, labels,fontsize=40, loc="lower left",bbox_to_anchor=(0, -0.215))
plt.text(10, 0.25, str(percentage)+"%", fontsize=60,color='#000000')

plt.ylim((-0.1,0.9))
# Make the chart fill out the figure better.
fig.tight_layout()
# Save figure
fig.savefig(opt.output_folder+'WaterfallDrugsSparseGO_pearson.png', transparent=True)  

artifact = wandb.Artifact("WaterfallDrugsSparseGO_pearson",type="plots")
artifact.add_file(opt.output_folder+'WaterfallDrugsSparseGO_pearson.png')
wandb.log_artifact(artifact)
  
# Top 10 drugs bar chart, pearson 
plt.rcParams['figure.figsize'] = (16, 22)
fig, ax = plt.subplots()
rhos_top=rhos[0:10]
drugs_top=drugs[0:10].copy()
# drugs_top.iloc[1] = drugs_top.iloc[1].replace("+","\n +")
#drugs_top.iloc[1] = 'navitoclax:piperlongumine'
# drugs_top.iloc[6] = 'docetaxel:tanespimycin'
#drugs_top.iloc[5] = 'docetaxel:tanespimycin'

colors = ['lightseagreen' if (x < 0.5) else '#B4D04F' for x in rhos_top ]
bars = ax.bar(
    x=drugs_top,
    height=rhos_top,
    edgecolor="none",
    linewidth=1,
    color = colors,
    width=0.9,
)
#plt.yticks(fontsize=30)
plt.yticks([])
plt.xticks([])

# First, let's remove the top, right and left spines (figure borders)
# which really aren't necessary for a bar chart.
# Also, make the bottom spine gray instead of black.
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['bottom'].set_color('#DDDDDD')

# Third, add a horizontal grid (but keep the vertical grid hidden).
# Color the lines a light gray as well.
ax.set_axisbelow(True)
#ax.yaxis.grid(False, color='#EEEEEE')
ax.xaxis.grid(False)

#plt.xticks(rotation=80,fontsize=40)

# Add text annotations to the top of the bars.
bar_color = bars[0].get_facecolor()
for bar in bars:
  ax.text(
      
      bar.get_x() + bar.get_width() / 2,
      bar.get_height() + 0.03,
      round(bar.get_height(), 3),
      horizontalalignment='center',
      #color=bar_color,
      color='#000000',
      weight='bold',
      fontsize=80,
      rotation="vertical"
  )

i=0
for bar in bars:
    ax.text(
      
      bar.get_x() + bar.get_width() / 2,
      #0.05,
      0.01,
      drugs_top.iloc[i],
      horizontalalignment='center',
      #color=bar_color,
      color='#000000',
      #weight='bold',
      #fontsize=87,
      fontsize=60,
      rotation="vertical",
    )
    i=i+1

ax.tick_params(bottom=True, left=False, axis='x', which='major', pad=-1)
# Add labels and a title. Note the use of `labelpad` and `pad` to add some
# extra space between the text and the tick labels.
ax.set_xlabel('', labelpad=15, color='#333333')
#ax.set_ylabel('r', labelpad=15, color='#333333',fontsize=30)
ax.set_title('', color='#333333',
             weight='bold')

# Make the chart fill out the figure better.
fig.tight_layout()
# Save
fig.savefig(opt.output_folder+'top10sparse_pearson.png', transparent=True)

artifact = wandb.Artifact("top10sparse_pearson",type="plots")
artifact.add_file(opt.output_folder+'top10sparse_pearson.png')
wandb.log_artifact(artifact)

# SPEARMAN ----------------------------------------------------
# Import all the results 
list_correlation = {sample: None for sample in opt.samples_folders}
for folder in opt.samples_folders:
    with open(opt.output_folder+folder+"/"+"drug_predictions_dataframe_sort_test_spearman.pkl", 'rb') as dictionary_file:
        list_correlation[folder] = pickle.load(dictionary_file)
        
# Join results   
# Start with an empty dataframe
all_df = pd.DataFrame()

# Loop through the list of dataframes and merge them
for key in list_correlation:
    df = list_correlation[key]
    if all_df.empty:
        all_df = df
    else:
        all_df = pd.merge(all_df, df, on=['Name'])
all_df = all_df.set_index("Name")

all_df_mean = all_df.mean(axis=1).reset_index(name='Mean')
all_df_std = all_df.std(axis=1).reset_index(name='Std')

mean_spearman = all_df_mean["Mean"].mean()
print("Average spearman correlations of all drugs: ",mean_spearman)
wandb.log({"Average spearman correlations of all drugs:": mean_spearman})

all_df_values = pd.merge(all_df_mean,all_df_std,on=['Name'])
all_df_values.columns = ["Name","Mean","Std"]
all_df_values =all_df_values.sort_values(by=["Mean"], ascending=False)

all_df_values.to_csv(opt.output_folder+'spearman_means.txt', sep='\t', index=False)

artifact = wandb.Artifact("means_spearman_drug",type="drug means")
artifact.add_file(opt.output_folder+'spearman_means.txt')
wandb.log_artifact(artifact)

# Waterfall plot Spearman
plt.rcParams['figure.figsize'] = (12, 9)
drugs = all_df_values["Name"]
rhos = all_df_values["Mean"]
error = all_df_values["Std"]

percentage = round((sum(rhos>0.5)/len(rhos))*100)

fig, ax = plt.subplots()
#colors = ['#208EA3' if (x < 0.5) else '#A4C61A' for x in rhos ]
colors = ['#C9C9C9' if (x < 0.5) else '#B4D04F' for x in rhos ]
ax.bar(
    x=drugs,
    height=rhos,
    edgecolor=colors,
    color=colors,
    linewidth=1
)
plt.xticks([])
plt.yticks(fontsize=28)

# First, let's remove the top, right and left spines (figure borders)
# which really aren't necessary for a bar chart.
# Also, make the bottom spine gray instead of black.
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['bottom'].set_visible(False)
#ax.spines['bottom'].set_color('#DDDDDD')

# Second, remove the ticks as well.
ax.tick_params(bottom=False, left=False)

# Third, add a horizontal grid (but keep the vertical grid hidden).
# Color the lines a light gray as well.
ax.set_axisbelow(False)
ax.yaxis.grid(False)
#ax.yaxis.grid(True, color='#EEEEEE')
ax.xaxis.grid(False)

# Add labels and a title. Note the use of `labelpad` and `pad` to add some
# extra space between the text and the tick labels.
ax.set_xlabel('Drugs', labelpad=-30, color='#333333',fontsize=50)
ax.set_ylabel('Spearman correlation', labelpad=15, color='#333333',fontsize=50)
ax.set_title('', color='#333333',
             weight='bold')

colors2 = {'High confidence drugs (r>0.5)':'#A4C61A'}  
labels = list(colors2.keys())
handles = [plt.Rectangle((0,0),1,1, color=colors2[label]) for label in labels]
#plt.legend(handles, labels,fontsize=40, loc="lower left",bbox_to_anchor=(0, -0.215))
plt.text(10, 0.25, str(percentage)+"%", fontsize=60,color='#000000')

plt.ylim((-0.1,0.9))
# Make the chart fill out the figure better.
fig.tight_layout()
fig.savefig(opt.output_folder+'WaterfallDrugsSparseGO_spearman.png', transparent=True)   

artifact = wandb.Artifact("WaterfallDrugsSparseGO_spearman",type="plots")
artifact.add_file(opt.output_folder+'WaterfallDrugsSparseGO_spearman.png')
wandb.log_artifact(artifact) 
    
# Top 10 drugs bar chart, spearman
plt.rcParams['figure.figsize'] = (16, 22)
fig, ax = plt.subplots()
rhos_top=rhos[0:10]
drugs_top=drugs[0:10].copy()
# drugs_top.iloc[1] = drugs_top.iloc[1].replace("+","\n +")
#drugs_top.iloc[1] = 'navitoclax:piperlongumine'
# drugs_top.iloc[6] = 'docetaxel:tanespimycin'
#drugs_top.iloc[5] = 'docetaxel:tanespimycin'

colors = ['lightseagreen' if (x < 0.5) else '#B4D04F' for x in rhos_top ]
bars = ax.bar(
    x=drugs_top,
    height=rhos_top,
    edgecolor="none",
    linewidth=1,
    color = colors,
    width=0.9,
)
#plt.yticks(fontsize=30)
plt.yticks([])
plt.xticks([])

# First, let's remove the top, right and left spines (figure borders)
# which really aren't necessary for a bar chart.
# Also, make the bottom spine gray instead of black.
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['bottom'].set_color('#DDDDDD')

# Third, add a horizontal grid (but keep the vertical grid hidden).
# Color the lines a light gray as well.
ax.set_axisbelow(True)
#ax.yaxis.grid(False, color='#EEEEEE')
ax.xaxis.grid(False)

#plt.xticks(rotation=80,fontsize=40)

# Add text annotations to the top of the bars.
bar_color = bars[0].get_facecolor()
for bar in bars:
  ax.text(
      
      bar.get_x() + bar.get_width() / 2,
      bar.get_height() + 0.03,
      round(bar.get_height(), 3),
      horizontalalignment='center',
      #color=bar_color,
      color='#000000',
      weight='bold',
      fontsize=80,
      rotation="vertical"
  )

i=0
for bar in bars:
    ax.text(
      
      bar.get_x() + bar.get_width() / 2,
      #0.05,
      0.01,
      drugs_top.iloc[i],
      horizontalalignment='center',
      #color=bar_color,
      color='#000000',
      #weight='bold',
      #fontsize=87,
      fontsize=60,
      rotation="vertical",
    )
    i=i+1

ax.tick_params(bottom=True, left=False, axis='x', which='major', pad=-1)
# Add labels and a title. Note the use of `labelpad` and `pad` to add some
# extra space between the text and the tick labels.
ax.set_xlabel('', labelpad=15, color='#333333')
#ax.set_ylabel('r', labelpad=15, color='#333333',fontsize=30)
ax.set_title('', color='#333333',
             weight='bold')

# Make the chart fill out the figure better.
fig.tight_layout()
fig.savefig(opt.output_folder+'top10sparse_spearman.png', transparent=True)

artifact = wandb.Artifact("top10sparse_spearman",type="plots")
artifact.add_file(opt.output_folder+'top10sparse_spearman.png')
wandb.log_artifact(artifact) 
    
# Calculate the density plot of all the models  
    
list_predictions = {sample: None for sample in opt.samples_folders}
list_models_pearsons = {sample: None for sample in opt.samples_folders}
list_models_spearmans = {sample: None for sample in opt.samples_folders}
for folder in opt.samples_folders:
    file_labels = opt.input_folder + folder + "/" + opt.labels_name
    file_predictions = opt.output_folder + folder + "/" + opt.predictions_name
    
    real_auc = []
    sparse_auc = []
    
    with open(file_labels, 'r') as fi:
        for line in fi:
            tokens = line.strip().split('\t')
            real_auc.append(float(tokens[2]))
            
    real_aucA = np.array(real_auc)
            
    with open(file_predictions, 'r') as fi:
        for line in fi:
            tokens = line.strip().split('\t')
            sparse_auc.append(float(tokens[0]))
    
    sparse_aucA = np.array(sparse_auc)
    sparse = (real_aucA, sparse_aucA)
    
    list_predictions[folder] = pd.DataFrame(list(zip(real_auc, sparse_auc,["SparseGO"]*len(real_auc))),columns =['Real AUC', 'Predicted AUC','Class',])
    list_models_pearsons[folder] = float(pearson_corr(torch.from_numpy(list_predictions[folder].loc[list_predictions[folder].loc[:,"Class"]=="SparseGO","Predicted AUC"].to_numpy()),torch.from_numpy(list_predictions[folder].loc[list_predictions[folder].loc[:,"Class"]=="SparseGO","Real AUC"].to_numpy())).numpy())
    list_models_spearmans[folder] = float(spearman_corr(torch.from_numpy(list_predictions[folder].loc[list_predictions[folder].loc[:,"Class"]=="SparseGO","Predicted AUC"].to_numpy()),torch.from_numpy(list_predictions[folder].loc[list_predictions[folder].loc[:,"Class"]=="SparseGO","Real AUC"].to_numpy())).numpy())


all_predictions = pd.DataFrame()
# Loop through the list of dataframes and concatenate them
for key in list_predictions:
    df = list_predictions[key]
    if all_predictions.empty:
        all_predictions = df
    else:
        all_predictions = pd.concat([all_predictions, df])
        
# Calculate overall correlations and loss
criterion=nn.MSELoss()

pe_overall = pearson_corr(torch.from_numpy(all_predictions.loc[all_predictions.loc[:,"Class"]=="SparseGO","Predicted AUC"].to_numpy()),torch.from_numpy(all_predictions.loc[all_predictions.loc[:,"Class"]=="SparseGO","Real AUC"].to_numpy())).numpy()
pe_overall = np.around(pe_overall,4)

sp_overall = spearman_corr(torch.from_numpy(all_predictions.loc[all_predictions.loc[:,"Class"]=="SparseGO","Predicted AUC"].to_numpy()).detach().numpy(),torch.from_numpy(all_predictions.loc[all_predictions.loc[:,"Class"]=="SparseGO","Real AUC"].to_numpy())).numpy()
sp_overall = np.around(sp_overall,4)

loss_overall = criterion(torch.from_numpy(all_predictions.loc[all_predictions.loc[:,"Class"]=="SparseGO","Predicted AUC"].to_numpy()),torch.from_numpy(all_predictions.loc[all_predictions.loc[:,"Class"]=="SparseGO","Real AUC"].to_numpy())).numpy()
loss_overall = np.around(loss_overall,4)
    
# Calculate average correlations
pe_average = np.around(statistics.mean(list(list_models_pearsons.values())),4)
sp_average =  np.around(statistics.mean(list(list_models_spearmans.values())),4)
    
# PLOT RESULTS   
plt.rcParams['figure.figsize'] = (10, 7)

x = all_predictions.loc[all_predictions.loc[:,"Class"]=="SparseGO","Real AUC"].to_numpy()
y = all_predictions.loc[all_predictions.loc[:,"Class"]=="SparseGO","Predicted AUC"].to_numpy()

#histogram definition
bins = [1000, 1000] # number of bins

# histogram the data
hh, locx, locy = np.histogram2d(x, y, bins=bins)

# Sort the points by density, so that the densest points are plotted last
z = np.array([hh[np.argmax(a<=locx[1:]),np.argmax(b<=locy[1:])] for a,b in zip(x,y)])
idx = z.argsort()
x2, y2, z2 = x[idx], y[idx], z[idx]


fig, ax = plt.subplots()
plt.scatter(x2, y2, c=z2, cmap='turbo', marker='.',s=4) 
m, b = np.polyfit(x, y, 1)
plt.plot(x, m*x+b,color='#333333') # line 

#plt.xticks([])
plt.yticks(fontsize=16)
plt.xticks(fontsize=16)

# First, let's remove the top, right and left spines (figure borders)
# which really aren't necessary for a bar chart.
# Also, make the bottom spine gray instead of black.
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_color('#DDDDDD')
ax.spines['left'].set_color('#DDDDDD')

# Second, remove the ticks as well.
ax.tick_params(bottom=False, left=False)

# Third, add a horizontal grid (but keep the vertical grid hidden).
# Color the lines a light gray as well.
ax.set_axisbelow(True)
ax.yaxis.grid(True, color='#EEEEEE')
ax.xaxis.grid(False)

# Add labels and a title. Note the use of `labelpad` and `pad` to add some
# extra space between the text and the tick labels.
ax.set_xlabel('Real response (AUDRC)', labelpad=18, color='#333333',fontsize=25)
ax.set_ylabel('Predicted response (AUDRC)', labelpad=18, color='#333333',fontsize=25)
ax.set_title('Density plot', color='#000000', weight='bold',fontsize=30)

plt.text(0.43, 0.18, "Overall spearman corr. = "+str(sp_overall), fontsize=15,color='#333333',weight='bold')
plt.text(0.43, 0.14, "Overall pearson corr. = "+str(pe_overall), fontsize=15,color='#333333',weight='bold')
plt.text(0.43, 0.10, "Overall MSE loss = "+str(loss_overall), fontsize=15,color='#333333',weight='bold')

plt.text(0.43, 0.06, "Average spearman corr. = "+str(sp_average), fontsize=15,color='#333333',weight='bold')
plt.text(0.43, 0.02, "Average pearson corr. = "+str(pe_average), fontsize=15,color='#333333',weight='bold')

plt.xlim((0,1))
plt.ylim((0,1))
# Make the chart fill out the figure better.
fig.tight_layout()
fig.savefig(opt.output_folder+'density_plot.png', transparent=True)
    
artifact = wandb.Artifact("density_plot",type="plots")
artifact.add_file(opt.output_folder+'density_plot.png')
wandb.log_artifact(artifact) 

wandb.finish()