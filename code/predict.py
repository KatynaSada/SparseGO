"""
    This script makes predictions using a pre-trained SparseGO network.
"""

import sys
import os
import pandas as pd
import numpy as np
import torch
import torch.utils.data as du
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import util
from util import *
from network import sparseGO_nn
import argparse

def predict(criterion, predict_data, gene_dim, drug_dim, model_file, batch_size, result_file, cell_features, drug_features, device):

    feature_dim = gene_dim + drug_dim

    model = torch.load(model_file)
    model.to(device)

    predict_feature, predict_label = predict_data

    predict_label_gpu = predict_label.to(device, non_blocking=True)

    model.eval()

    test_loader = du.DataLoader(du.TensorDataset(predict_feature,predict_label), batch_size=batch_size, shuffle=False)

    #Test
    test_predict = torch.zeros(0, 1, device=device)
    with torch.no_grad():
        for i, (inputdata, labels) in enumerate(test_loader):
            # Convert torch tensor to Variable
            features = build_input_vector(inputdata, cell_features, drug_features)

            cuda_features =  features.to(device)

            # make prediction for test data
            out = model(cuda_features)
            test_predict = torch.cat([test_predict, out])

    test_cum_loss=criterion(test_predict, predict_label_gpu)
    test_corr = pearson_corr(test_predict, predict_label_gpu)
    test_corr_spearman = spearman_corr(test_predict.cpu().detach().numpy(), predict_label_gpu.cpu())

    print('Test loss: {:.4f}'.format(test_cum_loss))
    print('Test Corr (p): {:.4f}'.format(test_corr))
    print('Test Corr (s): {:.4f}'.format(test_corr_spearman))

    print('Allocated:', round(torch.cuda.memory_allocated(0)/1024**3,1), 'GB')
    print('Cached:   ', round(torch.cuda.memory_cached(0)/1024**3,1), 'GB')

    np.savetxt(result_file+'test_predictions.txt', test_predict.cpu().detach().numpy(),'%.5e')
    np.savetxt(result_file+'test_result.txt', np.array(['Test Corr (p): {:.4f}'.format(test_corr),'Test Corr (s): {:.4f}'.format(test_corr_spearman),'Test loss: {:.4f}'.format(test_cum_loss)]),fmt='%s')

inputdir="../data/toy_example_DrugCell/samples1/" # CHANGE
outputdir="../results/toy_example_DrugCell/samples1/"

mutation = "cell2mutation.txt"

parser = argparse.ArgumentParser(description='Train SparseGO')
parser.add_argument('-predict', help='Dataset to be predicted',type=str, default=inputdir+"drugcell_test.txt")
parser.add_argument('-batchsize', help='Batchsize', type=int, default=5000)
parser.add_argument('-gene2id', help='Gene to ID mapping file', type=str, default=inputdir+"gene2ind.txt")
parser.add_argument('-drug2id', help='Drug to ID mapping file', type=str, default=inputdir+"drug2ind.txt")
parser.add_argument('-cell2id', help='Cell to ID mapping file', type=str, default=inputdir+"cell2ind.txt")
parser.add_argument('-load', help='Model file', type=str, default=outputdir + 'best_model_s.pt')
parser.add_argument('-result', help='Result file name', type=str, default=outputdir)
parser.add_argument('-cuda_id', help='Specify GPU', type=int, default=0)
parser.add_argument('-genotype', help='Mutation information for cell lines', type=str, default=inputdir+mutation)
parser.add_argument('-fingerprint', help='Morgan fingerprint representation for drugs', type=str, default=inputdir+"drug2fingerprint.txt")

opt = parser.parse_args()
torch.set_printoptions(precision=5)

predict_data, cell2id_mapping, drug2id_mapping = prepare_predict_data(opt.predict, opt.cell2id, opt.drug2id)
gene2id_mapping = load_mapping(opt.gene2id)

# load cell/drug features
cell_features = np.genfromtxt(opt.genotype, delimiter=',')
drug_features = np.genfromtxt(opt.fingerprint, delimiter=',')

num_cells = len(cell2id_mapping)
num_drugs = len(drug2id_mapping)
num_genes = len(gene2id_mapping)
drug_dim = len(drug_features[0,:])

criterion=nn.MSELoss()

device = torch.device(f"cuda:{opt.cuda_id}" if torch.cuda.is_available() else "cpu")
predict(criterion, predict_data, num_genes, drug_dim, opt.load, opt.batchsize, opt.result, cell_features, drug_features, device)
