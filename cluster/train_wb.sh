#!/bin/bash

# CHANGE THIS - parameters to modify
cudaid=0 # cuda id
epoch=100
drug_neurons='100,50,6'
# These parameters can also be tunned in train_gpu_wb.py using Weights & Biases (W&B) -----
batch=15000
lr=0.1
decay_rate=0.001
number_neurons_per_GO=6
number_neurons_per_final_GO=6
final_neurons=6
# -----

projectname="SparseGO_toy_example" # CHANGE THIS - name you want to give to your W&B project

gpu_name="RTX_A4000" # CHANGE THIS - not important, just a reminder of which gpu you used.

for samples in "samples1"  # CHANGE THIS - folder(s) where you have the data
do
  inputdir="../data/toy_example/"$samples"/" # CHANGE THIS - folder where you have the folder(s) of data
  modeldir="../results/toy_example/"$samples"/" # CHANGE THIS - folder to store results
  mkdir $modeldir

  type="" # CHANGE THIS - add something if files have different endings

  gene2idfile=$inputdir"gene2ind"$type".txt"
  cell2idfile=$inputdir"cell2ind"$type".txt"
  drug2idfile=$inputdir"drug2ind"$type".txt"
  traindatafile=$inputdir"drugcell_train"$type".txt"
  valdatafile=$inputdir"drugcell_val"$type".txt"
  drugfile=$inputdir"drug2fingerprint"$type".txt"

  ontfile=$inputdir"drugcell_ont.txt" # CHANGE THIS - ontology file
  #ontfile=$inputdir"ontology.txt"

  mutationfile=$inputdir"cell2mutation"$type".txt" # CHANGE THIS - expression/mutation file
  #mutationfile=$inputdir"cell2expression"$type".txt"

  testdatafile=$inputdir"drugcell_test"$type".txt"

  #WINDOWS
  source activate  C:/Users/ksada/Anaconda3/envs/SparseGO # CHANGE THIS - your environment
  wandb login b1f6d1cea53bb6557df3c1c0c0530b53cadeed3d # CHANGE THIS - your W&B account
  python -u "C:/Users/ksada/OneDrive - Tecnun/SparseGO_code/code/train_gpu_wb.py" -onto $ontfile -gene2id $gene2idfile -drug2id $drug2idfile -cell2id $cell2idfile -train $traindatafile -test $valdatafile -modeldir $modeldir -cuda $cudaid -genotype $mutationfile -fingerprint $drugfile -number_neurons_per_GO $number_neurons_per_GO -number_neurons_per_final_GO $number_neurons_per_final_GO -drug_neurons $drug_neurons -final_neurons $final_neurons -epoch $epoch -batchsize $batch -lr $lr -decay_rate $decay_rate -predict $testdatafile -result $modeldir -project $projectname -gpu_name $gpu_name  > $modeldir"train_correlation.log"

done

#conda info --envs
