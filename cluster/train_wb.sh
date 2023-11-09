#!/bin/bash

# CHANGE THIS - parameters to modify
cudaid=0 # cuda id
epoch=5
drug_neurons='100,50,6'
# These parameters can also be tunned in train_gpu_wb.py using Weights & Biases (W&B) -----
batch=5000
lr=0.1
decay_rate=0.001
number_neurons_per_GO=6
number_neurons_per_final_GO=6
final_neurons=6
# -----

projectname="Test" # CHANGE THIS - name you want to give to your W&B project
foldername="toy_example_DrugCell" # CHANGE THIS - folder where you have the folder(s) of data
mkdir "../results/"$foldername

# Activate python environment
source activate /Users/katyna/SparseGO # CHANGE THIS - your environment # source activate  C:/Users/ksada/Anaconda3/envs/SparseGO
wandb login b1f6d1cea53bb6557df3c1c0c0530b53cadeed3d # CHANGE THIS - your W&B account

# The loop was created to facilitate cross-validation, more than 1 folder can be provided to create different models using different samples for training and testing
for samples in "samples1"  # CHANGE THIS - folder(s) where you have the data
do
  inputdir="../data/"$foldername"/"$samples"/"
  modeldir="../results/"$foldername"/"$samples"/" # CHANGE THIS - folder to store results
  mkdir $modeldir # Create folder to store results

  type="" # CHANGE THIS - add something if files have different endings

  gene2idfile=$inputdir"gene2ind"$type".txt"
  cell2idfile=$inputdir"cell2ind"$type".txt"
  drug2idfile=$inputdir"drug2ind"$type".txt"
  traindatafile=$inputdir"drugcell_train"$type".txt" # CHANGE THIS - train file
  valdatafile=$inputdir"drugcell_val"$type".txt" # CHANGE THIS - validation file
  drugfile=$inputdir"drug2fingerprint"$type".txt"

  ontfile=$inputdir"drugcell_ont.txt" # CHANGE THIS - ontology file
  #ontfile=$inputdir"ontology.txt"

  mutationfile=$inputdir"cell2mutation"$type".txt" # CHANGE THIS - expression/mutation file
  #mutationfile=$inputdir"cell2expression"$type".txt"

  testdatafile=$inputdir"drugcell_test"$type".txt" # CHANGE THIS - test file

  python -u "../code/train_gpu_wb.py" -onto $ontfile -gene2id $gene2idfile -drug2id $drug2idfile -cell2id $cell2idfile -train $traindatafile -val $valdatafile -modeldir $modeldir -cuda $cudaid -genotype $mutationfile -fingerprint $drugfile -number_neurons_per_GO $number_neurons_per_GO -number_neurons_per_final_GO $number_neurons_per_final_GO -drug_neurons $drug_neurons -final_neurons $final_neurons -epoch $epoch -batchsize $batch -lr $lr -decay_rate $decay_rate -predict $testdatafile -result $modeldir -project $projectname -sweep_name $samples >$modeldir"train_correlation.log"
done
# ----- Create the plots/graphs/final metrics
input_folder="../data/"$foldername"/"
output_folder="../results/"$foldername"/"
model_name="best_model_p.pt"
predictions_name="ModelPearson_test_predictions.txt"
labels_name="drugcell_test.txt"
ontology_name="drugcell_ont.txt"
# samples_folders and txt_type has to be changed in the python file (for now)

python -u "../code/per_drug_correlation.py" -input_folder $input_folder -output_folder $output_folder -model_name $model_name -predictions_name $predictions_name -labels_name $labels_name -ontology_name $ontology_name -project $projectname>$modeldir"metrics.log"
















#conda info --envs
