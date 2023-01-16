#!/bin/bash
#inputdir="../data/data_expression_lincs1000/" # CHANGE THIS
inputdir="../data/cross_validation/samples1/"
type="" # CHANGE THIS

gene2idfile=$inputdir"gene2ind"$type".txt"
cell2idfile=$inputdir"cell2ind"$type".txt"
drug2idfile=$inputdir"drug2ind"$type".txt"
testdatafile=$inputdir"drugcell_val"$type".txt"
drugfile=$inputdir"drug2fingerprint"$type".txt"

ontfile=$inputdir"drugcell_ont.txt" # CHANGE THIS
#ontfile=$inputdir"lincs_ont.txt"

mutationfile=$inputdir"cell2mutation"$type".txt" # CHANGE THIS
#mutationfile=$inputdir"cell2expression"$type".txt"

directory="../results/DrugCell_sample1" # CHANGE THIS



modelfile=$directory"/best_model_p.pt"
resultdir=$directory

cudaid=$1

if [$cudaid = ""]; then
	cudaid=0
fi

#source activate /home/ksada/.conda/envs/SparseGoNew2
source activate  C:/Users/ksada/Anaconda3/envs/SparseGO


python -u ../code/predict_gpu.py -gene2id $gene2idfile -cell2id $cell2idfile -drug2id $drug2idfile -genotype $mutationfile -fingerprint $drugfile -result $resultdir -predict $testdatafile -load $modelfile -cuda $cudaid > $directory/ValCorrelation.log
