#' @description
#' The aim of this code is to obtain the expression matrix of the selected cell lines.
#' Expression data is downloaded with the depmap and ExperimentHub libraries. 
#'  
#' Four txt files are created: 
#'  - Expression matrix (cell2expression.txt)
#'  - List of cell lines (cell2ind.txt)
#'  - List of genes included in the expression file (gene2id.txt)
#'  - File containing the cell line-drug pairs (some cell lines are not on the depmap dataset) (drugcell_all.txt)
#'  *** Ontology file (ontology.txt) has to later be created with Gene
#'
library("depmap")
library("ExperimentHub")
library(dplyr)


mac <- "/Users/katyna/Library/CloudStorage/OneDrive-Tecnun/"
windows <- "C:/Users/ksada/OneDrive - Tecnun/"
computer <- windows

# cell line ids 
cell_id <- t(read.delim(paste(computer,"Tesis/Codigo/DrugCell/Data/cell2ind.txt",sep=""),header = FALSE,row.names=1))

## create ExperimentHub query object
eh <- ExperimentHub()
query(eh, "depmap")

TPM <- eh[["EH7292"]] # TPM_21Q4       

# Keep only wanted cell lines
TPM <- filter(TPM, cell_line %in% cell_id) # only 940 cell lines found 

# Keep only the annotated genes in mygene (step required to later create the ontology file)
library(mygene)
genes <- queryMany(gsub(" .*$", "",colnames(expression_matrix[,-1])), species="human")
genes_data <- as.data.frame(genes@listData)
genes_data <- filter(genes_data, is.na(genes_data$notfound))
TPM <- filter(TPM, gene_name %in% genes_data$query) # 19146 genes found


library(tidyr)
expression_matrix <- spread(TPM[,c("cell_line","gene","rna_expression")], gene, rna_expression)

# save expression file ---
write.table((round(expression_matrix[,-1],7)), "/Users/ksada/OneDrive - Tecnun/Tesis/Codigo/ExpressionData/data_expression_all_genes/cell2expression_depmap.txt", append = FALSE, quote=FALSE,sep = ",", dec = ".",row.names = FALSE, col.names = FALSE)

# make cell lines file ---
cells_txt <- cbind(0:(dim(expression_matrix)[1]-1),expression_matrix[,1])
write.table(cells_txt, "/Users/ksada/OneDrive - Tecnun/Tesis/Codigo/ExpressionData/data_expression_all_genes/cell2ind_depmap.txt", append = FALSE, quote=FALSE,sep = "\t", dec = ".",row.names = FALSE, col.names = FALSE)

# make genes file ---
genes_txt <- cbind(0:(dim(expression_matrix)[2]-2),gsub(" .*$", "",colnames(expression_matrix[,-1])))
write.table(genes_txt, "/Users/ksada/OneDrive - Tecnun/Tesis/Codigo/ExpressionData/data_expression_all_genes/gene2ind_depmap.txt", append = FALSE, quote=FALSE,sep = "\t", dec = ".",row.names = FALSE, col.names = FALSE)

# make train data file --- 
pairs <- read.csv("/Users/ksada/OneDrive - Tecnun/Tesis/Codigo/DrugCell/Data/drugcell_all.csv", header = FALSE)
selected_pairs <- pairs[which(!is.na(match(pairs$V1,expression_matrix$cell_line))),]
write.table(selected_pairs, "/Users/ksada/OneDrive - Tecnun/Tesis/Codigo/ExpressionData/data_expression_all_genes/drugcell_all_depmap.txt", append = FALSE, quote=FALSE,sep = "\t", dec = ".",row.names = FALSE, col.names = FALSE)

# extra...
# separate in train and test
file <- "/Users/ksada/OneDrive - Tecnun/Tesis/Codigo/ExpressionData/data_expression_all_genes/drugcell_all.csv"
a <- read.csv(file, header = FALSE, sep = ";")

spec = c(train = 0.8, test = 0.1, validate = 0.1)

g = sample(cut(seq(nrow(a)), 
               nrow(a)*cumsum(c(0,spec)),
               labels = names(spec)
))

res = split(a, g)

write.table(res[["train"]], file = "/Users/ksada/OneDrive - Tecnun/Tesis/Codigo/ExpressionData/data_expression_all_genes/drugcell_train.txt", sep = "\t", row.names = F, col.names=F, quote = FALSE)
write.table(res[["test"]], file = "/Users/ksada/OneDrive - Tecnun/Tesis/Codigo/ExpressionData/data_expression_all_genes/drugcell_test.txt", sep = "\t", row.names = F, col.names=F, quote = FALSE)
write.table(res[["validate"]], file = "/Users/ksada/OneDrive - Tecnun/Tesis/Codigo/ExpressionData/data_expression_all_genes/drugcell_val.txt", sep = "\t", row.names = F, col.names=F, quote = FALSE)

# 203.718 (152788 + 50929)
dim(res[["train"]])[1]+dim(res[["validate"]])[1]

dim(res[["train"]])[1]
dim(res[["validate"]])[1]
dim(res[["test"]])[1]


