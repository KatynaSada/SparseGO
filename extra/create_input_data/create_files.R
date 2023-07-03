library("PharmacoGx")
library(DescTools)
library(drda)
library(dr4pl)

mac <- "/Users/katyna/Library/CloudStorage/OneDrive-Tecnun/Tesis/Codigo/AUC"
windows <- "C:/Users/ksada/OneDrive - Tecnun/Tesis/Codigo/AUC/"
angel <- "C:/Users/arubio/OneDrive - Tecnun/AUC"
computer <- windows
setwd(computer) # choose directory

database <- readRDS("./DataSets/CTRPv2/CTRPv2_2015.rds")
# database <- readRDS("./DataSets/GDSC2/GDSC_2020(v2-8.2).rds")
database <- readRDS("./DataSets/GDSC1/GDSC_2020(v1-8.2).rds")
database <- readRDS("./DataSets/PRISM/PRISM_2020.rds")
# NCI60<- readRDS("./DataSets/NCI60/NCI60.rds")
# data("CCLEsmall")

compute_auc = function(dr4plmodel, xlim, ylim=c(0,1)) {
  # Compute the area under the curve of a DR4PL model
  # dr4plmodel: a DR4PL model object, typically created with the drm() function from the drc package
  # xlim: a vector of length 2 specifying the range of x values to integrate over
  # ylim: a vector of length 2 specifying the range of y values to integrate over (default is 0 to 1)
  
  # Convert xlim to a logarithmic scale
  xlim <- 10^(xlim)
  
  # Get the coefficients of the DR4PL model
  Theta <- coefficients(modelodr4pl)
  
  # Define a function for the DR4PL model
  f1 = function(x,ylim) pmax(pmin((Theta[1] + 
                                     (Theta[4] - Theta[1])/(1 + ((10^x)/Theta[2])^Theta[3])),ylim[2]),ylim[1])
  
  # Integrate the DR4PL model function over the specified range of x values
  integrate(f1, log10(xlim[1]),log10(xlim[2]),ylim)$value/(log10(xlim[2]/xlim[1]))
}

# COMPUTING THE AUC -------------
# # Extract concentration data
# x <- database@treatmentResponse[["raw"]][,,1]
# x <- apply(x, 2, as.numeric) # Convert to numeric in case some pSets are not numeric
# x_log <- log(x) # Convert to log 
# 
# # Extract viability data
# y <- database@treatmentResponse[["raw"]][,,2]
# y <- apply(y, 2, as.numeric) # Convert to numeric in case some pSets are not numeric
# y_normalized  <- y/100 # Divide to convert from percentage to decimal
# 
# # Initialize matrices for AUC values and sample/drug names
# AUCS <- matrix(NA, nrow = dim(x_log)[1], ncol = 1)
# Names <- matrix("", nrow = dim(x_log)[1], ncol = 2)
# 
# # Add column names for matrices
# colnames(Names) <- cbind("Sample","Drug")
# colnames(AUCS) <- cbind("AUC_dr4")
# 
# for(experiment_index in 1:dim(x_log)[1]){
#   if (experiment_index/100 == round(experiment_index/100)) print(experiment_index)
#   
#   # get non-empty concentration and viability indices for a given dose
#   ind_x <- names(x_log[experiment_index,!is.na(x_log[experiment_index,])]) # non empty concentrations
#   ind_y <- names(y_normalized[experiment_index,!is.na(y_normalized[experiment_index,])]) # non empty viability
#   ind_both <- intersect(ind_x,ind_y) # experiments that have both a concentration value and its viability
#   
#   if(is.null(ind_both)) next # skip if there are no experiments with both concentrations and viability values for a given dose
#   
#   # assign sample and treatment IDs
#   Names[experiment_index,1] <- database@treatmentResponse[["info"]][["sampleid"]][experiment_index] # sampleid
#   Names[experiment_index,2] <- database@treatmentResponse[["info"]][["treatmentid"]][experiment_index] # treatmentid
#   
#   top <- 2
#   modelodr4pl <- NULL
#   options(show.error.messages = FALSE)
#   
#   # attempt to fit DR4PL model with logistic initialization and Huber weighting
#   try(modelodr4pl <- dr4pl(y_normalized[experiment_index,ind_both] ~ I(10^x_log[experiment_index,ind_both]), method.init = "logistic", method.robust = "Huber"),TRUE)
#   
#   # if the above fitting fails, try fitting DR4PL model with default initialization
#   if (is.null(modelodr4pl)) try(modelodr4pl <- dr4pl(y_normalized[experiment_index,ind_both] ~ I(10^x_log[experiment_index,ind_both])),TRUE)
#   
#   options(show.error.messages = TRUE)
#   
#   # skip if the fitting still fails
#   if (is.null(modelodr4pl)) next
#   
#   # calculate AUC using the fitted DR4PL model
#   try({AUCS[experiment_index] <- compute_auc(modelodr4pl, xlim = c(-4,2), ylim = c(0,top))},TRUE)
# }
# 
# # Create a data frame with sample and drug names and their corresponding AUC values
# auc <- data.frame(Names, AUCS)
# # Remove rows with missing AUC values
# auc <- na.omit(auc)
# # Save the resulting data frame
# # save(auc, file = paste(computer, "./GDSC_auc.Rdata", sep = ""))

# # Remove rows with missing AUC values
# auc <- data.frame(database@treatmentResponse[["info"]][["sampleid"]], database@treatmentResponse[["info"]][["treatmentid"]],1-database@treatmentResponse[["profiles"]][["aac_recomputed"]])
# auc <- na.omit(auc)
# colnames(auc) <- c("Sample","Drug","AUC_dr4")


# DRUG SMILES -------------
load("./DataSets/PRISM/auc_PRISM.RData") # load precomputed auc
auc <- auc_PRISM
remove(auc_PRISM)
# Look for missing smiles
library(openxlsx) # Load openxlsx library to read Excel files

# Get drugs and their corresponding SMILES information from the database
drugs <- database@treatment[["treatmentid"]]
smiles <- database@treatment[["smiles"]]
drug_data <- as.data.frame(cbind(drugs,smiles))

drug_data$smiles[which(drug_data$smiles=="-666")] = NA # Replace any SMILES value of -666 (which is an error) with NA

look <- drug_data$drugs[which(is.na(drug_data$smiles))] # Store drugs with missing SMILES information in the "look" variable

library(webchem) # Load webchem library to interact with Pubchem
library(dplyr) # Load dplyr library for data manipulation

# Retrieve Pubchem Compound ID (CID) for each drug in "look"
cid_look <- get_cid(look,match = "first")
colnames(cid_look) <- c("drugs","CID") # Rename columns to "drugs" and "CID" for easier merging later

drug_data_look <- merge(drug_data,cid_look) # Merge CID information with drug names
drug_data_look <- drug_data_look[,-2] # Remove the second column since it is empty

# Retrieve SMILES information for each CID in "missing_smiles"
missing_smiles <- pc_prop(na.omit(cid_look$CID), "CanonicalSMILES")

# Merge SMILES information with drug names and CIDs
drug_data_look <- merge(drug_data_look,missing_smiles)

# Calculate the total number of non-NA values in the CanonicalSMILES column of both data frames
sum(!is.na(missing_smiles$CanonicalSMILES))+sum(!is.na(drug_data$smiles)) #405 smiles!!! (de 544) en CTRPv2 y 171 de 190 en GDSC

# add to original dataframe 
drug_data$smiles[match(drug_data_look$drugs,drug_data$drugs)] <- drug_data_look$CanonicalSMILES

# The datasets had missing SMILES, look for them in other sources...
#### CTRPv2 ####
look2 <- drug_data$drugs[which(is.na(drug_data$smiles))]
library(readxl)
ctrp_info <- read_excel(paste(computer,"DataSets/CTRPv2/CTRPv2.0._INFORMER_SET.xlsx",sep=""))
pruebas <- match(look2, ctrp_info$cpd_name)
missing_smiles3 <- ctrp_info$cpd_smiles[pruebas]
drug_data_look2 <- data.frame(look2, missing_smiles3)
colnames(drug_data_look2) <- c('drugs', 'smiles')
drug_data$smiles[match(drug_data_look2$drugs,drug_data$drugs)] <- drug_data_look2$smiles

#### GDSC2 ####
nas_gdsc2 <- data.frame(drug_data$drugs[which(is.na(drug_data$smiles))])
nas_gdsc2$smiles <- NA
colnames(nas_gdsc2) <- c('drugs', 'smiles')
nas_gdsc2$smiles[1] <- 'C1=CC=C(C(=C1)C2=CC3=C(C4=C(N3)C=CC(=C4)O)C5=C2C(=O)NC5=O)Cl'
nas_gdsc2$smiles[14] <- 'CC(C)C1=C(N2C(=O)N=C(N3CCN(CC3C)C(=O)C=C)C3=CC(F)=C(N=C23)C2=C(F)C=CC=C2O)C(C)=CC=N1'
nas_gdsc2 <- na.omit(nas_gdsc2)
drug_data$smiles[match(nas_gdsc2$drugs,drug_data$drugs)] <- nas_gdsc2$smiles

#### GDSC1 ####
nas_gdsc1 <- drug_data$drugs[which(is.na(drug_data$smiles))]
nas_gdsc1 <- data.frame(drug_data$drugs[which(is.na(drug_data$smiles))])
nas_gdsc1$smiles <- NA
colnames(nas_gdsc1) <- c('drugs', 'smiles')
nas_gdsc1$smiles[1] <- 'C1=CC=C(C(=C1)C2=CC3=C(C4=C(N3)C=CC(=C4)O)C5=C2C(=O)NC5=O)Cl'
nas_gdsc1$smiles[3] <- 'CCC1=C2CN3C(=CC4=C(C3=O)COC(=O)C4(CC)O)C2=NC5=C1C=C(C=C5)OC(=O)N6CCC(CC6)N7CCCCC7.C1C(N(C2=C(N1)NC(=NC2=O)N)C=O)CNC3=CC=C(C=C3)C(=O)NC(CCC(=O)O)C(=O)O.C1=C(C(=O)NC(=O)N1)F'
nas_gdsc1 <- na.omit(nas_gdsc1)
drug_data$smiles[match(nas_gdsc1$drugs,drug_data$drugs)] <- nas_gdsc1$smiles

### PRISM ####
drug_data$smiles <- sapply(drug_data$smiles, function(y) stringr::str_replace(y, ",.+", ''))
drug_data$smiles <- sapply(drug_data$smiles, function(y) stringr::str_replace(y, "///.+", ''))
drug_data$smiles[which(drug_data$drugs == 'Sevelamer')] <- 'NCC=C.ClCC1CO1'


# Create dataframe of results: number - smiles - drug name (some drug-cell pairs are going to be deleted)
compound_names <-matrix(nrow = dim(na.omit(drug_data))[1], ncol = 3) # Create a matrix with the number of rows equal to the number of non-missing SMILES values in the drug_data dataframe, and 3 columns for the number, SMILE, and Name of the drug
colnames(compound_names) <- c("number","SMILE","Name")
compound_names[,1] <- 0:(dim(na.omit(drug_data))[1]-1) # Fill in the "number" column with sequential integers starting from 0
compound_names[,2] <- drug_data$smiles[complete.cases(drug_data)] # Fill in the "SMILE" column with non-missing SMILES values from the drug_data dataframe
compound_names[,3] <- drug_data$drugs[complete.cases(drug_data)] # Fill in the "Name" column with drug names from the drug_data dataframe
compound_names <- as.data.frame(compound_names)

# For standard AND drug repositioning PRISM (only keep similar compounds)
file <- paste(computer,"/DataSets/train_sparseGO/allsamples/compound_names_both.txt",sep="")
compound_names_all <- read.delim(file, header = TRUE,sep = "\t")
# keep only same drugs
compound_names <- compound_names[which(!is.na(match(compound_names$Name, compound_names_all$Name))),]
compound_names[,1] <- 0:(dim(na.omit(compound_names))[1]-1)

# For in-silico drug sensitivity PRISM
# file <- paste(computer,"/DataSets/train_sparseGO/allsamples/compound_names_both.txt",sep="")
# compound_names_all <- read.delim(file, header = TRUE,sep = "\t")
# # keep only different drugs
# compound_names <- compound_names[which(is.na(match(compound_names$Name, compound_names_all$Name))),]
# compound_names[,1] <- 0:(dim(na.omit(compound_names))[1]-1)

# CELL LINES OMICS DATA  -------------
library(mygene)
CCLE<- readRDS("./DataSets/CCLE/CCLE_2015.rds")

# sampleid <- database@sample[["sampleid"]] # cell lines in current dataset
# CCLE_sampleid <- CCLE@molecularProfiles[["rna"]]@colData@listData[["sampleid"]] # row names of expression data
# sum(is.na(match(sampleid,CCLE_sampleid))) # 67 lineas de CTRPv2 no estan en CCLE, 402 cell lines of GDSC (out of 1084) are not in CCLe

# genes_exp <- (read.delim("C:/Users/ksada/OneDrive - Tecnun/SparseGO_code/data/cross_validation_expression/allsamples/gene2ind.txt",header = FALSE,row.names=1))
# genes_exp_info <- queryMany(t(genes_exp), scopes="symbol", fields="ensembl.gene", species="human", returnall=FALSE,return.as = "DataFrame") # map gene symbol to ensembl gene
# genes_exp_ensembl <- as.character(sapply(genes_exp_info@listData[["ensembl"]], function(x) x[[1]][1]))
# CTRPv2expression <- summarizeMolecularProfiles(CCLE,
#                                              cellNames(database),
#                                              mDataType="rna",
#                                              features=genes_exp_ensembl,
#                                              verbose=FALSE)

# sum(!is.na(match(t(genes_exp),CTRPv2expression@elementMetadata@listData[["Symbol"]])))
# HACEN MATCH MENOS GENES QUE SI USO DEPMAP...

# try with depmap 
library("depmap")
library("ExperimentHub")

eh <- ExperimentHub() # Create ExperimentHub query object
query(eh, "depmap") # Query ExperimentHub for "depmap" datasets

# EXPRESSION
# Get the 'TPM' dataset, it contains the 22Q2 CCLE "Transcript Per Million" RNAseq gene expression data for protein coding genes. This dataset includes data from 19221 genes, 1406 cell lines, 33 primary diseases and 30 lineages
TPM <- eh[["EH7556"]] 

# Match sample IDs from the current dataset to CCLE sample IDs and remove missing values
samples_to_ccleid <- na.omit(CCLE@sample[["CCLE.name"]][match(database@sample[["sampleid"]],CCLE@sample[["sampleid"]])])

# For standard and insilico PRISM
# import train file from standard train, only keep same cells
# file <- paste(computer,"/DataSets/train_sparseGO/allsamples/sparseGO_train.txt",sep="")
# train_all_samples <- read.delim(file, header = FALSE,sep = "\t")
# train_all_samples_cells <- unique(train_all_samples$V1)
# samples_to_ccleid <- intersect(samples_to_ccleid,train_all_samples_cells) # returns only the elements from samples_to_ccleid that are present in train_all_samples_cells
# setdiff(samples_to_ccleid,unique(TPM$cell_line))

# For drug-repositioning (cells) PRISM
# import train file from one of the cv of drug-repositioning
file <- paste(computer,"/DataSets/train_sparseGO_cells/samples1/sparseGO_train.txt",sep="")
train_samples1 <- read.delim(file, header = FALSE,sep = "\t")
train_samples1_cells <- unique(train_samples1$V1)
samples_to_ccleid <- setdiff(samples_to_ccleid,train_samples1_cells) # returns only the elements from samples_to_ccleid that are not present in train_samples1_cells
intersect(samples_to_ccleid,unique(TPM$cell_line))

# Filter the TPM dataset to only include the cell lines from the current dataset
TPM <- filter(TPM, cell_line %in% samples_to_ccleid) # only 793 cell lines found CTRPv2, 654 in GDSC

library(tidyr)
expression_matrix <- spread(TPM[,c("cell_line","gene_name","rna_expression")], gene_name, rna_expression) # Spread the TPM dataset to create an expression matrix with one row per cell line and one column per gene

# Make train data file --- 
# create test file with all data (Cell name - smile - auc)
pairs_cells_name <- CCLE@sample[["CCLE.name"]][match(auc$Sample,CCLE@sample[["sampleid"]])] # Get CCLE sample names (expression file has CCLE names)
pairs_drugs_name <- compound_names$SMILE[match(auc$Drug,compound_names$Name)] # Get SMILES 

# Create a matrix with cell lines, SMILES strings, and AUC values
cell_drug_auc <- as.data.frame(cbind(pairs_cells_name,pairs_drugs_name,auc$AUCS))
cell_drug_auc <- cell_drug_auc[cell_drug_auc$pairs_cells_name %in% expression_matrix$cell_line, ] # Remove rows with no expression data 
cell_drug_auc <- na.omit(cell_drug_auc) # Remove rows with missing SMILES 
write.table(cell_drug_auc, file=paste(computer,"DataSets/test_PRISM_cells/allsamples/sparseGO_all_PRISM.csv",sep=""), append = FALSE, quote=FALSE,sep = "\t", dec = ".",row.names = FALSE, col.names = FALSE)
dim(cell_drug_auc) # 351922 from CTRPv2 -- 135993 from GDSC2

# keep only cells and drugs that are part of one or more samples 
cells_with_samples <- unique(na.omit(cell_drug_auc$pairs_cells_name))
drugs_with_samples <- unique(na.omit(cell_drug_auc$pairs_drugs_name))

# Make expression file ---
subset_expression_matrix <- expression_matrix[expression_matrix$cell_line %in% cells_with_samples, ]
write.table((round(subset_expression_matrix[,-1],7)), paste(computer,"DataSets/test_PRISM_cells/allsamples/cell2expression_allgenes_PRISM.txt",sep = ""), append = FALSE, quote=FALSE,sep = ",", dec = ".",row.names = FALSE, col.names = FALSE)

# Make cell lines file ---
cells_txt <- cbind(0:(dim(subset_expression_matrix)[1]-1),subset_expression_matrix$cell_line)
write.table(cells_txt,file = paste(computer,"DataSets/test_PRISM_cells/allsamples/cell2ind_PRISM.txt",sep = ""), append = FALSE, quote=FALSE,sep = "\t", dec = ".",row.names = FALSE, col.names = FALSE)

# Make genes file ---
genes_txt <- cbind(0:(dim(subset_expression_matrix)[2]-2),gsub(" .*$", "",colnames(subset_expression_matrix[,-1]))) # remove everything after a space in the genes names 
write.table(genes_txt,  paste(computer,"DataSets/test_PRISM_cells/allsamples/gene2ind_allgenes_PRISM.txt",sep = ""), dec = ".",row.names = FALSE, col.names = FALSE,quote = FALSE)

# Make drugs file ---
subset_compound_names <- compound_names[compound_names$SMILE %in% drugs_with_samples, ]
subset_compound_names$number <- 0:(dim(subset_compound_names)[1]-1)
write.table(subset_compound_names, paste(computer,"DataSets/test_PRISM_cells/allsamples/compound_names_PRISM.txt",sep=""), sep = "\t", row.names = FALSE, col.names = TRUE,quote = FALSE)

# MUTATIONS
drugcell_cells <- read.table(paste(computer,"drugcell_cell2ind.txt",sep=""), header = FALSE)
sum(drugcell_cells$V2 %in% samples_to_ccleid)
# Many samples are the same, keep drugcell files, only verify auc file (keep only samples with SMILE and mutations) 
cell_drug_auc_mut <- as.data.frame(cbind(pairs_cells_name,pairs_drugs_name,auc$AUCS))
cell_drug_auc_mut <- cell_drug_auc_mut[cell_drug_auc_mut$pairs_cells_name %in% intersect(drugcell_cells$V2,samples_to_ccleid), ] # Remove rows with no expression data 
cell_drug_auc_mut <- na.omit(cell_drug_auc_mut) # Remove rows with missing SMILES 
write.table(cell_drug_auc_mut, file=paste(computer,"DataSets/test_PRISM_cells_mutations/allsamples/sparseGO_all_PRISM.csv",sep=""), append = FALSE, quote=FALSE,sep = "\t", dec = ".",row.names = FALSE, col.names = FALSE)
dim(cell_drug_auc_mut) # 358879 from CTRPv2

# Make drugs file ---
drugs_with_samples <- unique(na.omit(cell_drug_auc_mut$pairs_drugs_name))
subset_compound_names <- compound_names[compound_names$SMILE %in% drugs_with_samples, ]
subset_compound_names$number <- 0:(dim(subset_compound_names)[1]-1)
write.table(subset_compound_names, paste(computer,"DataSets/test_PRISM_cells_mutations/allsamples/compound_names_PRISM.txt",sep=""), sep = "\t", row.names = FALSE, col.names = TRUE,quote = FALSE)
# DONT FORGET FINGERPRINT FILE

# # The 'mutationCalls' dataset contains merged the 22Q2 mutation calls (for coding region, germline filtered) and includes data from 18784 genes, 1771 cell lines, 33 primary diseases and 30 lineages.
# mutationCalls <- eh[["EH7557"]] 
# # keep only damaging and non conserving mutations (change codification zone)
# mutationCalls <- mutationCalls[mutationCalls$var_annotation %in% c("damaging","other non-conserving"),]
# depmap_ids <- depmap_metadata()[,c("depmap_id","cell_line")] # ccle ids
# mutationCalls <- cbind(mutationCalls,array(1, dim = c(dim(mutationCalls)[1]))) # add a 1 to mark mutation
# colnames(mutationCalls)[33] <- "has_mutation"
# mutationCalls_short <- mutationCalls[,c("depmap_id","gene_name","has_mutation")]
# mutationCalls_short <- merge(depmap_ids,mutationCalls_short, by="depmap_id")[,-1] # change depmap_id to ccle_id
# mutations_matrix <- spread(mutationCalls_short[!duplicated(mutationCalls_short), ], gene_name, has_mutation, fill = 0)
# # keep only the 3008 drugcell genes 
# drugcell_genes <- read.table(paste(computer,"drugcell_gene2ind.txt",sep=""), header = FALSE)
# mutations_matrix_drugcell <- mutations_matrix[,colnames(mutations_matrix) %in% drugcell_genes$V2]
# # Missing "LGALS7"- podria ser LGALS7B... "P2RX5"  "VAMP2" = add 3 empty columns with 0s
# mutations_matrix_drugcell <- cbind(mutations_matrix_drugcell,array(0, dim= c(dim(mutations_matrix_drugcell)[1],3)))
# colnames(mutations_matrix_drugcell)[c(3006,3007,3008)] <- drugcell_genes$V2[!(drugcell_genes$V2 %in% colnames(mutations_matrix))]
# mutations_matrix_drugcell <- mutations_matrix_drugcell[,drugcell_genes$V2] # reorder
# 
# mutations_matrix_drugcell <- cbind(mutations_matrix$cell_line,mutations_matrix_drugcell) # add cell_line name
# colnames(mutations_matrix_drugcell)[1] <- "cell_line"
# 
# mutations_matrix_drugcell <- filter(mutations_matrix_drugcell, cell_line %in% samples_to_ccleid) # 806 cell lines found CTRPv2


file <- paste(computer,"/DataSets/train_sparseGO_mutations_insilico/allsamples/compound_names_both.txt",sep="")
A_compound <- read.delim(file, header = TRUE,sep = "\t")

file <- paste(computer,"/DataSets/train_sparseGO_mutations_insilico/allsamples/drug2fingerprint_both.txt",sep="")
A_fingerprint <- read.delim(file, header = FALSE,sep = ",")
