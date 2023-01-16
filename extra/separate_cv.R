mac <- "/Users/katyna/Library/CloudStorage/OneDrive-Tecnun/"
windows <- "C:/Users/ksada/OneDrive - Tecnun/"
computer <- windows

file <- "/Users/ksada/OneDrive - Tecnun/my_code/data/data_expression_all_genes_larger_graph/drugcell_all.csv"
a <- read.csv(file, header = FALSE,sep = ";")


spec = c(group1 = 0.2, group2 = 0.2, group3 = 0.2, group4=0.2, group5=0.2)

g = sample(cut(seq(nrow(a)), 
               nrow(a)*cumsum(c(0,spec)),
               labels = names(spec)
))

res = split(a, g)

train <- rbind(res[["group1"]],res[["group2"]],res[["group3"]],res[["group4"]])
nrow(train)
validate <- train[1:5000,]
train <- train[5001:nrow(train),]
nrow(train)
test <- res[["group5"]]

outputdir <- paste(computer,"my_code/data/cross_validation_expression/",sep="")

write.table(train, file = paste(outputdir,"samples5/drugcell_train.txt",sep=""), sep = "\t", row.names = F, col.names=F, quote = FALSE)
write.table(test, file =paste(outputdir,"samples5/drugcell_test.txt",sep=""), sep = "\t", row.names = F, col.names=F, quote = FALSE)
write.table(validate, file = paste(outputdir,"samples5/drugcell_val.txt",sep=""), sep = "\t", row.names = F, col.names=F, quote = FALSE)

# 203.718 (152788 + 50929)
dim(train)[1]+dim(validate)[1]

dim(train)[1]
dim(validate)[1]
dim(test)[1]

