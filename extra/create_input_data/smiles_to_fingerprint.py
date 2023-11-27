import rdkit
from rdkit import Chem
from rdkit.Chem import AllChem

import numpy as np
mac = "/Users/katyna/Library/CloudStorage/OneDrive-Tecnun/Tesis/Codigo/AUC/"
windows = "C:/Users/ksada/OneDrive - Tecnun/Tesis/Codigo/AUC/"
computer = windows # CHANGE

dataset = "PRISM"
file_handle = open(computer+'DataSets/test_'+dataset+"_cells_mutations"+'/allsamples/compound_names_'+dataset+'.txt') # function opens a file, and returns it as a file object.
compounds = {} # dictionary of values on required txt
for line in file_handle:
    line = line.rstrip().split() # quitar espacios al final del string y luego separar cada elemento
    compounds[line[0]] = line[1] # save drug index (to have the same order of the fingerprints file) and smiles 

file_handle.close()

del compounds[next(iter(compounds))] # delete first element, col names

fingerprints = []
for i in compounds.keys():
    m1 = Chem.MolFromSmiles(compounds[i]) # glucose 
    fp1 = AllChem.GetMorganFingerprintAsBitVect(m1,2,nBits=2048) # radious = 2
    finger = fp1.ToList()
    fingerprints.append(finger)

fingerprints_np = np.array(fingerprints, dtype=float)
    
savedir = computer+"DataSets/test_"+dataset+"_cells_mutations"+"/allsamples/drug2fingerprint_"+dataset+".txt" # CHANGE
np.savetxt(savedir, fingerprints_np, delimiter=',',fmt='%.0f')

# save SMILE with index
compounds_indices =  np.array(list(compounds.items()))
np.savetxt(computer+"DataSets/test_"+dataset+"_cells_mutations"+"/allsamples/drug2ind_"+dataset+".txt", compounds_indices, delimiter='\t', fmt='%s')

# compare files
drug_features = np.genfromtxt("C:/Users/ksada/OneDrive - Tecnun/SparseGO_code/data/toy_example/drug2fingerprint.txt", delimiter=',')
drug_features2 = np.genfromtxt(savedir, delimiter=',')