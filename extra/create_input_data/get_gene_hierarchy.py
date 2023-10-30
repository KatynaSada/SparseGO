"""
    This script creates the hierarchy of genes (genes were organized into a hierarchy of nested gene sets, representing cellular subsystems at different scales, based on terms extracted from the GO Biological Process hierarchy).

    Description:
        1. The annotation data from the list of provided genes is downloaded.
        2. The complete gene ontology network is downloaded from the .obo file.
        3. The network is pruned based on the following conditions: (different from DrugCell)
            a) All terms on the network must have at least "n" directly annotated genes.
            b) All terms must have more than "m" different annotated genes in comparison to each of their children (parents must be different from their children).
            c) The network can only have "p" parent-child relations above the bottom layer subsystems.
"""
n = 5 # lower n to have more specific GO terms 
m = 10 # it doesn't matter that m is larger than n, the terms connected to the genes (in the first layer) have no children
p = 8

import itertools
import mygene
import pandas as pd
import numpy as np
import obonet
import networkx as nx
import math
import sys # we require code from other folders
# Build hierarchy
mac = "/Users/katyna/Library/CloudStorage/OneDrive-Tecnun/"
windows = "C:/Users/ksada/OneDrive - Tecnun/"
computer = windows # CHANGE

sys.path.append(computer + "SparseGO_code/code")
import util
from util import *

# 1. Find gene-term pairs ---------
dataset_create = "GDSC2"
path = "C:/Users/ksada/OneDrive - Tecnun/Tesis/Codigo/AUC/DataSets/train_"+dataset_create+"/allsamples/"

genes_file = path+"gene2ind_allgenes_"+dataset_create+".txt"

gene2id_mapping = load_mapping(genes_file) 

# dG, terms_pairs, genes_terms_pairs = load_ontology(path+"sparseGO_ont_"+dataset_create+".txt", gene2id_mapping)

mg = mygene.MyGeneInfo()
# Change gene symbol/name/alias to entrezgene id
genes_ids = mg.querymany(gene2id_mapping.keys(), scopes='symbol', species="human",fields="entrezgene",as_dataframe=True)
# genes_ids["query"] # has the genes symbols
genes_ids.reset_index(level=0, inplace=True) # remove row name (change to column)
genes_ids.dropna(subset=['entrezgene'], inplace=True) # remove genes with nans in entrezgene
genes_ids.drop_duplicates(subset=['query'], inplace=True) # some input query terms found dup hits, remove duplicates

# Get the annotations of the genes (split them in 2 because sometimes it crashes, Bad Gateway error)
# Define the number of genes and the split point
total_genes = len(genes_ids['entrezgene'])
split_point = math.ceil(total_genes / 2)

# Split the gene IDs into two halves 
first_half_genes = genes_ids['entrezgene'][:split_point]
second_half_genes = genes_ids['entrezgene'][split_point:]

# Query for the first half of the genes
first_half_annotations = mg.getgenes(first_half_genes, fields='symbol,go.BP.id', as_dataframe=True)
# Query for the second half of the genes
second_half_annotations = mg.getgenes(second_half_genes, fields='symbol,go.BP.id', as_dataframe=True)

# Merge the resulting dataframes
genes_annotations = pd.concat([first_half_annotations, second_half_annotations])
genes_annotations["symbol_ori"] = genes_ids["query"].values # change to the symbol used on the expression matrix (should be the same one, but just to make sure)

# Create go-gene dataframe of BP processes
gene_go = []
for gene,row in genes_annotations.iterrows():
    annotations = row["go.BP"] # can be modified to MF or CC
    if isinstance(annotations,list): # Ensures that further processing is only performed if the value is a list.
        terms = [item for sublist in annotations for item in sublist.values()] 
        pairs = itertools.product(terms,list([row["symbol_ori"]])) # Generates all possible combinations (Cartesian product) between the values in terms and a list containing the value from the "symbol" 
        result = [pair for pair in pairs if pair[0] != pair[1]]
        gene_go.append(result)
    elif isinstance(row["go.BP.id"],str): # if it only has 1 GO term, it is stored in go.BP.id
        gene_go.append([(row["go.BP.id"],row["symbol_ori"])])    
        
# flatten list
gene_go = np.array([item for sublist in gene_go for item in sublist])
gene_go = pd.DataFrame(gene_go).drop_duplicates()
# gene_go["type"] = ["gene"]*len(gene_go)


# 2. Download all the gene ontology network --------------

# Read the Gene Ontology ------
url = 'http://purl.obolibrary.org/obo/go/go-basic.obo'
full_graph = obonet.read_obo(url)
full_graph = full_graph.reverse() # change the direction of nodes
[n for n in full_graph.nodes if full_graph.in_degree(n) == 0] # graph contains the 3 roots (BP,MF,CC)

# 3. Define our hierarchy by pruning ------ 

# Alternative #2: Create the full graph and remove the unwanted terms (non annotated terms, terms with less than # of annotations, terms that don't meet a criterion)
def remove_node(g, node):
    """
    Removes the node and connects edges (connects its parents with its childs)

        Parameters
        ----------
        g: A directed graph class that can store multiedges.

        node: str, name of term to remove

        Output
        ------
        g: The directed graph without the unwanted node

    """
    sources = [source for source, _ in g.in_edges(node)] # parents
    targets = [target for _, target in g.out_edges(node)] # children
    new_edges = itertools.product(sources, targets)
    new_edges = [(source, target) for source, target in new_edges if source != target] # remove self-loops
    g.add_edges_from(new_edges) # add new connections
    g.remove_node(node) # remove unwanted term

    return g

# 3.1. Remove non-annotated terms
all_nodes = set(full_graph.nodes()) # All nodes of the obo graph
keep_nodes = set(gene_go[0]) # Annotated nodes

# some terms are obsolete and have no parents, this can cause problems when creating the hierarchy (can't create all connections)
keep_nodes_aux = keep_nodes.copy()
for i in keep_nodes_aux:
    num_parents = len([source for source, _ in full_graph.in_edges(i)])
    if num_parents==0:
        # print(i)
        keep_nodes.remove(i)

keep_nodes.add("GO:0008150")

unwanted_nodes = all_nodes - keep_nodes

our_graph = full_graph.copy()

for term in unwanted_nodes:
    # print(term)
    remove_node(our_graph, term)

[n for n in our_graph.nodes if len(our_graph.in_edges(n)) == 0]  # roots HEREEE

# 3.2. Add the gene nodes
gene_go_list = list(gene_go.itertuples(index=False, name=None))
our_graph.add_edges_from(gene_go_list)

[n for n in our_graph.nodes if len(our_graph.in_edges(n)) == 0]

our_graph_copy = our_graph.copy()
# remove nodes that have genes but no parents (again, just to make sure), or genes whose annotations are not in the ontology (obsolete terms)
for node in our_graph_copy.nodes:
    if len(our_graph.in_edges(node)) == 0 and node != "GO:0008150":# if node has genes but no parents
        print(node)
        remove_node(our_graph, node)

del our_graph_copy # delete

[n for n in our_graph.nodes if len(our_graph.in_edges(n)) == 0]

# 3.3. Prune the graph, stay only with terms that meet the following conditions
#   a) Select the go terms with more than "n" annotated genes, delete those that have less than "n" genes
#   b) Verify terms are different from all children terms (more than m genes more than any child)

# Build list to see the actual level on the graph of each node
our_graph_copy = our_graph.copy() #  Copy the graph to avoid modifying the original
level_list = []   # level_list stores the elements on each level of the hierarchy

while True:
    leaves = [n for n in our_graph_copy.nodes() if our_graph_copy.out_degree(n) == 0] # looks for nodes with no children
    if len(leaves) == 0: break

    level_list.append(leaves) # add the terms on each level
    our_graph_copy.remove_nodes_from(leaves) # remove nodes with no children from graph, now the bottom terms are different

del our_graph_copy # delete empty variable
del leaves # delete empty variable

for terms_to_check in level_list[1:len(level_list)]: # Go through each level (excluding the genes level) to check if terms meet the conditions
    for node in terms_to_check: # Check each node
        genes = [] # store the annotated genes of term
        children = [] # store the children of the term

        for parent, child in our_graph.out_edges(node): # check all of its descendents
            if child[0:3]=="GO:": # if the node is a go term
                children.append(child)
                children = list(set(children))
            else:  # if not it is a gene
                genes.append(child)
                genes = list(set(genes))

        if len(genes)<n: # if the amount of genes is less than n, the term is deleted
            remove_node(our_graph, node)


        elif len(children)>0: # if the term has children (not only genes)
            for node_child in children:
                gene_counter=[] # stores the genes of a child
                for parent, child in our_graph.out_edges(node_child): #check all descendents of each child
                    if child[0:3]!="GO:":
                        gene_counter.append(child) # append genes
                # if there are less than m different genes between the parent and the child the parent is removed
                if len(set(genes) - set(gene_counter)) < m :
                    remove_node(our_graph, node)
                    break

[n for n in our_graph.nodes if len(our_graph.in_edges(n)) == 0]  # roots


# Observe the graph levels in a list...
our_graph_copy = our_graph.copy() #  Copy the graph to avoid modifying the original
level_list_pruned = []   # level_list stores the elements on each level of the hierarchy

while True:
    leaves = [n for n in our_graph_copy.nodes() if our_graph_copy.out_degree(n) == 0]
    if len(leaves) == 0: break

    level_list_pruned.append(leaves) # add the terms on each level
    our_graph_copy.remove_nodes_from(leaves)

del our_graph_copy
del leaves

#   c) Remove all subsystems more than "p" parent-child relations above the bottom layer subsystems (subsystems without any children)
for level in level_list_pruned[p+1:len(level_list_pruned)-1]:
    for term in level:
        remove_node(our_graph, term)

# Observe the final graph levels in a list...
our_graph_copy = our_graph.copy() #  Copy the graph to avoid modifying the original
level_list_final = []   # level_list stores the elements on each level of the hierarchy

while True:
    leaves = [n for n in our_graph_copy.nodes() if our_graph_copy.out_degree(n) == 0]
    if len(leaves) == 0: break

    level_list_final.append(leaves) # add the terms on each level
    our_graph_copy.remove_nodes_from(leaves)

del our_graph_copy
del leaves

# verify graph only has one root
import networkx.algorithms.components.connected as nxacc
uG = our_graph.to_undirected() # Returns an undirected representation of the digraph
connected_subG_list = list(nxacc.connected_components(uG)) #connected components
[n for n in our_graph.nodes if len(our_graph.in_edges(n)) == 0]  # roots


# Create go-gene/go dataframe and export to txt
edges = np.array(list(our_graph.edges())) # convert edges to a NumPy array
edges = np.unique(edges, axis=0) # drop duplicates
empty_col = np.empty((edges.shape[0], 1)) # add empty col
edges = np.hstack((edges, empty_col))
mask = np.char.startswith(edges[:,1], 'GO:') # create a mask for rows that start with 'GO:'
edges[:,2] = np.where(mask, 'default', 'gene') # create a new column based on the mask
#np.savetxt(r'/Users/katyna/Desktop/ont.txt', edges.values)

edges_genes = edges[edges[:, 2] == 'gene']

#edges.to_csv(path+"OneDrive - Tecnun/Tesis/Codigo/GeneOntology/lincs_ont_7.txt", sep='\t', index=False, header=False)

pd.DataFrame(edges).to_csv(path+"sparseGO_ont_"+dataset_create+".txt", sep='\t', index=False, header=False)

# modify txt documents, some genes are not annotated
expression = pd.read_csv(path+"cell2expression_allgenes_"+dataset_create+".txt", header = None)

keep_genes = sorted(set(edges_genes[:,1]))
keep_genes_idx = [gene2id_mapping.get(key) for key in keep_genes]

genes_new_mapping = pd.DataFrame(list(keep_genes))
expression = expression.iloc[:,keep_genes_idx]

# replace expression file
np.savetxt(path+"cell2expression_"+dataset_create+".txt", expression.values,fmt='%1.7f',delimiter=",")

# replace gene2ind file
genes_new_mapping["index"]=genes_new_mapping.index
np.savetxt(path+"gene2ind_"+dataset_create+".txt",genes_new_mapping.iloc[:,[1,0]].values,delimiter="\t", fmt="%s")


# CHANGE MORE DATASETS (so that they all have the same input terms and ontology)
dataset_change = "GDSC1"
path = "C:/Users/ksada/OneDrive - Tecnun/Tesis/Codigo/AUC/DataSets/train_"+dataset_change+"/allsamples/"

pd.DataFrame(edges).to_csv(path+"sparseGO_ont_"+dataset_change+".txt", sep='\t', index=False, header=False)

# modify txt documents, some genes are not annotated
expression = pd.read_csv(path+"cell2expression_allgenes_"+dataset_change+".txt", header = None)

keep_genes = sorted(set(edges_genes[:,1]))
genes_file = path+"gene2ind_allgenes_"+dataset_change+".txt"
gene2id_mapping = load_mapping(genes_file) 
keep_genes_idx = [gene2id_mapping.get(key) for key in keep_genes]

genes_new_mapping = pd.DataFrame(list(keep_genes))
expression = expression.iloc[:,keep_genes_idx]

# replace expression file
np.savetxt(path+"cell2expression_"+dataset_change+".txt", expression.values,fmt='%1.7f',delimiter=",")

# replace gene2ind file
genes_new_mapping["index"]=genes_new_mapping.index
np.savetxt(path+"gene2ind_"+dataset_change+".txt",genes_new_mapping.iloc[:,[1,0]].values,delimiter="\t", fmt="%s")

# CHANGE MORE DATASETS (so that they all have the same input terms and ontology)
dataset_change = "CTRPv2"
path = "C:/Users/ksada/OneDrive - Tecnun/Tesis/Codigo/AUC/DataSets/train_"+dataset_change+"/allsamples/"

pd.DataFrame(edges).to_csv(path+"sparseGO_ont_"+dataset_change+".txt", sep='\t', index=False, header=False)

# modify txt documents, some genes are not annotated
expression = pd.read_csv(path+"cell2expression_allgenes_"+dataset_change+".txt", header = None)

keep_genes = sorted(set(edges_genes[:,1]))
genes_file = path+"gene2ind_allgenes_"+dataset_change+".txt"
gene2id_mapping = load_mapping(genes_file) 
keep_genes_idx = [gene2id_mapping.get(key) for key in keep_genes]

genes_new_mapping = pd.DataFrame(list(keep_genes))
expression = expression.iloc[:,keep_genes_idx]

# replace expression file
np.savetxt(path+"cell2expression_"+dataset_change+".txt", expression.values,fmt='%1.7f',delimiter=",")

# replace gene2ind file
genes_new_mapping["index"]=genes_new_mapping.index
np.savetxt(path+"gene2ind_"+dataset_change+".txt",genes_new_mapping.iloc[:,[1,0]].values,delimiter="\t", fmt="%s")

# IMPORTING FILES!!
# CHANGE MORE DATASETS (so that they all have the same input terms and ontology)
edges = pd.read_csv("C:/Users/ksada/OneDrive - Tecnun/Tesis/Codigo/AUC/DataSets/train_sparseGO/allsamples/sparseGO_ont_both.txt",sep="\t", header = None)
edges_genes = pd.read_csv("C:/Users/ksada/OneDrive - Tecnun/Tesis/Codigo/AUC/DataSets/train_sparseGO/allsamples/gene2ind_both.txt",sep=" ", header = None)

dataset_change = "PRISM"
path = "C:/Users/ksada/OneDrive - Tecnun/Tesis/Codigo/AUC/DataSets/test_"+dataset_change+"_insilico"+"/allsamples/"

pd.DataFrame(edges).to_csv(path+"sparseGO_ont_"+dataset_change+".txt", sep='\t', index=False, header=False)

# modify txt documents, some genes are not annotated
expression = pd.read_csv(path+"cell2expression_allgenes_"+dataset_change+".txt", header = None)

keep_genes = sorted(set(edges_genes.iloc[:,1]))
genes_file = path+"gene2ind_allgenes_"+dataset_change+".txt"
gene2id_mapping = load_mapping(genes_file) 
keep_genes_idx = [gene2id_mapping.get(key) for key in keep_genes]

genes_new_mapping = pd.DataFrame(list(keep_genes))
expression = expression.iloc[:,keep_genes_idx]

# replace expression file
np.savetxt(path+"cell2expression_"+dataset_change+".txt", expression.values,fmt='%1.7f',delimiter=",")

# replace gene2ind file
genes_new_mapping["index"]=genes_new_mapping.index
np.savetxt(path+"gene2ind_"+dataset_change+".txt",genes_new_mapping.iloc[:,[1,0]].values,delimiter="\t", fmt="%s")



# export the names of the GO terms used --------
output_file = open('go_terms.txt', 'w')

# Iterate over nodes in the graph and write ID and name pairs to file
for node in keep_nodes:
    go_id = node
    go_name = full_graph.nodes[node]['name']
    output_file.write(f"{go_id}\t{go_name}\n")

# Close the output file
output_file.close()