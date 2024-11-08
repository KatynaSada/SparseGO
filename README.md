<!-- PROJECT LOGO -->
<br />
<div align="center">
  <a href="https://github.com/KatynaSada/SparseGO_code">
    <img src="images/logoSparseGO.png" width="400" alt="Logo" >
  </a>

  <h3 align="center">SparseGO</h3>

  <p align="center">
    A VNN to predict cancer drug response using a sparse explainable neural network
    +
    <br />
    a method to predict the mechanism of action of drugs.
    <br />
    <a href="https://github.com/KatynaSada/SparseGO_code/issues">Request Feature</a>
  </p>

[![DOI](https://zenodo.org/badge/589525718.svg)](https://zenodo.org/badge/latestdoi/589525718)
</div>


<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="#built-with">Built With</a></li>
      </ul>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#environment-set-up-for-training-and-testing-of-sparsego">Environment set up</a></li>
      </ul>
    </li>
    <li>
      <a href="#training-and-testing-sparsego">Training and testing SparseGO</a>
      <ul>
        <li><a href="#required-input-files">Required input files</a></li>
        <li><a href="#training">Training</a></li>
        <ul>
          <li><a href="#parameters-for-training">Parameters</a></li>
          <li><a href="#running-the-training-code">Run the code</a></li>
        </ul>
        <li><a href="#testing">Testing</a></li>
        <ul>
          <li><a href="#parameters-for-predicting">Parameters</a></li>
          <li><a href="#running-the-prediction-code">Run the code</a></li>
        </ul>
      </ul>
    </li>
    <li><a href="#deepmoa-method-to-predict-the-moa-of-drugs-using-deeplift">DeepMoA</a></li>
    <li><a href="#contributing">Contributing</a></li>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#acknowledgments">Acknowledgments</a></li>
  </ol>
</details>



<!-- ABOUT THE PROJECT -->
# About The Project

 <p align="center"><img src="images/network.png" width="700" alt="Logo"></p>

 <p align="center"><a href=https://www.thelancet.com/journals/ebiom/article/PIIS2352-3964(23)00333-X/fulltext>Discovering the mechanism of action of drugs with a sparse explainable network<a></p>

Although Deep Neural Networks (DDNs) have been successful in predicting the efficacy of cancer drugs, the lack of explainability in their decision-making process is a significant challenge. Previous research (<a href="https://pubmed.ncbi.nlm.nih.gov/33096023/DrugCell">DrugCell<a>) proposed mimicking the Gene Ontology structure to allow for interpretation of each neuron in the network. However, these previous approaches require huge amount of GPU resources and hinder its extension to genome-wide models.
We developed SparseGO, a sparse and interpretable neural network, for predicting drug response in cancer cell lines and their Mechanism of Action (MoA). To ensure model generalization, we trained it on multiple datasets and evaluated its performance using three cross-validation schemes. Its efficiency allows it to be used with gene expression. In addition, SparseGO integrates an eXplainable Artificial Intelligence (XAI) technique, DeepLIFT, with Support Vector Machines to computationally discover the MoA of drugs.   

This project includes instructions for:
* making predictions using a trained SparseGO network,
* training a SparseGO network,
* and using the DeepMoA method to predict the MoA of drugs.

<p align="right">(<a href="#about-the-project">back to top</a>)</p>

### Built With

*   <a href="https://www.python.org/">
      <img src="images/python.png" width="110" alt="python" >
    </a>
*   <a href="https://pytorch.org/">
      <img src="images/pytorch.png" width="105" alt="pytorch" >
    </a>
*   <a href="http://geneontology.org/">
      <img src="images/geneontology.png" width="105" alt="geneontology" >
    </a>

<p align="right">(<a href="#about-the-project">back to top</a>)</p>

<!-- GETTING STARTED -->
# Getting Started

```diff
- The original DrugCell repository served as the basis for these instructions. I'd like to take this chance to thank the Trey Ideker Lab at UCSD!
```

* <a href="https://github.com/idekerlab/DrugCell">DrugCell<a>
* <a href="https://github.com/idekerlab">Trey Ideker Lab<a>

## Prerequisites
SparseGO training/testing scripts require the following environmental setup:

* Recommended hardware
    * GPU server with CUDA installed (can also run in non-CUDA devices)

* Software
    * Python = 3.9
    * Miniconda or anaconda
        * Relevant information for installing Anaconda can be found in: https://docs.conda.io/projects/conda/en/latest/user-guide/install/.
    * Weights & Biases · MLOps platform (only for training)
      * Create wandb free account at <a href="https://wandb.ai/site">https://wandb.ai/site<a>

## Environment set up for training and testing of SparseGO
**1.** Create environment called SparseGO which uses python=3.9
  ```angular2
  conda create -p PATH_TO_SAVE_ENVIRONMENT/SparseGO python=3.9
  ```

**2.** Activate environment
  ```angular2
  conda activate PATH_TO_SAVE_ENVIRONMENT/SparseGO
  ```      
**3.** Install the cuda-toolkit (other versions= https://anaconda.org/nvidia/cuda-toolkit), don't necessarily choose the most recent version, make sure that it is a version compatible with PyTorch Sparse and Pytorch.
  ```angular2
  conda install -c "nvidia/label/cuda-11.8.0" cuda-toolkit
  ```
  * Tip: to verify your CUDA version ```nvcc --version```

**4.** Install PyTorch and PyTorch Sparse in the created SparseGO environment

  * Depending on the **specifications of your machine and cuda-toolkit installed**, run appropriate command to install PyTorch.
  The installation command line can be found in https://pytorch.org/get-started/locally/ or https://pytorch.org/get-started/previous-versions/.

   For cuda 11.8 and linux/windows OS...
   ```angular2
  pip install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.8 -c pytorch -c nvidia
  ```
**5.** After installing PyTorch install <a href="https://pypi.org/project/torch-sparse/">PyTorch Sparse<a>, to install the binaries for PyTorch simply run (make sure you install it for the correct PyTorch version)...
  ```angular2
  pip install torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.0.1+cu118.html
  ```

  For other versions...
  ```angular2
  pip install torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-${PYTORCH_VERSION}+${CUDA}.html and
  ```
  where ```${PYTORCH_VERSION}``` should be replaced by your version of PyTorch and ```${CUDA}``` should be replaced by ```cpu```, ```cu116```, ```cu117```, etc. depending on your PyTorch installation.

* Also install the following packages...
  ```angular2
  pip install pandas
  pip install wandb
  pip install mygene
  pip install obonet
  pip install scipy
  pip install matplotlib
  ```

* After setting up the conda virtual environment, make sure to activate environment before executing SparseGO scripts.
    When training or testing using the bash scripts provided (_train_wb.sh_ or _test.sh_), there's no need to run this as the example bash scripts already have the command line.
  ```angular2
  source activate SparseGO
  ```
<p align="right">(<a href="#about-the-project">back to top</a>)</p>

# Training and testing SparseGO
- The files needed to train the models are at https://gitlab.com/katynasada/sparsego_data and http://drugcell.ucsd.edu/downloads/.

## Required input files
Required input files:
1. Cell feature files:
    * **_gene2ind.txt_**: Gene to ID mapping file, a tab-delimited file where the 1st column is index of genes and the 2nd column is the symbol of the genes
    * **_cell2ind.txt_**: Cell to ID mapping file, a tab-delimited file where the 1st column is index of cells and the 2nd column is the name of cells (genotypes).
    * **_cell2mutation.txt_** OR **_cell2expression.txt_**: choose the file depending on whether you want to train with mutations or with expression
      * _cell2mutation.txt_: a comma-delimited file where each row has 3,008 binary values indicating each gene is mutated (1) or not (0).
        OR
      * _cell2expression.txt_: a comma-delimited file where each row has 15014 values indicating the expression of 15014 genes.
        * The script to download the expression data and create the file is provided in _extra_ folder (_get_expression_matrix.R_). It requires the _cell2ind.txt_ file.

        *The column index of each gene should match with those in _gene2ind.txt_ file. The line number should
      match with the indices of cells in _cell2ind.txt_ file.

2. Drug feature files:
    * **_drug2ind.txt_**: a tab-delimited file where the 1st column is index of drug and the 2nd column is
    identification of each drug (e.g., SMILES representation or name). The identification of drugs
    should match to those in _drug2fingerprint.txt_ file.
    * **_drug2fingerprint.txt_**: a comma-delimited file where each row has 2,048 binary values which would form
    , when combined, a Morgan Fingerprint representation of each drug.
    The line number should match with the indices of drugs in _drug2ind.txt_ file.

3. Training data file: **_sparseGO_train.txt_** / **_drugcell_train.txt_**
    * A tab-delimited file containing all data points that you want to use to train the model.
        The 1st column is identification of cells (genotypes), the 2nd column is identification of
        drugs and the 3rd column is an observed drug response in a floating number.

4. Validation data file: **_sparseGO_val.txt_** / **_drugcell_val.txt_**
    * A tab-delimited file that in the same format as the training data.

5. Test data file: **_sparseGO_test.txt_** / **_drugcell_test.txt_**
    * A tab-delimited file containing all data points that you want to estimate drug response for. A tab-delimited file that in the same format as the training data.

6. Ontology (hierarchy) file: **_sparsego_ont.txt_** / **_drugcell_ont.txt_**
    * A tab-delimited file that contains the ontology (hierarchy) that defines the structure of a branch
          of the model that encodes the genotypes. The first column is always a term (subsystem or pathway) or a gene,
          and the second column is another term or a gene.
          The third column should be set to "default" when the line represents a link between terms, and
          "gene" when the line represents an annotation link between a term and a gene.
          The following is an example describing a sample hierarchy.
          ![](https://github.com/idekerlab/DrugCell/blob/master/misc/drugcell_ont_image_sample.png)
          ```angular2

           GO:0045834	GO:0045923	default
           GO:0045834	GO:0043552	default
           GO:0045923	AKT2	gene
           GO:0045923	IL1B	gene
           GO:0043552	PIK3R4	gene
           GO:0043552	SRC	gene
           GO:0043552	FLT1	gene
          ```

        * _drugcell_ont.txt_ is the file used to create DrugCell and the SparseGO mutation models.
        * _sparsego_ont.txt_ is the file used to create the SparseGO expression model.
          * The script to create the ontology file is provided in _extra_ folder (_get_gene_hierarchy.py_). It requires the _gene2ind.txt_ and _cell2expression.txt_ files.

<p align="right">(<a href="#about-the-project">back to top</a>)</p>

## Training

The code used to train the model (<a href="https://github.com/KatynaSada/SparseGO_code/blob/main/code/train_gpu_wb.py">_code/train_gpu_wb.py_<a>) also uses the top-performing model to predict using the test data. Three files containing train, validation, and test data (cell line-drug pairs-AUC) must be provided.


### Parameters for training

There are a various optional parameters that you can provide in addition to the input files:

  1. **-epoch**: the number of epoch to run during the training phase (type=int).

  2. **-lr**: the learning rate (type=float).

  3. **-decay_rate**: the learning decay rate (type=float).

  4. **-batchsize**: the number of training examples in one forward/backward pass. The higher the batch size, the more memory space you'll need (type=int).

  5. **-modeldir**: the path of the folder to store trained models (type=str).

  6. **-cuda_id**: GPU ID (type=int).

  7. **-number_neurons_per_GO**: Mapping for the number of neurons in each term in genotype parts (type=int).

  8. **-Mapping for the number of neurons in the root term**: Mapping for the number of neurons in each term in genotype parts (type=int).

  9. **-drug_neurons**: Mapping for the number of neurons in each layer (type=str).

  10. **-final_neurons**: Number of neurons before the output and after concatenating (type=int).

  11. **-result**: the path of the folder to store the results of the predictions of test (type=str).

  12. **-project**: wandb project name that you want to use (type=str).

  13. **-gpu_name**: not important, just a reminder of which GPU you used (type=str).

### Running the training code
**1.** Activate SparseGO environment

**2.** Login to wandb account

```angular2
pip install wandb # If is not installed
wandb login APIKEY # APIKEY should be replaced by your API key.
```
**3.** In <a href="https://github.com/KatynaSada/SparseGO_code/blob/main/code/train_gpu_wb.py">train_gpu_wb.py_<a> change the entity parameter of the line 402 to your wandb username.
```angular2
sweep_id = wandb.sweep(sweep_config, entity="YOUR_USERNAME", project=opt.project)
```


**4.** Make folders for the models and the test results (you can use the same folder)

**5.** Finally, to train AND test the SparseGO model, execute a command line similar to the example provided in <a href="https://github.com/KatynaSada/SparseGO_code/blob/main/cluster/train_wb.sh">_cluster/train_wb.sh_<a>:


```diff
python -u code/train_gpu_wb.py -onto data/toy_example/samples1/drugcell_ont.txt -gene2id data/toy_example/samples1/gene2ind.txt -drug2id data/toy_example/samples1/drug2ind.txt -cell2id data/toy_example/samples1/cell2ind.txt -train data/toy_example/samples1/drugcell_train.txt -val data/toy_example/samples1/drugcell_val.txt -modeldir results/toy_example/samples1/ -cuda 0 -genotype data/toy_example/samples1/cell2mutation.txt -fingerprint data/toy_example/samples1/drug2fingerprint.txt -number_neurons_per_GO 6 -number_neurons_per_final_GO 6 -drug_neurons '100,50,6' -final_neurons 6 -epoch 100 -batchsize 5000 -lr 0.1 -decay_rate 0.001 -predict data/toy_example/samples1/drugcell_test.txt -result results/toy_example/samples1/ -project SparseGO_toy_example -gpu_name my_GPU  > results/toy_example/samples1/train.log

- This is a command for running a Python script, train_gpu_wb.py, with various command-line arguments. The -u option specifies to run the script in unbuffered mode, which means that output will be written to the terminal immediately, rather than being buffered.
- The remaining arguments provide values for the various options that the script requires.
```

## Testing
The code <a href="https://github.com/KatynaSada/SparseGO_code/blob/main/code/predict_gpu.py">_code/predict_gpu.py_<a> makes predictions using a pre-trained SparseGO network.

### Parameters for predicting
There are a few optional parameters that you can provide in addition to the input files:

  1. **-batchsize**: the number of training examples in one forward/backward pass. The higher the batch size, the more memory space you'll need (type=int).

  2. **-load**: the path of the pre-trained model (type=str).

  3. **-result**: the path of the folder where you want to store the results (type=str).

  4. **-cuda**: GPU ID (type=int).

### Running the prediction code
**1.** Activate SparseGO environment

**2.** Make folder for test results

**4.** Finally, to test the SparseGO model, execute a command line similar to the example provided in <a href="https://github.com/KatynaSada/SparseGO_code/blob/main/cluster/test.sh">_cluster/test.sh_<a>:

```angular2

python -u code/predict_gpu.py -gene2id data/toy_example/samples1/gene2ind.txt
                              -cell2id data/toy_example/samples1/cell2ind.txt
                              -drug2id data/toy_example/samples1/drug2ind.txt
                              -genotype data/toy_example/samples1/cell2mutation.txt
                              -fingerprint data/toy_example/samples1/drug2fingerprint.txt
                              -result results/toy_example/samples1/
                              -predict data/toy_example/samples1/cell2ind.txt
                              -load results/toy_example/samples1/best_model_p.pt
                              -cuda 0 > results/toy_example/samples1/test.log

```

<p align="right">(<a href="#about-the-project">back to top</a>)</p>

# DeepMoA method to predict the MoA of drugs using DeepLIFT
The method to predict the MoA of drugs is described in the next notebook...
<a href="https://katynasada.github.io/DeepMoA_SparseGO/">_DeepMoA_<a>

It requires the following packages for algorithm execution and graphing...
  ```angular2
  pip install seaborn
  pip install captum
  pip install ipywidgets
  pip install plotly
  pip install openpyxl
  pip install -U scikit-learn
  pip install nbformat==4.2.0
  pip install ipykernel
  pip install -U kaleido
  pip install colorlover
  ```


<!-- CONTRIBUTING -->
# Contributing

Any contributions you make are **greatly appreciated**.

If you have a suggestion that would make this better, please fork the repo and create a pull request. You can also simply open an <a href="https://github.com/KatynaSada/SparseGO_code/issues">issue</a> with the tag "enhancement".
Don't forget to give the project a star! Thanks again!

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

<p align="right">(<a href="#about-the-project">back to top</a>)</p>

<!-- CONTACT -->
# Contact

Katyna Sada - ksada@unav.es - ksada@tecnun.es

<p align="right">(<a href="#about-the-project">back to top</a>)</p>


<!-- ACKNOWLEDGMENTS -->
# Acknowledgments
* [DrugCell](https://github.com/idekerlab/DrugCell)
* [Weights & Biases](https://www.wandb.ai/)
* [Sparse Linear](https://github.com/hyeon95y/SparseLinear)

<p align="right">(<a href="#about-the-project">back to top</a>)</p>

<!-- MARKDOWN LINKS & IMAGES -->
