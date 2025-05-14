# GeneralDFT
Source code to complement the paper "General Chemically Intuitive Atom- and Bond-level DFT Descriptors for Machine Learning Approaches to Reaction Condition Prediction ". All results described in the paper should be reproducible using the scripts shown here and the embeddings available on Zenodo.

# Installation

-Clone the repository:

```
git clone https://github.com/moleculebits/GeneralDFT.git
```
-Go to the main directory:

```
cd GeneralDFT
```
-Make sure you satify all the requirements in the requirements.txt file. The range of module versions you can use should be fairly flexible, but do confirm that the PyTorch version
you have installed matches the GPU/CPU you have in your system.

-If you want to test the files in a blank virtual environment, for instance a conda environemnt, you can install Python 3.9 and then use pip to download the requirements:

```
pip install -r requirements.txt
```
-Again, make sure you download the appropriate version of PyTorch.

-After that, you can go to Zenodo and download the embeddings. Modify the paths according to where you decided to install the embeddings. 

-With that set up, all scripts necessary to reproduce the paper should run smoothly provided you first train/test a model before trying to plot your own results. The bmodel.py (base fnn for sample reactions), crossval.py (cross-validation for fnn), and rfcrossval.py (cross-validation for random forests) scripts should be run as follows:

```
./python crossval.py -p "path/to/embeddings/embedding.pt" -s "4000" -t "hybrid"
```
-Where -p takes the path to the embeddings, -s the embedding size, and -t the embedding type (hybrid, struct or dft). Notice that previous versions of the Zenodo embeddings use different prefixes to indicate the embedding type. This should not lead to issues provided you do not mix the nomenclatures. In any case, we strongly recommend using the newest available version.

# Plotting

-In case you want to reproduce the main figures included in the paper, you can simply run the python script included in the plotting directory, which makes use of the files under "results" to reproduce the figures. Notice that the "results" directory only contains data immediately relevant to the plots, whereas the rest of the obtained performance metrics were added directly to the paper's Supporting Information.

# Embedding Generation

-The embedding generation process is complex and includes hardware-specific steps performed on a high-performance computing cluster, as such, the entire process could not be included in this repository. Nevertheless, we aimed to provide enough support to allow you to generate your own new embeddings and generate the ones used in our paper. The "embgen" is a work-in-progress and includes sanitized code from our workflow that has been modified to remove system information and private third-party code.

# Replicating the DFT computations

-The directory "dftsample" includes the ORCA and NBO versions used in the paper and describes the commands used for the computations. All you need to run your jobs exactly like us is to use the same version and add the same ORCA keywords to your scripts.