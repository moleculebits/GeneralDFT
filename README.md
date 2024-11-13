# GeneralDFT
Source code to complement the paper "General Chemically Intuitive Atom- and Bond-level DFT Descriptors for Machine Learning Approaches to Reaction Condition Prediction ".

# Installation

-Clone the repository:

```
git clone https://github.com/moleculebits/GeneralDFT.git
```
-go to the main directory:

```
cd GeneralDFT
```
-make sure you satify all the requirements in the requirements.txt file. The range of module versions you can use should be fairly flexible, but do confirm that the PyTorch version
you have installed corresponds to the GPU/CPU you have in your system.

-If you want to test the files in a blank virtual environment, for instance a conda environemnt, you can install Python 3.10 and then use pip to download the requirements:

```
pip install -r requirements.txt
```
-Again, make sure you download the appropriate version of PyTorch.

-After that, you can go to Zenodo and download the embeddings. Modify the paths according to where you decide to install the embeddings. After that, all scripts should run smoothly provided you first train/test a model before trying to plot results.

