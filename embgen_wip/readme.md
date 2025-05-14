# THIS IS A WORK IN PROGRESS!

-If you simply want to reproduce our results, you don't need this directory. You can simply download the embeddings from zenodo and use them to train the models using bmodel.py, crossval.py, and rfcrossval.py. Parsing tools to help you analyze the output are available under utils.

-This directory contains a sanitized version of the scripts we use to parse the output and generate the embeddings needed to train all models. We do not guarantee this currently version will run on your system as all files have been modified to remove details about the HPC we use and dependencies relating to private third-party code.

-We kept all the features that allow you to parse output files, generate graphs with embedded dft descriptors, train and predict using gl2vec, and generate the final embeddings.

-notice that mol2vec models were trained using command line as described in the corresponding repo. If you don't want to spend a few days training and generating the corpus, we added the weights of our trained models to Zenodo to get you started. Those can, of course, also be used for other projects.

-if you have any issues with the embgen file, ensure that your gensim version is old enough. This dependency is particularly fragile since the original repo isn't being updated anymore.