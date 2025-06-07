# model outputs

-This directory contains a sample of cross validation model output (excluding model weights due to size). The file "hybridcvvalreacts48-4000.txt" is a truncated example of the actual output file obtained when running the crossval.py script with the hybrid4000 embedding. The file was truncated due to its large size and should NOT be opened using a text editor, rather it is used for debugging and labeling reactions through the cvvalsplit.py script provided under ./utils. 
 
-All other files included are actual model outputs. The file was truncated due to its large size. "cvvalreacts" files should NOT be opened using a text editor, rather they are used for debugging and labeling reactions through the cvvalsplit.py script provided under ./utils. 

-When reproducing the primary results of the paper (from your own model outputs), only "cvvlabs.pt", "cvtrain.xlsx" and "hybridcvval.xlsx" files are required. "cvvalreacts" files are only needed to label reactions (with reaction classes) if you wish to do so. They are not required to plot any of the figures.

-To reproduce additional results and the performance summary included in the Supporting Information, you can parse the "cvvlabs.pt", "cvtrain.xlsx" (or .csv) and "hybridcvval.xlsx" files, using the ./utils/parsing/cvmetrics.py script.
