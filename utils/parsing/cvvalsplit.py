#This utils file is used to generate per reaction class metrics for fnn model outputs.
#it is only needed if you wish to reproduce those results from scratch as opposed to 
# using the provided tables available in Supporting information and ./results/rxnclasses/

import os
import torch
import math
import numpy as np
import scipy.stats
import pandas as pd

def compute_mean_ci(values, mean_only=0):
    values = [float(value) for value in values if math.isnan(float(value)) != True] #handles NaNs (for classes with zero count)
    if len(values) == 0:
        return "N/A"  #Handles empty cases
    
    mean, se = np.mean(values), scipy.stats.sem(values)
    n = len(values)
    ci = se * scipy.stats.t.ppf((1 + 0.95) / 2., n-1) if n > 1 else 0
    
    if not mean_only:
        #Formats as "mean ± CI" with three decimal places
        return f"{mean:.3f} ± {ci:.3f}"
    else:
        return mean

def split_folds_and_write(val_dir, val_file, labs_dir, labs_file, output_dir):
    emb_fold = {}
    for i in range(5):
        _, truelabs, predlabs = torch.load(labs_dir+f'f{i}'+labs_file) 
        truelabs = list(map(lambda x: x[0].encode('unicode_escape').decode(), truelabs)) #changing format to list of strings
        predlabs = list(map(lambda x: x[0].encode('unicode_escape').decode(), predlabs)) 
        emb_fold[i] = pd.DataFrame({'predicted': predlabs, 'gtruth': truelabs}, dtype=str)
    chunks = {}
    current_fold = None
    current_lines = [] 

    #Ensures the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    #Splits file into fold chunks
    with open(os.path.join(val_dir, val_file), 'r') as f:
        for line in f:
            line = line.rstrip('\n')
            line.encode('unicode_escape').decode()
            line = line.replace(" ", "").replace("  ", "") #removes whitespaces and tabs
            if "predicted" in line:
                line = line.replace("predicted:['", ",").replace("']gtruth:['", ",").replace("']", "") #converts string into csv format
            if line.startswith("############FOLD"):
                if current_fold is not None:
                    chunks[current_fold] = current_lines
                current_fold = int(line.replace("############FOLD", "").replace("#############", ""))
                current_lines = []
            elif line:
                current_lines.append(line.split(","))

        if current_fold is not None:
            chunks[current_fold] = current_lines

    #Writes each fold to its own file
    for fold_number, rxnlist in chunks.items():
        df = pd.DataFrame(rxnlist, columns=['smiles', 'predicted', 'gtruth'])

        df = df.tail(len(emb_fold[fold_number]))
        df['match0'] = df['predicted'].values == emb_fold[fold_number]['predicted'].values
        df['match1'] = df['gtruth'].values == emb_fold[fold_number]['gtruth'].values
        if (not df["match0"].all() or not df["match1"].all()): #sanity check
            print('predictions in .pt file do not correspond to the .txt file given')
        else:
            output_path = os.path.join(output_dir, f"f{fold_number}"+val_file)
            df = df.drop(['match0', 'match1'], axis='columns')
            df.to_csv(output_path, index=False)

VALDIR = "./modelout"  #input dir with cvvalreacts.txt files
LABSDIR = "./modelout" #input dir with cvvlabs.pt files
OUTPUTFOLDDIR =  "./folds" #output for the chunked cvvalreacts file
RXNCLASSPATH = "./results/rxnclasses/reaction_classes.json" #path to reaction_classes json file
DESTDIR = "./output" #where to output the per reaction class metrics

if __name__ == "__main__":

    VALFILES = ['hybridcvvalreacts48-4000.txt'] #add the need reacts files to this list 
    LABSFILES = ['hybridcvvlabs48-4000.pt'] #add the needed labs files to this list

    genflag = 0
    if genflag:
        for valfile, embfile in zip(VALFILES, LABSFILES):
            split_folds_and_write(VALDIR, valfile, LABSDIR, embfile, OUTPUTFOLDDIR)

    rxnclassdf = pd.read_json(RXNCLASSPATH).transpose()
    rxnclassdf['smiles'] = rxnclassdf.index
    rxnclassdf = rxnclassdf.reset_index(drop=True)
    rxnclassdf['smiles'] = rxnclassdf['smiles'].map(lambda x: x.strip('\n').encode('unicode_escape').decode())
    rxnclassdf['reaction_classname'] = rxnclassdf['reaction_classname'].map(lambda x: x.strip('\n').encode('unicode_escape').decode())
    all_classes = rxnclassdf['reaction_classname'].unique()
    if rxnclassdf['reaction_classname'].isna().any():
        print('reaction class file contains nans!')
    else:
        supdf = pd.DataFrame(index=sorted(all_classes))
        supdf.index.name = 'reaction_classes'
        for valfile in VALFILES:
            macro_acc = []
            for i in range(5):
                df = pd.read_csv(os.path.join(OUTPUTFOLDDIR, f'f{i}'+valfile))
                df.columns = ['smiles', 'predicted', 'gtruth']
                df['smiles'] = df['smiles'].map(lambda x: x.strip('\n').encode('unicode_escape').decode())
                if not df['smiles'].isin(rxnclassdf['smiles']).all():
                    print('mismatch between reaction class file and predictions!')

                df_merged = df.merge(rxnclassdf, on='smiles', how='left')
                df_merged['correct_prediction'] = df_merged['gtruth'] == df_merged['predicted']
                df_merged['reaction_classname'] = df_merged['reaction_classname'].map(lambda x: str(x).strip('\n').encode('unicode_escape').decode())
                if df_merged['reaction_classname'].isna().any():
                    print('df merge mismatch when adding classes')

                result = df_merged.copy().groupby("reaction_classname").agg(
                    total_samples=pd.NamedAgg(column='smiles', aggfunc='size'),     # count of rows per class
                    correct_predictions=pd.NamedAgg(column='correct_prediction', aggfunc='sum')   # sum of bool values in 'flag' per class
                )
                result = result.reindex(supdf.index, fill_value=0)
                result['accuracy'] = result['correct_predictions'] / result['total_samples'].replace(0, np.nan)

                supdf[f'acc{i}'] = result['accuracy'].copy().round(3)
                supdf[f's{i}'] = result['total_samples'].copy()
                macro_acc.append(compute_mean_ci(result['accuracy'].tolist(), 1)) #macro-averaged mean accuracy

            supfile = valfile.replace('molgl2', 'combined').replace('cvvalreacts48-', 'rxnclasssup').replace('.txt', '.csv')
            os.makedirs(DESTDIR, exist_ok=True)
            supdf.to_csv(os.path.join(DESTDIR, supfile))

            cidf = pd.DataFrame({col: supdf.iloc[:, i::2].values.tolist() for i, col in enumerate(['accuracy', 'total_samples'])}, index=supdf.index)

            cidf['accuracy'] = cidf['accuracy'].apply(compute_mean_ci)
            cidf['total_samples'] = cidf['total_samples'].apply(lambda x: int(sum(x)))

            countsum = cidf['total_samples'].sum()
            print('total number of reactions: ', countsum)

            cidf.loc['macro avg'] = {'accuracy': compute_mean_ci(macro_acc)} #adds macro averaged accuracy with ci to df
            cifile = supfile.replace('sup', 'sum')
            cidf.to_csv(os.path.join(DESTDIR, cifile))
