#This utils file is used to convert the corresponding .csv and .pt model outputs into performance reports
#it is only needed if you wish to reproduce the results from scratch.

import torch
import numpy as np
import scipy.stats
import pandas as pd

from sklearn.metrics import classification_report
 
DIR_PATH = './modelout/' #path to the model output

#please change the output directories as needed!

def compute_mean_ci(values): #this works because we ensure that all folds contain all classes
    if len(values) == 0:
        return "N/A"  # Handle empty cases
    
    mean, se = np.mean(values), scipy.stats.sem(values)
    n = len(values)
    ci = se * scipy.stats.t.ppf((1 + 0.95) / 2., n-1) if n > 1 else 0
    
    return f"{mean:.3f} ± {ci:.3f}"

reports = {}
dfs     = []
accs    = []
for i in range(5):
    classes, truelabs, predlabs = torch.load(DIR_PATH+f'f{i}hybridcvvlabs48-4000.pt')
    report = classification_report(truelabs, predlabs, labels=classes, output_dict=True, digits=6)
    report = pd.DataFrame(report).transpose()
    newdf = report[['support', 'precision']].copy()
    report.rename(columns = {'precision': f'precision{i+1}', 'recall': f'recall{i+1}', 'f1-score': f'f1-score{i+1}', 'support': 'support'}, inplace=True)
    reports[f'fold{i}'] = report

    newdf['precision'] = newdf['precision'].map('{:.3f}'.format)
    newdf['support'] = newdf['support'].map(int)
    newdf.rename(columns = {'precision': f'acc{i+1}', 'support': f's{i+1}'}, inplace=True)
    newdf.drop(newdf.tail(3).index, inplace=True)
    accs.append(newdf)

    report['mask'] = report['support'].apply(bool)
    dfs.append(report.loc[report['mask'], report.columns.difference(['mask', 'support'])])

accs = pd.concat(accs, axis=1, sort=False)
combined_df = pd.concat(dfs, axis=1, sort=False)
num_cols = len(dfs[0].columns)

grouped_df = pd.DataFrame({col: combined_df.iloc[:, i::3].values.tolist() for i, col in enumerate(['f1-score', 'precision', 'recall'])}, index=combined_df.index)

ci_df = grouped_df.apply(compute_mean_ci)
supports = reports['fold0']['support'] + reports['fold1']['support'] + reports['fold2']['support'] + reports['fold3']['support'] + reports['fold4']['support']
ci_df['total_samples'] = supports

topvals = []
for i in range(5):
    loss  = pd.read_csv(DIR_PATH+f'f{i}hybridcvtrain48-4000.csv', delimiter= ',', header=None)
    report = pd.read_csv(DIR_PATH+f'f{i}hybridcvval48-4000.csv', delimiter= ',', header=None)
    report.columns = ['top-1', 'top-3']
    topvals.append(report[loss[1]==loss[1].min()])
top_df = pd.concat(topvals, axis=0)
citop_df = top_df.apply(compute_mean_ci)
ci_df.loc['top-3'] = citop_df.iloc[1] 
ci_df.to_csv('./results/hybridcvsum4000.csv')
accs.to_csv('./results/hybridsup4000.csv')