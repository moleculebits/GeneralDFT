import torch
import numpy as np
import scipy.stats
import pandas as pd

DIR_PATH = 'C:\\Users\\migno\\Documents\\ReactionContextDiscovery\\crossval\\'

def compute_mean_ci(values):
    
    if len(values) == 0:
        return "N/A"  # Handle empty cases
    
    mean, se = np.mean(values), scipy.stats.sem(values)
    n = len(values)
    ci = se * scipy.stats.t.ppf((1 + 0.95) / 2., n-1) if n > 1 else 0
    
    # Format as "mean ± CI" with two decimal places
    return f"{mean:.3f} ± {ci:.3f}"

reports500  = {}
reports1000 = {}
dfs500      = []
dfs1000     = []
accs500     = []
accs1000    = []
topvals500  = []
topvals1000 = []
reports = torch.load(DIR_PATH+f'fmol2rfreports1000.pt')
for i in range(5):
    report500, top3ac500   = reports[f'f{i}rf500']
    report1000, top3ac1000 = reports[f'f{i}rf1000']

    topvals500.append({f'top3': top3ac500})
    topvals1000.append({f'top3': top3ac1000})

    newdf1 = report500[['support', 'precision']].copy()
    report500.rename(columns = {'precision': f'precision{i+1}', 'recall': f'recall{i+1}', 'f1-score': f'f1-score{i+1}', 'support': 'support'}, inplace=True)
    reports500[f'fold{i}'] = report500

    newdf2 = report1000[['support', 'precision']].copy()
    report1000.rename(columns = {'precision': f'precision{i+1}', 'recall': f'recall{i+1}', 'f1-score': f'f1-score{i+1}', 'support': 'support'}, inplace=True)
    reports1000[f'fold{i}'] = report1000

    newdf1['precision'] = newdf1['precision'].map('{:.3f}'.format)
    newdf1['support'] = newdf1['support'].map(int)
    newdf1.rename(columns = {'precision': f'acc{i+1}', 'support': f's{i+1}'}, inplace=True)

    newdf1.drop(newdf1.tail(3).index, inplace=True)
    accs500.append(newdf1)

    report500['mask'] = report500['support'].apply(bool)
    dfs500.append(report500.loc[report500['mask'], report500.columns.difference(['mask', 'support'])])

    newdf2['precision'] = newdf2['precision'].map('{:.3f}'.format)
    newdf2['support'] = newdf2['support'].map(int)
    newdf2.rename(columns = {'precision': f'acc{i+1}', 'support': f's{i+1}'}, inplace=True)

    newdf2.drop(newdf2.tail(3).index, inplace=True)
    accs1000.append(newdf2)

    report1000['mask'] = report1000['support'].apply(bool)
    dfs1000.append(report1000.loc[report1000['mask'], report1000.columns.difference(['mask', 'support'])])


accs500 = pd.concat(accs500, axis=1, sort=False)
combined_df500 = pd.concat(dfs500, axis=1, sort=False)
num_cols500 = len(dfs500[0].columns)

grouped_df500 = pd.DataFrame({col: combined_df500.iloc[:, i::3].values.tolist() for i, col in enumerate(['f1-score', 'precision', 'recall'])}, index=combined_df500.index)

accs1000 = pd.concat(accs1000, axis=1, sort=False)
combined_df1000 = pd.concat(dfs1000, axis=1, sort=False)
num_cols1000 = len(dfs1000[0].columns)

grouped_df1000 = pd.DataFrame({col: combined_df1000.iloc[:, i::3].values.tolist() for i, col in enumerate(['f1-score', 'precision', 'recall'])}, index=combined_df1000.index)

ci_df500 = grouped_df500.apply(compute_mean_ci)
ci_df1000 = grouped_df1000.apply(compute_mean_ci)

supports500 = reports500['fold0']['support'] + reports500['fold1']['support'] + reports500['fold2']['support'] + reports500['fold3']['support'] + reports500['fold4']['support']
ci_df500['total_samples'] = supports500

supports1000 = reports1000['fold0']['support'] + reports1000['fold1']['support'] + reports1000['fold2']['support'] + reports1000['fold3']['support'] + reports1000['fold4']['support']
ci_df1000['total_samples'] = supports1000

top_df500 = pd.DataFrame(topvals500)
citop_df500  = top_df500.apply(compute_mean_ci)
ci_df500.loc['top-3'] = citop_df500.iloc[0]
ci_df500.to_csv('./tables/struct500treessum1000.csv')
accs500.to_csv('./tables/struct500treessup1000.csv')

top_df1000 = pd.DataFrame(topvals1000)
citop_df1000  = top_df1000.apply(compute_mean_ci)
ci_df1000.loc['top-3'] = citop_df1000.iloc[0]
ci_df1000.to_csv('./tables/struct1000treessum1000.csv')
accs1000.to_csv('./tables/struct1000treessup1000.csv')


