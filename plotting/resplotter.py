#this file containes plotting utilities to reproduce all figures in Results and Discussion along with
#the class distribution plots

import os
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from matplotlib import pyplot as plt
import matplotlib.ticker as ticker

def format_sumframe(classdf):
    macroavg = classdf.loc['macro avg', 'precision']
    mean_str, error_str = macroavg.split('±')
    mean = float(mean_str.strip())
    error = float(error_str.strip())
    macroavg = [mean, error]

    classdf = classdf.iloc[:-4, :]
    classdf.index.name = 'smiles'
    classdf['f1-score'] = classdf['f1-score'].apply(lambda x: float(x.split(' ')[0]))
    classdf['precision'] = classdf['precision'].apply(lambda x: float(x.split(' ')[0]))
    classdf['recall'] = classdf['recall'].apply(lambda x: float(x.split(' ')[0]))
    classdf['total_samples'] = classdf['total_samples'].apply(lambda x: float(x.split(' ')[0]))

    return (classdf, macroavg)

def format_rxnframe(rxnclassdf): 
    macroavg = rxnclassdf.loc['macro avg', 'accuracy']
    mean_str, error_str = macroavg.split('±')
    mean = float(mean_str.strip())
    error = float(error_str.strip())
    macroavg = [mean, error]

    rxnclassdf = rxnclassdf.iloc[:-1, :]
    rxnclassdf['accuracy'] = rxnclassdf['accuracy'].apply(lambda x: float(x.split(' ')[0]))
    return (rxnclassdf, macroavg)

PATH = './results'                #dir for the per condition class results
RXNPATH = './results/rxnclasses/' #dir for reaction class classification results (best model only)

#reads frames for the per condition results
rf_scores    = pd.read_csv(os.path.join(PATH,'rf_scores.csv')) #metrics summary for all rf models
rf_ci  = pd.read_csv(os.path.join(PATH,'rf_ci.csv'))
rfclass = format_sumframe(pd.read_csv(os.path.join(PATH,'struct500treessum2400.csv'), index_col=0))[0]

fnn_scores    = pd.read_csv(PATH+'fnn_scores.csv') #metrics summary for all fnn models
fnn_ci  = pd.read_csv(PATH+'fnn_ci.csv')
fnnclass, classmacro = format_sumframe(pd.read_csv(os.path.join(PATH,'hybridcvsum4000.csv'), index_col=0))

#reads frames for the per reaction class results
fnnrxn, rxnmacro = format_rxnframe(pd.read_csv(os.path.join(RXNPATH,'hybridrxnclasssum4000.csv'), index_col=0))

fig0, axes0 = plt.subplots(nrows=2, ncols=1)
bardf = pd.concat((rfclass['precision'], fnnclass['precision']), axis=1)
bardf.columns = ['RF2400', 'FNN4000']
bardf.index = list(np.arange(1, 297))
bar1 = bardf.iloc[:148]
bar2 = bardf.iloc[148:]

#bar1 = bar1.sort_values(by='FNN4000') in case you'd rather sort by value uncomment
#bar2 = bar2.sort_values(by='FNN4000') in case you'd rather sort by value uncomment

bar1.plot.bar(ax=axes0[0], color=['tab:orange', 'tab:blue'], width=1)
bar2.plot.bar(ax=axes0[1], color=['tab:orange', 'tab:blue'], width=1)
axes0[0].set_xticks(np.arange(0, 147, 20))
axes0[0].tick_params(axis='x', rotation=0)
axes0[1].set_xticks(np.arange(0, 147, 20))
axes0[1].set_xticklabels(np.arange(149, 296, 20))
axes0[1].tick_params(axis='x', rotation=0)
axes0[1].get_legend().remove()

axes0[0].set_ylabel('Class accuracy')
axes0[1].set_ylabel('Class accuracy')
axes0[1].set_xlabel('Classes')


#selecting corresponding values
fnn_hybrid = fnn_scores.iloc[-3::-3]
fnn_struct = fnn_scores.iloc[-2::-3]
fnn_dft    = fnn_scores.iloc[-1::-3]

fnn_ci_hybrid = fnn_ci.iloc[-3::-3]
fnn_ci_struct = fnn_ci.iloc[-2::-3]
fnn_ci_dft    = fnn_ci.iloc[-1::-3]

rf_hybrid = rf_scores.iloc[-3::-3]
rf_struct = rf_scores.iloc[-2::-3]
rf_dft    = rf_scores.iloc[-1::-3]

rfpm_hybrid = rf_ci.iloc[-3::-3]
rfpm_struct = rf_ci.iloc[-2::-3]
rfpm_dft    = rf_ci.iloc[-1::-3]

sizes = [1.0, 1.2, 2.0, 2.4, 4.0] #x values (embedding sizes)
fig1, axes1 = plt.subplots(nrows=1, ncols=2)
axes1[0].errorbar(sizes, fnn_hybrid['top-1'].values, yerr=fnn_ci_hybrid['top-1'], linestyle='dashed', marker='o', label='FNN Hybrid', capsize=4, elinewidth=1)
axes1[0].errorbar(sizes, fnn_struct['top-1'].values, yerr=fnn_ci_struct['top-1'],  linestyle='dashed', marker='o', label='FNN Struct', capsize=4, elinewidth=1)
axes1[0].errorbar(sizes, fnn_dft['top-1'].values, yerr=fnn_ci_dft['top-1'], linestyle='dashed', marker='o', label='FNN DFT',capsize=4, elinewidth=1)

axes1[0].errorbar(sizes, rf_hybrid['top-1'].values, yerr=rfpm_hybrid['top-1'], linestyle='dashed', marker='o', label='RF Hybrid', capsize=4, elinewidth=1)
axes1[0].errorbar(sizes, rf_struct['top-1'].values, yerr=rfpm_struct['top-1'], linestyle='dashed', marker='o', label='RF Struct',capsize=4, elinewidth=1)
axes1[0].errorbar(sizes, rf_dft['top-1'].values, yerr=rfpm_dft['top-1'], linestyle='dashed', marker='o', label='RF DFT',capsize=4, elinewidth=1)

axes1[1].errorbar(sizes, fnn_hybrid['top-3'].values, yerr=fnn_ci_hybrid['top-3'], linestyle='dashed', marker='o', label='FNN Hybrid', capsize=4, elinewidth=1)
axes1[1].errorbar(sizes, fnn_struct['top-3'].values, yerr=fnn_ci_struct['top-3'], linestyle='dashed', marker='o', label='FNN Struct',capsize=4, elinewidth=1)
axes1[1].errorbar(sizes, fnn_dft['top-3'].values, yerr=fnn_ci_dft['top-3'], linestyle='dashed', marker='o', label='FNN DFT', capsize=4, elinewidth=1)
axes1[1].errorbar(sizes, rf_hybrid['top-3'].values, yerr=rfpm_hybrid['top-3'], linestyle='dashed', marker='o', label='RF Hybrid', capsize=4, elinewidth=1)
axes1[1].errorbar(sizes, rf_struct['top-3'].values, yerr=rfpm_struct['top-3'], linestyle='dashed', marker='o', label='RF Struct',capsize=4, elinewidth=1)
axes1[1].errorbar(sizes, rf_dft['top-3'].values, yerr=rfpm_dft['top-3'], linestyle='dashed', marker='o', label='RF DFT', capsize=4, elinewidth=1)
axes1[1].legend().set_draggable(True)

axes1[0].set_ylabel('Top-1 accuracy (%)')
axes1[1].set_ylabel('Top-3 accuracy (%)')

axes1[0].set_yticks([25, 35, 45, 55, 65, 75])
axes1[1].set_yticks([25, 35, 45, 55, 65, 75])
axes1[0].set_xticks(sizes)
axes1[1].set_xticks(sizes)
axes1[0].xaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
axes1[0].set_xlabel('Embedding size ($10^3$)')
axes1[1].set_xlabel('Embedding size ($10^3$)')

fig2, axes2 = plt.subplots()
supportdf = fnnclass.sort_values(by='total_samples', ascending=False)
supportdf['total_samples'].plot.bar(ax=axes2, width=1)
axes2.set_xticks(np.arange(0, 295, 25))
axes2.hlines(y=15, xmin=0, xmax=296, colors='tab:orange')
axes2.set()
axes2.tick_params(axis='x', rotation=0)
axes2.set_xticklabels(np.arange(1, 296, 25))
axes2.set_yscale('log')
axes2.set_xlabel('Condition class number')
axes2.set_ylabel('Reaction count')
axes2.spines['right'].set_visible(False)
axes2.spines['top'].set_visible(False)

fig3, axes3 = plt.subplots()
rxnsupport = fnnrxn.sort_values(by='total_samples', ascending=True)
rxnsupport['total_samples'].plot.barh(ax=axes3, width=0.9, color='tab:brown')
axes3.set_ylabel('')
axes3.set_xlabel('Reaction count', fontsize=6)
axes3.tick_params(axis='both', labelsize=6)
axes3.set_xscale('log')
axes3.spines['right'].set_visible(False)
axes3.spines['top'].set_visible(False)
fig3.tight_layout()



fig4, axes4 = plt.subplots(nrows=1, ncols=2, sharex=True, sharey=True)

x = fnnrxn['total_samples']
y = fnnrxn['accuracy']

pdtinrxns = ( fnnrxn.loc['Sonogashira reaction', 'total_samples'] +  fnnrxn.loc['Suzuki coupling', 'total_samples'] +  
         fnnrxn.loc['Heck reaction', 'total_samples']  +  fnnrxn.loc['Palladium catalyzed reaction', 'total_samples'] +  
         fnnrxn.loc['Stille reaction', 'total_samples'])
print(pdtinrxns)

orgmetrxns = fnnrxn.loc['Other Organometallic C-C bond formation']['total_samples']


axes4[0].scatter(x, y*100, color='tab:red', s=15)
axes4[0].set_ylabel('Accuracy')
axes4[0].set_xlabel('Reaction count per reaction class')
axes4[0].set_xscale('log')
axes4[0].axhline(y=rxnmacro[0]*100, color='black')
axes4[0].text(0.95, 0.05, f"macro avg. = {rxnmacro[0]*100:.1f} ± {rxnmacro[1]:.1%}",
        transform=axes4[0].transAxes,
        fontsize=10,
        verticalalignment='top',
        horizontalalignment='right')

'''
model = LinearRegression().fit(x.values.reshape(-1, 1), y)
x_range = np.linspace(x.min(), x.max(), 500).reshape(-1, 1)
y_pred = model.predict(x_range)
r2 = model.score(x.values.reshape(-1, 1), y)
axes4[0].text(0.95, 0.95, f"$R^2$ = {r2:.3f}",
        transform=axes4[0].transAxes,
        fontsize=10,
        verticalalignment='top',
        horizontalalignment='right')
axes4[0].plot(x_range, y_pred, color='black')
''' #uncomment for (linear) regression

x = fnnclass['total_samples']
y = fnnclass['precision']

axes4[1].scatter(x, y*100, color='tab:red', s=15)
axes4[1].set_xlabel('Reaction count per condition class')
axes4[1].set_xscale('log')
axes4[1].yaxis.set_tick_params(labelleft=True)
axes4[1].axhline(y=classmacro[0]*100, color='black')
axes4[1].text(0.95, 0.05, f"macro avg. = {classmacro[0]*100:.1f} ± {classmacro[1]:.1%}",
        transform=axes4[1].transAxes,
        fontsize=10,
        verticalalignment='top',
        horizontalalignment='right')

'''
model1 = LinearRegression().fit(x.values.reshape(-1, 1), y)
x_range = np.linspace(x.min(), x.max(), 500).reshape(-1, 1)
y_pred = model1.predict(x_range)
r2 = model1.score(x.values.reshape(-1, 1), y)
axes4[1].text(0.95, 0.95, f"$R^2$ = {r2:.3f}",
        transform=axes4[1].transAxes,
        fontsize=10,
        verticalalignment='top',
        horizontalalignment='right')
axes4[1].scatter(x, y, color='tab:red', s=15)
axes4[1].plot(x_range, y_pred, color='black')
axes4[1].set_ylabel('Accuracy')
axes4[1].set_xlabel('Reaction count per condition class')
''' #uncomment for (linear) regression


plt.show()