#This file written by M. Nouman is part of the General DFT Descriptors project
#which is released under the MIT license. See LICENSE file for full license details.
import os
import torch
import pandas as pd
import numpy as np
import argparse
import dataset
from sklearn import preprocessing as sk
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, ConcatDataset

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import top_k_accuracy_score
from sklearnex import patch_sklearn

patch_sklearn()
torch.manual_seed(42)

parser = argparse.ArgumentParser()
parser.add_argument('-p', dest='path', type=str, help='specifies path to embedding')
parser.add_argument('-s', dest='size', type=int, help='specifies size of embedding')
parser.add_argument('-t', dest='type', type=str, help='specifies type of embedding')
args = parser.parse_args()

VALSIZE  = 6000
TESTSIZE = 6000
EMBPATH  = args.path
FOLDER   = './rfmodelout/'

trainingset   = dataset.EmbDataset(EMBPATH, VALSIZE, TESTSIZE, train=True, test=True)
validationset = dataset.EmbDataset(EMBPATH, VALSIZE, TESTSIZE, train=False, test=False)
testset       = dataset.EmbDataset(EMBPATH, VALSIZE, TESTSIZE, train=False, test=True)

print(len(trainingset))

labels = []
embs   = []
cvset = ConcatDataset((trainingset, validationset, testset)) #used later for cross validation
for emb, label, smarts in cvset:
    embs.append(emb.reshape(-1,).numpy())
    labels.append(label)
labels   = np.asarray(labels)

le = sk.LabelEncoder()
npembs   = np.vstack(embs)
nplabels = le.fit_transform(labels)

reports = {}
cv = StratifiedKFold(5)
for i, (train_indices, val_indices) in enumerate(cv.split(npembs, nplabels)):
    X_train, X_val = npembs[train_indices], npembs[val_indices] #val indices are not necessary as random forests are nonparametric 
    y_train, y_val = nplabels[train_indices], nplabels[val_indices]
    clf1 = RandomForestClassifier(n_estimators=500, n_jobs=24, max_depth=12, random_state=42)
    clf2 = RandomForestClassifier(n_estimators=1000, n_jobs=24, max_depth=12, random_state=42)

    clf1.fit(X_train, y_train)
    clf2.fit(X_train, y_train)
    
    y_prob1 = clf1.predict_proba(X_val)
    y_prob2 = clf2.predict_proba(X_val)
    y_pred1  = clf1.predict(X_val)
    y_pred2  = clf2.predict(X_val)


    top3_acc1 = top_k_accuracy_score(y_val, y_prob1, k=3, labels=np.unique(nplabels))
    top3_acc2 = top_k_accuracy_score(y_val, y_prob2, k=3, labels=np.unique(nplabels))

    report1 = classification_report(y_val, y_pred1, labels=np.unique(nplabels), target_names=le.classes_, output_dict=True)
    report2 = classification_report(y_val, y_pred2, labels=np.unique(nplabels), target_names=le.classes_, output_dict=True)

    df1 = pd.DataFrame(report1).transpose()
    df2 = pd.DataFrame(report2).transpose()

    reports[f'f{i}rf500']  = (df1.copy(), top3_acc1)
    reports[f'f{i}rf1000'] = (df2.copy(), top3_acc2)

    df1.to_csv(os.path.join(FOLDER, f'f{i}{args.type}rf500trees{args.size}.csv'))
    df2.to_csv(os.path.join(FOLDER, f'f{i}{args.type}rf1000trees{args.size}.csv'))
    torch.save((y_pred1, y_val, clf1), os.path.join(FOLDER, f'f{i}{args.type}rf500trees{args.size}.pt'))
    torch.save((y_pred2, y_val, clf2), os.path.join(FOLDER, f'f{i}{args.type}rf1000trees{args.size}.pt'))
torch.save(reports, os.path.join(FOLDER, f'{args.type}rfreports{args.size}.pt'))