#This file written by M. Nouman is part of the General DFT Descriptors project
#which is released under the MIT license. See LICENSE file for full license details.

import torch
from matplotlib import pyplot as plt
from matplotlib import cm
import copy
import numpy as np

from sklearn.manifold import TSNE
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import ConfusionMatrixDisplay, classification_report

DIR_PATH = ''
inputlayer = torch.load(DIR_PATH+'molgl2inlayer48-1000.pt')
lastlayer  = torch.load(DIR_PATH+'molgl2outlayer48-1000.pt')
classes, truelabs, predlabs = torch.load(DIR_PATH+'molgl2testlabs48-1000.pt')


le = LabelEncoder()
classes  = le.fit_transform(classes).reshape(-1,)
predlabs = le.transform(predlabs).reshape(-1,)
truelabs = le.transform(truelabs).reshape(-1,)

X_inp = TSNE(n_components=2, learning_rate='auto', init='random', random_state=42, perplexity=110).fit_transform(inputlayer)
X_out = TSNE(n_components=2, learning_rate='auto', init='random', random_state=42, perplexity=110).fit_transform(lastlayer)

fig = plt.figure(figsize=(12, 5))
ax = fig.add_subplot(1, 2, 1)

ax.scatter(X_inp[:, 0], X_inp[:, 1], c=truelabs, cmap='jet')
ax.set_xlabel('t-SNE 1')
ax.set_ylabel('t-SNE 2')
ax.set_title('Input t-SNE combined 4000')


ax = fig.add_subplot(1, 2, 2)
ax.scatter(X_out[:, 0], X_out[:, 1], c=truelabs, cmap='jet')
ax.set_xlabel('t-SNE 1')
ax.set_ylabel('t-SNE 2')
ax.set_title('Output t-SNE combined 4000')

my_cmap = copy.copy(cm.get_cmap('viridis'))
my_cmap.set_bad(my_cmap.colors[0])
print(classification_report(truelabs, predlabs, labels=classes))
ConfusionMatrixDisplay.from_predictions(truelabs, predlabs, include_values=False,  normalize=None, cmap=my_cmap, im_kw={'norm':'log'})
plt.xticks([], [])
plt.yticks([], [])

plt.title('Confusion Matrix combined 4000')
plt.show()