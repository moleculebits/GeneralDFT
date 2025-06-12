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
from torch.utils.data import DataLoader, WeightedRandomSampler, Subset, ConcatDataset

#improves performance
from sklearnex import patch_sklearn
patch_sklearn()

device = torch.device('cuda'if torch.cuda.is_available() else 'cpu')
print(device)
torch.manual_seed(42)
DROPOUT = 0.25

parser = argparse.ArgumentParser()
parser.add_argument('-p', dest='path', type=str, help='specifies path to embedding')
parser.add_argument('-s', dest='size', type=int, help='specifies size of embedding')
parser.add_argument('-t', dest='type', type=str, help='specifies type of embedding')
args = parser.parse_args()

if args.size==1000:
    weights = [3, 3, 2, 2]
elif args.size==1200:
    weights = [4, 3, 2, 2]
elif args.size==2000:
    weights = [6, 5, 4, 2]
elif args.size==2400:
    weights = [8, 6, 4, 2]
elif args.size in (3400, 3600, 4000):
    weights = [11, 9, 7, 4]
else: 
    weigths = []

class BNET(torch.nn.Module):
    def __init__(self, layerweigths, in_channels, out_channels):
        super().__init__()
        torch.set_default_dtype(torch.float32)
        self.fnn = torch.nn.Sequential(#Change the number of neurons according to the paper to reproduce the results obtained
            torch.nn.Linear(in_channels, layerweigths[0]*out_channels),
            torch.nn.LeakyReLU(),
            torch.nn.BatchNorm1d(layerweigths[0]*out_channels),
            torch.nn.Dropout(DROPOUT),
            torch.nn.Linear(layerweigths[0]*out_channels, layerweigths[1]*out_channels),
            torch.nn.LeakyReLU(),
            torch.nn.BatchNorm1d(layerweigths[1]*out_channels),
            torch.nn.Dropout(DROPOUT),
            torch.nn.Linear(layerweigths[1]*out_channels,layerweigths[2]*out_channels),
            torch.nn.LeakyReLU(),
            torch.nn.BatchNorm1d(layerweigths[2]*out_channels),
            torch.nn.Dropout(DROPOUT),
            torch.nn.Linear(layerweigths[2]*out_channels, layerweigths[3]*out_channels),
            torch.nn.LeakyReLU(),
            torch.nn.BatchNorm1d(layerweigths[3]*out_channels),
            torch.nn.Dropout(DROPOUT),
            torch.nn.Linear(layerweigths[3]*out_channels, out_channels),
        )

    def forward(self, fp):
        x = fp
        #feed forward for class prediction
        x = self.fnn(x)
        return x
    
def weight_reset(m):
    reset_parameters = getattr(m, "reset_parameters", None)
    if callable(reset_parameters):
        m.reset_parameters()


def top_correct(labels, predicted, k):
    _, topk   = torch.topk(predicted, k, 1)
    _, top1   = torch.max(predicted, 1)
    topcorrect = 0
    correct    = 0

    correct += float((labels == top1).sum().item())
    for i in range(0, labels.size(0)):
        topcorrect += int(labels[i] in topk[i, :])

    topkac = topcorrect
    top1ac = correct
    return top1ac, topkac

if __name__ == '__main__':

    VALSIZE   = 6000
    TESTSIZE  = 6000
    BATCHSIZE = 48

    LR = 0.001
    EPOCHS = 350

    #saving model information
    folder  = './modelout/outfiles/cvfiles/' #output directory
    EMBPATH        = args.path
    HRES_OUT       = args.type+f'cvv48-{args.size}.txt'            #test set accuracies (overview file)
    TESTLABS       = args.type+f'cvvlabs48-{args.size}.pt'         #test prediction and ground truth labels - IMPORTANT FOR SCORING (this is what you need)
    TESTING_OUT    = args.type+f'cvval48-{args.size}.csv'          #top-1 and top-3 acc results in csv (redundant)
    TRAINING_OUT   = args.type+f'cvtrain48-{args.size}.csv'        #training and validation loss in csv (redundant)
    TEST_REACTS    = args.type+f'cvvalreacts48-{args.size}.txt'    #test reaction SMILES + predictions (large file for debugging and reaction class labelling)
    TRAIN_REACTS   = args.type+f'cvtrreacts48-{args.size}.txt'     #train reaction SMILES + predictions (very large file for debugging, uncomment code to generate)
    INLAYER        = args.type+f'inlayer48-{args.size}.pt'         #input layer weights
    OUTLAYER       = args.type+f'outlayer48-{args.size}.pt'        #output layer weights

    BEST_STATE     = args.type+f'beststate48-{args.size}.pt'       #best model state 
    LAST_STATE     = args.type+f'laststate48-{args.size}.pt'       #last model state

    trainingset   = dataset.EmbDataset(EMBPATH, VALSIZE, TESTSIZE, train=True, test=True)
    validationset = dataset.EmbDataset(EMBPATH, VALSIZE, TESTSIZE, train=False, test=False)
    testset       = dataset.EmbDataset(EMBPATH, VALSIZE, TESTSIZE, train=False, test=True)

    labels = []
    embs   = []
    cvset = ConcatDataset((trainingset, validationset, testset)) #used later for cross validation
    print(len(trainingset))
    print(len(cvset))
    for emb, label, smarts in cvset:
        embs.append(emb.reshape(-1,).numpy())
        labels.append(label)

    le = sk.OneHotEncoder(handle_unknown='ignore')
    npembs   = np.vstack(embs)
    labels   = np.asarray(labels)
    nplabels = le.fit_transform(labels.reshape(-1, 1)).toarray()
    
    model = BNET(weights, int(np.shape(npembs)[1]), int(np.shape(nplabels)[1]))
    model.to(device)
    print(np.shape(npembs)[1], int(np.shape(nplabels)[1]))

    optimizer = torch.optim.SGD(model.parameters(), lr=LR, weight_decay=1e-5, momentum=0.9)
    loss_fun  = torch.nn.CrossEntropyLoss()
    scheduler1 = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, 30, T_mult=11)

    def train_epoch(trainingloader):
            running_loss = 0
            last_loss    = 0 
            total_loss   = 0

            total = 0.0
            top1correct = 0.0
            topkcorrect = 0.0

            rxnsmiles = []
            predlabs = []
            truelabs = []

            accum_iter = 1 #option for multiples of the batch size
            with torch.autocast(device_type='cuda', dtype=torch.float32):
                for idx, data in enumerate(trainingloader):
                    inputs, labels, smarts = data
                    outputs = model(inputs.to(device).float())

                    predlab  = le.inverse_transform(outputs.detach().cpu().numpy())
                    predlabs.extend(predlab.tolist())
                    truelabs.extend(labels.tolist())
                    rxnsmiles.extend(smarts)

                    labels = le.transform(labels).toarray()
                    labels = torch.from_numpy(labels)
                    loss = loss_fun(outputs, labels.to(device))
                    loss.backward()

                    _, labels   = torch.max(labels, 1)
                    total += float(labels.size(0))
                    res1, res3 = top_correct(labels, outputs.cpu(), 3)
                    top1correct += res1
                    topkcorrect += res3

                    top1ac = float(top1correct/total)
                    topkac = float(topkcorrect/total) 

                    running_loss += float(loss.item())
                    total_loss += float(loss.item())
                    if ((idx+1) % accum_iter == 0) or (idx+1 == len(trainingloader)):
                        optimizer.step()
                        optimizer.zero_grad()
                        last_loss    = running_loss/(accum_iter)
                        #print(f'batch: {idx+1} last_loss: {last_loss} top1accuracy: {top1ac:.4f}, top3accuracy: {topkac}')
                        running_loss = 0
                #uncomment to save more training output information (around 10-12 GB)
                #with open(os.path.join(folder, TRAIN_REACTS), 'a') as fout:
                    #for idx, (smarts, predlab, truelab) in enumerate(zip(rxnsmiles, predlabs, truelabs)):
                        #fout.write(smarts + f' predicted: {predlab} gtruth: {truelab} \n')

                return total_loss/len(trainingloader)



    #5-fold cross validation (+ fixed test set)
    cv = StratifiedKFold(n_splits=5, shuffle=False)
    for i, (train_indices, val_indices) in enumerate(cv.split(npembs, labels)):

        model.apply(weight_reset)

        fold_train = Subset(cvset, train_indices)
        fold_val   = Subset(cvset, val_indices)

        cvlabs = []
        for emb, label, smarts in fold_train:
            cvlabs.append(label)

        cvlabs = np.asarray(cvlabs)
        classes, class_sample_count = np.unique(cvlabs, return_counts=True)
        weight = 1.0/class_sample_count
        indices = np.array([np.where(classes == cvlabs[i])[0] for i in range(0, cvlabs.size)]).reshape(-1,)
        sample_weights = np.array([weight[idx] for idx in indices])
        sampler = WeightedRandomSampler(torch.from_numpy(sample_weights), len(fold_train), replacement=True)

        trainingloader   = DataLoader(fold_train, BATCHSIZE, collate_fn=dataset.EmbDataset.collate_fun, sampler=sampler, drop_last=False)
        validationloader = DataLoader(fold_val,  1000, collate_fn=dataset.EmbDataset.collate_fun, drop_last=False)

        foldstring = f'''\n
        ############ FOLD{i} #############\n
        \n        
'''
        print(foldstring)
        with open(os.path.join(folder, HRES_OUT), 'a') as trout:
            trout.write(foldstring)
        with open(os.path.join(folder, TEST_REACTS),'a') as fout:
            fout.write(foldstring)
      #  with open(os.path.join(folder, TRAIN_REACTS), 'a') as rxnout: uncomment for more debug info (10-12gb)
      #      rxnout.write(foldstring)

        best_vloss = 1e6
        for epoch in range(EPOCHS):
            model.train()
            avg_loss = train_epoch(trainingloader)
            scheduler1.step()
            model.eval()
            running_vloss = 0.0
            vtotal = 0.0
            top1vcorrect = 0.0
            topkvcorrect = 0.0
            with torch.no_grad():
                vrxnsmiles  = []
                vpredlabs   = []
                vtruelabs   = []
                vinputlayer = []
                vlastlayer  = []
                for idx, vdata in enumerate(validationloader):
                    vinputs, vlabels, vsmarts = vdata
                    voutputs = model(vinputs.to(device).float())

                    vinputlayer.append(vinputs.cpu().numpy())
                    vlastlayer.append(voutputs.cpu().numpy())

                    vpredlab  = le.inverse_transform(voutputs.detach().cpu().numpy())
                    vpredlabs.extend(vpredlab.tolist())
                    vtruelabs.extend(vlabels.tolist())
                    vrxnsmiles.extend(vsmarts)

                    vlabels = le.transform(vlabels).toarray()
                    vlabels = torch.from_numpy(vlabels)
                    vloss = loss_fun(voutputs, vlabels.to(device))
                    running_vloss += float(vloss)

                    _, vlabels   = torch.max(vlabels, 1)
                    vtotal += float(vlabels.size(0))
                    vres1, vres3 = top_correct(vlabels, voutputs.cpu(), 3)
                    top1vcorrect += vres1
                    topkvcorrect += vres3

                    top1vac = float(top1vcorrect/vtotal)
                    topkvac = float(topkvcorrect/vtotal) 

                avg_vloss = running_vloss/(idx+1)
            with open(os.path.join(folder, HRES_OUT), 'a') as trout:
                    trout.write('epoch: {}  train: {} valid: {} ratio: {} \n'.format(epoch+1, avg_loss, avg_vloss, avg_vloss/avg_loss))
            with open(os.path.join(folder, f'f{i}'+TRAINING_OUT), 'a') as csvout:
                    csvout.write(f'{avg_loss}, {avg_vloss} \n')
            with open(os.path.join(folder, HRES_OUT), 'a') as trout:
                trout.write(f'valid top1 accuracy: {top1vac} valid top3 accuracy: {topkvac}\n')
            with open(os.path.join(folder, f'f{i}'+TESTING_OUT), 'a') as csvout:
                csvout.write(f'{top1vac}, {topkvac}\n')
            
            if avg_vloss < best_vloss:
                best_vloss = avg_vloss
                torch.save(model.state_dict(), os.path.join(folder, f'f{i}'+BEST_STATE))
                vinputlayer = np.concatenate(vinputlayer, axis=0)
                vlastlayer  = np.concatenate(vlastlayer, axis=0)
                torch.save(vinputlayer, os.path.join(folder, f'f{i}'+INLAYER))
                torch.save(vlastlayer, os.path.join(folder, f'f{i}'+OUTLAYER))
                torch.save((classes, vtruelabs, vpredlabs), os.path.join(folder, f'f{i}'+TESTLABS)) #IMPORTANT FOR SCORING!!!
                with open(os.path.join(folder, TEST_REACTS), 'a') as fout:
                    for idx, (smarts, predlab, truelab) in enumerate(zip(vrxnsmiles, vpredlabs, vtruelabs)):
                        fout.write(smarts + f' predicted: {predlab} gtruth: {truelab} \n')
            else:
                torch.save(model.state_dict(), os.path.join(folder, f'f{i}'+LAST_STATE))