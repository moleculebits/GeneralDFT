#This file written by M. Nouman is part of the General DFT Descriptors project
#which is released under the MIT license. See LICENSE file for full license details.
import torch
import os
import numpy as np
import argparse
import dataset
from sklearn import preprocessing as sk
from torch.utils.data import DataLoader, WeightedRandomSampler, ConcatDataset

#improves performance
from sklearnex import patch_sklearn
patch_sklearn()

device = torch.device('cuda'if torch.cuda.is_available() else 'cpu')
print(device)
torch.manual_seed(21)
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

    VALSIZE   = 6993
    TESTSIZE  = 6993
    BATCHSIZE = 48

    LR = 0.001
    EPOCHS = 300

    #saving model information
    folder  = './modelout/'
    EMBPATH        = args.path
    HRES_OUT       = args.type+f'test48-{args.size}.txt'           #training loss in txt
    TESTLABS       = args.type+f'testlabs48-{args.size}.pt'        #test prediction and ground truth labels - IMPORTANT FOR SCORING
    TRAINING_OUT   = args.type+f'train48-{args.size}.csv'          #training loss in csv (redundant)
    TEST_REACTS    = args.type+f'testreacts48-{args.size}.txt'     #test reaction SMILES + predictions (for test)
    TRAIN_REACTS   = args.type+f'trreacts48-{args.size}.txt'       #train reaction SMILES + predictions (for debug)
    INLAYER        = args.type+f'inlayer48-{args.size}.pt'         #input layer weights
    OUTLAYER       = args.type+f'outlayer48-{args.size}.pt'        #output layer weights

    BEST_STATE     = args.type+f'beststate48-{args.size}.pt'       #best model state 
    LAST_STATE     = args.type+f'laststate48-{args.size}.pt'       #last model state

    trainingset   = dataset.EmbDataset(EMBPATH, VALSIZE, TESTSIZE, train=True, test=True)
    validationset = dataset.EmbDataset(EMBPATH, VALSIZE, TESTSIZE, train=False, test=False)
    testset       = dataset.EmbDataset(EMBPATH, VALSIZE, TESTSIZE, train=False, test=True)
    print(len(trainingset))

    labels = []
    embs   = []
    dset = ConcatDataset((trainingset, validationset, testset)) #used later for validation
    print(len(trainingset))
    print(len(dset))
    for emb, label, smarts in dset:
        embs.append(emb.reshape(-1,).numpy())
        labels.append(label)

    le = sk.OneHotEncoder(handle_unknown='ignore')
    npembs   = np.vstack(embs)
    labels   = np.asarray(labels)
    classes, class_sample_count = np.unique(labels, return_counts=True)
    nplabels = le.fit_transform(labels.reshape(-1, 1)).toarray()
    
    model = BNET(weights, int(np.shape(npembs)[1]), int(np.shape(nplabels)[1]))
    model.to(device)
    print(np.shape(npembs)[1], int(np.shape(nplabels)[1]))

    trlabs = []
    for emb, label, smarts in trainingset:
        trlabs.append(label)
    trlabs = np.asarray(trlabs)
    trclasses, trclass_sample_count = np.unique(trlabs, return_counts=True)
    trweight = 1.0/trclass_sample_count
    trindices = np.array([np.where(trclasses == trlabs[i])[0] for i in range(0, trlabs.size)]).reshape(-1,)
    sample_weights = np.array([trweight[idx] for idx in trindices])
    sampler = WeightedRandomSampler(torch.from_numpy(sample_weights), len(trainingset), replacement=True)

    trainingloader   = DataLoader(trainingset, BATCHSIZE, collate_fn=dataset.EmbDataset.collate_fun, sampler=sampler, drop_last=False)
    validationloader = DataLoader(validationset,  1000, collate_fn=dataset.EmbDataset.collate_fun, drop_last=False)
    testingloader    = DataLoader(testset,  1000, collate_fn=dataset.EmbDataset.collate_fun, drop_last=False)

    optimizer = torch.optim.SGD(model.parameters(), lr=LR, weight_decay=1e-5, momentum=0.9)
    loss_fun  = torch.nn.CrossEntropyLoss()
    scheduler1 = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, 30, T_mult=11)

    def train_epoch():
        running_loss = 0
        last_loss    = 0 
        total_loss   = 0

        total = 0.0
        top1correct = 0.0
        topkcorrect = 0.0

        rxnsmiles = []
        predlabs = []
        truelabs = []

        accum_iter = 1
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
                    print(f'batch: {idx+1} last_loss: {last_loss} top1accuracy: {top1ac:.4f}, top3accuracy: {topkac}')
                    running_loss = 0
            with open(os.path.join(folder,TRAIN_REACTS), 'w') as fout:
                for idx, (smarts, predlab, truelab) in enumerate(zip(rxnsmiles, predlabs, truelabs)):
                    fout.write(smarts + f' predicted: {predlab} gtruth: {truelab} \n')

            return total_loss/len(trainingloader)
        
    epochidx = 0
    best_vloss = 1e6
    for epoch in range(EPOCHS):
        model.train()
        avg_loss = train_epoch()
        model.eval()
        running_vloss = 0.0
        with torch.no_grad():
            for idx, vdata in enumerate(validationloader):
                vinputs, vlabels, _ = vdata
                voutputs  = model(vinputs.to(device))
                vlabels = le.transform(vlabels).toarray()
                vlabels = torch.from_numpy(vlabels)
                vloss = loss_fun(voutputs, vlabels.to(device))
                running_vloss += float(vloss)

            avg_vloss = running_vloss/(idx+1)
            scheduler1.step()
        with open(os.path.join(folder, HRES_OUT), 'a') as trout:
            with open(os.path.join(folder, TRAINING_OUT), 'a') as csvout:
                csvout.write(f'{avg_loss}, {avg_vloss} \n')
                trout.write('epoch: {}  train: {} valid: {} ratio: {} \n'.format(epochidx+1, avg_loss, avg_vloss, avg_vloss/avg_loss))

        if avg_vloss < best_vloss:
            best_vloss = avg_vloss
            bestflag = 1
            torch.save(model.state_dict(), os.path.join(folder,BEST_STATE))
        else:
            bestflag = 0
            torch.save(model.state_dict(), os.path.join(folder,LAST_STATE))
        epochidx += 1

        with torch.no_grad():
            total = 0.0

            top1correct = 0.0
            topkcorrect = 0.0
            trxnsmiles  = []
            tpredlabs   = []
            ttruelabs   = []
            tlastlayer  = []
            tinputlayer = []

            for idx, tdata in enumerate(testingloader):
                tinputs, tlabels, smarts = tdata
                toutputs = model(tinputs.to(device))

                tinputlayer.append(tinputs.cpu().numpy())
                tlastlayer.append(toutputs.cpu().numpy())

                predlab  = le.inverse_transform(toutputs.cpu().numpy())
                tpredlabs.extend(predlab.tolist())
                ttruelabs.extend(tlabels.tolist())
                trxnsmiles.extend(smarts)
                tlabels  = le.transform(tlabels).toarray()
                tlabels  = torch.from_numpy(tlabels)

                _, tlabels   = torch.max(tlabels, 1)
                total += float(tlabels.size(0))
                res1, res3 = top_correct(tlabels, toutputs.cpu(), 3)
                top1correct += res1
                topkcorrect += res3
            top1ac = float(top1correct/total)
            topkac = float(topkcorrect/total)

        if bestflag == 1:
            tinputlayer = np.concatenate(tinputlayer, axis=0)
            tlastlayer  = np.concatenate(tlastlayer, axis=0)
            torch.save(tinputlayer, os.path.join(folder,INLAYER))
            torch.save(tlastlayer, os.path.join(folder, OUTLAYER))
            torch.save((classes, ttruelabs, tpredlabs), os.path.join(folder, TESTLABS))
            with open(os.path.join(folder,TEST_REACTS), 'w') as fout:
                for idx, (smarts, predlab, truelab) in enumerate(zip(trxnsmiles, tpredlabs, ttruelabs)):
                    fout.write(smarts + f' predicted: {predlab} gtruth: {truelab} \n')
        with open(os.path.join(folder, HRES_OUT), 'a') as trout:
            trout.write(f'test top1 accuracy: {top1ac} test top3 accuracy: {topkac}\n')

