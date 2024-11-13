#This file written by M. Nouman is part of the General DFT Descriptors project
#which is released under the MIT license. See LICENSE file for full license details.

import torch
import numpy as np
import pickle
import dataset
from sklearn import preprocessing as sk
from torch.utils.data import DataLoader, WeightedRandomSampler

device = torch.device('cuda'if torch.cuda.is_available() else 'cpu')
print(device)

torch.manual_seed(42)
DROPOUT = 0.25

class BNET(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        torch.set_default_dtype(torch.float32)
        self.fnn = torch.nn.Sequential(#Change the number of neurons according to the paper to reproduce the results obtained
            torch.nn.Linear(in_channels, 8*out_channels),
            torch.nn.LeakyReLU(),
            torch.nn.BatchNorm1d(8*out_channels),
            torch.nn.Dropout(DROPOUT),
            torch.nn.Linear(8*out_channels, 6*out_channels),
            torch.nn.LeakyReLU(),
            torch.nn.BatchNorm1d(6*out_channels),
            torch.nn.Dropout(DROPOUT),
            torch.nn.Linear(6*out_channels,4*out_channels),
            torch.nn.LeakyReLU(),
            torch.nn.BatchNorm1d(4*out_channels),
            torch.nn.Dropout(DROPOUT),
            torch.nn.Linear(4*out_channels, 2*out_channels),
            torch.nn.LeakyReLU(),
            torch.nn.BatchNorm1d(2*out_channels),
            torch.nn.Dropout(DROPOUT),
            torch.nn.Linear(2*out_channels, out_channels),
        )

    def forward(self, fp):
        x = fp
        #feed forward for class prediction
        x = self.fnn(x)
        return x

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
    EPOCHS = 300

    #saving model information

    HRES_OUT       = 'molgl2htest48-3400.txt'           #test set accuracies
    TESTLABS       = 'mol2gl2htestlabs48-3400.pt'       #test prediction and ground truth labels (redundant to prevent data-loss in case of model corruption)
    TRAINING_OUT   = 'molgl2htrainres48-3400.csv'       #training and validation loss
    TEST_REACTS    = 'molgl2htestreacts48-3400.txt'     #test reaction SMILES + predictions
    TRAIN_REACTS   = 'molgl2htrainreacts48-3400.txt'    #train reaction SMILES + predictions
    INLAYER       = 'molgl2hinlayer48-3400.pt'          #input layer weights
    OUTLAYER       = 'molgl2houtlayer48-3400.pt'        #output layer weights
    ONEHOT         = 'onehoth3400.pkl'                  #one-hot model

    BEST_STATE     = 'mol2gl2hbeststate48-3400.pt'      #best model state 
    LAST_STATE     = 'mol2gl2hlaststate48-3400.pt'      #last model state

    trainingset   = dataset.EmbDataset(VALSIZE, TESTSIZE, train=True, test=True)
    validationset = dataset.EmbDataset(VALSIZE, TESTSIZE, train=False, test=True)
    testset       = dataset.EmbDataset(VALSIZE, TESTSIZE, train=False, test=False)

    print(len(trainingset))

    labels = []
    for emb, label, smarts in trainingset:
        labels.append(label)

    nplabels = np.asarray(labels)
    classes, class_sample_count = np.unique(labels, return_counts=True)
    weight = 1.0/class_sample_count
    indices = np.array([np.where(classes == nplabels[i])[0] for i in range(0, nplabels.size)]).reshape(-1,)
    sample_weights = np.array([weight[idx] for idx in indices])

    sampler = WeightedRandomSampler(torch.from_numpy(sample_weights), len(trainingset), replacement=True)
    weight = torch.from_numpy(weight)

    for emb, label, smarts in (*validationset, *testset):
        labels.append(label)

    nplabels = np.array(labels)
    classes, class_sample_count = np.unique(nplabels, return_counts=True)
    class_weights = 1.0/class_sample_count
    le = sk.OneHotEncoder(handle_unknown='ignore')
    nplabels = le.fit_transform(nplabels.reshape(-1, 1)).toarray()

    with open(ONEHOT, 'wb') as handle:
        pickle.dump(le, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    trainingloader   = DataLoader(trainingset, BATCHSIZE, collate_fn=dataset.EmbDataset.collate_fun, sampler=sampler, drop_last=True)
    validationloader = DataLoader(validationset,  1000, collate_fn=dataset.EmbDataset.collate_fun)
    testloader       = DataLoader(testset,  1000, collate_fn=dataset.EmbDataset.collate_fun)

    sampleemb = trainingset[0][0]
    model = BNET(sampleemb.size(dim=1), int(np.shape(nplabels)[1]))
    model.to(device)
    print(sampleemb.size(dim=1), int(np.shape(nplabels)[1]))

    optimizer = torch.optim.SGD(model.parameters(), lr=LR, weight_decay=1e-5, momentum=0.9)
    loss_fun  = torch.nn.CrossEntropyLoss()
    scheduler1 = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.9)
    scheduler2 = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, 30)

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
            with open(TRAIN_REACTS, 'w') as fout:
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
            scheduler2.step(avg_loss)
        with open(HRES_OUT, 'a') as trout:
            with open(TRAINING_OUT, 'a') as csvout:
                csvout.write(f'{avg_loss}, {avg_vloss} \n')
                trout.write('epoch: {}  train: {} valid: {} ratio: {} \n'.format(epochidx+1, avg_loss, avg_vloss, avg_vloss/avg_loss))

        if avg_vloss < best_vloss:
            best_vloss = avg_vloss
            bestflag = 1
            torch.save(model.state_dict(), BEST_STATE)
        else:
            bestflag = 0
            torch.save(model.state_dict(), LAST_STATE)
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

            for idx, tdata in enumerate(testloader):
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
            torch.save(tinputlayer, INLAYER)
            torch.save(tlastlayer, OUTLAYER)
            torch.save((classes, ttruelabs, tpredlabs), TESTLABS)
            with open(TEST_REACTS, 'w') as fout:
                for idx, (smarts, predlab, truelab) in enumerate(zip(trxnsmiles, tpredlabs, ttruelabs)):
                    fout.write(smarts + f' predicted: {predlab} gtruth: {truelab} \n')
        with open(HRES_OUT, 'a') as trout:
            trout.write(f'test top1 accuracy: {top1ac} test top3 accuracy: {topkac}\n')

