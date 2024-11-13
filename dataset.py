#This file written by M. Nouman is part of the General DFT Descriptors project
#which is released under the MIT license. See LICENSE file for full license details.

import torch
import random
import numpy as np
from torch.utils.data import Dataset

class EmbDataset(Dataset):
    def __init__(self, valsize=0, testsize=0, train=True, test=False,  rand=True):
        self.embpath   = 'gl2inputembs500.pt'
        self.tuplelist = torch.load(self.embpath)
        if rand == True:
            random.seed(42)
            random.shuffle(self.tuplelist)
        if test == True and train == True:
            self.tuplelist = self.tuplelist[valsize+testsize:]
        elif test == False and train == False:
            self.tuplelist = self.tuplelist[0:valsize]
        elif train == True:
            self.tuplelist = self.tuplelist[valsize:]
        else:
            self.tuplelist = self.tuplelist[valsize:testsize+valsize]

    def __len__(self):
        return len(self.tuplelist)
      
    def __getitem__(self, idx):
        molemb = self.tuplelist[idx][0]
        label  = self.tuplelist[idx][1][0]
        smarts = self.tuplelist[idx][2]

        return molemb, label, smarts
    
    @staticmethod
    def collate_fun(batch):
        molembs  = [item[0] for item in batch]
        labels   = [item[1] for item in batch]
        smarts   = [item[2] for item in batch]

        molembs = torch.concat(molembs, dim=0)
        labels  = np.asarray(labels).reshape(-1, 1) #using np instead of torch to apply sklearn's encoding functions during training
 
        return molembs, labels, smarts

