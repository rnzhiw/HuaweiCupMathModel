import pandas as pd
import numpy as np
from torch.utils import data
import torch.nn as nn
import torch
import os

pd.set_option('display.max_columns', None)


class DataLoader(data.Dataset):
    def __init__(self, split):
        self.split = split
        if split == 'train' or split == 'valid':
            feature = pd.read_csv('data/Molecular_Descriptor.csv', index_col='SMILES').values.tolist()
            label = pd.read_csv('data/ADMET.csv', index_col='SMILES').values.tolist()
            
            feature_train, label_train = [], []
            feature_valid, label_valid = [], []

            for i in range(0, 1974):
                if i % 10 != 0:
                    feature_train.append(feature[i])
                    label_train.append(label[i])
                else:
                    feature_valid.append(feature[i])
                    label_valid.append(label[i])
                    
            if split == 'train':
                self.feature = np.array(feature_train)
                self.label = np.array(label_train)
            elif split == 'valid':
                self.feature = np.array(feature_valid)
                self.label = np.array(label_valid)
            else:
                print("split must in [train, valid]")
              
        elif split == 'test':
            feature = pd.read_csv("data/Molecular_Descriptor_test.csv", index_col='SMILES').values.tolist()
            self.feature = np.array(feature)
        else:
            print('Error: split must be train, valid or test!')
        

    def __len__(self):
        return len(self.feature)
    
    def __getitem__(self, index):
        if self.split == 'train' or self.split == 'valid':
            x = torch.from_numpy(self.feature[index]).float()
            y = torch.from_numpy(np.array(self.label[index])).float()
            return x, y

        elif self.split == 'test':
            x = torch.from_numpy(self.feature[index]).float()
            return x

    
if __name__ == '__main__':
    dataloader = DataLoader(split='train')
    print(len(dataloader))
    x, y = dataloader.__getitem__(0)
    print(x.shape, y.shape)
    
    dataloader = DataLoader(split='valid')
    print(len(dataloader))
    x, y = dataloader.__getitem__(0)
    print(x.shape, y.shape)

    dataloader = DataLoader(split='test')
    x = dataloader.__getitem__(0)
    print(x.shape)
