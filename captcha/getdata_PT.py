"""
File name: train_pnetcls_PT.py
Author: Hugo Rechatin
Date created: Feburary 6, 2022

This script defines the Dataset class that gets patches and labels from a given directory.

Because i wasn't sure how the patches/labels would be organised I tried to cover multiple cases.

Seeing as the preprocessing script seg_patch_extraction.py generats .npy and train_pnetcls.py 
loads the patches from npy files I assumed in any case the data would be storred in npy files
 
"""

import numpy as np
import os
import torch
from torch.utils.data import Dataset
from torchvision.transforms import Normalize

#%%
'''
__init__
    Inputs
        patch_dir : folder with .npy files containing patches
        label_dir : folder with .npy files containing either 
                        - an array of patch-wise labels | (shape = (number of patches))
                        - an array of 2D arrays of pixel-wise labeled patches | (shape = (number of patches, patch_size, patch_size))
        normalize : bool, if True, loaded patch tensors will be normalized
        pixel_wise : bool, if True, considers label_dir as contaning .npy of 2D arrays of pixel-wise labeled patches
                     else, 
                     either converts 2D array of pixel-wise labels to 1D array of patch wise labels
                     either, if the labels were already 1D array changes nothing
__getitem__
    Output
        patch : if normalize
                    normalized tensor of size [1, patch_size, patch_size]
                else
                    tensor of size [1, patch_size, patch_size]
        label : if pixel_wise
                    tensor of size [1,patch_size,patch_size] | 1 => vessel else 0 
                else
                    tensor of size [1] | 1 => vessel else 0
'''
class Dataset_np_1(Dataset):
    def __init__(self, patch_dir, label_dir, normalize = False, pixel_wise = False):
        patch_list = []
        label_list = []
        i = 0
        for idx, patch_file in enumerate(os.listdir(patch_dir)):
            patch = (np.load(os.path.join(patch_dir,patch_file)))
            label = (np.load(os.path.join(label_dir, os.listdir(label_dir)[idx])))
            patch_list[i:i+patch.shape[0]] = patch[:]
            label_list[i:i+label.shape[0]] = label[:]
            i += patch.shape[0]
        self.patches =  np.array(patch_list)
        self.labels = np.array(label_list)
        self.normalize = normalize
        self.pixel_wise = pixel_wise
    def __len__(self):
        return len(self.patches)
    
    def __getitem__(self, idx):
        patch = torch.from_numpy(self.patches[idx]).float() 
        patch = patch[None, : , :] # create a dummy dimension for the channels(=1) to get the conventional [B,C,H,W] tensors when using the dataloader
        if self.pixel_wise:
            label = torch.from_numpy(self.labels[idx])[None, : , :].float()
        else :
            if (len(self.labels.shape)==1) : #check if labels are pixel wise or patch wise 
                label = torch.tensor(np.full((1), self.labels[idx])).float()
            else : 
                label = torch.tensor(np.full((1), np.max(self.labels[idx]))).float() # the np.max allows us to get patch wise regardless of the shape of the labels .npy
        if self.normalize:
            patch = Normalize(torch.mean(patch), torch.std(patch))(patch)
        return patch, label

#%%
'''
__init__
    Inputs
        patches_dir : folder with .npy files containing patches
                      each file name indicates it's patches labels 
        Normalize : bool, if True loaded patch tensors will be normalized
__getitem__
    Output
        patch : if normalize
                    normalized tensor of size [1, patch_size, patch_size]
                else
                    tensor of size [1, patch_size, patch_size]
        label : tensor of size [1] with 1 <=> vessel else 0 
'''
class Dataset_np_2(Dataset):
    def __init__(self, patch_dir, Normalize = False):
        patch_list = []
        label_list = []
        i = 0
        for patch_file in os.listdir(patch_dir):
            patch = (np.load(os.path.join(patch_dir,patch_file)))
            patch_list[i:i+patch.shape[0]] = patch[:]

            if patch_file.find('_vessel') == -1: 
                label_list[i:i+patch.shape[0]] = np.zeros(i+patch.shape[0] - i)
            else : label_list[i:i+patch.shape[0]] = np.ones(i+patch.shape[0] - i)
            
            i += patch.shape[0]
        self.patches =  np.array(patch_list)
        self.labels = np.array(label_list)
        self.Normalize = Normalize
        
    def __len__(self):
        return len(self.file_list)
        
    def __getitem__(self, idx):
        patch = torch.from_numpy(self.patches[idx]).float() 
        patch = patch[None, : , :] 
        label = torch.tensor([self.labels[idx]])
        if self.Normalize:
            patch = Normalize(torch.mean(patch), torch.std(patch))(patch)
        return patch, label
