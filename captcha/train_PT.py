"""
File name: train_PT.py
Author: Rechatin Hugo
Date created: Jun 13, 2022

This script can trains any model given te proper HP as arguments. By default trains a Pnet
"""
from utils.wnet_PT import Wnet  
from utils.pnetcls_PT import Pnet


from getdata_PT import Dataset_np_1

from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision.transforms import Normalize
from torchvision.transforms import Compose 
from torchvision.transforms import ToTensor 
import torch.optim as optim
from tqdm import tqdm
import torch
import torch.nn as nn
import numpy as np
from utils.metrics_PT import dice_coeff, dice_loss,GDLoss
import time

def train_PT_model(train_patch_dir, label_patch_dir, model_filepath, 
               validation_split = 0.2,batch_size = 64, 
               patch_size = 96, normalize = True, pixel_wise = False, 
               epochs = 100, Model = Pnet,
               lr = 0.01, momentum = 0, criterion = nn.BCELoss(), optimizer = torch.optim.SGD):
    
#DATA RETRIVAL
    data = Dataset_np_1(train_patch_dir, label_patch_dir, normalize, pixel_wise)
    dataset_size = data.__len__()
    
#MODEL, CRITERION & OPTIMIZER#
    if Model == Pnet:
        model = Model(patch_size)
        opt = optimizer(model.parameters(), lr, momentum)
    elif Model == Wnet :
        model = Model()
        opt = optimizer(model.parameters(), lr)

#TRAIN/VALIDATION SPLIT#    
    idxs = list(range(dataset_size))
    split = int(np.floor(validation_split*dataset_size))
    train_idxs ,val_idxs = idxs[split:], idxs[:split]

#SAMPLER & DATALOADER#
    train_sampler = SubsetRandomSampler(train_idxs)
    val_sampler = SubsetRandomSampler(val_idxs)
    
    train_dataloader = DataLoader(data, batch_size = batch_size, sampler = train_sampler)
    val_dataloader = DataLoader(data, batch_size=batch_size, sampler = val_sampler)
    
#TRANING#
    min_val_loss = np.inf
    for epoch in tqdm(range(epochs)):
#MODEL FITTING#
        train_loss = 0.0
        start_loading = time.time()
        for samples, labels in train_dataloader:
    #FIT
            #samples.to(device) #for colab
            end_loading = time.time()
            print("loading: ", end_loading - start_loading)
            print(train_loss)
            start_process = time.time()
            opt.zero_grad()
            out = model(samples)
            loss = criterion(out, labels)
            loss.backward()
            print(loss.item())
            opt.step()
            end_process = time.time()
            print("process: ", end_process - start_process)
            start_loading = time.time()
            
            train_loss += loss.item()
#VALIDATION#
        val_loss = 0.0
        for samples, labels in val_dataloader:
    #VALIDATE
            out = model(samples)
            loss = criterion(out, labels)
            val_loss += loss.item()
        if(min_val_loss > val_loss):
            min_val_loss = val_loss
    torch.save(model.state_dict(), model_filepath)
