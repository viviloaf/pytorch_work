# -*- coding: utf-8 -*-
"""
Created on Thu Aug 16 14:41:25 2018

@author: Vinh
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
#import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from torch.autograd import Variable
import time
import copy

#from resnet import resnet18

# Sanity check for CUDA or CPU 
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Load MINST dataset

kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
# This is something I copied from example code online. I do not understand why
# they would do this other than to make that extremely long line below, declaring
# the dataset, shorter. What do you think?
 
# Load data here under one variable and 2 attributes to easily reference later in code
mnist ={
    'train': torch.utils.data.DataLoader(
            datasets.MNIST(
                    './data', 
                    train=True, 
                    download=True, 
                    transform=transforms.Compose([
                            transforms.ToTensor(), 
                            transforms.Normalize((0.1307,), (0.3081,))])), 
#batch_size=100, 
shuffle=True, 
**kwargs),
    
    'test': torch.utils.data.DataLoader(
            datasets.MNIST(
                    './data',
                    train=False,
                    download=False,
                    transform=transforms.Compose([
                            transforms.ToTensor(), 
                            transforms.Normalize((0.1307,), (0.3081,))])), 
#batch_size=100, 
shuffle=True, 
**kwargs),
}

# Convert MNIST greyscale into something Resnet can read
#im = numpy.dstack((im,im,im))

# Easily load MNIST data based on state dataloaders.train or dataloaders.test
dataloaders = {x: torch.utils.data.DataLoader(mnist[x], batch_size=4, shuffle=True, num_workers=4) for x in ['train', 'test']}

# Load the size of datasets to use for statistics later 
dataset_sizes = {x: len(mnist[x]) for x in ['train', 'test']}

# **not sure what this is yet 
#class_names = mnist['train']

# Get a batch of training data to push to the model
#inputs, value = next(iter(dataloaders['train']))

print("Testset: ",len(mnist['train']))
print("Trainset: ", len(mnist['test']))

def __getitem__(index):
    data, target = mnist[index]
    return data, target, index

    x=type(data)
    print(x)
    x=type(target)
    print(x)
    x=type(index)
    print(x)
    