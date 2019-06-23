#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 28 18:33:40 2019

@author: vinh
"""
from __future__ import print_function, division
import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


import warnings
warnings.filterwarnings("ignore")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
cuda = torch.cuda.is_available()

jtx2_values = pd.read_csv('data/jtx2/data_2018_11_11_13- Q 2nd floor training set.csv')

n = 30
img_name = jtx2_values.iloc[n, 0]
#jtx2 = jtx2_values.iloc[n, 1:].as_matrix()
#jtx2 = jtx2.astype('int').reshape(1, 4)

#def show_img(image, jtx2):
#    """Show images with landmarks"""
#    print('Data: {}'.format(jtx2))
#    plt.imshow(image)
#    #plt.scatter(landmarks[:, 0], landmarks[:, 1], s=10, marker='.', c='r')
#    plt.pause(0.001) #pause so plots are updated
    
class RobotDataset(Dataset):
    """Movement Dataset"""
    
    def __init__(self, csv_file, root_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file containing the motor/encoder
                values
            root_dir (string): Directory with all the images
            transform (callable, optional): Optional image transforms to be applied
                on a sample
        """
        self.jtx2_values = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        
    def __len__(self):
        return len(self.jtx2_values)
    
    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.jtx2_values.iloc[idx, 0])
        image = io.imread(img_name)
        jtx2 = self.jtx2_values.iloc[idx, 3:].as_matrix()
        jtx2 = jtx2.astype('double').reshape(1, 2)
        sample = {'image': image, 'jtx2': jtx2}
        
        if self.transform:
            sample = self.transform(sample)
            
        return sample
    
class ToTensor(object):
        
    def __call__(self, sample):
        image, jtx2 = sample['image'], sample['jtx2']
        
        #Swap color axis because:
        # Numpy is H x W x C
        # PyTorch is C x H x W
        image = image.transpose((2, 0, 1))
        return {'image': torch.from_numpy(image).float(), 
                'jtx2': torch.from_numpy(jtx2).float()}

        
transformed_dataset = RobotDataset(csv_file='data/jtx2/data_2018_11_11_13- Q 2nd floor training set.csv',
                             root_dir='data/jtx2/', 
                             transform=transforms.Compose([ToTensor()]))

batch_size = 4
total_imgs = len(jtx2_values)

dataloader = DataLoader(transformed_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

# Helper function to show a batch so we can see what is happening

# ==============================================================================
#def show_jtx2_batch(sample_batched):
#    """Show images with motor data for a batch of samples"""
#    images_batch, jtx2_batch = \
#        sample_batched['image'], sample_batched['jtx2']
#    batch_size = len(images_batch)
#
#    
#    grid = utils.make_grid(images_batch, nrow = 1, padding = 3, )
#    plt.imshow(grid.numpy().transpose((1, 2, 0)))
#    #print('Data: {}'.format(jtx2_batch))
#    
#    for i in range(batch_size):
#        plt.title('Batch #{} from DataLoader'.format(i_batch))
#        
#for i_batch, sample_batched in enumerate(dataloader):
#    print('Batch #{}'.format(i_batch), 
#          'Batch Size, C, H, W : {}'.format(sample_batched['image'].size()))
#    print('Batch #{}'.format(i_batch),
#          'Batch Size, Tensor Dimension: {}'.format(sample_batched['jtx2'].size()))
#    
#    # Observe 4th batch and stop
#    if i_batch == 4: #(total_imgs/batch_size): #Only use this if iterating the whole dataset
#        plt.figure()
#        show_jtx2_batch(sample_batched)
#        plt.axis('off')
#        plt.ioff()
#        plt.show()
#        print("left_encoder,right_encoder,left_speed,right_speed: \n{}".format(sample_batched['jtx2']))
#        #print('This shows successful conversion of datatype from a numpy array to a Tensor array')
#        break
# =============================================================================
    
class EndToEndNet(nn.Module):
    def __init__(self):
        super(EndToEndNet, self).__init__()
# =============================================================================
#       1 Input image channel
#       2 output channels      
# =============================================================================
        self.num_outputs = 2
        self.pool  = nn.MaxPool2d(2, 2)
        self.conv1 = nn.Conv2d(3, 24, 5, padding = (0,2))
        self.conv2 = nn.Conv2d(24, 36, 5, padding = (0,2))
        self.conv3 = nn.Conv2d(36, 48, 5, padding = (0,2))
        self.conv4 = nn.Conv2d(48, 64, 5, padding = (0,2))
        self.conv5 = nn.Conv2d(64, 128, 3, padding = (0,1))
        self.fc1   = nn.Linear(1280,100)
        self.fc2   = nn.Linear(100,50)
        self.fc3   = nn.Linear(50,10)
        self.fc4   = nn.Linear(10,self.num_outputs)

    def forward(self, x):
        x = x.type(torch.cuda.FloatTensor)
        x = self.pool(x)
        #print(x.shape)
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        #print(x.shape)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        #print(x.shape)
        x = F.relu(self.conv3(x))
        x = self.pool(x)
        #print(x.shape)
        x = F.relu(self.conv4(x))
        x = self.pool(x)
        #print(x.shape)
        x = F.relu(self.conv5(x))
        #print(x.shape)
        x = x.view(x.shape[0],-1)
        #print(x.shape)
        x = F.relu(self.fc1(x))
        #print(x.shape)
        x = F.relu(self.fc2(x))
        #print(x.shape)
        x = F.relu(self.fc3(x))
        #print(x.shape)
        x = self.fc4(x)
        #print(x.shape)
        return x

# Initialize Model
net = EndToEndNet().to(device)
    
# Initialize Optimizer
# Placeholder

for i in range(2):
    count = 0
    total = 0
    for i_batch, sample_batched in enumerate(dataloader):
        sample_batched['image'] = sample_batched['image'].cuda()
        sample_batched['jtx2'] = sample_batched['jtx2'].cuda()
        criterion = nn.MSELoss()                              # Define loss function
        optimizer = optim.Adam(net.parameters(),lr=0.00006)   # Initialize optimizer
        optimizer.zero_grad()                                 # Clear gradients
        predict = net(sample_batched['image'])                        # Forward pass
        loss = criterion(predict, sample_batched['jtx2'])            # Calculate loss
        loss.backward()                                       # Backward pass (calculate gradients)
        optimizer.step()                                      # Update tunable parameters                                # Forward pass
        print('Epoch',i,'batch',i_batch,'loss', float(loss.item()))
        total += float(loss.item())
        count += 1
    print('Epoch',i,'AVG loss',total/count)
torch.save(net.state_dict(), './net.pt')
