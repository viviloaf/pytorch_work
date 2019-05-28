#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 27 12:49:54 2019

@author: vinh
Create a folder structure called /data/jtx2

Rename folder of images to "018_11_11_01"

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

import warnings
warnings.filterwarnings("ignore")

jtx2_values = pd.read_csv('data/jtx2/data_2018_11_11_13- Q 2nd floor training set.csv')

n = 30
img_name = jtx2_values.iloc[n, 0]
jtx2 = jtx2_values.iloc[n, 1:].as_matrix()
jtx2 = jtx2.astype('int').reshape(1, 4)

#print('Image name: {}'.format(img_name))
#print('Array shape: {}'.format(jtx2.shape))
#print('Data: {}'.format(jtx2[:4]))

def show_img(image, jtx2):
    """Show images with landmarks"""
    print('Data: {}'.format(jtx2))
    plt.imshow(image)
    #plt.scatter(landmarks[:, 0], landmarks[:, 1], s=10, marker='.', c='r')
    plt.pause(0.001) #pause so plots are updated
    
#plt.figure()
#show_img(io.imread(os.path.join('data/jtx2/', img_name)), jtx2)

#plt.show()
#print(jtx2)
"""
Set up a class to define the dataset so we have some simple functions to give it a name,
find out how many files are in the dataset,
and include both the image and the values (left and right motor) as a callable function so we can know the 
value at any chosen datapoint
"""

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
        jtx2 = self.jtx2_values.iloc[idx, 1:].as_matrix()
        jtx2 = jtx2.astype('int').reshape(1, 4)
        sample = {'image': image, 'jtx2': jtx2}
        
        if self.transform:
            sample = self.transform(sample)
            
        return sample

# Instantiate the class and iterate through the data samples

robot_dataset = RobotDataset(csv_file='data/jtx2/data_2018_11_11_13- Q 2nd floor training set.csv',
                             root_dir='data/jtx2/')

fig = plt.figure()

for i in range(len(robot_dataset)):
    num = 4
    sample = robot_dataset[i]
    
    print('Sample: #{},'.format(i), 'H x W x C: {},'.format(sample['image'].shape), 
          'Array Size: {}'.format(sample['jtx2'].shape))
    
    #ax = plt.subplot(1, num, i+1)
    #plt.tight_layout()
    #ax.set_title('Sample #{}'.format(i))
    #ax.axis('off')
    #show_img(**sample)
    
    if i == (num-1):
        plt.show()
        break
    
# Below is a transform function, just in case

class Rescale(object):
    """
    Rescale the image in a sample to a given size
    
    Args:
        Output size (tuple or itn): Desired Outsize size.
        If Tuple:
            output is matched to output_size
        If Int:
            Smaller of image_edges is matched ot output_size keeping aspect
            ratio the same
    """
    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size
    
    def __call__(self, sample):
        image, jtx2 = sample['image'], sample['jtx2']
        
        h, w = image.shape[:2]
        if isinstance(self,output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size
            
        new_h, new_w = int(new_h), int(new_w)
        
        img = transform.resize(image, (new_h, new_w))
        
        return{'image': img, 'jtx2': jtx2}
        
class RandomCrop(object):
    """
    Ramdomly crops the image in a sample
    
    Args:
        Output_size (tuple or int), Desired output size
        if int: square crop
    """
    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size == 2)
            self.output_size = output_size
    
    def __call__(self, sample):
        image, jtx2 = sample['image'], sample['jtx2']
        
        h, w = image.shape[:2]
        new_h, new_w = self.output_size
        
        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)
        
        image = image[top: top + new_h,
                      left: left + new_w]
        
        jtx2 = jtx2
        
        return {'image': image, 'jtx2': jtx2}
    
# Converts our numpy array (image and jtx2 data) into a tensor so we can feed
#        it into the neural network
class ToTensor(object):
        
    def __call__(self, sample):
        image, jtx2 = sample['image'], sample['jtx2']
        
        #Swap color axis because:
        # Numpy is H x W x C
        # PyTorch is C x H x W
        
        image = image.transpose((2, 0, 1))
        return {'image': torch.from_numpy(image), 
                'jtx2': torch.from_numpy(jtx2)}
    
"""
Load the samples into a DataLoader in batches to be submitted into a NeuralNet
"""
transformed_dataset = RobotDataset(csv_file='data/jtx2/data_2018_11_11_13- Q 2nd floor training set.csv',
                             root_dir='data/jtx2/', 
                             transform=transforms.Compose([ToTensor()]))

#for i in range(len(transformed_dataset)):
#    sample = transformed_dataset[i]
#
#    print(i, sample['image'].size(), sample['jtx2'].size())
#
#    if i == 3:
#        break
batch_size = 4
total_imgs = len(jtx2_values)

dataloader = DataLoader(transformed_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

# Helper function to show a batch so we can see what is happening

def show_jtx2_batch(sample_batched):
    """Show images with motor data for a batch of samples"""
    images_batch, jtx2_batch = \
        sample_batched['image'], sample_batched['jtx2']
    batch_size = len(images_batch)
    im_size = images_batch.size(2)
    grid_border_size = 2
    
    grid = utils.make_grid(images_batch, nrow = 1, padding = 3, )
    plt.imshow(grid.numpy().transpose((1, 2, 0)))
    #print('Data: {}'.format(jtx2_batch))
    
    for i in range(batch_size):
        plt.title('Batch #{} from DataLoader'.format(i_batch))
        
for i_batch, sample_batched in enumerate(dataloader):
    print('Batch #{}'.format(i_batch), 
          'Batch Size, C, H, W : {}'.format(sample_batched['image'].size()))
    print('Batch #{}'.format(i_batch),
          'Batch Size, Tensor Dimension: {}'.format(sample_batched['jtx2'].size()))
    
    # Observe 4th batch and stop
    if i_batch == 4: #(total_imgs/batch_size): #Only use this if iterating the whole dataset
        plt.figure()
        show_jtx2_batch(sample_batched)
        plt.axis('off')
        plt.ioff()
        plt.show()
        print("left_encoder,right_encoder,left_speed,right_speed: \n{}".format(sample_batched['jtx2']))
        print('This shows successful conversion of datatype from a numpy array to a Tensor array')
        break