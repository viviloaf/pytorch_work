from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy

plt.ion()   # interactive mode

# Load MINST

# Data to train neural network
train_data = torch.utils.data.DataLoader(
torchvision.datasets.MNIST('./files/', train=True, download=True,
						 transform=torchvision.transforms.Compose([
						   torchvision.transforms.ToTensor(),
						   torchvision.transforms.Normalize(
							 (0.1307,), (0.3081,))
						 ])),
batch_size=1000, shuffle=True)

# Separate data to test against network
test_data = torch.utils.data.DataLoader(
torchvision.datasets.MNIST('./files/', train=False, download=True,
						 transform=torchvision.transforms.Compose([
						   torchvision.transforms.ToTensor(),
						   torchvision.transforms.Normalize(
							 (0.1307,), (0.3081,))
						 ])),
batch_size=100, shuffle=True)
examples = enumerate(test_data)
batch_idx, (data, target) = next(examples)
data.shape

print("There are:",data.shape,";1000 examples of 28x28 pixels in greyscale")

# Sanity check for CUDA or CPU 
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Train the model

					 
def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
	model.train()
	
	for batch_idx, (data, target) in enumerate(train_data):
		optimizer.zero_grad()
		
		if torch.device("cuda:0"):
			data, target = data.cuda(), target.cuda()
			
		data, target = Variable(data), Variable(target)
		output = model(data)
		loss = F.nll_loss(output, target)
		loss.backward()
		optimizer.step()
		
		if batch_idx % log_interval == 0:
		  print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
			epoch, batch_idx * len(data), len(train_loader.dataset),
			100. * batch_idx / len(train_loader), loss.item()))

# Display predictions at the end
def visualize_model(model, num_images=6):
    was_training = model.training
    model.eval()
    images_so_far = 0
    fig = plt.figure()

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloaders['val']):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            for j in range(inputs.size()[0]):
                images_so_far += 1
                ax = plt.subplot(num_images//2, 2, images_so_far)
                ax.axis('off')
                ax.set_title('predicted: {}'.format(class_names[preds[j]]))
                imshow(inputs.cpu().data[j])

                if images_so_far == num_images:
                    model.train(mode=was_training)
                    return
        model.train(mode=was_training)
        
# Load Pretrained model, Reset18, reset the final fully connected layer
# aka "chop off"
model_ft = models.resnet18(pretrained=True)
num_ftrs = model_ft.fc.in_features
model_ft.fc = nn.Linear(num_ftrs, 2)

model_ft = model_ft.to(device)

criterion = nn.CrossEntropyLoss()

# Observe that all parameters are being optimized
optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)

# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

# Train and evaluate the data
model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler,
                       num_epochs=25)
                       
# show sample between loaded images and predictions                       
visualize_model(model_ft)
