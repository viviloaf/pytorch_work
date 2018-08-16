import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
#import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
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



###############################################################################
# Function to train a model

def train_model(model, criterion, optimizer, scheduler, num_epochs = 25):
    since = time.time()
    
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs -1))
        print('-' * 10)
        
        # Each epoch has a training and validation phase
        for phase in ['train', 'test']:
            if phase == 'train':
                scheduler.step()
                model.train() # Send data to model 1
            else:
                model.eval() # Set model to evaluate mode
                
            running_loss = 0.0
            running_corrects = 0
            
            # Training Model
            for batch_idx, (data, target) in enumerate(mnist['train']):
                data, target = data.to(device), target.to(device)
                
                # Zero the gradients
                optimizer.zero_grad()
                
                # forward pass
                # track history only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(data)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, target)
                    
                    # backward + optimize only if training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                
                # statistics
                running_loss +=loss.item() * data.size(0)
                running_corrects += torch.sum(preds == target.data)
                
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]
            
            print('{} Loss" {:.4f} Accuracy: {:.4f}'.format(phase, epoch_loss, epoch_acc))
            
            # Deep copy the model
            # Copy the model and make a new model
            if phase == 'test' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                
        print()
    
    time_elpased = time.sim - since
    print('Training complete in {:.0f}m {;.0f}s'.format(time_elpased // 60, time_elpased % 60))
    print('Best test accuracy: {;.4f}'.format(best_acc))
    
    # Load best model weighs
    model.load_state_dict(best_model_wts)
    return model
###############################################################################
# Finetuning the Convolution NN
# Load a pretrained model and reset final fully connected layer
# =============================================================================
# NOTE:
# All pre-trained models expect input images normalized in the same way, i.e. 
# mini-batches of 3-channel RGB images of shape (3 x H x W), where H and W are 
# expected to be at least 224. The images have to be loaded in to a range of 
# [0, 1] and then normalized using mean = [0.485, 0.456, 0.406] and 
# std = [0.229, 0.224, 0.225]. You can use the following transform to 
#normalize:
#
#normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                             std=[0.229, 0.224, 0.225])
# =============================================================================
#Creating a Convolutional Layer to accept 1 channel input and output 3 channels
    
class channel_add(torch.nn.Module):
    def __init__(self):
        super(channel_add, self).__init__()
        self.conv1 = nn.Conv2d(1, 3, kernel_size = 7)
    
    def forward(self, output):
        output = self.conv1(output)
        return output
    
#
#conversion_model = channel_add()
#conversion_model = conversion_model.to(device)

model_ft = models.resnet18(pretrained=True)
#model_ft = resnet18(pretrained=False)
num_ftrs = model_ft.fc.in_features
model_ft.fc = nn.Linear(num_ftrs, 10)

model_ft = model_ft.to(device)



criterion = nn.CrossEntropyLoss()

# Observe that all parameters are being optimized
optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)

# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

###############################################################################
# Use ConvNet as a fixed feature extrator

# Freeze all the network except the final layer
# requires_grad == False
# This freezes the prarameters so that gradients are not computed in 'backwards'\
# See here: http://pytorch.org/docs/notes/autograd.html#excluding-subgraphs-from-backward

#model_conv = resnet18(pretrained=False)
model_conv = models.resnet18(pretrained=True)
for param in model_conv.parameters():
    param.requires_grad = False
    
# Parameters of newly constructed modules have requires_grad true by default

num_ftrs = model_conv.fc.in_features
model_conv.fc = nn.Linear(num_ftrs, 2)

model_conv = model_conv.to(device)

criterion = nn.CrossEntropyLoss()

# Final layer being optimized
optimizer_conv = optim.SGD(model_conv.fc.parameters(), lr=0.001, momentum=0.9)

# Decay LR by factor of 0.1 every 7 epochs
# **note, why?
ex_lr_scheduler = lr_scheduler.StepLR(optimizer_conv, step_size=7, gamma=0.1)

###############################################################################
# Train and Evaluate

#model_ft = train_model(conversion_model, model_ft, criterion, optimizer_ft, exp_lr_scheduler, num_epochs = 25)
model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler, num_epochs = 25)