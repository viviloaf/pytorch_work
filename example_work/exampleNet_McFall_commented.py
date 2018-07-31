# -*- coding: utf-8 -*-
"""
Created on Tue May 29 08:33:28 2018

@author: kmcfall
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plot
import torch.optim as optim

class Net(nn.Module): # Define class for the desired neural network architecture
    def __init__(self, H=10):                      # Default to 10 hidden nodes
    #i read online that the number of hidden layers doesn't matter as much and you keep adding more until you start getting the results you want. I think this part of development would be avoided by using a pre-existing model with its own weights or by transfer learning by taking the previous weights of a similar, larger dataset and then training it on our specific data
    # Choosing architecture is a bit trial and error. Indeed a good approach is to start small and keep adding until you get acceptable results.
    # However, failure to get good results could be with other hyperparameters like learning rate or the data itself.
    # Transfer learning with an established architecture is a good way to go, or at least base our architecture on something already known to solve similar problems.
  
        super(Net, self).__init__()                # Inherit from torch Module class
        self.num_inputs = 2                        # Number of inputs (x and y)
        #in the context of image processing, there would be 3 inputs: a [m,n] 2d matrix representing the input image after aby resizing calculations, a throttle value, a theta steering value? Or would it be an input for each pixel value
        # Our inputs will be 2D, and indeed all pixels are inputs. This number of inputs however would refer to the number of channels. If the input image is grayscale, then there would be a single input, which is a 2D matrix. If the input image is RGB then the number of inputs would be 3, again each a 2D matrix.
        # The input tensor x (in the forward function) would have 4 axes, the first is for each sample in the batch, the second the number of channels (1 for grayscale, 3 for RGB), and the third and fourth are row and column for each pixel in each input channel
        
        self.num_classes = 2                       # Number of outputs (two classes) 
        #unsure what total outputs would be. There would be theta and throttle, but the actual nagivation would be handled in the hidden layers and we don't really care what goes on in there, right?
        # The outputs would be steering angle and throttle (for Ackerman steering) or the two wheel speeds (for differential steering)
        
        self.hidden = nn.Linear(self.num_inputs,H) # Fully connected layer
        #this allows for back prop, so the data knows where it came from, right?
        # Not sure what you mean - indeed the weights and biases of fully connected layers are optimized using backpropogation, but you can do backpropagation on other structures like convolutions
        self.out = nn.Linear(H,self.num_classes)   # One output for each class
        self.soft = nn.Softmax(1)                  # Apply 1D softmax so that outputs sum to one
        #not sure what a 1d softmax is
        # Softmax is a tool for classification problems (our problem will be regression, not classification) where the output for each class are constrained to be between 0 and 1, and sum to 1 so they can be interpreted as probabilities

    def forward(self, x):
        x = self.hidden(x)  # Hidden layer
        x = F.relu(x)       # Activation function
        x = self.out(x)     # Output layer
        #x = F.relu(x)       # Activation function
        return self.soft(x) # Outputs are probabilities for each class

# Training input data is the exclusive OR function with some noise
xTrain = [0,0,0,0,1,1,1,1,1,1,1,1,0,0,0,0] + np.random.rand(16)/3 - 1/3/2
yTrain = [0,0,0,0,1,1,1,1,0,0,0,0,1,1,1,1] + np.random.rand(16)/3 - 1/3/2
# Torch expects samples in the first dimension (batch) and feature values in second dimension
trainInput = np.column_stack([xTrain,yTrain])
trainInput = torch.from_numpy(trainInput).float() # Convert numpy to torch tensor
trainLabel = torch.tensor([0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1]) # Ground truth labels for inputs
# Loss does not get sufficiently small for some choices of starting weights,
# so loop until the loss is acceptably low.
loss = 1 # Ensure loop starts
while loss > -0.9:                           # Check for low loss
#why
# After observing the loss with networks that didn't learn completely, I noticed that successful networks always had loss lower than -0.9 so I set the threshld there.
# I also tried all sorts of different numbers of hidden nodes and even added more data but still had the same problem. It would appear that the error surface for this particular problem has some large plateaus that are difficult to avoid.
    net = Net()                              # Instantiate object for the network
    criterion = nn.NLLLoss()                 # Identify the loss function
    optimizer = optim.SGD(net.parameters(), lr=0.1, momentum=0.9) # Choose an optimizer
    for i in range(100):                     # Loop through 100 training epochs
    #this one doesn't immediately make sense because you input 100 but are stating there will be 200 training sessions, epochs
    # My bad, I changed the number of epochs but did not update the comment
    
        optimizer.zero_grad()                # Zero out any old gradients
        output = net(trainInput)             # Forward pass
        #this is so the data knows where it came from?
        # This is the forward pass, also called inference. The input data in trainInput is fed into the network, which retuns the output for the current value of the weights (and biases) in the variable output.
        
        loss = criterion(output, trainLabel) # Compute loss
        loss.backward()                      # Comptue gradients in backward pass
        optimizer.step()                     # Update weights
res = 60 # Resolution for image of decision boundary
# Get all (x,y) combinations in the region of interest
x,y = np.meshgrid(np.linspace(-0.5,1.5,res),np.linspace(-0.5,1.5,res))
# Format inputs to be samples in first dimension, features in second dimension
testInput = np.column_stack([x.reshape(-1,1),y.reshape(-1,1)])
testInput = torch.from_numpy(testInput).float()     # Convert numpy to torch tensor
#why is part of the training done in nunpy before using pytorch tensors?
# No training is actually happening here, simply creating the input data (which would be loaded from image files in our case) and then formatting it correctly the way pyTorch likes it.
# Perhaps there is a way to do this directly in pyTorch, but I already knew how to do it in numpy so I just converted. It seems like this is typical since pyTorch makes a big deal of easily converting to and from numpy.

testOutput = net(testInput)                         # Forward pass
z = testOutput[:,1]                                 # Take only class 1 output (class 0 output is (1 - class 1 output))
z = z.view(res,res)                                 # Reshape back to meshgrid format
z = z.detach().numpy()                              # Convert to numpy
plot.figure(1)                                      # Open figure
plot.clf()                                          # Clear figure
plot.imshow(z,cmap='gray')                          # Display decision boundary
plot.plot(xTrain[0:8]*30+15,yTrain[0:8]*30+15,'*m') # Display class 0 training data in magenta
plot.plot(xTrain[8: ]*30+15,yTrain[8: ]*30+15,'*c') # Display class 1 training data in cyan

