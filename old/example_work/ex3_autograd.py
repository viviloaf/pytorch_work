'''
In the above examples, we had to manually implement both the forward 
and backward passes of our neural network. Manually implementing the 
backward pass is not a big deal for a small two-layer network, but can 
quickly get very hairy for large complex networks.

Thankfully, we can use automatic differentiation to automate the
 computation of backward passes in neural networks. The autograd package 
 in PyTorch provides exactly this functionality. When using autograd, 
 the forward pass of your network will define a computational graph; 
 nodes in the graph will be Tensors, and edges will be functions that
 produce output Tensors from input Tensors. Backpropagating through this 
 graph then allows you to easily compute gradients.

This sounds complicated, it’s pretty simple to use in practice. Each 
Tensor represents a node in a computational graph. If x is a Tensor 
that has x.requires_grad=True then x.grad is another Tensor holding the 
gradient of x with respect to some scalar value.

Here we use PyTorch Tensors and autograd to implement our two-layer
 network; now we no longer need to manually implement the backward pass 
 through the network:
'''

# -*- coding: utf-8 -*-
import torch

dtype = torch.float
device = torch.device("cuda:0")
#device = torch.device("cpu") #uncomment this to run on CPU

# N is bath size: D_in is input dimension:
# H is hidden dimension: D_out is output dimension
N, D_in, H, D_out = 64, 1000, 100, 10

# Create random Tensors for weights
# Setting requires_grad=False indicates that we do not need ot compute gradients
# with respect to these Tenspors during the backward pass

x = torch.randn(N, D_in, device=device, dtype=dtype)
y = torch.randn(N, D_out, device=device, dtype=dtype)

# Create random tensors for weights
# Setting requires_grad=True indicates that we want to compute gradients with
# respect to these Tensors during the backward pass.

w1 = torch.randn(D_in, H, device=device, dtype=dtype, requires_grad=True)
w2 = torch.randn(H, D_out, device=device, dtype=dtype, requires_grad=True)

learning_rate = 1e-6
for t in range(500):
	# Forward pass: compute predicted y using operations on tensors: these
	# are exactly the same operations we used to cpkpute the forward pass using
	# tensors, but we do not need to keep references to intermediate values since
	# we are not implement the backward pass by hand
	
	y_pred = x.mm(w1).clamp(min=0).mm(w2)
	
	# Compute and print loss using operations on Tensors.
	# Now loss is a Tensor of shape (1,)
	# loss.item() gets the a scalar value held in the loss.
	
	loss = (y_pred - y).pow(2).sum()
	print(t, loss.item())
	
	# use autograd to compute with backwards pass. This call will compute the
	# gradient of loss with respect to all Tensors with requires_grad=True.
	# After this call w1.grad and w2.grad will be Tensors holding the gradient
	# of the loss with respect to w1 and w2 respectivetly
	loss.backward()
	
	# Manually update weights using gradient descent. Wrap in torch.no_grad()
	# because weights have requires_grad=True, but we don't need to track this
	# in autograd.
	# An alternative way is to operate on weight.data and weight.grad.data.tensor
	# but that doesn't track history.
	# You can also use torch.optim.SGD to achieve this.
	with torch.no_grad():
		w1 -= learning_rate * w1.grad
		w2 -= learning_rate * w2.grad
	
		# Manually zero the gradients after updating weights
		w1.grad.zero_()
		w2.grad.zero_()
