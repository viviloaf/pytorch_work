'''
Under the hood, each primitive autograd operator is really two functions 
that operate on Tensors. The forward function computes output Tensors 
from input Tensors. The backward function receives the gradient of the 
output Tensors with respect to some scalar value, and computes the 
gradient of the input Tensors with respect to that same scalar value.

In PyTorch we can easily define our own autograd operator by defining a 
subclass of torch.autograd.Function and implementing the forward and 
backward functions. We can then use our new autograd operator by 
constructing an instance and calling it like a function, passing Tensors 
containing input data.

In this example we define our own custom autograd function for performing 
the ReLU nonlinearity, and use it to implement our two-layer network:
'''
# -*- coding: utf-8 -*-

import torch

class MyReLU(torch.autograd.Function):
	'''
	We can implement our own custom autograd Functions by subclassing 
	torch.autograd.Function and implemeting the forward and backward 
	passes which operate on tensors
	'''
	@staticmethod
	def forward(ctx, input):
		"""
		In the forward pass we recieve a Tensor containing the input and 
		return a Tensor containing the output. ctx is a context object
		that can be used to stash information for use in the backwards 
		pass using the ctx.save_for_backward method
		"""
		ctx.save_for_backward(input)
		return input.clamp(min=0)
		
	@staticmethod
	def backward(ctx, grad_output):
		'''
		In the backward pass we recieve a Tensor containing the gradient
		of the loss with respect to the output. and we need to compute 
		the gradient of the loss with respect to the input
		'''
		input, = ctx.saved_tensors
		grad_input = grad_output.clone()
		grad_input[input < 0] = 0
		return grad_input
		
dtype = torch.float
cuda = torch.device("cuda:0")

# N is batch size; D_in is input dimensions
# H is hidden dimension; D_out is output dimension
N, D_in, H, D_out = 64, 1000, 100, 10

# Create random Tensors to hold input and output
x = torch.randn(N, D_in, device=cuda, dtype=dtype)
y = torch.randn(N, D_out, device=cuda, dtype=dtype)

# Create random Tensors for weights
w1 = torch.randn(D_in, H, device=cuda, dtype=dtype, requires_grad=True)
w2 = torch.randn(H, D_out, device=cuda, dtype=dtype, requires_grad=True)

learning_rate = 1e-6
for t in range(500):
	# To apply our function, we use Function.apply method. We alias this as 
	# 'relu'
	relu = MyReLU.apply
	
	# Forward pass: Compute predicted y using operations: we compute
	# ReLU using our custom autograd operation
	y_pred = relu(x.mm(w1)).mm(w2)
	
	# Compute and print loss
	loss = (y_pred - y).pow(2).sum()
	print(t, loss.item())
	
	# Use autograd to compute the backward pass
	loss.backward()
	
	#Update Weights using gradient descent
	with torch.no_grad():
		w1 -= learning_rate * w1.grad
		w2 -= learning_rate * w2.grad
		
		# Manually zero the gradients after updating weights
		w1.grad.zero_()
		w2.grad.zero_()
