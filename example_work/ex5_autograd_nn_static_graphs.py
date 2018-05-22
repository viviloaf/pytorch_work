'''
Computational graphs and autograd are a very powerful paradigm for 
defining complex operators and automatically taking derivatives; 
however for large neural networks raw autograd can be a bit too low-level.

When building neural networks we frequently think of arranging the 
computation into layers, some of which have learnable parameters which 
will be optimized during learning.

In TensorFlow, packages like Keras, TensorFlow-Slim, and TFLearn provide 
higher-level abstractions over raw computational graphs that are useful 
for building neural networks.

In PyTorch, the nn package serves this same purpose. The nn package 
defines a set of Modules, which are roughly equivalent to neural network 
layers. A Module receives input Tensors and computes output Tensors, but 
may also hold internal state such as Tensors containing learnable 
parameters. The nn package also defines a set of useful loss functions 
that are commonly used when training neural networks.

In this example we use the nn package to implement our two-layer network:
'''
# -*- coding: utf-8 -*-
import torch

# N is batch size; D_in is input dimension:
# H is hidden dimension; D_out is output dinemsion
N, D_in, H, D_out = 64, 1000, 100, 10

# Create Random tensors to hold inputs and outputs
x = torch.randn(N, D_in)
y = torch.randn(N, D_out)

# Use the nn package to define our model as a sequence of layers. 
# nn.Sequential is a moduel which contains other Modules, and applies them in
# sequence to produce its output. Each linear model computes output from
# input using a linear function, and holds Internal Tensors for its weight 
# and bias

model = torch.nn.Sequential(
	torch.nn.Linear(D_in, H),
	torch.nn.ReLU(),
	torch.nn.Linear(H, D_out),
)

# The NN package also contains definitions of popular loss functions: in this
# case we will use Mean Squared Error (MSE) as our loss function

loss_fn = torch.nn.MSELoss(size_average=False)

learning_rate = 1e-4 #why is this larger than the previous examples, 1e-6?
for t in range(500):
	'''
	Forward Pass: compute predicted y by passing x to the model. Module
	objects override the __call__ operator so you can call them like
	functions. When doing so, you pass a Tenspr of input data to the Moduel
	and it produces a Tensor of otuput data
	'''
	y_pred = model(x)
	
	'''
	Compute and print loss. We pass Tensors containing the predicted and true
	values of y, and the loss function returns a Tensor containing the
	loss
	'''
	loss = loss_fn(y_pred, y)
	print(t, loss.item())
	
	#Zero the gradients before running the backwards pass
	model.zero_grad()
	
	'''
	Backwards pass: compute gradient of the loss with respect to all of 
	the learnable parameters of the model. Internally, the parameters
	of each moduel are stored in Tensors with requres_grad=True, so this call
	will compute gradients for all learnable parameters in the model
	'''
	loss.backward()
	
	'''
	Update the weights using gradient descent. Each parameter is a Tensor,
	so we can access and gradients like we did before
	'''
	with torch.no_grad():
		for param in model.parameters():
			param -= learning_rate * param.grad
