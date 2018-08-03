import torch #Import the pytorch enviroment to python
import torchvision #load datasets

#import and load data
#
train_loader = torch.utils.data.DataLoader(
torchvision.datasets.MNIST('./files/', train=True, download=True,
						 transform=torchvision.transforms.Compose([
						   torchvision.transforms.ToTensor(),
						   torchvision.transforms.Normalize(
							 (0.1307,), (0.3081,))
						 ])),
batch_size=64, shuffle=True)

test_loader = torch.utils.data.DataLoader(
torchvision.datasets.MNIST('./files/', train=False, download=True,
						 transform=torchvision.transforms.Compose([
						   torchvision.transforms.ToTensor(),
						   torchvision.transforms.Normalize(
							 (0.1307,), (0.3081,))
						 ])),
batch_size=1000, shuffle=True)
examples = enumerate(test_loader)
batch_idx, (data, target) = next(examples)
data.shape
print("There are:",data.shape,";1000 examples of 28x28 pixels in greyscale")


#neural network is here
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim
use_cuda = torch.cuda.is_available()

#define neural network
class Net(nn.Module):
	def __init__(self):
		super(Net, self).__init__()
		self.conv1 = nn.Conv2d(1, 100, kernel_size=5)
		self.conv2 = nn.Conv2d(100, 20, kernel_size=5)
		self.conv2_drop = nn.Dropout2d()
		self.fc1 = nn.Linear(320, 50)
		self.fc2 = nn.Linear(50, 10)

	def forward(self, x):
		x = F.relu(F.max_pool2d(self.conv1(x), 2))
		x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
		x = x.view(-1, 320)
		x = F.relu(self.fc1(x))
		x = F.dropout(x, training=self.training)
		x = self.fc2(x)
		return F.log_softmax(x)
	
n_epochs = 3
learning_rate = 0.01
momentum = 0.5
random_seed = 1
log_interval = 10
				  
model = Net()
if use_cuda:
	model = model.cuda()
	
optimizer = optim.SGD(model.parameters(), lr=learning_rate,
				  momentum=momentum)
					 
def train(epoch):
	model.train()
	for batch_idx, (data, target) in enumerate(train_loader):
		optimizer.zero_grad()
		if use_cuda:
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
	
def test():
	model.eval()
	test_loss = 0
	correct = 0
	for data, target in test_loader:
		if use_cuda:
			data, target = data.cuda(), target.cuda()
		data, target = Variable(data, volatile=True), Variable(target)
		output = model(data)
		test_loss += F.nll_loss(output, target, size_average=False).item()
		pred = output.data.max(1, keepdim=True)[1]
		correct += pred.eq(target.data.view_as(pred)).sum()
		test_loss /= len(test_loader.dataset)
		print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
		test_loss, correct, len(test_loader.dataset),
		100. * correct / len(test_loader.dataset)))

for epoch in range(1, n_epochs + 1):
	train(epoch)
	test()
