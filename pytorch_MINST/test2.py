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

import matplotlib.pyplot as plt
fig = plt.figure()
for i in range(6):
    ax = fig.add_subplot(231+i)
    ax.imshow(data[i][0])
fig
