import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

# Load MINST dataset
mnist_trainset = datasets.MNIST(root='./data', train=True, download=True, transform=None)

mnist_testset = datasets.MNIST(root='./data', train=False, download=True, transform=None)

print("Testset: ",len(mnist_trainset))
print("Trainset: ", len(mnist_testset))

#
# Sanity check for CUDA or CPU 
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])

#load in pretrained network

# =============================================================================
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

resnet18 = models.resnet18(pretrained=True)
