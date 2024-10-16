
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import v2 
import torch 
from torchvision import datasets
from torch.utils.data import DataLoader, Dataset
from vae_architacture import VAE


# Load MNIST dataset
batch_size = 64

# Define transformations
transform = v2.Compose([
    v2.ToImage(), 
    v2.ToDtype(torch.float32, scale=True), 
    v2.Lambda(lambda x: x.view(-1) - 0.3),  # Add some noise
])

# Download and load the training and test data
train_data = datasets.MNIST(
    '~/.pytorch/MNIST_data/', 
    download=True, 
    train=True, 
    transform=transform,
)
test_data = datasets.MNIST(
    '~/.pytorch/MNIST_data/', 
    download=True, 
    train=False, 
    transform=transform,
)

# Create data loaders
train_loader = DataLoader(
    train_data, 
    batch_size=batch_size, 
    shuffle=True,
)
test_loader = DataLoader(
    test_data, 
    batch_size=batch_size, 
    shuffle=False,
)

