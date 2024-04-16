import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np64
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from PIL import Image
import torch.optim as optim
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
import torch.nn.functional as F

import tarfile
from PIL import Image


class MNIST(Dataset):

    def load_data(self, data_dir):
        images = []
        labels = []
        resize_transform = transforms.Resize((32, 32))  
        with tarfile.open(data_dir, 'r') as tar:
            for member in tar.getmembers():
                if member.isfile() and member.name.endswith('.png'):
                    image = Image.open(tar.extractfile(member))
                    image = resize_transform(image)  
                    image = self.transform(image)
                    images.append(image)
                    labels.append(int(os.path.basename(member.name).split('_')[-1].split('.')[0]))  # Extract label from filename
        return images, labels

    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        self.images, self.labels = self.load_data(data_dir)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img, label = self.images[idx], self.labels[idx]
        return img, label

if __name__ == '__main__':
    
    train_data_dir = '/content/train.tar'
    test_data_dir = '/content/test.tar'

    
    mnist_train_dataset = MNIST(train_data_dir)
    mnist_test_dataset = MNIST(test_data_dir)

    
    print(f'Training dataset: Number of images: {len(mnist_train_dataset)}, Number of labels: {len(set(mnist_train_dataset.labels))}')
    print(f'Test dataset: Number of images: {len(mnist_test_dataset)}, Number of labels: {len(set(mnist_test_dataset.labels))}')
