import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import torchvision.transforms.functional as adjust

def loading_imgs(root_path):
    transform = transforms.Compose(

    [
    #Downsample
     transforms.Resize(60),
     transforms.ToTensor(),
    ])
    #Loading Dataset
    imgs = torchvision.datasets.ImageFolder(root= root_path, transform=transform )
    return imgs

def loading_imgs_without_downsample(root_path):
    transform = transforms.Compose(

    [
     transforms.ToTensor(),
    ])
    #Loading Dataset
    imgs = torchvision.datasets.ImageFolder(root= root_path, transform=transform )
    return imgs