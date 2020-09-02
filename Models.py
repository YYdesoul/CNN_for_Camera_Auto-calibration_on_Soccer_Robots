import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

class Net_3_layer(nn.Module):
    def __init__(self, kernel):
        super(Net_3_layer, self).__init__()
        self.conv1 = nn.Conv2d(in_channels = 3, out_channels = 32, kernel_size = kernel, padding = int((kernel-1)/2))
        self.pool1 = nn.MaxPool2d(2,2)
        self.fc = nn.Linear(32 * 30 * 40, 3) 
        
    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = x.view(-1, 32 * 30 * 40)
        x = F.dropout(x)
        x = self.fc(x)
        return x
    
class Net_5_layer(nn.Module):
    def __init__(self, kernel):
        super(Net_5_layer, self).__init__()
        self.conv1 = nn.Conv2d(in_channels = 3, out_channels = 32, kernel_size = kernel, padding = int((kernel-1)/2))
        self.conv2 = nn.Conv2d(in_channels = 32, out_channels = 64, kernel_size = kernel, padding = int((kernel-1)/2))
        self.pool1 = nn.MaxPool2d(2,2)
        self.fc = nn.Linear(64 * 15 * 20, 3) 
        
    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool1(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 15 * 20)
        x = F.dropout(x)
        x = self.fc(x)
        return x
    
class Net_7_layer(nn.Module):
    def __init__(self, kernel):
        super(Net_7_layer, self).__init__()
        self.conv1 = nn.Conv2d(in_channels = 3, out_channels = 32, kernel_size = kernel, padding = int((kernel-1)/2))
        self.conv2 = nn.Conv2d(in_channels = 32, out_channels = 64, kernel_size = kernel, padding = int((kernel-1)/2))
        self.conv3 = nn.Conv2d(in_channels = 64, out_channels = 96, kernel_size = kernel, padding = int((kernel-1)/2))
        self.pool1 = nn.MaxPool2d(2,2)
        self.fc = nn.Linear(96 * 7 * 10, 3) 
        
    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool1(F.relu(self.conv2(x)))
        x = self.pool1(F.relu(self.conv3(x)))
        x = x.view(-1, 96 * 7 * 10)
        x = F.dropout(x)
        x = self.fc(x)
        return x
    
class Net_9_layer(nn.Module):
    def __init__(self, kernel):
        super(Net_9_layer, self).__init__()
        self.conv1 = nn.Conv2d(in_channels = 3, out_channels = 32, kernel_size = kernel, padding = int((kernel-1)/2))
        self.conv2 = nn.Conv2d(in_channels = 32, out_channels = 64, kernel_size = kernel, padding = int((kernel-1)/2))
        self.conv3 = nn.Conv2d(in_channels = 64, out_channels = 96, kernel_size = kernel, padding = int((kernel-1)/2))
        self.conv4 = nn.Conv2d(in_channels = 96, out_channels = 128, kernel_size = kernel, padding = int((kernel-1)/2))

        self.pool1 = nn.MaxPool2d(2,2)
        self.fc = nn.Linear(128 * 3 * 5, 3) 
        
    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool1(F.relu(self.conv2(x)))
        x = self.pool1(F.relu(self.conv3(x)))
        x = self.pool1(F.relu(self.conv4(x)))
        x = x.view(-1, 128 * 3 * 5)
        x = F.dropout(x)
        x = self.fc(x)
        return x    