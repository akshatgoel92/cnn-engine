# Import packages
import torchvision

import os
import random
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import torchvision.datasets as datasets
import torchvision.transforms as transforms


from tqdm import tqdm
from torch.utils import data
from tensorboardX import SummaryWriter


    
class AlexNet(nn.Module):
    
    def __init__(self, num_classes):
        
        super(AlexNet, self).__init__()

        self.net = nn.Sequential(
            # 3 * 224 * 224
            nn.Conv2d(in_channels=3, out_channels=96, kernel_size=11, stride=4), 
            # Size: (224 + 2*2 - 8) / 2 + 1 = (110 + 1) = 111
            # 64 * 111 * 111
            nn.ReLU(inplace=True),
            # 64 * 111 * 111
            nn.MaxPool2d(kernel_size=3, stride=2),
            # Size: (111 + 2*0 - 3) / 1 + 1 = (108 + 1) = 109
            # 64 * 109 * 109 
            nn.Conv2d(in_channels=96, out_channels=256, kernel_size=5, padding=2),
            # Size: (109 + 2*2 - 5) / 1 + 1 = (108 + 1) = 109
            # 64 * 109 * 109
            nn.ReLU(inplace=True),
            # Size: (109 + 2*2 - 5) / 1 + 1 = (108 + 1) = 109
            # 64 * 109 * 109
            nn.MaxPool2d(kernel_size=3, stride=2), 
            # Size: (109 + 2*0 - 3) / 2 + 1 = (53 + 1) = 54
            # 128 * 54 * 54
            nn.Conv2d(in_channels=256, out_channels=384, kernel_size=3, padding=1),
            # Size: (54 + 2*1 - 3) / 1 + 1 = (53 + 1) = 54
            # 128 * 54 * 54
            nn.ReLU(inplace=True),
            # Size: (54 + 2*1 - 3) / 1 + 1 = (53 + 1) = 54
            # 256 * 54 * 54
            nn.Conv2d(in_channels=384, out_channels=384, kernel_size=3, padding=1),
            # Size: (54 + 2*1 - 3) / 1 + 1 = (53 + 1) = 54
            # 256 * 54 * 54
            nn.ReLU(inplace=True),
            # Size: (54 + 2*1 - 3) / 1 + 1 = (53 + 1) = 54
            # 256 * 54 * 54
            nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, padding=1),
            # Size: (54 + 2*1 - 3) / 1 + 1 = (53 + 1) = 54
            # 256 * 54 * 54
            nn.ReLU(inplace=True),
            # Size: (54 + 2*1 - 3) / 1 + 1 = (53 + 1) = 54
            # 256 * 54 * 54
            nn.MaxPool2d(kernel_size=3, stride=2),
            # Size: (54 - 3) / 2 + 1 = (53 + 1) = 54
            # 256 * 54 * 54
            ) 
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(),
            nn.Linear(in_features=256 * 6 * 6, out_features=4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(in_features=4096, out_features=4096), 
            nn.ReLU(inplace=True),
            nn.Linear(in_features=4096, out_features=num_classes)
            )

        self.init_weights()


    def init_weights(self):
        for layer in self.net:
            if isinstance(layer, nn.Conv2d):
                nn.init.normal_(layer.weight, mean=0, std=0.1)
                nn.init.constant_(layer.bias, 0)
            if isinstance(layer, nn.Linear):
                nn.init.normal_(layer.weight, mean=0, std=0.1)
                nn.init.constant_(layer.bias, 1)

        nn.init.constant_(self.net[3].bias, 1)
        nn.init.constant_(self.net[6].bias, 1)
        nn.init.constant_(self.net[8].bias, 1)


    def forward(self, x):
        x = self.net(x)
        return self.classifier(x)