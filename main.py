from __future__ import print_function
from __future__ import division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
from torch.nn import Parameter
import numpy as np
import torchvision
from torchvision import  datasets, models, transforms
import matplotlib.pyplot as plt

class DetectionNetwork(nn.Module):
    def __init__(self):
        super(DetectionNetwork, self).__init__()
        resnet = models.resnet50(pretrained=True)
        modules = list(resnet.children())[:8] # delete the last layers.
        self.resnet_features = nn.Sequential(*modules)
        B = 5 #anchor boxes
        K = 1 #number of classes
        self.last_conv = nn.Conv2d(2048, B*(K+1+4), kernel_size=3, stride=2, padding=1, bias=False)
        
    def forward(self, x):
        x = self.resnet_features(x)
        x = self.last_conv(x)
        return x

model = DetectionNetwork().cuda()

for i in range(448, 449):
    inp = Variable(torch.ones(2,3,i,i)).cuda()
    features = model(inp)
    print(i, features.size())
