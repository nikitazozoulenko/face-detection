import torch
import torch.nn as nn
import torchvision
from torchvision import models

class DetectionNetwork(nn.Module):
    def __init__(self):
        super(DetectionNetwork, self).__init__()
        resnet = models.resnet50(pretrained=True)
        modules = list(resnet.children())[:8] # delete the last layers.
        self.resnet_features = nn.Sequential(*modules)
        B = 5 #anchor boxes
        K = 1 #number of classes
        self.last_conv = nn.Conv2d(2048, B*(K+1+4), kernel_size=3, stride=2, padding=1, bias=False)
        self.BN1 = nn.BatchNorm3d(3)
        
    def forward(self, x):
        x = self.BN1(x)
        x = self.resnet_features(x)
        x = self.last_conv(x)
        return x
