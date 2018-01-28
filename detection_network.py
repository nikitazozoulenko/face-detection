import torch
import torch.nn as nn
import torchvision
from torchvision import models
from torch.autograd import Variable
from torch.nn import Parameter

import numpy as np

from util_detection import *

class CreateAnchorsAndBoxes(nn.Module):
    def __init__(self):
        super(CreateAnchorsAndBoxes, self).__init__()

    def forward(self, offsets, classes, anchors_hw, height, width):
        #offsets shape [batch_size, 4A, S, S]
        #anchors shape [A, 2]
        #classes shape [batch_size, (K+1)A, S, S]
        R, C, H, W = list(classes.size())
        A, _ = list(anchors_hw.size())

        #RESHAPE OFFSETS
        offsets = offsets.view(R,-1, A*H*W).permute(0,2,1)
        
        #RESHAPE CLASSES
        classes = classes.view(R,-1, A*H*W).permute(0,2,1)
        
        #EXPAND CENTER COORDS
        x_coords = ((torch.arange(H).cuda()/H*height+0.5)).expand(W, H)
        y_coords = ((torch.arange(W).cuda()/W*width+0.5)).expand(H, W).t()
        coord_grid = Variable(torch.stack((x_coords, y_coords), dim = 0))
        coords = coord_grid.view(2,-1).t().expand(A, -1, -1)
        anch = anchors_hw.unsqueeze(1).expand(-1,H*W,-1)

        anchors_min = coords - anch/2
        anchors_max = anchors_min + anch
        anchors_min = anchors_min.view(-1,2)
        anchors_max = anchors_max.view(-1,2)

        anchors = torch.cat((anchors_min, anchors_max), dim = 1)
        boxes = offsets + anchors

        return boxes, classes, anchors

class RegressorHead(nn.Module):
    def __init__(self):
        super(RegressorHead, self).__init__()
        A = 6

        self.conv0 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias = False)
        self.BN0 = nn.BatchNorm3d(256)
        self.regressor = nn.Conv2d(256, A*4, kernel_size=3, stride=1, padding=1, bias = False)

    def forward(self, x):
        #x shape [batch_size, 256, H, W]
        x = self.conv0(x)
        x = self.BN0(x)
        x = self.regressor(x)
        return x

class ClassificationHead(nn.Module):
    def __init__(self):
        super(ClassificationHead, self).__init__()
        K = 1
        A = 6
        
        pi = 0.05
        bias = np.log(K*(1-pi)/pi)
        self.prior = Parameter(torch.FloatTensor([[bias]]).expand(A, -1, -1))
        self.foreground_bias = Parameter(torch.FloatTensor([[0]]).expand(A*K, -1, -1))

        self.conv0 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias = False)
        self.BN0 = nn.BatchNorm3d(256)
        self.background = nn.Conv2d(256,   A, kernel_size=3, stride=1, padding=1, bias = False)
        self.foreground = nn.Conv2d(256, A*K, kernel_size=3, stride=1, padding=1, bias = False)

    def forward(self, x):
        #x shape [batch_size, 256, H, W]
        x = self.conv0(x)
        x = self.BN0(x)
        background = self.background(x) + self.prior
        foreground = self.foreground(x)
        return torch.cat((background, foreground), dim=1)
    
class FaceNet(nn.Module):
    def __init__(self):
        super(FaceNet, self).__init__()
        resnet = models.resnet50(pretrained=True)
        modules_conv3 = list(resnet.children())[:6]
        modules_conv4 = list(resnet.children())[6]
        modules_conv5 = list(resnet.children())[7]
        
        self.BN = nn.BatchNorm3d(3)

        self.conv3 = nn.Sequential(*modules_conv3)
        self.bottleneck_conv3 = nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0, bias = False)
        
        self.conv4 = nn.Sequential(*modules_conv4)
        self.bottleneck_conv4 = nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding=0, bias = False)

        self.conv5 = nn.Sequential(*modules_conv5)
        self.bottleneck_conv5 = nn.Conv2d(2048, 256, kernel_size=1, stride=1, padding=0, bias = False)

        self.conv6 = nn.Conv2d(2048, 1024, kernel_size=3, stride=2, padding=1)
        self.bottleneck_conv6 = nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding=0, bias = False)

        self.upsample = nn.Upsample(scale_factor=2, mode = "nearest")

        self.out6BN = nn.BatchNorm3d(256)
        self.out5BN = nn.BatchNorm3d(256)
        self.out4BN = nn.BatchNorm3d(256)
        self.out3BN = nn.BatchNorm3d(256)

        self.regressor_head =  RegressorHead()
        self.classification_head = ClassificationHead()
        
        self.softmax = nn.Softmax(dim=2)

        #scales = [16, 21, 27, 36, 47, 62, 82, 107, 141, 186, 244, 320]
        self.anchors_hw3 = Variable(torch.Tensor([[16, 16],  [16/1.5, 16],
                                                  [21, 21],  [21/1.5, 21],
                                                  [27, 27],  [27/1.5, 27]]),requires_grad=False).cuda()
        self.anchors_hw4 = Variable(torch.Tensor([[36, 36],  [36/1.5, 36],
                                                  [47, 47],  [47/1.5, 47],
                                                  [62, 62],  [62/1.5, 62]]),requires_grad=False).cuda()
        self.anchors_hw5 = Variable(torch.Tensor([[82, 82],  [82/1.5, 82],
                                                  [107, 107],  [107/1.5, 107],
                                                  [141, 141],  [141/1.5, 141]]),requires_grad=False).cuda()
        self.anchors_hw6 = Variable(torch.Tensor([[186, 186],  [186/1.5, 186],
                                                  [244, 244],  [244/1.5, 244],
                                                  [320, 320],  [320/1.5, 320]]),requires_grad=False).cuda()
        self.create_anchors_and_boxes = CreateAnchorsAndBoxes()
        
    def forward(self, x, phase = "train"):
        _, _, height, width = x.size()
        x = self.BN(x)
        conv3 = self.conv3(x)
        bottleneck_conv3 = self.bottleneck_conv3(conv3)
        conv4 = self.conv4(conv3)
        bottleneck_conv4 = self.bottleneck_conv4(conv4)
        conv5 = self.conv5(conv4)
        bottleneck_conv5 = self.bottleneck_conv5(conv5)
        conv6 = self.conv6(conv5)
        bottleneck_conv6 = self.bottleneck_conv6(conv6)

        # FPN Feature pyramid structure described in paper
        # "Feature Pyramid Networks for Object Detection"
        out6 = bottleneck_conv6
        out6 = self.out6BN(out6)
        
        out5 = bottleneck_conv5 + self.upsample(out6)
        out5 = self.out5BN(out5)
        
        out4 = bottleneck_conv4 + self.upsample(out5)
        out4 = self.out4BN(out4)

        out3 = bottleneck_conv3 + self.upsample(out4)
        out3 = self.out3BN(out3)

        offsets6 = self.regressor_head(out6)
        classes6 = self.classification_head(out6)
        boxes6, classes6, anchors6 = self.create_anchors_and_boxes(offsets6, classes6, self.anchors_hw6, height, width)

        offsets5 = self.regressor_head(out5)
        classes5 = self.classification_head(out5)
        boxes5, classes5, anchors5 = self.create_anchors_and_boxes(offsets5, classes5, self.anchors_hw5, height, width)

        offsets4 = self.regressor_head(out4)
        classes4 = self.classification_head(out4)
        boxes4, classes4, anchors4 = self.create_anchors_and_boxes(offsets4, classes4, self.anchors_hw4, height, width)

        offsets3 = self.regressor_head(out3)
        classes3 = self.classification_head(out3)
        boxes3, classes3, anchors3 = self.create_anchors_and_boxes(offsets3, classes3, self.anchors_hw3, height, width)

        #concat all the predictions
        boxes = torch.cat((boxes3, boxes4, boxes5, boxes6), dim=1)
        classes = torch.cat((classes3, classes4, classes5, classes6), dim=1)
        anchors = torch.cat((anchors3, anchors4, anchors5, anchors6), dim=0)

        sizes = [boxes3.size(1), boxes4.size(1), boxes5.size(1), boxes6.size(1)]

        if phase == "train":
            return boxes, classes, anchors
        if phase == "test":
            classes = self.softmax(classes)
            return boxes, classes

