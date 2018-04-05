import torch
import torch.nn as nn
import torchvision
from torchvision import models
from torch.autograd import Variable
from torch.nn import Parameter

import numpy as np

class ResidualBlock(nn.Module):
    def __init__(self, channels, expansion = 4, cardinality = 1):
        super(ResidualBlock, self).__init__()

        self.block = nn.Sequential(nn.Conv2d(channels*expansion, channels, kernel_size=1, bias=False),
                                   nn.BatchNorm2d(channels),
                                   nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups = cardinality, bias=False),
                                   nn.BatchNorm2d(channels),
                                   nn.Conv2d(channels, channels*expansion, kernel_size=1, bias=False),
                                   nn.BatchNorm2d(channels*expansion))
        
        self.relu = nn.ReLU(inplace = True)
        
        
    def forward(self, x):
        res = x

        out = self.block(x)
        out = self.relu(out+res)
        
        return out
    

def make_anchors_and_bbox(offsets, classes, anchors_hw, height, width):
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
    x_coords = ((torch.arange(W).cuda()+0.5)/W*width).expand(H, W)
    y_coords = ((torch.arange(H).cuda()+0.5)/H*height).expand(W, H).t()
    coord_grid = torch.stack((x_coords, y_coords), dim = 0)
    coords = coord_grid.view(2,-1).t().expand(A, -1, -1)
    anch = anchors_hw.unsqueeze(1).expand(-1,H*W,-1)

    anchors_min = coords - anch/2
    anchors_max = anchors_min + anch
    anchors_min = anchors_min.view(-1,2)
    anchors_max = anchors_max.view(-1,2)
            
    anchors = Variable(torch.cat((anchors_min, anchors_max), dim = 1))
    boxes = offsets + anchors

    return boxes, classes, anchors        


class PredictionHead(nn.Module):
    def __init__(self):
        super(PredictionHead, self).__init__()
        self.regressor = RegressionHead()
        self.classifyer = ClassificationHead()


    def forward(self, x):
        offsets = self.regressor(x)
        confidences = self.classifyer(x)
        return offsets, confidences


class RegressionHead(nn.Module):
    def __init__(self):
        super(RegressionHead, self).__init__()
        A = 6
        self.regressor = nn.Conv2d(256, A*4, kernel_size=3, stride=1, padding=1, bias = True)

        channels = 64
        expansion = 4
        cardinality = 1
        block_depth = 3

        res_0 = [ResidualBlock(channels, expansion, cardinality) for _ in range(block_depth)]
        #upsample = nn.ConvTranspose2d(channels*expansion, channels*expansion, 3, stride=2, padding=1)
        #upsample = nn.Upsample(scale_factor=2, mode="bilinear")
        res_1 = [ResidualBlock(channels, expansion, cardinality) for _ in range(block_depth)]
        self.residual = nn.Sequential(*res_0, *res_1)

    def forward(self, x):
        #x shape [batch_size, 256, H, W]
        x = self.residual(x)
        x = self.regressor(x)
        return x


class ClassificationHead(nn.Module):
    def __init__(self):
        super(ClassificationHead, self).__init__()
        K = 1
        A = 6
        pi = 0.001
        bias = np.log(K*(1-pi)/pi)
        self.prior = Parameter(torch.cuda.FloatTensor([[bias]]).expand(A, -1, -1))
        
        self.background = nn.Conv2d(256,   A, kernel_size=3, stride=1, padding=1, bias = False)
        self.foreground = nn.Conv2d(256, A*K, kernel_size=3, stride=1, padding=1, bias = True)

        channels = 64
        expansion = 4
        cardinality = 1
        block_depth = 3

        res_0 = [ResidualBlock(channels, expansion, cardinality) for _ in range(block_depth)]
        #upsample = nn.ConvTranspose2d(channels*expansion, channels*expansion, 3, stride=2, padding=1)
        #upsample = nn.Upsample(scale_factor=2, mode="bilinear")
        res_1 = [ResidualBlock(channels, expansion, cardinality) for _ in range(block_depth)]
        self.residual = nn.Sequential(*res_0, *res_1)


    def forward(self, x):
        #x shape [batch_size, 256, H, W]
        x = self.residual(x)
        background = self.background(x) + self.prior
        foreground = self.foreground(x)
        return torch.cat((background, foreground), dim=1)
    

class FaceNet(nn.Module):
    def __init__(self):
        super(FaceNet, self).__init__()
        resnet = models.resnet18(pretrained=True)
        modules_conv2 = list(resnet.children())[:5]
        modules_conv3 = list(resnet.children())[5]
        modules_conv4 = list(resnet.children())[6]
        modules_conv5 = list(resnet.children())[7]
        
        self.input_BN = nn.BatchNorm3d(3)
        
        self.conv2 = nn.Sequential(*modules_conv2)
        self.bottleneck_conv2 = nn.Conv2d(64, 256, kernel_size=1, stride=1, padding=0, bias = True)

        self.conv3 = nn.Sequential(*modules_conv3)
        self.bottleneck_conv3 = nn.Conv2d(128, 256, kernel_size=1, stride=1, padding=0, bias = True)
        
        self.conv4 = nn.Sequential(*modules_conv4)
        self.bottleneck_conv4 = nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0, bias = True)

        self.conv5 = nn.Sequential(*modules_conv5)
        self.bottleneck_conv5 = nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0, bias = True)

        self.conv6 = nn.Sequential(*[ResidualBlock(128, expansion=4) for _ in range(2)])
        self.bottleneck_conv6 = nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0, bias = True)

        self.prediction_head =  PredictionHead()

        self.anchors_hw2 = torch.Tensor([[16, 16],  [16/1.6, 16],
                                         [20, 20],  [20/1.6, 20],
                                         [25, 25],  [25/1.6, 25]]).cuda()
        self.anchors_hw3 = torch.Tensor([[32, 32],  [32/1.6, 32],
                                         [40, 40],  [40/1.6, 40],
                                         [51, 51],  [51/1.6, 51]]).cuda()
        self.anchors_hw4 = torch.Tensor([[64, 64],  [64/1.6, 64],
                                         [81, 81],  [81/1.6, 81],
                                         [102, 102],  [102/1.6, 102]]).cuda()
        self.anchors_hw5 = torch.Tensor([[128, 128],  [128/1.6, 128],
                                         [161, 161],  [161/1.6, 161],
                                         [203, 203],  [203/1.6, 203]]).cuda()
        self.anchors_hw6 = torch.Tensor([[256, 256],  [256/1.6, 256],
                                         [322, 322],  [322/1.6, 322],
                                         [406, 406],  [406/1.6, 406]]).cuda()
        
    def forward(self, x, phase = "train"):
        _, _, height, width = x.size()
        x = self.input_BN(x)

        conv2 = self.conv2(x)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)
        conv5 = self.conv5(conv4)
        conv6 = self.conv6(conv5)
        
        conv2 = self.bottleneck_conv2(conv2)
        conv3 = self.bottleneck_conv3(conv3)
        conv4 = self.bottleneck_conv4(conv4)
        conv5 = self.bottleneck_conv5(conv5)
        conv6 = self.bottleneck_conv6(conv6)

        offsets6, classes6 = self.prediction_head(conv6)
        boxes6, classes6, anchors6 = make_anchors_and_bbox(offsets6, classes6, self.anchors_hw6, height, width)
        offsets5, classes5 = self.prediction_head(conv5)
        boxes5, classes5, anchors5 = make_anchors_and_bbox(offsets5, classes5, self.anchors_hw5, height, width)
        offsets4, classes4 = self.prediction_head(conv4)
        boxes4, classes4, anchors4 = make_anchors_and_bbox(offsets4, classes4, self.anchors_hw4, height, width)
        offsets3, classes3 = self.prediction_head(conv3)
        boxes3, classes3, anchors3 = make_anchors_and_bbox(offsets3, classes3, self.anchors_hw3, height, width)
        offsets2, classes2 = self.prediction_head(conv2)
        boxes2, classes2, anchors2 = make_anchors_and_bbox(offsets2, classes2, self.anchors_hw3, height, width)

        #concat all the predictions
        #boxes = [boxes3, boxes4, boxes5, boxes6, boxes7]
        #classes = [classes3, classes4, classes5, classes6, classes7]
        #anchors = [anchors3, anchors4, anchors5, anchors6, anchors7]
        boxes = torch.cat((boxes2, boxes3, boxes4, boxes5, boxes6), dim=1)
        classes =torch.cat((classes2, classes3, classes4, classes5, classes6), dim=1)
        anchors = torch.cat((anchors2, anchors3, anchors4, anchors5, anchors6), dim=0)
        return boxes, classes, anchors