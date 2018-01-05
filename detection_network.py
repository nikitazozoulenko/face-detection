import torch
import torch.nn as nn
import torchvision
from torchvision import models
from torch.autograd import Variable

import numpy as np

from util_detection import *

class CreateBoxesWithAnchors(nn.Module):
    def __init(self):
        super(CreateBoxesWithAnchorsAndOffsets, self).__init()
        self.A = 15

    def forward(self, batch_offsets, anchors, classes):
        #offsets       shape [batch_size, 4A, S, S]
        #anchors       shape [A, 2]
        #batch_classes shape [batch_size, (K+1)A, S, S]
        R, C, H, W = list(batch_offsets.size())
        A, _ = list(anchors.size())
        
        #EXPAND CLASSES
        classes = classes.permute(0,2,3,1).contiguous().view(R, A*H*W, -1)
        
        #EXPAND OFFSETS
        offsets = batch_offsets.permute(0,2,3,1).contiguous().view(R, A*H*W, 4)

        #EXPAND CENTERS
        x_coords = ((torch.arange(H)+0.5)/H).expand((W, H))
        y_coords = ((torch.arange(W)+0.5)/W).expand((H, W)).t()
        stacked = torch.stack((y_coords, x_coords), dim = 2)
        new_stacked = stacked.view(H*W, -1)
        newer_stacked = Variable(new_stacked.expand(A,-1, -1).contiguous().view(A*H*W, 2)).cuda()

        #EXPAND ANCHORS
        stacked_anchors = anchors.expand(H*W, -1, -1).contiguous().view(A*H*W, 2)

        #do the math
        l_left = offsets[:, :, 0:1]
        l_right = offsets[:, :, 1:2]
        l_up = offsets[:, :, 2:3]
        l_down = offsets[:, :, 3:4]

        center_x = newer_stacked[:, 0:1]
        center_y = newer_stacked[:, 1:2]

        a_w = stacked_anchors[:, 0:1]
        a_h = stacked_anchors[:, 1:2]

        xmin = -l_left + center_x - a_w/2
        ymin = -l_up + center_y - a_h/2
        width = l_left + l_right + a_w
        height = l_up + l_down + a_h

        # xmin =  center_x - a_w/2
        # ymin = center_y - a_h/2
        # width =  a_w
        # height =  a_h

        #boxes = torch.cat((xmin, ymin, width, height), dim=1).expand((R, -1, -1))
        boxes = torch.cat((xmin, ymin, width, height), dim=2)
        return boxes, classes
    
    
class DetectionNetwork(nn.Module):
    def __init__(self):
        super(DetectionNetwork, self).__init__()
        resnet = models.resnet50(pretrained=True)
        modules = list(resnet.children())[:8] # delete the last layers.
        self.resnet_features = nn.Sequential(*modules)
        A = 15 #anchor boxes
        K = 1 #number of classes
        self.regressor_head = nn.Conv2d(2048, A*4, kernel_size=3, stride=1, padding=1, bias=False)
        self.classification_head = nn.Conv2d(2048, A*(K+1), kernel_size=3, stride=1, padding=1, bias=False)
        self.BN1 = nn.BatchNorm3d(3)
        self.BNFeatures = nn.BatchNorm3d(2048)
        self.softmax = nn.Softmax(dim=2)
        self.create_boxes_with_anchors = CreateBoxesWithAnchors()

        self.anchors = Variable(torch.Tensor([[0.8,0.4],    [0.8,0.8],   [0.4,0.8],
                                              [0.4,0.2],    [0.4,0.4],   [0.2,0.4],
                                              [0.2,0.1],    [0.2,0.2],   [0.1,0.2],
                                              [0.1,0.05],   [0.1,0.1],   [0.05,0.1],
                                              [0.05,0.025], [0.05,0.05], [0.025,0.05]]), requires_grad=False).cuda()
        
    def forward(self, x):
        x = self.BN1(x)
        x = self.resnet_features(x)
        x = self.BNFeatures(x)
        offsets = self.regressor_head(x) / 100
        classes = self.classification_head(x)
        boxes, classes = self.create_boxes_with_anchors(offsets, self.anchors, classes)
        classes = self.softmax(classes)
        return boxes, classes

class ClassLoss(nn.Module):
    def __init__(self):
        super(ClassLoss, self).__init__()

    def forward(self, classes, positive_idx):
        gamma = 2
        alpha = 1
        
        positive_idx = positive_idx[:,0]

        gather_pos = torch.zeros(classes.size(0), out=torch.ByteTensor()).cuda()
        gather_pos.index_fill_(0, positive_idx.data, 1)
        gather_neg = torch.ones(classes.size(0), out=torch.ByteTensor()).cuda()
        gather_neg.index_fill_(0, positive_idx.data, 0)

        mask = torch.stack((gather_neg, gather_pos), dim = 1)

        probs = classes[mask]

        #focal loss described in paper "Focal Loss for Dense Object Detection"
        focal_loss = -alpha * (1-probs).pow(gamma) * probs.log()
        return torch.sum(focal_loss)

class CoordLoss(nn.Module):
    def __init__(self):
        super(CoordLoss, self).__init__()
        self.smooth_l1_loss = nn.SmoothL1Loss(size_average=False)

    def forward(self, boxes, gt, positive_idx):
        #boxes,       size [S*S*A,       4]
        #gt,          size [num_obj, 4]
        #positive_idx size [num_matches, 2]
        pred_idx = positive_idx[:, 0]
        gt_idx = positive_idx[:, 1]

        selected_pred = boxes.index_select(0, pred_idx)
        selected_gt = gt.index_select(0, gt_idx)

        coord_loss = self.smooth_l1_loss(selected_pred, selected_gt)
        return coord_loss

def match(threshhold, boxes, classes, gt, num_objects):
    pred_boxes, gt_boxes = from_xywh_to_xyxy(boxes, gt)
    ious = jaccard(pred_boxes, gt_boxes)

    best_gt_iou, best_gt_idx = ious.max(0)
    arange = torch.arange(0, num_objects, out=torch.LongTensor())
    arange = Variable(arange).cuda()
    arange = torch.stack((best_gt_idx, arange), dim=1)

    positive_mask = ious > threshhold

    for i in range(arange.size(0)):
        # # There is a gather function388 for that.

        # # m = torch.randn(4,2)
        # # ids = torch.Tensor([1,1,0,0]).long()
        # # print(m.gather(1, ids.view(-1,1)))
        box_idx = arange[i, 0]
        arange_idx = arange[i, 1]
        positive_mask[box_idx, arange_idx] = 1

    positive_idx = torch.nonzero(positive_mask)
    return positive_idx
    
class Loss(nn.Module):
    def __init__(self):
        super(Loss, self).__init__()
        self.class_loss = ClassLoss()
        self.coord_loss = CoordLoss()

    def forward(self, threshhold, batch_boxes, batch_classes, batch_gt, batch_num_objects):
        #batch_boxes,      size [batch_size,       S*S*A,  4]
        #batch_classes,    size [batch_size,       S*S*A,  K+1]
        #batch_gt,         size [batch_size, max_num_obj,  4]
        #batch_num_objects size [batch_size, max_num_obj]
        threshhold = 0.8
        R = list(batch_boxes.size())[0]
        class_loss = Variable(torch.zeros(1)).cuda()
        coord_loss = Variable(torch.zeros(1)).cuda()
        
        for boxes, classes, gt, num_objects in zip(batch_boxes, batch_classes, batch_gt, batch_num_objects):
            gt = gt[:num_objects]
            positive_idx = match(threshhold, boxes, classes, gt, int(num_objects))
            class_loss += self.class_loss(classes, positive_idx)
            coord_loss += self.coord_loss(boxes, gt, positive_idx)

        total_loss = class_loss + coord_loss
        return total_loss, class_loss, coord_loss


        
        
