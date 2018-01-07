import torch
import torch.nn as nn
import torchvision
from torchvision import models
from torch.autograd import Variable

import numpy as np

from util_detection import *

class CreateAnchorsAndBoxes(nn.Module):
    def __init__(self):
        super(CreateAnchorsAndBoxes, self).__init__()

    def forward(self, classes, anchors_hw):
        #offsets shape [batch_size, 4A, S, S]
        #anchors shape [A, 2]
        #classes shape [batch_size, (K+1)A, S, S]
        R, _, H, W = list(classes.size())
        A, _ = list(anchors_hw.size())

        ##RESHAPE OFFSETS
        #offsets = offsets.permute(0,2,3,1).contiguous().view(R, A*H*W, 4)
        
        #RESHAPE CLASSES
        classes = classes.permute(0,2,3,1).contiguous().view(R, A*H*W, -1)

        #EXPAND CENTERS
        x_coords = ((torch.arange(H).cuda()+0.5)/H).expand(W, H)
        y_coords = ((torch.arange(W).cuda()+0.5)/W).expand(H, W).t()
        coord_grid = Variable(torch.stack((x_coords, y_coords), dim = 2))
        coords = coord_grid.view(-1, 2).expand(A,-1,-1).permute(1,0,2)

        anchors_min = (coords - anchors_hw/2)
        anchors_max = anchors_min + anchors_hw

        anchors_min = anchors_min.view(-1,2)
        anchors_max = anchors_max.view(-1,2)

        anchors = torch.cat((anchors_min, anchors_max), dim = 1)

        return classes, anchors

class DetectionNetwork(nn.Module):
    def __init__(self):
        super(DetectionNetwork, self).__init__()
        resnet = models.resnet50(pretrained=True)
        modules = list(resnet.children())[:8] # delete the last layers.
        self.resnet_features = nn.Sequential(*modules)
        A = 9 #anchor boxes
        K = 1 #number of classes
        self.regressor_head = nn.Conv2d(2048, A*4, kernel_size=3, stride=1, padding=1, bias=False)
        self.classification_head = nn.Conv2d(2048, A*(K+1), kernel_size=3, stride=1, padding=1, bias=False)
        self.BN1 = nn.BatchNorm3d(3)
        self.BNFeatures = nn.BatchNorm3d(2048)
        self.softmax = nn.Softmax(dim=2)
        self.anchors_hw = Variable(torch.Tensor([[0.30,0.30], [0.15,0.30],
                                                 [0.18,0.18], [0.09,0.18],
                                                 [0.10,0.10], [0.05,0.10],
                                                 [0.05,0.05], [0.025,0.05],
                                                 [0.025,0.025]]), requires_grad=False).cuda()
        self.create_anchors_and_boxes = CreateAnchorsAndBoxes()
    def forward(self, x, phase = "train"):
        x = self.BN1(x)
        x = self.resnet_features(x)
        x = self.BNFeatures(x)
        classes = self.classification_head(x)
        classes, anchors = self.create_anchors_and_boxes(classes, self.anchors_hw)
        classes = self.softmax(classes)
        if phase == "train":
            return classes, anchors
        if phase == "test":
            selected_boxes, selected_classes =  process_boxes(anchors, classes)
            return selected_classes, selected_boxes

class ClassLoss(nn.Module):
    def __init__(self):
        super(ClassLoss, self).__init__()

    def forward(self, classes, positive_idx):
        gamma = 2
        
        positive_idx = positive_idx[:,0]

        gather_pos = torch.zeros(classes.size(0), out=torch.ByteTensor()).cuda()
        gather_pos.index_fill_(0, positive_idx.data, 1)
        gather_neg = torch.ones(classes.size(0), out=torch.ByteTensor()).cuda()
        gather_neg.index_fill_(0, positive_idx.data, 0)

        mask = torch.stack((gather_neg, gather_pos), dim = 1)

        probs = classes[mask]

        #focal loss described in paper "Focal Loss for Dense Object Detection"
        focal_loss = -(1-probs).pow(gamma) * probs.log()
        extra = focal_loss.index_select(0, positive_idx)
        
        return torch.sum(focal_loss) + 10*torch.sum(extra)

class CoordLoss(nn.Module):
    def __init__(self):
        super(CoordLoss, self).__init__()
        self.l1_loss = nn.L1Loss(size_average=False)

    def forward(self, boxes, gt, positive_idx):
        #boxes,       size [S*S*A,       4]
        #gt,          size [num_obj, 4]
        #positive_idx size [num_matches, 2]
        pred_idx = positive_idx[:, 0]
        gt_idx = positive_idx[:, 1]

        selected_pred = boxes.index_select(0, pred_idx)
        selected_gt = gt.index_select(0, gt_idx)
        selected_gt = from_xywh_to_xyxy(selected_gt)

        coord_loss = self.l1_loss(selected_pred, selected_gt)
        return coord_loss

def match(threshhold, anchors, gt):
    gt_boxes = from_xywh_to_xyxy(gt)
    ious = jaccard(anchors, gt_boxes)
    best_gt_iou, best_gt_idx = ious.max(0)
    
    num_objects = best_gt_idx.size(0)
    arange = torch.arange(0, num_objects, out=torch.LongTensor())
    arange = Variable(arange).cuda()
    arange = torch.stack((best_gt_idx, arange), dim=1)

    positive_mask = ious > threshhold
    print("mask ious", ious[ious>threshhold])

    for i in range(num_objects):
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

    def forward(self, threshhold, batch_classes, anchors, batch_gt, batch_num_objects):
        #batch_boxes,      size [batch_size,       S*S*A,  4]
        #batch_classes,    size [batch_size,       S*S*A,  K+1]
        #batch_gt,         size [batch_size, max_num_obj,  4]
        #batch_num_objects size [batch_size, max_num_obj]
        ALPHA_CLASS = 0.01
        ALPHA_COORD = 1
        R = batch_classes.size(0)
        class_loss = Variable(torch.zeros(1)).cuda()
        coord_loss = Variable(torch.zeros(1)).cuda()
        
        for classes, gt, num_objects in zip(batch_classes, batch_gt, batch_num_objects):
            gt = gt[:num_objects]
            positive_idx = match(threshhold, anchors, gt)
            class_loss += self.class_loss(classes, positive_idx)
            #coord_loss += self.coord_loss(boxes, gt, positive_idx)
            print("GT", gt)

        class_loss = class_loss * ALPHA_CLASS / R
        coord_loss = coord_loss * ALPHA_COORD / R
        total_loss = class_loss + coord_loss
        return total_loss, class_loss, coord_loss
