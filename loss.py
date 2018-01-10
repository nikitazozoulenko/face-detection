import torch
import torch.nn as nn
import torchvision
from torchvision import models
import torch.autograd as autograd
from torch.autograd import Variable
import torch.nn.functional as F

import numpy as np

from util_detection import *

def from_xywh_to_xyxy(boxes):
    xmin = boxes[:, 0:1]
    ymin = boxes[:, 1:2]
    xmax = xmin + boxes[:, 2:3]
    ymax = ymin + boxes[:, 3:4]
    return torch.cat((xmin, ymin, xmax, ymax), dim=1)
    
class ClassLoss(nn.Module):
    def __init__(self):
        super(ClassLoss, self).__init__()

    def forward(self, classes, positive_idx):
        positive_idx = positive_idx[:,0]
        gather_pos = torch.zeros(classes.size(0), out=torch.LongTensor()).cuda()
        gather_pos.index_fill_(0, positive_idx.data, 1)
        indices = Variable(gather_pos)

        gamma = Variable(torch.cuda.FloatTensor([2]))
        weight = Variable(torch.cuda.FloatTensor([1,5]))
        loss = FocalLoss.apply(classes, indices, gamma, weight)
        return loss

class CoordLoss(nn.Module):
    def __init__(self):
        super(CoordLoss, self).__init__()
        self.l1_loss = nn.L1Loss(size_average=True)

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

    def forward(self, batch_boxes, batch_classes, anchors, batch_gt, batch_num_objects):
        #batch_boxes,      size [batch_size,       S*S*A,  4]
        #batch_classes,    size [batch_size,       S*S*A,  K+1]
        #batch_gt,         size [batch_size, max_num_obj,  4]
        #batch_num_objects size [batch_size, max_num_obj]
        threshhold = 0.5
        ALPHA_CLASS = 1
        ALPHA_COORD = 1
        R = batch_classes.size(0)
        class_loss = Variable(torch.zeros(1)).cuda()
        coord_loss = Variable(torch.zeros(1)).cuda()
        
        for boxes, classes, gt, num_objects in zip(batch_boxes, batch_classes, batch_gt, batch_num_objects):
            gt = gt[:num_objects]
            positive_idx = match(threshhold, anchors, gt)
            class_loss += self.class_loss(classes, positive_idx)
            coord_loss += self.coord_loss(boxes, gt, positive_idx)

        class_loss = class_loss * ALPHA_CLASS / R
        coord_loss = coord_loss * ALPHA_COORD / R
        total_loss = class_loss + coord_loss
        return total_loss, class_loss, coord_loss

class FocalLoss(autograd.Function):
    @staticmethod
    def forward(ctx, classes, indices, gamma, weight):
        """Compute the focal loss with indices
        Args:
            classes:    (Variable) Shape: [num_bboxes, C+1] Tensor with C+1 classes, (NOT SOFTMAXED)
            indices:    (Variable) Shape: [num_bboxes]      Tensor with GT indices, 0<= value < C+1
            gamma:      (Variable) Shape: [1]               The exponent for FL
            weight:     (Variable) Shape: [C+1]             The "alpha" described in the paper, weight per class
        Return:
            focal_loss: (Variable) Shape: [1] Focal loss
        """
        ctx.save_for_backward(classes, indices, gamma, weight)
        
        #get one_hot representation of indices
        one_hot = torch.cuda.ByteTensor(classes.size()).zero_()
        one_hot.scatter_(1, indices.unsqueeze(1), 1)

        #calc softmax and logsoftmax
        probs = F.softmax(Variable(classes), 1).data
        probs = probs[one_hot]
        logs = probs.log()

        #get weights into the right shape
        weights = torch.index_select(weight, 0, indices)

        #calculate FL and sum
        focal_loss = -weights * torch.pow((1-probs), gamma) * logs
        return torch.mean(focal_loss, 0)

    @staticmethod
    def backward(ctx, grad_output):
        classes, indices, gamma, weight = ctx.saved_variables

        #get one_hot representation of indices
        one_hot = torch.cuda.ByteTensor(classes.size()).zero_()
        one_hot.scatter_(1, indices.data.unsqueeze(1), 1)
        one_hot = Variable(one_hot)

        #calc softmax and logsoftmax
        probs = F.softmax(classes, 1)
        probs_mask = probs[one_hot].unsqueeze(1)
        logs_mask = probs_mask.log()

        #get weights into the right shape
        weights = torch.index_select(weight, 0, indices).unsqueeze(1)
        
        #gradient derived by hand, CE is when focal_change == 1
        focal_factor = torch.pow((1-probs_mask), gamma-1) * (1 - probs_mask - gamma * logs_mask * probs_mask)
        grad = weights * (probs - one_hot.float()) * focal_factor

        N = classes.size(0)
        return grad * grad_output / N, None, None, None
    
if __name__ == "__main__":
    print(torch.cuda.is_available())
    classes = Variable(torch.Tensor([[0.6, 0.4], [0.3, 0.7], [0.9, 0.1]]), requires_grad = True)
    indices = Variable(torch.LongTensor([0, 1, 0]))
    weight = Variable(torch.FloatTensor([0.5,20]))
    loss = torch.nn.CrossEntropyLoss(weight = weight, size_average=False)(classes, indices)
    print("CE", loss)
    loss.backward()
    print("CE", classes.grad)

    fl_classes = Variable(torch.cuda.FloatTensor([[0.6, 0.4], [0.3, 0.7], [0.9, 0.1]]), requires_grad = True)
    print("test")
    fl_indices = Variable(torch.cuda.LongTensor([0, 1, 0]))
    fl_gamma = Variable(torch.cuda.FloatTensor([0]))
    fl_weight = Variable(torch.cuda.FloatTensor([0.5,20]))
    print("classes", fl_classes)
    fl_loss = FocalLoss.apply(fl_classes, fl_indices, fl_gamma, fl_weight)
    print("FL", fl_loss)
    fl_loss.backward()
    print("FL", fl_classes.grad)

    from torch.autograd import gradcheck
    # gradchek takes a tuple of tensor as input, check if your gradient
    # evaluated with these tensors are close enough to numerical
    # approximations and returns True if they all verify this condition.
    input = (Variable(torch.cuda.FloatTensor([[0.6, 0.4], [0.3, 0.7], [0.9, 0.1]]), requires_grad = True),  Variable(torch.cuda.LongTensor([0, 1, 0])),
             Variable(torch.cuda.FloatTensor([4])), Variable(torch.cuda.FloatTensor([1,20])))
    #input = Variable(torch.Tensor([[0.6, 0.4], [0.3, 0.7], [0.9, 0.1]]), requires_grad = True)
    test = gradcheck(FocalLoss.apply, input, eps=1e-5, atol=1e-4)
    print(test)





