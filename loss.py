import torch
import torch.nn as nn
import torchvision
from torchvision import models
import torch.autograd as autograd
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np

from util_detection import jaccard

# # focal loss
# class ClassLoss(nn.Module):
#     def __init__(self):
#         super(ClassLoss, self).__init__()

#     def forward(self, classes, positive_idx):
#         gather_pos = torch.zeros(classes.size(0), out=torch.LongTensor()).cuda()
#         if len(positive_idx) != 0:
#             positive_idx = positive_idx[:,0]
#             gather_pos.index_fill_(0, positive_idx.data, 1)
#         indices = Variable(gather_pos)

#         gamma = Variable(torch.cuda.FloatTensor([2]))
#         weight = Variable(torch.cuda.FloatTensor([1,3]))
#         loss = FocalLoss.apply(classes, indices, gamma, weight)

#         return torch.mean(loss)

class ClassLoss(nn.Module):
    def __init__(self):
        super(ClassLoss, self).__init__()
        self.cross_entropy = nn.CrossEntropyLoss(weight=torch.cuda.FloatTensor([1,3]), size_average=True, reduce=True)

    def forward(self, classes, positive_idx):
        gather_pos = torch.zeros(classes.size(0), out=torch.LongTensor()).cuda()
        if len(positive_idx) != 0:
            positive_idx = positive_idx[:,0]
            gather_pos.index_fill_(0, positive_idx.data, 1)
        indices = Variable(gather_pos)

        loss = self.cross_entropy(classes, indices)
        return loss


class CoordLoss(nn.Module):
    def __init__(self):
        super(CoordLoss, self).__init__()
        self.l1_loss = nn.SmoothL1Loss(size_average=True)

    def forward(self, boxes, gt, pos, idx):
        #boxes,       size [S*S*A,       4]
        #gt,          size [num_obj, 4]
        #positive_idx size [num_matches, 2]
        if len(pos) != 0:
            pred_idx = pos.squeeze()
            gt_idx = idx.squeeze()

            selected_pred = boxes.index_select(0, pred_idx)
            selected_gt = gt.index_select(0, gt_idx)

            coord_loss = self.l1_loss(selected_pred, selected_gt)
            return coord_loss
        else:
            return 0

def match(threshhold, anchors, gts):
    pos = []
    idx = []
    for i, gt in enumerate(gts):
        ious = jaccard(anchors, gt.unsqueeze(0))
        pos_mask = ious.squeeze() >= threshhold
        indices = torch.nonzero(pos_mask)
        if len(indices) != 0:
            pos += [indices]
            idx += [torch.cuda.LongTensor(indices.size()).fill_(i)]
    if len(pos) != 0:
        return Variable(torch.cat(pos, dim=0)), Variable(torch.cat(idx, dim=0))
    else:
        return pos, idx
    
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
        threshhold = 0.54
        R = batch_gt.size(0)
        class_loss = Variable(torch.zeros(1).cuda())
        coord_loss = Variable(torch.zeros(1).cuda())
        
        for boxes, classes, gt, num_objects in zip(batch_boxes, batch_classes, batch_gt, batch_num_objects):
            gt = gt[:num_objects]
            pos, idx = match(threshhold, anchors.data, gt.data)
            class_loss += self.class_loss(classes, pos)
            coord_loss += self.coord_loss(boxes, gt, pos, idx)
        class_loss = class_loss / R
        coord_loss = coord_loss / R / 1000
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
        eps = 0.00001
        
        #get one_hot representation of indices
        one_hot = torch.cuda.ByteTensor(classes.size()).zero_()
        one_hot.scatter_(1, indices.unsqueeze(1), 1)

        #calc softmax and logsoftmax
        probs = F.softmax(Variable(classes), 1).data
        probs = probs[one_hot]
        logs = (probs+eps).log()

        #get weights into the right shape
        weights = torch.index_select(weight, 0, indices)

        #calculate FL and sum
        focal_loss = -weights * torch.pow((1-probs), gamma) * logs
        #return torch.mean(focal_loss, 0)
        return focal_loss

    @staticmethod
    def backward(ctx, grad_output):
        classes, indices, gamma, weight = ctx.saved_variables
        eps = 0.00001

        #get one_hot representation of indices
        one_hot = torch.cuda.ByteTensor(classes.size()).zero_()
        one_hot.scatter_(1, indices.data.unsqueeze(1), 1)
        one_hot = Variable(one_hot)

        #calc softmax and logsoftmax
        probs = F.softmax(classes, 1)
        probs_mask = probs[one_hot].unsqueeze(1)
        logs_mask = (probs_mask+eps).log()

        #get weights into the right shape
        weights = torch.index_select(weight, 0, indices).unsqueeze(1)
        
        #gradient derived by hand, CE is when focal_change == 1
        focal_factor = torch.pow((1-probs_mask), gamma-1) * (1 - probs_mask - gamma * logs_mask * probs_mask)
        grad = weights * (probs - one_hot.float()) * focal_factor

        return grad * grad_output.expand(2, -1).t(), None, None, None
    
if __name__ == "__main__":
    print(torch.cuda.is_available())
    classes = Variable(torch.Tensor([[0.6, 0.4], [0.3, 0.7], [0.9, 0.1]]), requires_grad = True)
    indices = Variable(torch.LongTensor([0, 1, 0]))
    weight = Variable(torch.FloatTensor([0.5,20]))
    loss = torch.nn.CrossEntropyLoss(weight = weight, size_average=False, reduce = False)(classes, indices)
    print("CE", loss)
    torch.autograd.backward([loss], [torch.ones(loss.size())])
    print("CE", classes.grad)

    fl_classes = Variable(torch.cuda.FloatTensor([[0.6, 0.4], [0.3, 0.7], [0.9, 0.1]]), requires_grad = True)
    print("test")
    fl_indices = Variable(torch.cuda.LongTensor([0, 1, 0]))
    fl_gamma = Variable(torch.cuda.FloatTensor([0]))
    fl_weight = Variable(torch.cuda.FloatTensor([0.5,20]))
    fl_loss = FocalLoss.apply(fl_classes, fl_indices, fl_gamma, fl_weight)
    print("FL", fl_loss)
    torch.autograd.backward([fl_loss], [torch.ones(fl_loss.size()).cuda()])
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





