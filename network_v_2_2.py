#cross conv3-conv7 with bilinear interpolation upsampling, new anchors, 0.55

import torch
import torch.nn as nn
import torchvision
from torchvision import models
import torch.autograd as autograd
from torch.autograd import Variable
from torch.nn import Parameter
import torch.nn.functional as F
import numpy as np

from util_detection import jaccard

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
        block_depth = 2

        res_0 = [ResidualBlock(channels, expansion, cardinality) for _ in range(block_depth)]
        res_1 = [ResidualBlock(channels, expansion, cardinality) for _ in range(block_depth)]
        self.residual0 = nn.Sequential(*res_0)
        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear")
        self.residual1 = nn.Sequential(*res_1)

    def forward(self, x):
        #x shape [batch_size, 256, H, W]
        x = self.residual0(x)
        x = self.upsample(x)
        x = self.residual1(x)
        x = self.regressor(x)
        return x


class ClassificationHead(nn.Module):
    def __init__(self):
        super(ClassificationHead, self).__init__()
        A = 6
        pi = 0.001
        bias = -np.log((1-pi)/pi)
        self.prior = Parameter(torch.FloatTensor([[bias]]).expand(A, -1, -1)).contiguous()
        
        self.conf_predictions = nn.Conv2d(256,   A, kernel_size=3, stride=1, padding=1, bias = False)

        channels = 64
        expansion = 4
        cardinality = 1
        block_depth = 2

        res_0 = [ResidualBlock(channels, expansion, cardinality) for _ in range(block_depth)]
        res_1 = [ResidualBlock(channels, expansion, cardinality) for _ in range(block_depth)]
        self.residual0 = nn.Sequential(*res_0)
        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear")
        self.residual1 = nn.Sequential(*res_1)

    def forward(self, x):
        #x shape [batch_size, 256, H, W]
        x = self.residual0(x)
        x = self.upsample(x)
        x = self.residual1(x)
        return self.conf_predictions(x) + self.prior
    

class FaceNet(nn.Module):
    def __init__(self):
        super(FaceNet, self).__init__()
        resnet = models.resnet18(pretrained=True)
        modules_conv3 = list(resnet.children())[:6]
        modules_conv4 = list(resnet.children())[6]
        modules_conv5 = list(resnet.children())[7]
        
        self.input_BN = nn.BatchNorm3d(3)

        self.conv3 = nn.Sequential(*modules_conv3)
        self.bottleneck_conv3 = nn.Conv2d(128, 256, kernel_size=1, stride=1, padding=0, bias = True)
        
        self.conv4 = nn.Sequential(*modules_conv4)
        self.bottleneck_conv4 = nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0, bias = True)

        self.conv5 = nn.Sequential(*modules_conv5)
        self.bottleneck_conv5 = nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0, bias = True)

        self.conv6 = nn.Sequential(nn.Conv2d(512, 512, kernel_size=1, stride=2, padding=0, bias = True),
                                   *[ResidualBlock(128, expansion=4) for _ in range(2)])
  
        self.conv7 = nn.Sequential(nn.Conv2d(512, 512, kernel_size=1, stride=2, padding=0, bias = True),
                                   *[ResidualBlock(128, expansion=4) for _ in range(2)])
        self.bottleneck_conv6 = nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0, bias = True)
        self.bottleneck_conv7 = nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0, bias = True)

        self.prediction_head =  PredictionHead()

        self.anchors_wh3 = torch.Tensor([[16, 16],  [16, 16*1.5],
                                         [20, 20],  [20, 20*1.5],
                                         [25, 25],  [25, 25*1.5]]).cuda()
        self.anchors_wh4 = torch.Tensor([[32, 32],  [32, 32*1.5],
                                         [40, 40],  [40, 40*1.5],
                                         [51, 51],  [51, 51*1.5]]).cuda()
        self.anchors_wh5 = torch.Tensor([[64, 64],  [64, 64*1.5],
                                         [81, 81],  [81, 81*1.5],
                                         [102, 102],  [102, 102*1.5]]).cuda()
        self.anchors_wh6 = torch.Tensor([[128, 128],  [128, 128*1.5],
                                         [161, 161],  [161, 161*1.5],
                                         [203, 203],  [203, 203*1.5]]).cuda()
        self.anchors_wh7 = torch.Tensor([[256, 256],  [256, 256*1.5],
                                         [322, 322],  [322, 322*1.5],
                                         [406, 406],  [406, 406*1.5]]).cuda()

        self.upsampling = nn.Upsample(scale_factor=2, mode="bilinear")
        
    def forward(self, x, phase = "train"):
        _, _, height, width = x.size()
        x = self.input_BN(x)

        conv3 = self.conv3(x)
        conv4 = self.conv4(conv3)
        conv5 = self.conv5(conv4)
        conv6 = self.conv6(conv5)
        conv7 = self.conv7(conv6)

        conv7 = self.bottleneck_conv7(conv7)
        conv6 = self.bottleneck_conv6(conv6) + self.upsampling(conv7)
        conv5 = self.bottleneck_conv5(conv5) + self.upsampling(conv6)
        conv4 = self.bottleneck_conv4(conv4) + self.upsampling(conv5)
        conv3 = self.bottleneck_conv3(conv3) + self.upsampling(conv4)

        offsets7, classes7 = self.prediction_head(conv7)
        boxes7, classes7, anchors7 = make_anchors_and_bbox(offsets7, classes7, self.anchors_wh7, height, width)
        offsets6, classes6 = self.prediction_head(conv6)
        boxes6, classes6, anchors6 = make_anchors_and_bbox(offsets6, classes6, self.anchors_wh6, height, width)
        offsets5, classes5 = self.prediction_head(conv5)
        boxes5, classes5, anchors5 = make_anchors_and_bbox(offsets5, classes5, self.anchors_wh5, height, width)
        offsets4, classes4 = self.prediction_head(conv4)
        boxes4, classes4, anchors4 = make_anchors_and_bbox(offsets4, classes4, self.anchors_wh4, height, width)
        offsets3, classes3 = self.prediction_head(conv3)
        boxes3, classes3, anchors3 = make_anchors_and_bbox(offsets3, classes3, self.anchors_wh3, height, width)


        #concat all the predictions
        #boxes = [boxes3, boxes4, boxes5, boxes6, boxes7]
        #classes = [classes3, classes4, classes5, classes6, classes7]
        #anchors = [anchors3, anchors4, anchors5, anchors6, anchors7]
        #boxes = [boxes2, boxes3, boxes4, boxes5, boxes6]
        #classes =[classes2, classes3, classes4, classes5, classes6]
        #anchors = [anchors2, anchors3, anchors4, anchors5, anchors6]
        boxes = torch.cat((boxes3, boxes4, boxes5, boxes6, boxes7), dim=1)
        classes =torch.cat((classes3, classes4, classes5, classes6, classes7), dim=1)
        anchors = torch.cat((anchors3, anchors4, anchors5, anchors6, anchors7), dim=0)
        return boxes, classes, anchors


def make_anchors_and_bbox(offsets, classes, anchors_wh, height, width):
    #offsets shape [batch_size, 4A, H, W]
    #anchors shape [A, 2]
    #classes shape [batch_size, A, H, W]
    R, A, H, W = classes.size()

    #RESHAPE OFFSETS
    offsets = offsets.view(R, 4, A*H*W).permute(0,2,1)
            
    #RESHAPE CLASSES
    classes = classes.view(R, A*H*W)
            
    #EXPAND CENTER COORDS
    x_coords = ((torch.arange(W).cuda()+0.5)/W*width).expand(H, W)
    y_coords = ((torch.arange(H).cuda()+0.5)/H*height).expand(W, H).t()
    coord_grid = torch.stack((x_coords,y_coords), dim = 2) #H-dim, W-dim, (x,y)
    coord_grid = coord_grid.expand(A,-1,-1,-1) #A-dim, H-dim, W-dim, (x,y)
    coords = coord_grid.contiguous().view(-1, 2) #AHW, 2
    anch = anchors_wh.unsqueeze(1).expand(-1,H*W,-1).contiguous().view(-1, 2) #AHW, 2

    anchors_min = coords - anch/2
    anchors_max = anchors_min + anch
            
    anchors = Variable(torch.cat((anchors_min, anchors_max), dim = 1), requires_grad = False)
    boxes = offsets + anchors

    return boxes, classes, anchors


class ClassLoss(nn.Module):
    def __init__(self):
        super(ClassLoss, self).__init__()
        self.sigmoid = nn.Sigmoid()
        self.cross_entropy = nn.BCEWithLogitsLoss(size_average=False)

    def forward(self, classes, positive_idx):
        gather_pos = torch.zeros(classes.size(0), out=torch.LongTensor()).cuda()
        num_pos = 1
        if len(positive_idx) != 0:
            positive_idx = positive_idx[:,0]
            num_pos = float(positive_idx.size(0))
            gather_pos.index_fill_(0, positive_idx.data, 1)
        indices = Variable(gather_pos.float())

        #eps = 0.0000000001
        #gamma = 3
        #pred = self.sigmoid(classes)
        #loss = -indices*((1-pred)**gamma)*torch.log(pred+eps) - (1-indices)*(pred**gamma)*torch.log(1-pred+eps)
        #loss = -indices*torch.log(pred+eps) - (1-indices)*torch.log(1-pred+eps)
        loss = self.cross_entropy(classes, indices) / num_pos
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
        threshhold = 0.55
        R = batch_gt.size(0)
        class_loss = Variable(torch.zeros(1).cuda())
        coord_loss = Variable(torch.zeros(1).cuda())
        
        for boxes, classes, gt, num_objects in zip(batch_boxes, batch_classes, batch_gt, batch_num_objects):
            gt = gt[:num_objects]
            pos, idx = match(threshhold, anchors.data, gt.data)
            class_loss += self.class_loss(classes, pos)
            coord_loss += self.coord_loss(boxes, gt, pos, idx)
        class_loss = class_loss / R
        coord_loss = coord_loss / R
        total_loss = class_loss + coord_loss
        return total_loss, class_loss, coord_loss


if __name__ == "__main__":
    A = 6
    height = 128
    width = 128
    offsets = Variable(torch.Tensor(3, 4*A, 32, 32))
    classes = Variable(torch.Tensor(3, A, 32, 32))
    anchors_wh2 = torch.Tensor([[16, 16],  [16, 16*2],
                                [20, 20],  [20, 20*2],
                                [25, 25],  [25, 25*2]])
    result = make_anchors_and_bbox(offsets, classes, anchors_wh2, height, width)