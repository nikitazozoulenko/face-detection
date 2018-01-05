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

from detection_network import *
from data_feeder import DataFeeder
from util_detection import *

model = DetectionNetwork().cuda()
loss = Loss().cuda()
data_feeder = DataFeeder(preprocess_workers = 12, cuda_workers = 1, numpy_size = 8, cuda_size = 2, batch_size = 2)
data_feeder.start_queue_threads()

for i in range(1):
    _, (batch) = data_feeder.get_batch()
    images, gt, num_objects = batch
    boxes, classes = model(images)
    total_loss, class_loss, coord_loss = loss(0.001, boxes, classes, gt, num_objects)
    
    #draw_and_show_boxes(images.cpu().data.numpy(), boxes.cpu().data.numpy(), 1, "red")
    
    draw_big(images.cpu().data.numpy(), gt[0, 0:num_objects[0]].cpu().data.numpy(), 3, "red")
    draw_big(images.cpu().data.numpy(), boxes[0].cpu().data.numpy(), 3, "blue")
    

data_feeder.kill_queue_threads()
