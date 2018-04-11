from __future__ import print_function
from __future__ import division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt

from network_v_1_1 import FaceNet
from data_feeder import DataFeeder
from process_data import get_paths_train, get_paths_val
from util_detection import process_draw

train_data_feeder = DataFeeder(get_paths_train, preprocess_workers = 8, cuda_workers = 1, numpy_size = 20, cuda_size = 3, batch_size = 1, volatile = True)
val_data_feeder = DataFeeder(get_paths_val, preprocess_workers = 4, cuda_workers = 1,
                               numpy_size = 10, cuda_size = 2, batch_size = 1, jitter = False, volatile = True)
train_data_feeder.start_queue_threads()
val_data_feeder.start_queue_threads()

model = FaceNet().cuda()
model.load_state_dict(torch.load("savedir/facenet_01_it30k.pth"))
model.eval()

model2 = FaceNet().cuda()
model2.load_state_dict(torch.load("savedir/facenet_01_it70k.pth"))
model2.eval()


def test_model(images, model):
    boxes, classes, anchors = model(images)
    #process_draw(0.05, images, boxes, classes, use_nms = False)
    #process_draw(0.2, images, boxes, classes, use_nms = False)
    #process_draw(0.3, images, boxes, classes, use_nms = False)
    #process_draw(0.4, images, boxes, classes, use_nms = False, border_size = 1)
    #process_draw(0.5, images, boxes, classes, use_nms = False)
    #process_draw(0.6, images, anchors, classes, use_nms = False, border_size = 1)
    process_draw(0.4, images, anchors, classes, use_nms = False, softmax=True)
    #process_draw(0.8, images, boxes, classes, use_nms = True)
    #process_draw(0.9, images, boxes, classes, use_nms = False)
    

num_iterations = 5
for i in range(num_iterations):
    print(i)
    _, batch = train_data_feeder.get_batch()
    images, gt, num_objects = batch
    test_model(images, model)
    test_model(images, model)
    test_model(images, model2)

    
train_data_feeder.kill_queue_threads()
val_data_feeder.kill_queue_threads()


