from __future__ import print_function
from __future__ import division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt

from detection_network import *
from data_feeder import DataFeeder
from util_detection import *
from process_data import *
from loss import *

class_losses = []
coord_losses = []
total_losses = []
val_class_losses = []
val_coord_losses = []
val_total_losses = []
def graph():
    plt.plot(class_losses, "r", label="Class Loss")
    plt.plot(coord_losses, "g", label="Coord Loss")
    plt.plot(total_losses, "b", label="Total Loss")
    plt.legend(loc=1)
    plt.show()

model = FaceNet().cuda()
loss = Loss().cuda()

train_data_feeder = DataFeeder(get_paths_train, read_single_example, make_batch_from_list,
                               preprocess_workers = 8, cuda_workers = 1,
                               numpy_size = 20, cuda_size = 2, batch_size = 7)
val_data_feeder = DataFeeder(get_paths_val, read_single_example, make_batch_from_list,
                               preprocess_workers = 4, cuda_workers = 1,
                               numpy_size = 10, cuda_size = 1, batch_size = 2)
train_data_feeder.start_queue_threads()
val_data_feeder.start_queue_threads()

learning_rate = 0.000001
optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)

num_iterations = 60000
#0.005
try:
    for i in range(num_iterations):
        #optimizer.zero_grad()
        _, batch = train_data_feeder.get_batch()
        images, gt, num_objects = batch
        offsets, classes, anchors = model(images)
        total_loss, class_loss, coord_loss = loss(offsets, classes, anchors, gt, num_objects)
        
        total_loss.backward()
        optimizer.step()

        class_losses += [class_loss.cpu().data.numpy()[0]]
        coord_losses += [coord_loss.cpu().data.numpy()[0]]
        total_losses += [total_loss.cpu().data.numpy()[0]]

        if i % 100 == 0:
            print(i)
        if i == 300 or i == 30000:
            learning_rate /= (np.sqrt(10)**2)
            print("updated learning rate: current lr:", learning_rate)
            for param_group in optimizer.param_groups:
                param_group['lr'] = learning_rate
except Exception:
    print("exception")
graph()

_, batch = train_data_feeder.get_batch()
images, gt, num_objects = batch
boxes, classes = model(images, phase = "test")
#draw_and_show_boxes(images.cpu().data.numpy(), anchors.cpu().data.numpy(), 1, "red")
draw_and_show_boxes(images, boxes, 1, "red")
_, val_batch = val_data_feeder.get_batch()
val_images, val_gt, val_num_objects = val_batch
val_boxes, val_classes = model(val_images, phase = "test")
#draw_and_show_boxes(val_images.cpu().data.numpy(), val_anchors.cpu().data.numpy(), 1, "red")
draw_and_show_boxes(val_images, val_boxes, 1, "red")
    
train_data_feeder.kill_queue_threads()
val_data_feeder.start_queue_threads()


#plt.plot(total_losses, "b", label="Train Loss")
#plt.plot(val_total_losses, "r", label="Validation Loss")

