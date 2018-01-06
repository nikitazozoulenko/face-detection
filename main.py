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

model = DetectionNetwork().cuda()
loss = Loss().cuda()
data_feeder = DataFeeder(preprocess_workers = 6, cuda_workers = 1, numpy_size = 8, cuda_size = 2, batch_size = 12)
data_feeder.start_queue_threads()

learning_rate = 0.0001

optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)

class_losses = []
coord_losses = []
total_losses = []
val_class_losses = []
val_coord_losses = []
val_total_losses = []
num_iterations = 2000
for i in range(num_iterations):
    optimizer.zero_grad()
    _, (batch) = data_feeder.get_batch()
    images, gt, num_objects = batch
    boxes, classes = model(images, phase = "train")
    total_loss, class_loss, coord_loss = loss(0.3, boxes, classes, gt, num_objects)
    
    total_loss.backward()
    optimizer.step()

    class_losses += [class_loss.cpu().data.numpy()[0]]
    coord_losses += [coord_loss.cpu().data.numpy()[0]]
    total_losses += [total_loss.cpu().data.numpy()[0]]

    if i == 1000 or i == 2000:
        learning_rate /= np.sqrt(10)
        for param_group in optimizer.param_groups:
            param_group['lr'] = learning_rate
images, gt, num_objects = batch
boxes, classes = model(images, phase = "test")
draw_and_show_boxes(images.cpu().data.numpy(), boxes.cpu().data.numpy(), 1, "red")
    
data_feeder.kill_queue_threads()

plt.plot(class_losses, "r", label="Class Loss")
plt.plot(coord_losses, "g", label="Coord Loss")
plt.plot(total_losses, "b", label="Total Loss")

# Place a legend above this subplot, expanding itself to
# fully use the given bounding box.
plt.legend(loc=1)

plt.show()
