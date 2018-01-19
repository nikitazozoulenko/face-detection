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

train_data_feeder = DataFeeder(get_paths_train, read_single_example, make_batch_from_list,
                               preprocess_workers = 8, cuda_workers = 1,
                               numpy_size = 20, cuda_size = 3, batch_size = 8)
val_data_feeder = DataFeeder(get_paths_val, read_single_example, make_batch_from_list,
                               preprocess_workers = 4, cuda_workers = 1,
                               numpy_size = 10, cuda_size = 2, batch_size = 1)
train_data_feeder.start_queue_threads()
val_data_feeder.start_queue_threads()

#model = torch.load("savedir/facenet0o001.pt")
model = torch.load("savedir/facenet200k.pt")
model.eval()

num_iterations = 100
for i in range(num_iterations):
    print(i)
    _, batch = val_data_feeder.get_batch()
    images, gt, num_objects = batch
    boxes, classes = model(images, phase = "test")
    process_draw(0.35, images, boxes, classes)
    
train_data_feeder.kill_queue_threads()
val_data_feeder.start_queue_threads()


