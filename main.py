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

model = DetectionNetwork().cuda()
data_feeder = DataFeeder(preprocess_workers = 12, cuda_workers = 1, numpy_size = 12, cuda_size = 3, batch_size = 128)
data_feeder.start_queue_threads()

for i in range(100):
    _, batch = data_feeder.get_batch()
    print("BATCH", _)

data_feeder.kill_queue_threads()
