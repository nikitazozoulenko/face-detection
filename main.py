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

model = DetectionNetwork().cuda()
loss = Loss().cuda()
train_data_feeder = DataFeeder(get_paths_train, read_single_example, make_batch_from_list,
                               preprocess_workers = 8, cuda_workers = 1,
                               numpy_size = 20, cuda_size = 2, batch_size = 8)
val_data_feeder = DataFeeder(get_paths_train, read_single_example, make_batch_from_list,
                               preprocess_workers = 4, cuda_workers = 1,
                               numpy_size = 10, cuda_size = 1, batch_size = 2)

train_data_feeder.start_queue_threads()
val_data_feeder.start_queue_threads()

learning_rate = 0.0001

optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)

class_losses = []
coord_losses = []
total_losses = []
val_class_losses = []
val_coord_losses = []
val_total_losses = []
num_iterations = 4000
for i in range(num_iterations):
    optimizer.zero_grad()
    _, batch = train_data_feeder.get_batch()
    images, gt, num_objects = batch
    classes, anchors = model(images, phase = "train")
    total_loss, class_loss, coord_loss = loss(0.5, classes, anchors, gt, num_objects)
    
    total_loss.backward()
    optimizer.step()

    class_losses += [class_loss.cpu().data.numpy()[0]]
    coord_losses += [coord_loss.cpu().data.numpy()[0]]
    total_losses += [total_loss.cpu().data.numpy()[0]]

    # _, val_batch = val_data_feeder.get_batch()
    # images, gt, num_objects = val_batch
    # val_boxes, val_classes = model(images, phase = "train")
    # val_total_loss, val_class_loss, val_coord_loss = loss(0.5, val_boxes, val_classes, gt, num_objects)
    # val_class_losses += [val_class_loss.cpu().data.numpy()[0]]
    # val_coord_losses += [val_coord_loss.cpu().data.numpy()[0]]
    # val_total_losses += [val_total_loss.cpu().data.numpy()[0]]

    if i == 1500:
         learning_rate /= (np.sqrt(10)**2)
         for param_group in optimizer.param_groups:
             param_group['lr'] = learning_rate
_, batch = train_data_feeder.get_batch()
images, gt, num_objects = batch
classes, anchors = model(images, phase = "test")
#draw_big(images.cpu().data.numpy(), anchors.cpu().data.numpy(), 1, "red")
draw_and_show_boxes(images.cpu().data.numpy(), anchors.cpu().data.numpy(), 1, "red")
    
train_data_feeder.kill_queue_threads()
val_data_feeder.start_queue_threads()

plt.plot(class_losses, "r", label="Class Loss")
plt.plot(coord_losses, "g", label="Coord Loss")
plt.plot(total_losses, "b", label="Total Loss")

#plt.plot(total_losses, "b", label="Train Loss")
#plt.plot(val_total_losses, "r", label="Validation Loss")

# Place a legend above this subplot, expanding itself to
# fully use the given bounding box.
plt.legend(loc=1)

plt.show()



# BEST_GT_IDX Variable containing:
#  614
# [torch.cuda.LongTensor of size 1 (GPU 0)]

# best iou Variable containing:
#  0.5766
# [torch.cuda.FloatTensor of size 1 (GPU 0)]

# mask ious Variable containing:
#  0.5295
#  0.5325
#  0.5766
#  0.5109
#  0.5097
# [torch.cuda.FloatTensor of size 5 (GPU 0)]

# BEST_GT_IDX Variable containing:
#  786
#  669
#  543
#  462
# [torch.cuda.LongTensor of size 4 (GPU 0)]

# best iou Variable containing:
#  0.6185
#  0.4656
#  0.6632
#  0.7608
# [torch.cuda.FloatTensor of size 4 (GPU 0)]

# mask ious Variable containing:
#  0.5783
#  0.5001
#  0.7608
#  0.6632
#  0.5001
#  0.5258
#  0.6185
#  0.5238
# [torch.cuda.FloatTensor of size 8 (GPU 0)]

# BEST_GT_IDX Variable containing:
#   544
#  1354
#  1068
#  1201
#  1231
#  1076
#  1066
#   925
#  1326
#  1050
#   898
#   907
#   734
#   736
#   724
#  1079
#   895
# [torch.cuda.LongTensor of size 17 (GPU 0)]

# best iou Variable containing:
#  0.4331
#  0.4597
#  0.1747
#  0.2630
#  0.2640
#  0.2157
#  0.0300
#  0.1751
#  0.5588
#  0.5215
#  0.5599
#  0.2041
#  0.1573
#  0.1871
#  0.0601
#  0.1768
#  0.0462
# [torch.cuda.FloatTensor of size 17 (GPU 0)]

# mask ious Variable containing:
#  0.5599
#  0.5215
#  0.5588
# [torch.cuda.FloatTensor of size 3 (GPU 0)]

# ANCHORS Variable containing:
# -0.1188 -0.1188  0.1813  0.1813
# -0.0438 -0.1188  0.1063  0.1813
# -0.0588 -0.0588  0.1213  0.1213
#                ⋮                
#  0.9438  0.9438  0.9938  0.9938
#  0.9563  0.9438  0.9812  0.9938
#  0.9563  0.9563  0.9812  0.9812
# [torch.cuda.FloatTensor of size 2304x4 (GPU 0)]

























































# GT Variable containing:
#  0.1523  0.4993  0.0312  0.0563
#  0.0527  0.3755  0.0127  0.0225
#  0.1016  0.3854  0.0127  0.0267
#  0.2695  0.4402  0.0156  0.0464
#  0.3545  0.5049  0.0361  0.0661
#  0.3975  0.5921  0.0332  0.0900
#  0.4570  0.4740  0.0176  0.0422
#  0.4902  0.4627  0.0176  0.0366
#  0.5342  0.4332  0.0137  0.0295
#  0.3428  0.4205  0.0195  0.0281
#  0.7617  0.4712  0.0195  0.0309
#  0.8271  0.4459  0.0215  0.0520
#  0.6885  0.3952  0.0146  0.0338
#  0.8506  0.5274  0.0342  0.0675
#  0.5117  0.4641  0.0117  0.0323
# [torch.cuda.FloatTensor of size 15x4 (GPU 0)]

# BEST_GT_IDX Variable containing:
#  651
# [torch.cuda.LongTensor of size 1 (GPU 0)]

# best iou Variable containing:
#  0.4902
# [torch.cuda.FloatTensor of size 1 (GPU 0)]

# mask ious Variable containing:[torch.cuda.FloatTensor with no dimension]

# GT Variable containing:
#  0.4648  0.1849  0.0918  0.1438
# [torch.cuda.FloatTensor of size 1x4 (GPU 0)]

# BEST_GT_IDX Variable containing:
#  648
# [torch.cuda.LongTensor of size 1 (GPU 0)]

# best iou Variable containing:
#  0.5549
# [torch.cuda.FloatTensor of size 1 (GPU 0)]

# mask ious Variable containing:
#  0.5549
#  0.5549
#  0.5549
#  0.5549
# [torch.cuda.FloatTensor of size 4 (GPU 0)]

# GT Variable containing:
#  0.3467  0.1248  0.4199  0.3863
# [torch.cuda.FloatTensor of size 1x4 (GPU 0)]

# BEST_GT_IDX Variable containing:
#  783
# [torch.cuda.LongTensor of size 1 (GPU 0)]

# best iou Variable containing:
#  0.7879
# [torch.cuda.FloatTensor of size 1 (GPU 0)]

# mask ious Variable containing:
#  0.5592
#  0.6252
#  0.5380
#  0.7879
#  0.5836
#  0.5765
#  0.5453
# [torch.cuda.FloatTensor of size 7 (GPU 0)]

# GT Variable containing:
#  0.3389  0.1995  0.2490  0.2848
# [torch.cuda.FloatTensor of size 1x4 (GPU 0)]

# BEST_GT_IDX Variable containing:
#   634
#  1066
#   940
#   985
# [torch.cuda.LongTensor of size 4 (GPU 0)]

# best iou Variable containing:
#  0.5238
#  0.5465
#  0.6532
#  0.4116
# [torch.cuda.FloatTensor of size 4 (GPU 0)]

# mask ious Variable containing:
#  0.5238
#  0.6532
#  0.5171
#  0.5465
# [torch.cuda.FloatTensor of size 4 (GPU 0)]

# GT Variable containing:
#  0.3818  0.2241  0.0957  0.0955
#  0.3584  0.4037  0.0898  0.0841
#  0.4775  0.3534  0.0850  0.0876
#  0.7959  0.3513  0.0635  0.0718
# [torch.cuda.FloatTensor of size 4x4 (GPU 0)]

