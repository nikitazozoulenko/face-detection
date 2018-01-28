import torch
import torch.optim as optim
import matplotlib.pyplot as plt

from detection_network import *
from data_feeder import DataFeeder
from util_detection import *
from process_data import *
from loss import *

x_indices = []
class_losses = []
coord_losses = []
total_losses = []
val_x_indices = []
val_class_losses = []
val_coord_losses = []
val_total_losses = []

def graph():
    plt.figure(1)
    plt.plot(x_indices, class_losses, "r", label="Class Loss")
    plt.plot(x_indices, coord_losses, "g", label="Coord Loss")
    plt.plot(x_indices, total_losses, "b", label="Total Loss")
    plt.legend(loc=1)

    plt.figure(2)
    plt.plot(val_x_indices, val_class_losses, "r--", label="Val Class Loss")
    plt.plot(val_x_indices, val_coord_losses, "g--", label="Val Coord Loss")
    plt.plot(val_x_indices, val_total_losses, "b--", label="Val Total Loss")
    plt.legend(loc=1)

    plt.figure(3)
    plt.plot(x_indices, total_losses, "b", label="Loss")
    plt.plot(val_x_indices, val_total_losses, "g--", label="Val Loss")
    plt.legend(loc=1)

    plt.figure(4)
    plt.plot(x_indices, coord_losses, "b", label="coord_losses")
    plt.plot(val_x_indices, val_coord_losses, "g--", label="val_coord_losses")
    plt.legend(loc=1)

    plt.figure(5)
    plt.plot(x_indices, class_losses, "b", label="class_losses")
    plt.plot(val_x_indices, val_class_losses, "g--", label="val_class_losses")
    plt.legend(loc=1)
    plt.show()

model = FaceNet().cuda()
loss = Loss().cuda()

train_data_feeder = DataFeeder(get_paths_train, read_single_example, make_batch_from_list,
                               preprocess_workers=8, cuda_workers=1,
                               numpy_size=20, cuda_size=1, batch_size=2)
val_data_feeder = DataFeeder(get_paths_val, read_single_example, make_batch_from_list,
                             preprocess_workers=4, cuda_workers=1,
                             numpy_size=6, cuda_size=1, batch_size=1,
                             volatile=True)
train_data_feeder.start_queue_threads()
val_data_feeder.start_queue_threads()
#learning_rate = 0.001
learning_rate = 0.01
optimizer = optim.SGD(model.parameters(), lr=learning_rate,
                      momentum=0.9, weight_decay=0.0001)


def train_batch(data_feeder):
    _, batch = data_feeder.get_batch()
    images, gt, num_objects = batch
    offsets, classes, anchors = model(images)
    total_loss, class_loss, coord_loss = loss(
        offsets, classes, anchors, gt, num_objects)
    return total_loss, class_loss, coord_loss


num_iterations = 3000
for i in range(num_iterations):
    # training loss
    optimizer.zero_grad()
    total_loss, class_loss, coord_loss = train_batch(train_data_feeder)
    try:
        total_loss.backward()
        optimizer.step()
    except Exception as e:
        print(e)

    class_losses += [class_loss.cpu().data.numpy()[0]]
    coord_losses += [coord_loss.cpu().data.numpy()[0]]
    total_losses += [total_loss.cpu().data.numpy()[0]]
    x_indices += [i]

    # validation loss
    if i % 10 == 0:
        model.eval()
        val_total_loss, val_class_loss, val_coord_loss = train_batch(
            val_data_feeder)

        val_class_losses += [val_class_loss.cpu().data.numpy()[0]]
        val_coord_losses += [val_coord_loss.cpu().data.numpy()[0]]
        val_total_losses += [val_total_loss.cpu().data.numpy()[0]]
        val_x_indices += [i]
        model.train()

    # print progress
    if i % 100 == 0:
        print(i)

    # decrease learning rate
    if i == 3000000000:
        learning_rate /= 10
        print("updated learning rate: current lr:", learning_rate)
        for param_group in optimizer.param_groups:
            param_group['lr'] = learning_rate
    if i % 10000 == 0 and i != 0:
        torch.save(model, "savedir/facenet_1_"+str(i//1000)+"k.pt")
model.eval()
_, batch = val_data_feeder.get_batch()
images, gt, num_objects = batch
boxes, classes = model(images, phase="test")
process_draw(0.9, images, boxes, classes)
process_draw(0.8, images, boxes, classes)
process_draw(0.7, images, boxes, classes)
process_draw(0.6, images, boxes, classes)
process_draw(0.5, images, boxes, classes)
process_draw(0.4, images, boxes, classes)
process_draw(0.3, images, boxes, classes)
process_draw(0.2, images, boxes, classes)
process_draw(0.1, images, boxes, classes)
process_draw(0,   images, boxes, classes)

graph()

train_data_feeder.kill_queue_threads()
val_data_feeder.start_queue_threads()
