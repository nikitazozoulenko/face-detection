import torch
import torch.optim as optim
import matplotlib.pyplot as plt

from network_v_2_3 import FaceNet, Loss
from data_feeder import DataFeeder
from util_detection import process_draw
from process_data import get_paths_train, get_paths_val
from logger import Logger

def train(batch_loss, optimizer):
    optimizer.zero_grad()
    batch_loss.backward()
    optimizer.step()


def calc_loss(model, loss, data_feeder, i, total_logger, class_logger, coord_logger):
    _, batch = data_feeder.get_batch()
    images, gt, num_objects = batch
    offsets, classes, anchors = model(images)
    total_loss, class_loss, coord_loss = loss(offsets, classes, anchors, gt, num_objects)
    total_logger.write_log(total_loss, i)
    class_logger.write_log(class_loss, i)
    coord_logger.write_log(coord_loss, i)
    return total_loss


def increase_lr(optimizer):
    for param_group in optimizer.param_groups:
        print("updated learning rate: new lr:", param_group['lr']*10)
        param_group['lr'] = param_group['lr']*10


def decrease_lr(optimizer):
    for param_group in optimizer.param_groups:
        print("updated learning rate: new lr:", param_group['lr']/10)
        param_group['lr'] = param_group['lr']/10
    

def main():
    train_data_feeder = DataFeeder(get_paths_train, preprocess_workers=4, cuda_workers=1,
                                numpy_size=14, cuda_size=3, batch_size=16, jitter=True)
    val_data_feeder = DataFeeder(get_paths_val, preprocess_workers=1, cuda_workers=1,
                                numpy_size=6, cuda_size=1, batch_size=16, jitter = False,
                                volatile=True)
    train_data_feeder.start_queue_threads()
    val_data_feeder.start_queue_threads()

    version = "01"
    model = FaceNet().cuda()
    #model.load_state_dict(torch.load("savedir/facenet_01_it60k.pth"))
    loss = Loss().cuda()

    optimizer = optim.SGD(model.parameters(), lr=0.0005,
                      momentum=0.9, weight_decay=0.00001)

    t_total_logger= Logger("train_total_losses.txt")
    t_class_logger= Logger("train_class_losses.txt")
    t_coord_logger= Logger("train_coord_losses.txt")
    v_total_logger= Logger("val_total_losses.txt")
    v_class_logger= Logger("val_class_losses.txt")
    v_coord_logger= Logger("val_coord_losses.txt")

    n_acc_grad = 1
    for i in range(150001):
        optimizer.zero_grad()
        for j in range(n_acc_grad):
            batch_loss = calc_loss(model, loss, train_data_feeder, i*n_acc_grad + j, t_total_logger, t_class_logger, t_coord_logger)
            #train(batch_loss, optimizer)
            batch_loss.backward()
        optimizer.step()
        if i % 20 == 0:
            print(i)
            model.eval()
            calc_loss(model, loss, val_data_feeder, i*n_acc_grad, v_total_logger, v_class_logger, v_coord_logger)
            model.train()
        if i in [99999999999]:
            decrease_lr(optimizer)
        if i % 5000 == 0 and i!=0:
            torch.save(model.state_dict(), "savedir/facenet_"+version+"_it"+str(i//1000)+"k.pth")
    
    model.eval()
    for i in range(10):
        _, batch = train_data_feeder.get_batch()
        images, gt, num_objects = batch
        images = images[0:1]
        boxes, classes, anchors = model(images)
        #process_draw(0.05, images, anchors, classes, use_nms = False, softmax=True)
        process_draw(0.10, images, anchors, classes, use_nms = False, softmax=False, border_size=1)

    train_data_feeder.kill_queue_threads()
    val_data_feeder.kill_queue_threads()

    
if __name__ == "__main__":
    main()