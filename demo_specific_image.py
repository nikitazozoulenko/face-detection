from __future__ import print_function
from __future__ import division

import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np

from network_v_2_4 import FaceNet
from data_feeder import DataFeeder
from util_detection import process_draw
from PIL import Image


def test_model(images, model):
    boxes, classes, anchors = model(images, phase="test")
    for i in [0.15, 0.7, 0.9]:
        im = process_draw(i, images, boxes, classes, use_nms = True, border_size = 4, softmax = False)
        im.save("results" + str(i) +".png")


def main():
    model = FaceNet().cuda()
    model.load_state_dict(torch.load("savedir/facenet_v_2_4.pth"))
    model.eval()

    num_iterations = 1
    for i in range(num_iterations):
        #im = Image.open("/hdd/Images/crowd.jpg")
        im = Image.open("/hdd/Data/WIDERFace/WIDER_val/images/12--Group/12_Group_Group_12_Group_Group_12_935.jpg")
        width, height = im.size
        width = width + (128 - width%128)
        height = height + (128 - height%128)
        im.show()
        im = im.resize((width,height))
        array = np.asarray(im).astype(np.float32)
        tensor = torch.from_numpy(array).permute(2,0,1).unsqueeze(0).cuda()
        image = Variable(tensor, requires_grad = False, volatile = True)
        test_model(image, model)

if __name__ == "__main__":
    main()