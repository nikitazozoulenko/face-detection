from __future__ import print_function
from __future__ import division

import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np

from network_v_2_3 import FaceNet
from data_feeder import DataFeeder
from util_detection import process_draw
from PIL import Image


def test_model(images, model):
    boxes, classes, anchors = model(images, phase="test")
    for i in [ 0.2, 0.25]:
        im = process_draw(i, images, boxes, classes, use_nms = True, border_size = 10, softmax = False)
        im.save("results" + str(i) +".png")

def path2cudaimage(filepath):
    im = Image.open(filepath)
    orig_width, orig_height = im.size
    ratio = orig_width/orig_height

    if ratio >1:
        height = 1408+ 128*4
        width = height * ratio
    else:
        width = 1408+ 128*4
        height = width/ratio

    if width % 128 > 128//2:
        width = width + 128 - width % 128
    else:
        width = width- width % 128

    if height % 128 > 128//2:
        height = height + 128 - height % 128
    else:
        height = height - height % 128
    height = int(height)
    width = int(width)

    im = im.resize((width, height))
    im_array = np.asarray(im)
    tensor = torch.from_numpy(im_array).cuda().permute(2,0,1).unsqueeze(0).float()
    return Variable(tensor, volatile=True)


def main():
    model = FaceNet().cuda()
    model.load_state_dict(torch.load("savedir/facenet_pref.pth"))
    model.eval()

    num_iterations = 1
    for i in range(num_iterations):
        #image = path2cudaimage("/hdd/Images/crowd.jpg")

        image = path2cudaimage("/hdd/Data/WIDERFace/WIDER_val/images/12--Group/12_Group_Group_12_Group_Group_12_935.jpg")
        #image = path2cudaimage("/hdd/Data/WIDERFace/WIDER_val/images/2--Demonstration/2_Demonstration_Demonstration_Or_Protest_2_476.jpg")
        #image = path2cudaimage("/hdd/Data/WIDERFace/WIDER_val/images/0--Parade/0_Parade_marchingband_1_267.jpg")
        #image = path2cudaimage("/hdd/Data/WIDERFace/WIDER_val/images/12--Group/12_Group_Large_Group_12_Group_Large_Group_12_315.jpg")
        image = path2cudaimage("/hdd/Data/WIDERFace/WIDER_val/images/12--Group/12_Group_Team_Organized_Group_12_Group_Team_Organized_Group_12_60.jpg")
        image = path2cudaimage("/hdd/Data/WIDERFace/WIDER_val/images/0--Parade/0_Parade_marchingband_1_490.jpg")
        
        #image = path2cudaimage("/hdd/Data/WIDERFace/WIDER_val/images/6--Funeral/6_Funeral_Funeral_6_696.jpg")
        test_model(image, model)


# def main():
#     model = FaceNet().cuda()
#     model.load_state_dict(torch.load("savedir/facenet_01_it60k.pth"))
#     model.eval()

#     num_iterations = 1
#     for i in range(num_iterations):
#         im = Image.open("/hdd/Images/crowd.jpg")
#         #im = Image.open("/hdd/Data/WIDERFace/WIDER_val/images/12--Group/12_Group_Group_12_Group_Group_12_935.jpg")
#         width, height = im.size
#         width = width + (128 - width%128)
#         height = height + (128 - height%128)
#         im.show()
#         im = im.resize((width,height))
#         array = np.asarray(im).astype(np.float32)
#         tensor = torch.from_numpy(array).permute(2,0,1).unsqueeze(0).cuda()
#         image = Variable(tensor, requires_grad = False, volatile = True)
#         test_model(image, model)

if __name__ == "__main__":
    main()