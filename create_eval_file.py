import os

from PIL import Image
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable

from network_v_1_0 import FaceNet
from util_detection import nms

def path2cudaimage(filepath):
    im = Image.open(filepath)
    width, height = im.size
    width = width + 64 - width%64
    height = height + 64 - height%64
    im = im.resize((width, height))
    im_array = np.asarray(im)
    tensor = torch.from_numpy(im_array).cuda().permute(2,0,1).unsqueeze(0).float()
    return Variable(tensor, volatile=True)


def run_model_on_img(model, cuda_img):
    boxes, classes, anchors = model(cuda_img)
    processed_boxes, processed_conf = nms(boxes, classes, threshhold=0.3, use_nms=True, softmax=True)
    return processed_boxes, processed_conf


def create_eval_txt(processed_boxes, processed_conf, cat, image_path):
    if not os.path.exists("savedir/pred/"+cat):
        os.makedirs("savedir/pred/"+cat)
    with open("savedir/pred/"+cat+"/"+image_path.strip(".jpg")+".txt", mode="w") as f:
        f.write(image_path.strip(".jpg")+"\n")
        f.write(str(len(processed_boxes))+"\n")
        for box, conf in zip(processed_boxes, processed_conf):
            x0 = int(box[0])
            y0 = int(box[1])
            x1 = int(box[2])
            y1 = int(box[3])
            c = float(conf)
            f.write(str(x0)+" "+str(y0)+" "+str(x1-x0)+" "+str(y1-y0)+" "+str(c)+"\n")


def create_txts():
    model = FaceNet().cuda()
    model.load_state_dict(torch.load("savedir/facenet_01_it50k.pth"))
    model.eval()

    if not os.path.exists("savedir/pred"):
        os.makedirs("savedir/pred")

    im_dir = "/hdd/Data/WIDERFace/WIDER_val/images/"
    for cat in os.listdir(im_dir):
        for image_path in os.listdir(im_dir + cat):
            cuda_img = path2cudaimage(im_dir + cat + "/"+ image_path)
            processed_boxes, processed_conf = run_model_on_img(model, cuda_img)
            create_eval_txt(processed_boxes, processed_conf, cat, image_path)


if __name__ == "__main__":
    create_txts()