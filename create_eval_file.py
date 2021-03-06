import os

from PIL import Image
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable

from network_v_2_4 import FaceNet
from util_detection import nms
from util_detection import process_draw
ctr = 0
def path2cudaimage(filepath):
    im = Image.open(filepath)
    orig_width, orig_height = im.size
    ratio = orig_width/orig_height

    if ratio >1:
        height = 1408- 128*5
        width = height * ratio
    else:
        width = 1408- 128*5
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
    return Variable(tensor, volatile=True), orig_width, orig_height, width, height


def run_model_on_img(model, cuda_img, threshold, orig_width, orig_height, width, height):
    boxes, classes, anchors = model(cuda_img, phase="test")
    #process_draw(threshold, cuda_img, boxes, classes, use_nms = True, softmax=False)
    boxes[:,:,0:1] = boxes[:,:,0:1]*orig_width/width
    boxes[:,:,1:2] = boxes[:,:,1:2]*orig_height/height
    boxes[:,:,2:3] = boxes[:,:,2:3]*orig_width/width
    boxes[:,:,3:4] = boxes[:,:,3:4]*orig_height/height
    processed_boxes, processed_conf = nms(boxes, classes, threshhold=threshold, use_nms=True, softmax=False)
    # global ctr
    # print(ctr)
    # ctr += 1
    # if ctr > 10:
    #     boxes, classes, anchors = model(cuda_img, phase="test")
    #     process_draw(threshold, cuda_img, boxes, classes, use_nms = True, softmax=False)
    # if ctr == 50:
    #     assert True == False
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
    #model.load_state_dict(torch.load("savedir/facenet_01_it70k.pth"))
    model.load_state_dict(torch.load("savedir/facenet_v_2_4.pth"))
    model.eval()

    if not os.path.exists("savedir/pred"):
        os.makedirs("savedir/pred")

    im_dir = "/hdd/Data/WIDERFace/WIDER_val/images/"
    for cat in os.listdir(im_dir):
        for image_path in os.listdir(im_dir + cat):
            cuda_img, orig_width, orig_height, width, height = path2cudaimage(im_dir + cat + "/"+ image_path)
            processed_boxes, processed_conf = run_model_on_img(model, cuda_img, 0.01, orig_width, orig_height, width, height)
            create_eval_txt(processed_boxes, processed_conf, cat, image_path)


if __name__ == "__main__":
    create_txts()