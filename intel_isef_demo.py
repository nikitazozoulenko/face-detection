from __future__ import print_function
from __future__ import division

from multiprocessing import Process, Queue
import time

import torch
import torch.nn as nn
from torch.autograd import Variable
import cv2
import numpy as np

from util_detection import nms
from network_v_2_3 import FaceNet


def numpy_to_cuda(numpy_array):
    return Variable(torch.from_numpy(numpy_array).cuda().permute(2,0,1).float().unsqueeze(0), volatile=True)


def thread_func(cap, queue):
    while True:
        success, frame = cap.read()
        frame = cv2.resize(frame, (640,384))
        frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        queue.put(frame, block=True)

def main():
    model = FaceNet().cuda()
    model.load_state_dict(torch.load("savedir/facenet_pref.pth"))
    model.eval()
    
    cap = cv2.VideoCapture(0)
    cap.set(3,640) #width
    cap.set(4,384) #height

    queue = Queue(maxsize=10)

    p = Process(target=thread_func, args=(cap, queue))
    p.daemon = True
    p.start()

    canvas = np.zeros((800, 1000, 3), dtype=np.uint8)
    while True:
        now = time.time()
        frame = queue.get(block=True)
        
        cuda_frame = numpy_to_cuda(frame)
        boxes, classes, anchors = model(cuda_frame)
        processed_boxes, processed_classes = nms(boxes, classes, 0.3, use_nms = True, softmax=False)
        frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)

        for box in processed_boxes:
            box = box.int()
            xmin = box[0]
            ymin = box[1]
            xmax = box[2]
            ymax = box[3]
            cv2.rectangle(frame,(xmin,ymin),(xmax,ymax),(0,255,0),2)
        
        canvas[0:384, 0:640, :] = frame
        # check to see if the frame should be displayed to our screen
        cv2.imshow("Frame", canvas)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        then = time.time()
        print(1/(then-now))

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()