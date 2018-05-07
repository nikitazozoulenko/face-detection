from __future__ import print_function
from __future__ import division

from multiprocessing import Process
import time

import torch
import torch.nn as nn
from torch.autograd import Variable
import cv2
import numpy as np

from util_detection import nms
from network_v_2_3 import FaceNet

def func():
    i = 0
    while True:
        print(i)
        i += 1
        time.sleep(1)
if __name__ == "__main__":
    Process(target=func, args=()).start()
    Process(target=func, args=()).start()