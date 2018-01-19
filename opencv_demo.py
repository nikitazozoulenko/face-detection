from __future__ import print_function
from __future__ import division

import torch
import torch.nn as nn
import torch.optim as optim
# from torch.optim import lr_scheduler
# from torch.autograd import Variable
# import numpy as np
# import matplotlib.pyplot as plt

# from detection_network import *
# from data_feeder import DataFeeder
# from util_detection import *
# from process_data import *
# from loss import *

import cv2

cap = cv2.VideoCapture(0)

ret, frame = cap.read()
print(frame.__class__)
    
while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Our operations on the frame come here
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Display the resulting frame
    #cv2.imshow('frame',gray)
    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
