import os

im_dir = "/hdd/Data/WIDERFace/WIDER_val/images/"

for cat in os.listdir(im_dir):
    print(cat)