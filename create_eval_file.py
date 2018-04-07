import os

def create_txts():
    im_dir = "/hdd/Data/WIDERFace/WIDER_val/images/"

    for cat in os.listdir(im_dir):
        print(cat)

if __name__ == "__main__":
    create_txts()