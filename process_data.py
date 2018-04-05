import numpy as np
from PIL import Image, ImageOps

def read_WIDERFace(txt_dir = "/hdd/Data/WIDERFace/wider_face_split/wider_face_train_bbx_gt.txt",
                   img_dir = "/hdd/Data/WIDERFace/WIDER_train/images/", LIST_LENGTH = 12880,
                   MAX_NUM_OBJECTS = 1968):
    
    images_filenames = []
    im_num_objects = []
    gt_unprocessed = np.zeros((LIST_LENGTH, MAX_NUM_OBJECTS, 4))
    image_count = -1
    i = 0
    read_num_obj = False
    next_num_objects = 0
    with open(txt_dir, "r") as f:
        for line in f:
            if ".jpg" in line:
                images_filenames.append(img_dir+line.rstrip())
                read_num_obj = True
            elif read_num_obj == True:
                next_num_objects = int(line.rstrip())
                im_num_objects.append(next_num_objects)
                image_count += 1
                i = 0
                read_num_obj = False
            else:
                if i < MAX_NUM_OBJECTS:
                    #parse line
                    line = line.split()
                    gt_unprocessed[image_count, i, 0] = float(line[0]) #xmin
                    gt_unprocessed[image_count, i, 1] = float(line[1]) #ymin
                    gt_unprocessed[image_count, i, 2] = float(line[2]) + float(line[0]) #xmax
                    gt_unprocessed[image_count, i, 3] = float(line[3]) + float(line[1]) #ymax
                    i += 1
                    
    paths = [[images_filenames[i], gt_unprocessed[i], im_num_objects[i]] for i in range(len(images_filenames))]
    return paths #paths[3125:3126] is the max number of objs


def get_paths_train():
    return read_WIDERFace(txt_dir = "/hdd/Data/WIDERFace/wider_face_split/wider_face_train_bbx_gt.txt",
                          img_dir = "/hdd/Data/WIDERFace/WIDER_train/images/", LIST_LENGTH = 12880,
                          MAX_NUM_OBJECTS = 1968)


def get_paths_val():
    return read_WIDERFace(txt_dir = "/hdd/Data/WIDERFace/wider_face_split/wider_face_val_bbx_gt.txt",
                          img_dir = "/hdd/Data/WIDERFace/WIDER_val/images/", LIST_LENGTH = 3226,
                          MAX_NUM_OBJECTS = 709)


def sizes_WIDERFace(paths):
    plot_sizes = []
    for filename, gts, num_obj in paths:
        image = Image.open(filename)
        im_width, im_height = image.size
        gts[:num_obj, 0:1] = gts[:num_obj, 0:1] / im_width
        gts[:num_obj, 1:2] = gts[:num_obj, 1:2] / im_height
        gts[:num_obj, 2:3] = gts[:num_obj, 2:3] / im_width
        gts[:num_obj, 3:4] = gts[:num_obj, 3:4] / im_height

        gts[:num_obj] = gts[:num_obj] * 1200

        sizes = np.zeros((num_obj,2))
        sizes[:,0:1] = gts[:num_obj, 2:3] - gts[:num_obj, 0:1]
        sizes[:,1:2] = gts[:num_obj, 3:4] - gts[:num_obj, 1:2]
        sizes = np.absolute(sizes)
        for gt in sizes:
            size = np.sqrt(gt[0]*gt[1])
            plot_sizes.append(size)
            
    return plot_sizes


if __name__ == "__main__":
    paths = read_WIDERFace()
    sizes = np.array(sizes_WIDERFace(paths[:]))
    print(sizes)
    print(sizes.shape)

    pows = [(2**0.5)**x for x in range(20)]
    import matplotlib.pyplot as plt
    plt.hist(sizes, bins=pows)
    plt.show()
