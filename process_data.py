import numpy as np
from PIL import Image, ImageOps

def read_WIDERFace(txt_dir = "/hdd/Data/wider_face_split/wider_face_train_bbx_gt.txt",
                   img_dir = "/hdd/Data/WIDER_train/images/", LIST_LENGTH = 12880,
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
    return paths

def get_paths_train():
    return read_WIDERFace(txt_dir = "/hdd/Data/wider_face_split/wider_face_train_bbx_gt.txt",
                          img_dir = "/hdd/Data/WIDER_train/images/", LIST_LENGTH = 12880,
                          MAX_NUM_OBJECTS = 1968)

def get_paths_val():
    return read_WIDERFace(txt_dir = "/hdd/Data/wider_face_split/wider_face_val_bbx_gt.txt",
                          img_dir = "/hdd/Data/WIDER_val/images/", LIST_LENGTH = 3226,
                          MAX_NUM_OBJECTS = 709)

def fake_read_single_example(path):
    fake = np.array([[ 0.78125,     0.22894168,  0.04589844,  0.06695464],
        [ 0.88769531,  0.39092873,  0.03320312,  0.04427646],
        [ 0.51171875,  0.29805616,  0.03710938,  0.05075594],
        [ 0.32519531,  0.10043197,  0.05859375,  0.10367171],
        [ 0.1796875 ,  0.39308855,  0.03027344,  0.0399568 ],
        [ 0.08105469,  0.44492441,  0.02636719,  0.03131749],
        [ 0.38671875,  0.42548596,  0.02539062,  0.03023758]])
    fake[:, 2:3] += fake[:,0:1]
    fake[:, 3:4] += fake[:,1:2]
    num_objects = 7

    gt = np.copy(fake)
    #random number for if to flip horizontally or not
    random = np.random.randint(0,2)

    #read corresponding jpeg
    image = Image.open("/hdd/Data/WIDER_train/images/44--Aerobics/44_Aerobics_Aerobics_44_803.jpg")

    if(random == 0):
        image = ImageOps.mirror(image)
        #xmax = 1-xmin
        xmax_temp = np.copy(gt[:num_objects, 2:3])
        gt[:num_objects, 2:3] = 1 - gt[:num_objects, 0:1]
        #xmin = 1-xmax
        gt[:num_objects, 0:1] = 1 - xmax_temp
        
    image_array = np.asarray(image)
    gt = gt.astype(np.float32)
    
    return (image_array, gt, 7)

def read_single_example(path):
    #path is (filename, gt)
    image_path, gt, num_objects = path
    gt = np.copy(gt)
    #random number for if to flip horizontally or not
    random = np.random.randint(0,2)

    #read corresponding jpeg
    image = Image.open(image_path)
    im_width, im_height = image.size
    gt[:, 0:1] = gt[:, 0:1] / im_width
    gt[:, 1:2] = gt[:, 1:2] / im_height
    gt[:, 2:3] = gt[:, 2:3] / im_width
    gt[:, 3:4] = gt[:, 3:4] / im_height

    if(random == 0):
        image = ImageOps.mirror(image)
        #xmax = 1-xmin
        xmax_temp = np.copy(gt[:num_objects, 2:3])
        gt[:num_objects, 2:3] = 1 - gt[:num_objects, 0:1]
        #xmin = 1-xmax
        gt[:num_objects, 0:1] = 1 - xmax_temp
        
    image_array = np.asarray(image)
    gt = gt.astype(np.float32)
    
    return [image_array, gt, num_objects]

    
def make_batch_from_list(cumulative_batch):
    images = [x[0] for x in cumulative_batch]
    gt = [x[1] for x in cumulative_batch]
    num_objects = [x[2] for x in cumulative_batch]
    width = 512
    random = np.random.randint(0,3)
    resize_size = (width + 64*random, width + 64*random)
    resized_images = [np.asarray(Image.fromarray(image).resize(resize_size)) for image in images]
    
    max_batch_objects  = max(num_objects)
    gt = np.array(gt)[:, 0:max_batch_objects, :]
    
    return np.array(resized_images).astype(np.float32), gt, np.array(num_objects)
