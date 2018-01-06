import numpy as np
from PIL import Image, ImageOps

def get_paths():
    MAX_NUM_OBJECTS = 1968
    LIST_LENGTH = 12880
    directory = "/hdd/Data/wider_face_split/wider_face_train_bbx_gt.txt"

    images_filenames = []
    im_num_objects = []
    gt_unprocessed = np.zeros((LIST_LENGTH, MAX_NUM_OBJECTS, 4))

    image_count = -1
    i = 0
    read_num_obj = False
    next_num_objects = 0
    with open(directory, "r") as f:
        for line in f:
            if ".jpg" in line:
                images_filenames.append("/hdd/Data/WIDER_train/images/"+line.rstrip())
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
                    gt_unprocessed[image_count, i, 2] = float(line[2]) #width
                    gt_unprocessed[image_count, i, 3] = float(line[3]) #height
                    i += 1
                    
    paths = [[images_filenames[i], gt_unprocessed[i], im_num_objects[i]] for i in range(len(images_filenames))]
    np.random.shuffle(paths)
    return paths

def fake_read_image_and_label(path):
    #path is (filename, gt)
    image_path, gt, num_objects = path

    fake = np.array([[ 0.78125,     0.22894168,  0.04589844,  0.06695464],
        [ 0.88769531,  0.39092873,  0.03320312,  0.04427646],
        [ 0.51171875,  0.29805616,  0.03710938,  0.05075594],
        [ 0.32519531,  0.10043197,  0.05859375,  0.10367171],
        [ 0.1796875 ,  0.39308855,  0.03027344,  0.0399568 ],
        [ 0.08105469,  0.44492441,  0.02636719,  0.03131749],
        [ 0.38671875,  0.42548596,  0.02539062,  0.03023758]])

    fake_gt = np.zeros(gt.shape)
    fake_gt[0:7, :] = fake
    gt = fake_gt
    #random number for if to flip horizontally or not
    random = np.random.randint(0,2)

    #read corresponding jpeg
    image = Image.open("/hdd/Data/WIDER_train/images/44--Aerobics/44_Aerobics_Aerobics_44_803.jpg")

    if(random == 0):
        image = ImageOps.mirror(image)    
        #xmin = 1-xmin-width
        gt[:num_objects, 0:1] = 1 - gt[:num_objects, 0:1] - gt[:num_objects, 2:3]
        
    image_array = np.asarray(image)
    gt = gt.astype(np.float32)
    
    return image_array, gt, 7

def read_image_and_label(path):
    #path is (filename, gt)
    image_path, gt, num_objects = path

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
        #xmin = 1-xmin-width
        gt[:num_objects, 0:1] = 1 - gt[:num_objects, 0:1] - gt[:num_objects, 2:3]
        
    image_array = np.asarray(image)
    gt = gt.astype(np.float32)
    
    return image_array, gt, num_objects

def make_batch_from_list(images, gt, num_objects):
    width = 320
    random = np.random.randint(0,10)
    resize_size = (width + 32*random, width + 32*random)
    resized_images = [np.asarray(Image.fromarray(image).resize(resize_size)) for image in images]
    
    max_batch_objects  = max(num_objects)
    gt = np.array(gt)[:, 0:max_batch_objects, :]
    
    return np.array(resized_images).astype(np.float32), gt, np.array(num_objects)
    
    

