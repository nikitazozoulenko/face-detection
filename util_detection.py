import numpy as np
from PIL import Image, ImageDraw

import torch
from torch.autograd import Variable

def draw_and_show_boxes(cuda_image, cuda_boxes, border_size, color):
    try:
        image = cuda_image[0].data.cpu().numpy()
        boxes = cuda_boxes.data.cpu().numpy()
        image = np.transpose(image, (1, 2, 0))
    
        im = Image.fromarray((image).astype(np.uint8))
        width, height = im.size
        width = 600
        im = im.resize((width, width))
        im.show()
    
        dr = ImageDraw.Draw(im)
        boxes = np.copy(boxes)
        boxes = (boxes * width).astype(int)
        for box in boxes:
            x0 = box[0]
            y0 = box[1]
            x1 = box[2]
            y1 = box[3]

            for j in range(border_size):
                final_coords = [x0+j, y0+j, x1-j, y1-j]
                dr.rectangle(final_coords, outline = color)
        im.show()
    except Exception:
        print("no boxes")

###I CANT FIGURE OUT HOW TO FIX MY OWN IOU FUNCTION,
###IT IS CORRECT EXCEPT FOR SOME EXAMPLES WHICH IT GIVES IOU>1,
###MAJOIRTY OF EXAMPLES ARE CORRECT THOUGH, FROM NOW ON USING

### NEW IOU CODE FROM https://github.com/amdegroot/ssd.pytorch

def intersect(box_a, box_b):
    """ We resize both tensors to [A,B,2] without new malloc:
    [A,2] -> [A,1,2] -> [A,B,2]
    [B,2] -> [1,B,2] -> [A,B,2]
    Then we compute the area of intersect between box_a and box_b.
    Args:
      box_a: (tensor) bounding boxes, Shape: [A,4].
      box_b: (tensor) bounding boxes, Shape: [B,4].
    Return:
      (tensor) intersection area, Shape: [A,B].
    """
    A = box_a.size(0)
    B = box_b.size(0)
    max_xy = torch.min(box_a[:, 2:].unsqueeze(1).expand(A, B, 2),
                       box_b[:, 2:].unsqueeze(0).expand(A, B, 2))
    min_xy = torch.max(box_a[:, :2].unsqueeze(1).expand(A, B, 2),
                       box_b[:, :2].unsqueeze(0).expand(A, B, 2))
    inter = torch.clamp((max_xy - min_xy), min=0)
    return inter[:, :, 0] * inter[:, :, 1]


def jaccard(box_a, box_b):
    """Compute the jaccard overlap of two sets of boxes.  The jaccard overlap
    is simply the intersection over union of two boxes.  Here we operate on
    ground truth boxes and default boxes.
    E.g.:
        A ∩ B / A ∪ B = A ∩ B / (area(A) + area(B) - A ∩ B)
    Args:
        box_a: (tensor) Ground truth bounding boxes, Shape: [num_objects,4]
        box_b: (tensor) Prior boxes from priorbox layers, Shape: [num_priors,4]
    Return:
        jaccard overlap: (tensor) Shape: [box_a.size(0), box_b.size(0)]
    """
    inter = intersect(box_a, box_b)
    area_a = ((box_a[:, 2]-box_a[:, 0]) *
              (box_a[:, 3]-box_a[:, 1])).unsqueeze(1).expand_as(inter)  # [A,B]
    area_b = ((box_b[:, 2]-box_b[:, 0]) *
              (box_b[:, 3]-box_b[:, 1])).unsqueeze(0).expand_as(inter)  # [A,B]
    union = area_a + area_b - inter
    return inter / union  # [A,B]
