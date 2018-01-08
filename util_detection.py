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

def process_boxes(boxes, classes):
    #batch_boxes,      size [batch_size,       S*S*A,  4]
    #batch_classes,    size [batch_size,       S*S*A,  K+1]
    boxes = boxes[0, :, :]
    classes = classes[0, :, :]
    
    mask = classes > 0.5
    idx = mask[:, 1].nonzero().squeeze()
    print("idx", idx)
    print("classes", classes)
    try:
        selected_boxes = boxes.index_select(0, idx)
        selected_classes = classes.index_select(0, idx)
        print("SELECTED BOXES", selected_boxes)
        print("SELECTED CLASSES", selected_classes)
        return selected_boxes, selected_classes
    except Exception:
        print("EXCEPTION")
        fake = Variable(torch.zeros(1)).cuda()
        return fake, fake
    
def from_xywh_to_xyxy(boxes):
    xmin = boxes[:, 0:1]
    ymin = boxes[:, 1:2]
    xmax = xmin + boxes[:, 2:3]
    ymax = ymin + boxes[:, 3:4]
    return torch.cat((xmin, ymin, xmax, ymax), dim=1)


# def box_set_iou(boxes0, boxes1):
#     """
#     Args:
#         boxes1: shape [num_boxes_0, 4] xmin, ymin, width, height
#         boxes2: shape [num_boxes_1, 4] xmin, ymin, width, height
#     Returns:
#         ious: shape (num_boxes_0, num_boxes_1)
#         where ious[i, j] is the iou of boxes0[i], boxes1[j]
#     """
#     xmin0 = boxes0[:, 0:1]
#     ymin0 = boxes0[:, 1:2]
#     xmax0 = xmin0 + boxes0[:, 2:3]
#     ymax0 = ymin0 + boxes0[:, 3:4]

#     xmin1 = boxes1[:, 0:1]
#     ymin1 = boxes1[:, 1:2]
#     xmax1 = xmin1 + boxes1[:, 2:3]
#     ymax1 = ymin1 + boxes1[:, 3:4]

#     x00 = torch.min(xmin0, xmax0)
#     y00 = torch.min(ymin0, ymax0)
#     x01 = torch.max(xmin0, xmax0)
#     y01 = torch.max(ymin0, ymax0)

#     x10 = torch.min(xmin1, xmax1)
#     y10 = torch.min(ymin1, ymax1)
#     x11 = torch.max(xmin1, xmax1)
#     y11 = torch.max(ymin1, ymax1)

#     xI0 = torch.max(x00, x10.t())
#     yI0 = torch.max(y00, y10.t())
    
#     xI1 = torch.min(x01, x11.t())
#     yI1 = torch.min(y01, y11.t())

#     xI0 = torch.max(x00, x10.t())
#     yI0 = torch.max(y00, y10.t())
    
#     xI1 = torch.min(x01, x11.t())
#     yI1 = torch.min(y01, y11.t())
    
#     zero = Variable(torch.zeros(1)).cuda()

#     inter_area = torch.max((xI1 - xI0) * (yI1 - yI0), other = zero)
#     boxes0_area = (x01 - x00) * (y01 - y00)
#     boxes1_area = (x11 - x10) * (y11 - y10)
    
#     ious = inter_area/((boxes0_area + boxes1_area.t()) - inter_area)
    
#     #print("xI0", xI0)
#     #print("yI0", yI0)
#     #print("inter_area", inter_area)
#     #print("boxes0_area", boxes0_area)
#     #print("boxes1_area", boxes1_area)
#     #print("UNION", (boxes0_area + boxes1_area.t()) - inter_area)

#     return torch.max(ious, other = zero)

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
