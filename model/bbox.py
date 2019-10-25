import torch
import random

import numpy as np
import cv2


def confidence_filter(result, confidence):
    conf_mask = (result[:, :, 4] > confidence).float().unsqueeze(2)
    result = result*conf_mask

    return result


def confidence_filter_cls(result, confidence):
    max_scores = torch.max(result[:, :, 5:25], 2)[0]
    res = torch.cat((result, max_scores), 2)
    print(res.shape)

    cond_1 = (res[:, :, 4] > confidence).float()
    cond_2 = (res[:, :, 25] > 0.995).float()

    conf = cond_1 + cond_2
    conf = torch.clamp(conf, 0.0, 1.0)
    conf = conf.unsqueeze(2)
    result = result*conf
    return result


def get_abs_coord(box):
    box[2], box[3] = abs(box[2]), abs(box[3])
    x1 = (box[0] - box[2]/2) - 1
    y1 = (box[1] - box[3]/2) - 1
    x2 = (box[0] + box[2]/2) - 1
    y2 = (box[1] + box[3]/2) - 1
    return x1, y1, x2, y2


def sanity_fix(box):
    if (box[0] > box[2]):
        box[0], box[2] = box[2], box[0]

    if (box[1] > box[3]):
        box[1], box[3] = box[3], box[1]

    return box


def bbox_iou(box1, box2):
    """
    Returns the IoU of bounding boxes of tensor box1 vs tensor box2
    IoU equals (area of rectangular overlap of the boxes / total area of both boxes combined)

    Params:
        box1: comparison box 1 - tensor
        box2: comparison box 2 - tensor

    Returns:
        1d tensor containing IoUs of the bounding box represented by the first input 
        with each of the bounding boxes present in the second input

    """
    # Get the coordinates of bounding boxes, top-left corners, bottom-right corners
    # for nms, b1 is a single tensor, so the values are single tensor values like [87.8459],
    # where b2's values are each larger tensors like [63.6384, 59.0261, 91.1992]
    b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
    b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]

    # Get the coordinates of the intersection rectangle. This compares the x or y value in the
    # first box to all the values in the other box. For NMS, and a class prediction size of 4 like we do above, then each of these tensors is size[3]. In the case of
    # getting passed [87.8459] and [63.6384, 59.0261, 91.1992], inner_rect_x1 is [87.8459, 87.8459, 91.1992]
    # Small example:
    # ----------------------------
    # --1111111111----------------
    # --1--------1----------------
    # --1--------1----------------
    # --1---b2222c2222------------
    # --1---2////1---2------------
    # --1111a1111d---2------------
    # ------2--------2------------
    # ------2222222222------------
    # ----------------------------
    # Box1 = 1, Box2 = 2
    # a = max x1s of boxes
    # b = max y1s of boxes
    # c = min x2s of boxes
    # d = min y2s of boxes
    # //// = area of intersecting rectangle
    inter_rect_x1 = torch.max(b1_x1, b2_x1)
    inter_rect_y1 = torch.max(b1_y1, b2_y1)
    inter_rect_x2 = torch.min(b1_x2, b2_x2)
    inter_rect_y2 = torch.min(b1_y2, b2_y2)

    # Intersection area
    if torch.cuda.is_available():
        # As above, if a class prediction size of 4 like we do above, then this tensor is size[3]
        # This is really inefficient, and needs to be cleaned up.
        # PyTorch doesn't seem to hav a way to subtract tensors properly, so you can cuda() it.
        # So, you have to do a max() (against a zero tensor so you get the vars you input anyway),
        # and this returns the original tensor, but hey, it was done on the GPU. Then, guess what,
        # we just are really multiplying width x height here.
        inter_area = torch.max(inter_rect_x2 - inter_rect_x1 + 1, torch.zeros(inter_rect_x2.shape).cuda(
        ))*torch.max(inter_rect_y2 - inter_rect_y1 + 1, torch.zeros(inter_rect_x2.shape).cuda())

    else:
        # CPU version
        inter_area = (inter_rect_x2 - inter_rect_x1 + 1) * \
            (inter_rect_y2 - inter_rect_y1 + 1)

    # Union Area - Numerator
    # b1 & b2 area = w + 1 (so there's always a width), times height + 1 (so there's always a height)
    b1_area = (b1_x2 - b1_x1 + 1)*(b1_y2 - b1_y1 + 1)
    b2_area = (b2_x2 - b2_x1 + 1)*(b2_y2 - b2_y1 + 1)

    # Pretty self-explanatory. We add the area of the two rectangles, and then subtract the part where
    # they overlap, and boom, the denominator
    iou = inter_area / (b1_area + b2_area - inter_area)

    return iou


def pred_corner_coord(prediction):
    # Get indices of non-zero confidence bboxes
    ind_nz = torch.nonzero(prediction[:, :, 4]).transpose(0, 1).contiguous()

    box = prediction[ind_nz[0], ind_nz[1]]

    box_a = box.new(box.shape)
    box_a[:, 0] = (box[:, 0] - box[:, 2]/2)
    box_a[:, 1] = (box[:, 1] - box[:, 3]/2)
    box_a[:, 2] = (box[:, 0] + box[:, 2]/2)
    box_a[:, 3] = (box[:, 1] + box[:, 3]/2)
    box[:, :4] = box_a[:, :4]

    prediction[ind_nz[0], ind_nz[1]] = box

    return prediction


def write(x, batches, results, colors, classes):
    c1 = tuple(x[1:3].int())
    c2 = tuple(x[3:5].int())
    img = results[int(x[0])]
    cls = int(x[-1])
    label = "{0}".format(classes[cls])
    color = random.choice(colors)
    cv2.rectangle(img, c1, c2, color, 1)
    t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 1, 1)[0]
    c2 = c1[0] + t_size[0] + 3, c1[1] + t_size[1] + 4
    cv2.rectangle(img, c1, c2, color, -1)
    cv2.putText(img, label, (c1[0], c1[1] + t_size[1] + 4),
                cv2.FONT_HERSHEY_PLAIN, 1, [225, 255, 255], 1)
    return img
