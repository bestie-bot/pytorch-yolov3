import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import cv2
import matplotlib.pyplot as plt
from model.bbox import bbox_iou


def predict_transform(prediction, inp_dim, anchors, num_classes, CUDA=True):
    """
    Takes a detection layer feature map and turns it into a 2D tensor where
    every row is a list with objectness (P0) bounding box coordinates (Bx, By, Bw, Bh),
    and class probabilities for each anchor.

    Params:
        prediction: tensor from prediction, Batch x Channels x Width x Height
            example: 416 x 416 input, first YOLO detection prediction: torch.Size([1, 255, 13, 13])
        inp_dim: input image dimensions
        anchors: list of anchors
        num_classes: number of classes for prediction. VOC is usually 20, COCO=80
        CUDA: CUDA flag

    Returns:
        prediction for classes
    """
    # Let's move predictions to GPU if available
    if CUDA:
        prediction = prediction.cuda()
    # size of the current batch of the detection layer
    batch_size = prediction.size(0)
    # Determined by dividing the full image size by the width of the feature map,
    # which is also the size of the region maps. 416 x 416 = 13, 26, 52
    # for region map sizes among the 3 detection layers.
    stride = inp_dim // prediction.size(2)
    # This is the number of regions, so for an input image of
    # 416 x 416, goes in sizes of 13, 26, 52
    grid_size = inp_dim // stride
    # Boxes always have at least 5 slots, 1 for the objectness score (what is the
    # probability an object is in this grid box) and 4 dimensional attributes
    # of center coodinates + width + height or Bx, By, Bw, Bh
    bbox_attrs = 5 + num_classes
    # Obviously, number of anchor boxes we decided to use
    num_anchors = len(anchors)

    # We need to reduce the anchors from their size on the full image
    # to the reduced grid sizes, and resave them in a list. I thought
    # we would use normal anchor box sizes, so still unsure of this
    anchors = [(a[0]/stride, a[1]/stride) for a in anchors]

    # We need to get the prediction map data down to a 2D tensor
    # with tensor data turned into bounding boxes
    # on 416 x 416 image, generates tensor of 1 deep (flat) x 255 x 169
    # Can check for deletion. Seems we only need the last view.
    prediction = prediction.view(
        batch_size, bbox_attrs*num_anchors, grid_size*grid_size)
    # Generates tensor and rotates it to 1 x 169 x 255
    # Again, can check for deletion. Seems we only need the last view.
    prediction = prediction.transpose(1, 2).contiguous()
    # On the 416 image, creates a 3D tensor of 1 x 507 x 85. This accounts
    # for every anchor in every grid region. In the case of a 0 threshold, all
    # of these boxes would appear on an image.
    prediction = prediction.view(
        batch_size, grid_size*grid_size*num_anchors, bbox_attrs)

    # Sigmoid the  centre_X, centre_Y. and object confidence
    # This turns the values to a scale of 0 to 1, specifically so
    # these bounding box center offsets don't go into another grid cell.
    # Objectness also need to be deterined as a probability, so
    # it gets the sigmoid treatment. Currently it's a tensor value.
    prediction[:, :, 0] = torch.sigmoid(prediction[:, :, 0])
    prediction[:, :, 1] = torch.sigmoid(prediction[:, :, 1])
    # width and height would be prediction[:, : 2 or 3]
    prediction[:, :, 4] = torch.sigmoid(prediction[:, :, 4])

    # We need to create all the offsets for every bounding box center.
    # This means we need to get to a tensor shape we cna add to everything.
    # Let's start with a center offsets mesh grid to hold all the values. It's
    # a 2D tensor by grid size, so for 416 image, it's 13, 26, and 52
    # squared for each YOLO layer
    grid_len = np.arange(grid_size)
    a, b = np.meshgrid(grid_len, grid_len)

    # Convert from the meshgrid from i.e 13x13 array to a single column
    # tensor of 169 x 1. The -1 flattens the tensor first, and
    # the second value is the number of columns
    # Also size 26, 52 as well for 416, which gives us 676x1, and 2704x1.
    x_offset = torch.FloatTensor(a).view(-1, 1)
    y_offset = torch.FloatTensor(b).view(-1, 1)

    # Let's move as many calculations to the GPU as possible at this point
    if CUDA:
        x_offset = x_offset.cuda()
        y_offset = y_offset.cuda()

    # 1. Make one big tensor adding x and y offset along their depth axis. This creates
    # for example with 13 regions a 169 x 2 tensor
    # 2. Repeat these values for each anchor (since each grid cell has a set number
    # of anchors). Continuing example, this turns 169 x 2 tensor into a 169 x 6 tensor
    # 3. We need two columns, so by dropping the 6 columns to 2 columns, we have to add
    # three times the number of rows, hence this tensor goes from 169 x 6 to 507 x 2
    # 4. Unsqueeze adds an extra dimenion along the specific dimension. In this case, we use 0
    # so, this is turns it into a 3D tensor, all in 1 row, or a 1 x 507 x 2 tensor,
    # matching our prediction tensor
    x_y_offset = torch.cat((x_offset, y_offset), 1).repeat(
        1, num_anchors).view(-1, 2).unsqueeze(0)

    # Adds the x and y offsets to the prediction tensors. Remember that the first 5 values
    # for the 3rd dimension are Bx, By, Bw, Bh, P0. Here, ':2' = Bx and By. The x_y center
    # coordinates for each prediction anchor are therefore bumped up by their region value,
    # i.e., the 13th region has values of 12.x, 12.x (remember index 0)for the offsets
    prediction[:, :, :2] += x_y_offset

    # Conver the anchor boxes from a list of n paired tuples (however many anchor boxes we have
    # which is by default 3), to a tensor of n x 2. So, for by default, goes to 3 x 2 tensor
    anchors = torch.FloatTensor(anchors)

    # send the transformed anchors to the GPU now that they're tensors
    if CUDA:
        anchors = anchors.cuda()

    # Let's create enough anchor values for every empty anchor value in our prediction tensor,
    # so this needs to have 507 rows, therefore a 507 x 2 array. Then, we  turn it into
    # a 3D tensor so we can add it to the predictions of our width and heights of our center
    # coordinates, so a 1 x 507 x 2 tensor. (We'll try every anchor box in this space, and later
    # we'll test which ones have better coverage of our predicted object (IoU))
    anchors = anchors.repeat(grid_size*grid_size, 1).unsqueeze(0)

    # We take the log of the height and width of each bounding box. This converts any negative value
    # into a positive one and shrinks the anchor box by that percentage
    prediction[:, :, 2:4] = torch.exp(prediction[:, :, 2:4]) * anchors

    # Softmax the class scores. Currently a P0 score, 5th column, so we do the sigmoid for every class
    prediction[:, :, 5: 5 +
               num_classes] = torch.sigmoid((prediction[:, :, 5: 5 + num_classes]))

    # The bounding boxes are currently sized per region. Let's resize this to the size of the image
    # (I'm not sure why this is)
    prediction[:, :, :4] *= stride

    return prediction


def unique(tensor):
    tensor_np = tensor.cpu().numpy()
    unique_np = np.unique(tensor_np)
    unique_tensor = torch.from_numpy(unique_np)

    tensor_res = tensor.new(unique_tensor.shape)
    tensor_res.copy_(unique_tensor)
    return tensor_res


def write_results(prediction, confidence, num_classes, nms=True, nms_conf=0.4):
    """
    This function performs the objectness confidence thresholding and the 
    non-max suppression (nms), which gives us our true boxes to display.

    Params:
        prediction: ?
        confidence: the objectness score threshold
        num_classes: classes for our model (VOC: 20, COCO: 80, etc)
        nms: should we do nms (defaults to true)
        nms_conf: the NMS IoU threshold

    Returns:
        ?tensor of output predtions
    """

    conf_mask = (prediction[:, :, 4] > confidence).float().unsqueeze(2)
    prediction = prediction*conf_mask

    try:
        ind_nz = torch.nonzero(
            prediction[:, :, 4]).transpose(0, 1).contiguous()
    except:
        return 0

    box_a = prediction.new(prediction.shape)
    box_a[:, :, 0] = (prediction[:, :, 0] - prediction[:, :, 2]/2)
    box_a[:, :, 1] = (prediction[:, :, 1] - prediction[:, :, 3]/2)
    box_a[:, :, 2] = (prediction[:, :, 0] + prediction[:, :, 2]/2)
    box_a[:, :, 3] = (prediction[:, :, 1] + prediction[:, :, 3]/2)
    prediction[:, :, :4] = box_a[:, :, :4]

    batch_size = prediction.size(0)

    output = prediction.new(1, prediction.size(2) + 1)
    write = False

    for ind in range(batch_size):
        # select the image from the batch
        image_pred = prediction[ind]

        # Get the class having maximum score, and the index of that class
        # Get rid of num_classes softmax scores
        # Add the class index and the class score of class having maximum score
        max_conf, max_conf_score = torch.max(
            image_pred[:, 5:5 + num_classes], 1)
        max_conf = max_conf.float().unsqueeze(1)
        max_conf_score = max_conf_score.float().unsqueeze(1)
        seq = (image_pred[:, :5], max_conf, max_conf_score)
        image_pred = torch.cat(seq, 1)

        # Get rid of the zero entries
        non_zero_ind = (torch.nonzero(image_pred[:, 4]))

        image_pred_ = image_pred[non_zero_ind.squeeze(), :].view(-1, 7)

        # Get the various classes detected in the image
        try:
            img_classes = unique(image_pred_[:, -1])
        except:
            continue
        # WE will do NMS classwise
        for cls in img_classes:
            # get the detections with one particular class
            cls_mask = image_pred_ * \
                (image_pred_[:, -1] == cls).float().unsqueeze(1)
            class_mask_ind = torch.nonzero(cls_mask[:, -2]).squeeze()

            image_pred_class = image_pred_[class_mask_ind].view(-1, 7)

            # sort the detections such that the entry with the maximum objectness
            # confidence is at the top
            conf_sort_index = torch.sort(
                image_pred_class[:, 4], descending=True)[1]
            image_pred_class = image_pred_class[conf_sort_index]
            idx = image_pred_class.size(0)

            # if nms has to be done
            if nms:
                # For each detection
                for i in range(idx):
                    # Get the IOUs of all boxes that come after the one we are looking at
                    # in the loop
                    try:
                        ious = bbox_iou(image_pred_class[i].unsqueeze(
                            0), image_pred_class[i+1:])
                    except ValueError:
                        break

                    except IndexError:
                        break

                    # Zero out all the detections that have IoU > treshhold
                    iou_mask = (ious < nms_conf).float().unsqueeze(1)
                    image_pred_class[i+1:] *= iou_mask

                    # Remove the non-zero entries
                    non_zero_ind = torch.nonzero(
                        image_pred_class[:, 4]).squeeze()
                    image_pred_class = image_pred_class[non_zero_ind].view(
                        -1, 7)

            # Concatenate the batch_id of the image to the detection
            # this helps us identify which image does the detection correspond to
            # We use a linear straucture to hold ALL the detections from the batch
            # the batch_dim is flattened
            # batch is identified by extra batch column

            batch_ind = image_pred_class.new(
                image_pred_class.size(0), 1).fill_(ind)
            seq = batch_ind, image_pred_class
            if not write:
                output = torch.cat(seq, 1)
                write = True
            else:
                out = torch.cat(seq, 1)
                output = torch.cat((output, out))

    return output


def letterbox_image(img, inp_dim):
    '''resize image with unchanged aspect ratio using padding'''
    img_w, img_h = img.shape[1], img.shape[0]
    w, h = inp_dim
    new_w = int(img_w * min(w/img_w, h/img_h))
    new_h = int(img_h * min(w/img_w, h/img_h))
    resized_image = cv2.resize(
        img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

    canvas = np.full((inp_dim[1], inp_dim[0], 3), 128)

    canvas[(h-new_h)//2:(h-new_h)//2 + new_h, (w-new_w) //
           2:(w-new_w)//2 + new_w, :] = resized_image

    return canvas


def prep_image(img, inp_dim):
    """
    Prepare image for inputting to the neural network. 

    Returns a Variable 
    """
    img = (letterbox_image(img, (inp_dim, inp_dim)))
    img = img[:, :, ::-1].transpose((2, 0, 1)).copy()
    img = torch.from_numpy(img).float().div(255.0).unsqueeze(0)
    return img


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())


def count_learnable_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def convert2cpu(matrix):
    if matrix.is_cuda:
        return torch.FloatTensor(matrix.size()).copy_(matrix)
    else:
        return matrix


def load_classes(namesfile):
    fp = open(namesfile, "r")
    names = fp.read().split("\n")[:-1]
    return names


def get_im_dim(im):
    im = cv2.imread(im)
    w, h = im.shape[1], im.shape[0]
    return w, h
