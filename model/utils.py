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


def write_results(prediction, confidence, num_classes, nms=True, nms_conf=0.4):
    """
    This function performs the objectness confidence thresholding and the
    non-max suppression (nms), which gives us our true boxes to display.

    Params:
        prediction: tensor of size [1, 10647, 85] for 416 image
        confidence: the objectness score threshold
        num_classes: classes for our model (VOC: 20, COCO: 80, etc)
        nms: should we do nms (defaults to true)
        nms_conf: the NMS IoU threshold

    Returns:
        ?tensor of output predtions
    """
    # We set the threshold image wide for object prediction scores (P0) for
    # all 10647 boxes (416 image). We turn the 2d tensor we pull from
    # the prediction (1 x 10647) and convert to float. We then use unsqueeze
    # to convert to (1 x 10647 x 1) tensor. We can then multiply the predictions
    # so that if the conf_mask = 0, all the values are zeroed out for the class
    # predictions
    conf_mask = (prediction[:, :, 4] > confidence).float().unsqueeze(2)
    prediction = prediction * conf_mask

    # Currently done in bbox, not sure why it's here at the moment
    # try:
    #     ind_nz = torch.nonzero(
    #         prediction[:, :, 4]).transpose(0, 1).contiguous()
    # except:
    #     return 0

    # We need to turn our Bx, By, Bw, Bh into:
    # 1. top-left corner x
    # 2. top-left corner y
    # 3. right-bottom corner x
    # 4. right-bottom corner y
    # This way we can do an IoU comparison of predicted vs truth
    # There has to be a way to speed this process up even if
    # we're running batches of images and have various true detections
    box_a = prediction.new(prediction.shape)
    # Top-left corner x = center x - (width / 2)
    box_a[:, :, 0] = (prediction[:, :, 0] - prediction[:, :, 2]/2)
    # Top-left corner y = center y - (height / 2)
    box_a[:, :, 1] = (prediction[:, :, 1] - prediction[:, :, 3]/2)
    # Bottom-right corner y = center x + (width / 2)
    box_a[:, :, 2] = (prediction[:, :, 0] + prediction[:, :, 2]/2)
    # Bottom-right corner y = center y + (height / 2)
    box_a[:, :, 3] = (prediction[:, :, 1] + prediction[:, :, 3]/2)
    # Take the new values and replace the previous bounding box values
    # in the prediction tensor
    prediction[:, :, :4] = box_a[:, :, :4]

    # How many images are we processing at a time
    batch_size = prediction.size(0)

    # output = prediction.new(1, prediction.size(2) + 1)
    # Write will be true when we're ready to do the entire final output
    # for all images
    write = False

    for ind in range(batch_size):
        # The prediction tensor at ind is saying select the current image
        # in a batch. If this batch is one, then you get one image.
        # This converts from a prediction tensor of 1 x 10647 x 85 to a
        # 2D tensor of 10647 x 85
        image_pred = prediction[ind]

        # ** Confidence threshholding **

        # values, indices = max(input, dim)
        # Go through every row and indexes 5 - 85 (All class probablities in COCO),
        # of that row and return the maximum values only and index of that value. The max
        # method reduces the tensor space by 1, so max_conf and max_conf_score
        # each become a single 10647 tensor
        max_conf, max_conf_score = torch.max(
            image_pred[:, 5:5 + num_classes], 1)
        # Convert max_conf tensor from 10647 single tensor to a 2d
        # 10647 x 1 tensor of float values
        max_conf = max_conf.float().unsqueeze(1)
        # Convert max_conf_score tensor from 10647 single tensor to a 2d
        # 10647 x 1 tensor of float values
        max_conf_score = max_conf_score.float().unsqueeze(1)

        # Groups 3 tensors together as size:
        # (([10647, 5]), ([10647, 1]), ([10647, 1]))
        seq = (image_pred[:, :5], max_conf, max_conf_score)

        # Creates a prediction list of size ([10647, 7])
        image_pred = torch.cat(seq, 1)

        # Find all the predictions with a value greater than 0 and get
        # their index. Comes back as a 2d tensor like [16, 1]
        non_zero_ind = (torch.nonzero(image_pred[:, 4]))

        # Let's drop all the  values at 0 or below in our predictions
        try:
            # Reduce non_zero_ind to 1d tensor size of ([16]), which selects all the
            # rows at those indexes (since we're passing a list to the image_pred tensor
            # row param), plus all the columns of data in those rows (the ':'). That
            # returns a [16 x 7] tensor.
            image_pred_ = image_pred[non_zero_ind.squeeze(), :].view(-1, 7)

        except:
            continue

        # If there are no detections, then keep going. May be able to drpo this in
        # PyTorch 1.3, will have to check with an image and nothing in it.
        if image_pred_.shape[0] == 0:
            continue

        # Get the various classes detected in the image
        # -1 index (the last column value) which holds the class index
        # This tells you what you have predicted in the image, and now
        # we just need to sort the boxes out
        img_classes = torch.unique(image_pred_[:, -1])

        # ** Perform NMS **
        # This is non-maximum suppression, where we take the top values
        # and IoU to determine which multiples of our prediction classes
        # are true or not (I have 4 predicted people overlapping for one
        # person, but which one is the actual predicted box)
        # In the standard dog-bicycle-truck example for a static image
        # from Darknet, we enter with 3 classes to figure out
        for cls in img_classes:
            # Get the detections with one particular class
            # This will only select the rows that have a value equal to the
            # class listed in image_pred last column. Torch.where() returns
            # a [4] tensor with the indices of condition. The image_pred_class
            # tensors is those row indexes, modified to a [4 x 7] tensor. We may
            # be able to get ride of the view. Candidate for optimization deletion.
            class_mask_ind = torch.where(image_pred_[:, -1] == cls)
            image_pred_class = image_pred_[class_mask_ind].view(-1, 7)

            # Sort the detections such that the entry with the maximum objectness
            # confidence is at the top. sort returns val, index, which is why we
            # want the [1] element.
            conf_sort_index = torch.sort(
                image_pred_class[:, 4], descending=True)[1]

            # This is the sorted tensor, a [4 x 7] tensor
            image_pred_class = image_pred_class[conf_sort_index]

            # Number of detections in the tensor (the number of rows, so 0 index)
            idx = image_pred_class.size(0)

            # Go through each class prediction, and compare the boxes to the IoU's
            for i in range(idx):
                # Get the IOUs of all boxes that come after the one we are looking at
                # in the loop
                try:
                    # Add the sorted tensor's highest value. When you pull that row out, it becomes a 1d tensor.
                    # Need to turn it back into a 2D tensor with the unsqueeze. Then you add all the other
                    # predicted values as the other tensor
                    # Returns the IoU of the object with highest objectness score in relation to
                    # all the other tensors.
                    ious = bbox_iou(image_pred_class[i].unsqueeze(
                        0), image_pred_class[i+1:])

                except ValueError:
                    break

                except IndexError:
                    break

                # Zero out all the detections that have IoU  < nms treshhold
                # Pytorch has no way of selecting row by a value in that
                # row an only returning a tensor with approved rows.
                # This is the long way around.
                # Create a mask, which returns an array of 0's or values
                iou_mask = (ious < nms_conf).float().unsqueeze(1)
                # Return the image_pred_class, which has everything after the
                # 1st index as a comparison in the IoU BBox method earlier,
                # with 0's or values
                image_pred_class[i+1:] *= iou_mask

                # Remove the non-zero entries, returns [nx7] of 4 corner coordinates,
                # objectness score, the score of class with maximum confidence,
                # and the index of that class where n equals IoU's less than
                # nms threshold
                non_zero_ind = torch.nonzero(image_pred_class[:, 4]).squeeze()
                image_pred_class = image_pred_class[non_zero_ind].view(-1, 7)

            # Repeat the batch_id for as many detections of the class cls in the image
            batch_ind = image_pred_class.new(
                image_pred_class.size(0), 1).fill_(ind)
            seq = batch_ind, image_pred_class

            if not write:
                # On first time through, initiaize the output
                output = torch.cat(seq, 1)
                write = True
            else:
                # Every other time through, concatenate the tensors
                out = torch.cat(seq, 1)
                output = torch.cat((output, out))

    try:
        # In the example of the dog-bicycle-truck.png, the return is
        # a [3 x 8] tensor, where the batch number is added to the beginning
        # of the [3 x 7] image_pred_class
        return output
    except:
        return 0


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


def load_classes(namesfile):
    fp = open(namesfile, "r")
    names = fp.read().split("\n")[:-1]
    return names
