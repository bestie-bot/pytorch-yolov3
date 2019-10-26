import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from model.layers import *


def parse_cfg(cfgfile):
    """
    Takes a configuration file

    Returns a list of blocks. Each block describes a layer (usually) in the neural
    network to be built. Block is represented as a dictionary in the list

    """
    file = open(cfgfile, 'r')

    # store the lines in a list
    lines = file.read().split('\n')
    # get rid of the empty lines
    lines = [x for x in lines if len(x) > 0]
    # get rid of comments
    lines = [x for x in lines if x[0] != '#']
    # get rid of fringe whitespaces
    lines = [x.rstrip().lstrip() for x in lines]

    # To parse the files, we're going to create a key/value pair, and then
    # store each pair in a Python list
    block = {}
    blocks = []

    # Now that we collapsed the config file, let's go through each line
    for line in lines:
        # This marks the start of a new block based on the CFG filename, i.e. [convolutional]
        if line[0] == "[":
            # If block is not empty, this means we're still building the previous block dict.
            # Since we're in an obviously new block with the "[", the first time through we'll
            # need to send the current block to the list, and empty the current block dictionary
            # so it can receive the next values. This is how we separate dict items for our list.
            if len(block) != 0:
                # Add the current block info to the list
                blocks.append(block)
                # Reset the block
                block = {}
            # Store in a key called 'type' the text of the line minus the brackets,
            # deleting any trailing characters. Since we cleared the block earlier the first time
            # through (denoted by the "["), the first layer of the block will always be a type
            # that is a text value of layer type
            block["type"] = line[1:-1].rstrip()
        else:
            # Since we're somewhere in the block where we're not figuring out layer type,
            # split the line pairs into a tuple of key and value using "=" as the break point
            key, value = line.split("=")

            # Converting the CFG key value pairs into block keys with values
            block[key.rstrip()] = value.lstrip()

    # This is so the last time through the loop the last block gets added
    blocks.append(block)

    return blocks


def create_modules(blocks):
    """
    Takes all of the blocks from parse_config(yolo_cfg_file) and turns them into a proper
    Pytorch sequential model. Need to account for every block type.
    Those are: Convolutional, Upsample, Downsample, Route, Skip, Net, Yolo

    Params:
        List of blocks from Yolo Config

    Returns:
        Tuple-> (Network Info, List of Modules)
    """
    # Captures the information about the network parameters
    net_info = blocks[0]
    # Set up a PyTorch list of modules
    module_list = nn.ModuleList()
    # Initial filters is 3 since we start with 3 channels on the input image
    prev_filters = 3
    # What will store output features for every layer
    output_filters = []

    # indexing blocks helps with implementing route layers (skip connections)
    index = 0

    # Go through each of the items in the block list
    for x in blocks:
        # Create a Neural Network Sequential module
        module = nn.Sequential()

        # If it's the net layer, this layer is for info only, and should be skipped for
        # the purposes of creating the network
        if (x["type"] == "net"):
            continue

        # If it's a convolutional layer
        if (x["type"] == "convolutional"):
            # Get the info about the layer
            activation = x["activation"]
            try:
                # To increase the stability of a neural network, batch normalization normalizes
                # the output of a previous activation layer by subtracting the batch mean and
                # dividing by the batch standard deviation. It helps make sure all values are
                # within a certain level. For our purposes here, bias is already included in
                # batch norm, so we don't want to add any extra in our nn module
                batch_normalize = int(x["batch_normalize"])
                bias = False
            except:
                # Batch norm includes bias, but if there's no batch norm, make sure the bias
                # is added
                batch_normalize = 0
                bias = True

            filters = int(x["filters"])
            padding = int(x["pad"])
            kernel_size = int(x["size"])
            stride = int(x["stride"])

            # Due to the YOLO Cfg setup, we really should turn all the paddings to 0 where
            # they equal 1, i.e, there shouldn't be any padding. We'll leave this in here
            # for now
            if padding:
                # Calculates the zeroes of padding to add around the image. '//' == Rounded floor.
                # So in case of a kernel size of 3, padding is 1
                pad = (kernel_size - 1) // 2
            else:
                pad = 0

            # Add the convolutional layer
            conv = nn.Conv2d(prev_filters, filters,
                             kernel_size, stride, pad, bias=bias)
            module.add_module("conv_{0}".format(index), conv)

            # Add the Batch Norm Layer
            if batch_normalize:
                bn = nn.BatchNorm2d(filters)
                module.add_module("batch_norm_{0}".format(index), bn)

            # Check the activation.
            # It is either Linear or a Leaky ReLU for YOLO
            if activation == "leaky":
                activn = nn.LeakyReLU(0.1, inplace=True)
                module.add_module("leaky_{0}".format(index), activn)

        # If it's an upsampling layer we use Bilinear2dUpsampling
        elif (x["type"] == "upsample"):
            stride = int(x["stride"])
            upsample = nn.Upsample(scale_factor=2, mode="nearest")
            module.add_module("upsample_{}".format(index), upsample)

        # If it is a route layer
        elif (x["type"] == "route"):
            # Comes in a tuple, so we need to split the pair
            x["layers"] = x["layers"].split(',')

            # Start  of a route
            start = int(x["layers"][0])

            # end, if there exists one.
            try:
                end = int(x["layers"][1])
            except:
                end = 0

            # This is pointless since the start is NEVER greater than 0 in
            # the YOLO cg. Candidate for deletion on optimization.
            if start > 0:
                start = start - index

            # If there is a value for end, then we need to know which
            # layer this is supposed to connect to, since otherwise
            # it will just be the current index.
            if end > 0:
                end = end - index

            # We create an empty layer here as a placeholder, and then
            # as we do the forward pass, we'll adjust the code to
            # do bring forward the features from the start to appropriate index
            route = EmptyLayer()
            module.add_module("route_{0}".format(index), route)

            # This is pointless since the end is NEVER less than than 0 in
            # the YOLO cg. Candidate for deletion on optimization.
            if end < 0:
                filters = output_filters[index +
                                         start] + output_filters[index + end]
            else:
                # When you we create the output_filters for this module,
                # which act as the input filters for the next layer, we're
                # essentially adding the output layers from the layer designated
                # by the start variable. Not sure if this is a full bring forward
                # or a concatenation
                filters = output_filters[index + start]

        # Shortcut corresponds to skip connection. In the forward pass, we
        # make sure it brings the feature maps from the previous layer forward
        # via addition.
        elif x["type"] == "shortcut":
            from_ = int(x["from"])
            # Same reason for empty layer as in route type
            shortcut = EmptyLayer()
            module.add_module("shortcut_{}".format(index), shortcut)

        # Yolo is the detection layer
        elif x["type"] == "yolo":
            # Mask is which anchor boxes to use by index from the anchor
            # keys list in the YOLO CFG file. We add them to a list so
            # we can iterate over them.
            mask = x["mask"].split(",")
            mask = [int(x) for x in mask]

            # Split entire list of anchor box tuples
            anchors = x["anchors"].split(",")
            # Convert them to integers
            anchors = [int(a) for a in anchors]
            # We're taking the anchors list and turning it into
            # a set of tupled values, for as long as the list is,
            # in increments of 2 (since these are pairs of coordinates)
            # We can then index pairs of values by a single list index
            anchors = [(anchors[i], anchors[i+1])
                       for i in range(0, len(anchors), 2)]
            # Get the paired tuples that are listed by index in the mask list
            anchors = [anchors[i] for i in mask]

            # This is our detection layer from our detection class in the
            # models.layers
            detection = DetectionLayer(anchors)
            module.add_module("Detection_{}".format(index), detection)

        else:
            print(
                "Something not in our layer list. If you see this, you probably modified the YOLO cfg file. Dont' do that.")
            assert False

        # Append the moduel
        module_list.append(module)
        # Set up the previous filters for the next layer
        prev_filters = filters
        # Set up the output filters for the next layer
        output_filters.append(filters)
        # Move to the next layer
        index += 1

    return (net_info, module_list)
