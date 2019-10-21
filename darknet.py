import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from model.layers import *
from model.build import *
import cv2
from model.utils import *


def get_test_input():
    img = cv2.imread("images/dog-cycle-car.png")
    img = cv2.resize(img, (416, 416))  # Resize to the input dimension
    # BGR -> RGB | H X W C -> C X H X W
    img_ = img[:, :, ::-1].transpose((2, 0, 1))
    # Add a channel at 0 (for batch) | Normalise
    img_ = img_[np.newaxis, :, :, :]/255.0
    img_ = torch.from_numpy(img_).float()  # Convert to float
    img_ = Variable(img_)                     # Convert to Variable
    return img_


class Darknet(nn.Module):
    """
    Main Darknet class. It is a subclass of nn.Module
    """

    def __init__(self, cfgfile):
        super(Darknet, self).__init__()
        # Translate our YOLOv3 CFG file to blocks
        self.blocks = parse_cfg(cfgfile)
        # Convert those blocks to a module list for Pytorch
        self.net_info, self.module_list = create_modules(self.blocks)
        # self.header = torch.IntTensor([0, 0, 0, 0])
        # self.seen = 0

    def get_blocks(self):
        """
        Getter function for blocks

        Returns:
            blocks
        """
        return self.blocks

    def get_module_list(self):
        """
        Getter function for module_list

        Returns:
            module_list
        """
        return self.module_list

    # Main forward pass
    def forward(self, x, CUDA):
        """
        Does the forward pass

        Params:
            x: The input
            CUDA: Use GPU to accelerate task
        """
        detections = []
        # We don't want the first block, that contains the network info
        modules = self.blocks[1:]
        # We cache the output feature maps of every layer in a dict outputs.
        # The keys are the the indices of the layers, and the values are
        # the feature maps. We can then search through the keys to look up
        # a layers feature maps for route or shortcuts.
        outputs = {}

        write = 0
        # Go through every module (layer)
        for i in range(len(modules)):
            # Get the module type value from the current index
            module_type = (modules[i]["type"])
            if module_type == "convolutional" or module_type == "upsample" or module_type == "maxpool":
                # Not 100% sure, but I think because the module list is a
                # Pytorch nn.ModuleList(), you can multiply the index of this list,
                # that is, the block, by the inputs to this function (x), to get the output.
                # I believe this is the matrix multiplication part.
                x = self.module_list[i](x)
                # Set the key to the index, and set the value to the computed
                # calculation of the block and the input
                outputs[i] = x

            elif module_type == "route":
                layers = modules[i]["layers"]
                # The two layers designated in the layer get turned into a list with indexes
                # of 0 and 1
                layers = [int(a) for a in layers]

                # Route layers[0] is never greater than 0, so candidate for optimization deletion
                if (layers[0]) > 0:
                    layers[0] = layers[0] - i

                # This happens only on the 2 smaller detection laters, i.e. on a 416x416 image,
                # the 13x13 and 26x26 detection region levels
                if len(layers) == 1:
                    # Grab the out put from the index plus the first value, usually
                    # a -4 in this situation. This is what allows a kind of independent route
                    # for the detection region layers. This will then go back and take the layer
                    # where the split happen, pull those weights forward past the detection
                    # layer, and prepare them as a piece of input for the next convolution.
                    x = outputs[i + (layers[0])]

                else:
                    # These are the two large skip connections, from layers 37 -> 99 and 62 -> 87
                    if (layers[1]) > 0:
                        # Reset layer 1 to the difference between the desired layer index
                        # and the current layer. So, from 37 - 99 = (-62). We then add
                        # it to the current layer below in map2
                        layers[1] = layers[1] - i

                    # map1 is the output of the previous layer (layers[0] is always a
                    # negative number), here an upsample layer in the YOLO Cfg
                    map1 = outputs[i + layers[0]]
                    # map2 is the previous convolution to pull the data from
                    map2 = outputs[i + layers[1]]

                    # We're adding together the values of the outputs from the routed layers
                    # along the depth of the tensor since the param of 1 corresponds to
                    # the depth dimension. `Cat` method stands for concatenate.
                    x = torch.cat((map1, map2), 1)

                # Set the key to the current module index, and set the dict value to the computed
                # calculation of the block x variable
                outputs[i] = x

            elif module_type == "shortcut":
                from_ = int(modules[i]["from"])
                # Grab the output from the previous layer, as well as the `from` layer (which
                # is always -3) before. This is either a downsampling, upsampling or shortcut
                # connection.This simply adds the weights together without the tensor
                # concatenation you find in the routings. The is what creates the residual
                # blocks throughout the YOLO network
                # x = outputs[i-1] + outputs[i+from_]
                x = outputs[i-1] + outputs[i+from_]

                # Set the key to the current module index, and value to x variable calculation
                outputs[i] = x

            elif module_type == 'yolo':
                # Get the anchor list
                anchors = self.module_list[i][0].anchors
                # Get the input dimensions
                inp_dim = int(self.net_info["height"])
                # Get the number of classes
                num_classes = int(modules[i]["classes"])
                # Output the result
                x = x.data
                # Run a prediction on a particular region size
                x = predict_transform(x, inp_dim, anchors, num_classes, CUDA)

                if type(x) == int:
                    continue

                # If write = 0, that means this is the first detection
                if not write:
                    detections = x
                    write = 1
                # Otherise, concatenate the different predictions together along the
                # depth of the tensor
                else:
                    detections = torch.cat((detections, x), 1)

                # Since this is a detection layer, we still need to pull the weights from the previous layer
                # output, so that we can use it as input to the next later
                outputs[i] = outputs[i-1]

        try:
            # After all the modules have been gone through, return the detections tensor, which is a
            # combined tensor for all three region size
            return detections
        except:
            return 0

    def load_weights(self, weightfile):
        # Open the weights file
        fp = open(weightfile, "rb")

        # The first 4 values are header information
        # 1. Major version number
        # 2. Minor Version Number
        # 3. Subversion number
        # 4. Images seen
        header = np.fromfile(fp, dtype=np.int32, count=5)
        print(f"Header: {header}")
        self.header = torch.from_numpy(header)
        self.seen = self.header[3]

        # The rest of the values are the weights
        # Let's load them up
        weights = np.fromfile(fp, dtype=np.float32)

        ptr = 0
        for i in range(len(self.module_list)):
            module_type = self.blocks[i + 1]["type"]

            if module_type == "convolutional":
                model = self.module_list[i]
                try:
                    batch_normalize = int(self.blocks[i+1]["batch_normalize"])
                except:
                    batch_normalize = 0

                conv = model[0]

                if (batch_normalize):
                    bn = model[1]

                    # Get the number of weights of Batch Norm Layer
                    num_bn_biases = bn.bias.numel()

                    # Load the weights
                    bn_biases = torch.from_numpy(
                        weights[ptr:ptr + num_bn_biases])
                    ptr += num_bn_biases

                    bn_weights = torch.from_numpy(
                        weights[ptr: ptr + num_bn_biases])
                    ptr += num_bn_biases

                    bn_running_mean = torch.from_numpy(
                        weights[ptr: ptr + num_bn_biases])
                    ptr += num_bn_biases

                    bn_running_var = torch.from_numpy(
                        weights[ptr: ptr + num_bn_biases])
                    ptr += num_bn_biases

                    # Cast the loaded weights into dims of model weights.
                    bn_biases = bn_biases.view_as(bn.bias.data)
                    bn_weights = bn_weights.view_as(bn.weight.data)
                    bn_running_mean = bn_running_mean.view_as(bn.running_mean)
                    bn_running_var = bn_running_var.view_as(bn.running_var)

                    # Copy the data to model
                    bn.bias.data.copy_(bn_biases)
                    bn.weight.data.copy_(bn_weights)
                    bn.running_mean.copy_(bn_running_mean)
                    bn.running_var.copy_(bn_running_var)

                else:
                    # Number of biases
                    num_biases = conv.bias.numel()

                    # Load the weights
                    conv_biases = torch.from_numpy(
                        weights[ptr: ptr + num_biases])
                    ptr = ptr + num_biases

                    # reshape the loaded weights according to the dims of the model weights
                    conv_biases = conv_biases.view_as(conv.bias.data)

                    # Finally copy the data
                    conv.bias.data.copy_(conv_biases)

                # Let us load the weights for the Convolutional layers
                num_weights = conv.weight.numel()

                # Do the same as above for weights
                conv_weights = torch.from_numpy(weights[ptr:ptr+num_weights])
                ptr = ptr + num_weights

                conv_weights = conv_weights.view_as(conv.weight.data)
                conv.weight.data.copy_(conv_weights)

    def save_weights(self, savedfile, cutoff=0):
        if cutoff <= 0:
            cutoff = len(self.blocks) - 1

        fp = open(savedfile, 'wb')

        # Attach the header at the top of the file
        self.header[3] = self.seen
        header = self.header

        header = header.numpy()
        header.tofile(fp)

        # Now, let us save the weights
        for i in range(len(self.module_list)):
            module_type = self.blocks[i+1]["type"]

            if (module_type) == "convolutional":
                model = self.module_list[i]
                try:
                    batch_normalize = int(self.blocks[i+1]["batch_normalize"])
                except:
                    batch_normalize = 0

                conv = model[0]

                if (batch_normalize):
                    bn = model[1]

                    # If the parameters are on GPU, convert them back to CPU
                    # We don't convert the parameter to GPU
                    # Instead. we copy the parameter and then convert it to CPU
                    # This is done as weight are need to be saved during training
                    cpu(bn.bias.data).numpy().tofile(fp)
                    cpu(bn.weight.data).numpy().tofile(fp)
                    cpu(bn.running_mean).numpy().tofile(fp)
                    cpu(bn.running_var).numpy().tofile(fp)

                else:
                    cpu(conv.bias.data).numpy().tofile(fp)

                # Let us save the weights for the Convolutional layers
                cpu(conv.weight.data).numpy().tofile(fp)


model = Darknet("cfg/yolov3.cfg")
model.load_weights("yolov3.weights")
inp = get_test_input()
pred = model(inp, torch.cuda.is_available())


print(f"pred {pred}, {pred.shape}")
