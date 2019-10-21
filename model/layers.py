import torch
import torch.nn as nn


class MaxPoolStride1(nn.Module):
    def __init__(self, kernel_size):
        super(MaxPoolStride1, self).__init__()
        self.kernel_size = kernel_size
        self.pad = kernel_size - 1

    def forward(self, x):
        padded_x = F.pad(x, (0, self.pad, 0, self.pad), mode="replicate")
        pooled_x = nn.MaxPool2d(self.kernel_size, self.pad)(padded_x)
        return pooled_x


class EmptyLayer(nn.Module):
    def __init__(self):
        super(EmptyLayer, self).__init__()


class DetectionLayer(nn.Module):
    def __init__(self, anchors):
        super(DetectionLayer, self).__init__()
        self.anchors = anchors

    def forward(self, x, inp_dim, num_classes, confidence):
        x = x.data
        global CUDA
        prediction = x
        prediction = predict_transform(
            prediction, inp_dim, self.anchors, num_classes, confidence, CUDA)
        return prediction


class Upsample(nn.Module):
    def __init__(self, stride=2):
        super(Upsample, self).__init__()
        self.stride = stride

    def forward(self, x):
        stride = self.stride
        assert(x.data.dim() == 4)
        B = x.data.size(0)
        C = x.data.size(1)
        H = x.data.size(2)
        W = x.data.size(3)
        ws = stride
        hs = stride
        x = x.view(B, C, H, 1, W, 1).expand(B, C, H, stride, W,
                                            stride).contiguous().view(B, C, H*stride, W*stride)
        return x
#


class ReOrgLayer(nn.Module):
    def __init__(self, stride=2):
        super(ReOrgLayer, self).__init__()
        self.stride = stride

    def forward(self, x):
        assert(x.data.dim() == 4)
        B, C, H, W = x.data.shape
        hs = self.stride
        ws = self.stride
        assert(H % hs == 0),  "The stride " + str(self.stride) + \
            " is not a proper divisor of height " + str(H)
        assert(W % ws == 0),  "The stride " + str(self.stride) + \
            " is not a proper divisor of height " + str(W)
        x = x.view(B, C, H // hs, hs, W // ws,
                   ws).transpose(-2, -3).contiguous()
        x = x.view(B, C, H // hs * W // ws, hs, ws)
        x = x.view(B, C, H // hs * W // ws, hs *
                   ws).transpose(-1, -2).contiguous()
        x = x.view(B, C, ws*hs, H // ws, W // ws).transpose(1, 2).contiguous()
        x = x.view(B, C*ws*hs, H // ws, W // ws)
        return x
