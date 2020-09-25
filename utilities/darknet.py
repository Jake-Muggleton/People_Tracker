from __future__ import division
from utilities.util import *

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np


class EmptyLayer(nn.Module):
    """
    Empty layers are useful for when a layer's (eg route) operation 
    is so simple we can implement it straight in the main network
    """

    def __init__(self):
        super(EmptyLayer, self).__init__()


class DetectionLayer(nn.Module):
    """
    Like the EmptyLayer, don't worry about defining the forward function here
    """

    def __init__(self, anchors):
        super(DetectionLayer, self).__init__()
        self.anchors = anchors


def get_test_input():
    img = cv2.imread("dog-cycle-car.png")
    img = cv2.resize(img, (416, 416))  # Resize to the input dimension
    # BGR -> RGB | H X W C -> C X H X W
    img_ = img[:, :, ::-1].transpose((2, 0, 1))
    # Add a channel at 0 (for batch) | Normalise
    img_ = img_[np.newaxis, :, :, :]/255.0
    img_ = torch.from_numpy(img_).float()  # Convert to float
    img_ = Variable(img_)                     # Convert to Variable
    return img_


def parse_cfg(cfgfile):
    """
    Takes config file and returns a list of blocks. 
    Each block describes a block in the neural network to be built.
    Block is represented as a dictionary in the list.
    """

    file = open(cfgfile, 'r')
    lines = file.read().split('\n')  # returns a list of lines
    lines = [x for x in lines if len(x) > 0]  # remove empty lines
    lines = [x for x in lines if x[0] != '#']  # remove comments
    # remove leading and trailing whitespace
    lines = [x.rstrip().lstrip() for x in lines]

    block = {}  # dict
    blocks = []  # list

    for line in lines:
        if line[0] == "[":  # start of new block
            if len(block) != 0:  # if block is not empty, we are still storing prev block
                blocks.append(block)  # add prev block to list
                block = {}  # reinint block
            block["type"] = line[1:-1].rstrip()
        else:
            key, value = line.split("=")
            block[key.rstrip()] = value.lstrip()
    blocks.append(block)

    return blocks


class MaxPoolStride1(nn.Module):
    def __init__(self, kernel_size):
        super(MaxPoolStride1, self).__init__()
        self.kernel_size = kernel_size
        self.pad = kernel_size - 1

    def forward(self, x):
        padded_x = F.pad(x, (0, self.pad, 0, self.pad),  mode="replicate")
        pooled_x = nn.MaxPool2d(self.kernel_size, self.pad)(padded_x)
        return pooled_x


def create_modules(blocks):
    net_info = blocks[0]
    # stores a list of nn.Module objects, like a normal list but gives the parameters of each item for our main network
    module_list = nn.ModuleList()
    prev_filters = 3  # depth of kernal = prevous layer's filters, initially 3 for RGB img
    output_filters = []  # keeps track of the depth of each layer

    for index, x in enumerate(blocks[1:]):  # skip net block,
        # sequential() is used to sequentially execute multiple modules, used for blocks that contain multi-layers
        module = nn.Sequential()

        if(x["type"] == "convolutional"):
            # get layer info
            activation = x["activation"]

            try:
                batch_normalize = int(x["batch_normalize"])
                bias = False
            except Exception:
                batch_normalize = 0
                bias = True

            filters = int(x["filters"])
            padding = int(x["pad"])
            kernel_size = int(x["size"])
            stride = int(x["stride"])

            if padding:
                pad = (kernel_size - 1) // 2
            else:
                pad = 0

            # add conv later
            conv = nn.Conv2d(
                prev_filters,
                filters,
                kernel_size,
                stride,
                pad,
                bias=bias
            )
            # add_module(name, module)
            module.add_module("conv_{0}".format(index), conv)

            # Add Batch Norm Layer
            if batch_normalize:
                bn = nn.BatchNorm2d(filters)
                module.add_module("batch_norm_{0}".format(index), bn)

            # Add activation (either leaky ReLU or just straight up linear for YOLO)
            if activation == "leaky":
                act = nn.LeakyReLU(0.1, inplace=True)
                module.add_module("leaky_{0}".format(index), act)

        elif (x["type"] == "upsample"):
            stride = int(x["stride"])
            upsample = nn.Upsample(scale_factor=2, mode="nearest")
            module.add_module("upsample_{0}".format(index), upsample)

        elif (x["type"] == "route"):
            x["layers"] = x["layers"].split(',')

            start = int(x["layers"][0])  # start of route (concat source)
            try:  # end of route (if there is one)
                end = int(x["layers"][1])
            except Exception:
                end = 0

            if start > 0:
                start = start - index
            if end > 0:
                end = end - index

            route = EmptyLayer()
            module.add_module("route_{0}".format(index), route)

            # output size of layer after concatinations
            if end < 0:
                filters = output_filters[index +
                                         start] + output_filters[index + end]
            else:
                filters = output_filters[index+start]

        elif (x["type"] == "shortcut"):  # skip connection
            shortcut = EmptyLayer()
            module.add_module("shortcut_{0}".format(index), shortcut)

        elif x["type"] == "maxpool":
            stride = int(x["stride"])
            size = int(x["size"])
            if stride != 1:
                maxpool = nn.MaxPool2d(size, stride)
            else:
                maxpool = MaxPoolStride1(size)

            module.add_module("maxpool_{}".format(index), maxpool)

        elif (x["type"] == "yolo"):  # detection layer
            mask = x["mask"].split(',')
            mask = [int(x) for x in mask]

            anchors = x["anchors"].split(',')
            anchors = [int(a) for a in anchors]  # cast to int
            anchors = [(anchors[i], anchors[i+1])
                       for i in range(0, len(anchors), 2)]  # get anchor pairs
            anchors = [anchors[i] for i in mask]

            detection = DetectionLayer(anchors)
            module.add_module("Detection_{0}".format(index), detection)

        else:
            assert False, "Unknown layer type in cfg file"

        module_list.append(module)
        prev_filters = filters
        output_filters.append(filters)

    return(net_info, module_list)


class Darknet(nn.Module):
    """main module"""

    def __init__(self, cfgfile):
        super(Darknet, self).__init__()
        self.blocks = parse_cfg(cfgfile)
        self.net_info, self.module_list = create_modules(self.blocks)

    def load_weights(self, weightfile):
        fp = open(weightfile, 'rb')

        # first 5 values are header info
        # 1. Major Version Num
        # 2. Minor Version Num
        # 3. Sub Version Num
        # 4,5. Images seen by the network during training

        # read 5 int32 values
        header = np.fromfile(fp, dtype=np.int32, count=5)
        self.header = torch.from_numpy(header)
        self.seen = self.header[3]

        # the rest of the file is weights
        weights = np.fromfile(fp, dtype=np.float32)

        ptr = 0
        for i in range(len(self.module_list)):
            module_type = self.blocks[i+1]["type"]

            # if module is convolutional load weights, otherwise ignore

            if module_type == "convolutional":
                model = self.module_list[i]
                try:
                    batch_normalize = int(self.blocks[i+1]["batch_normalize"])
                except Exception:
                    batch_normalize = 0

                # index operator gets a particular layer in a module
                conv = model[0]

                if(batch_normalize):
                    bn = model[1]
                    num_bn_biases = bn.bias.numel()  # get number of weights in bn layer

                    # load weights
                    bn_biases = torch.from_numpy(
                        weights[ptr:ptr + num_bn_biases])
                    ptr += num_bn_biases

                    bn_weights = torch.from_numpy(
                        weights[ptr:ptr + num_bn_biases])
                    ptr += num_bn_biases

                    bn_running_mean = torch.from_numpy(
                        weights[ptr:ptr + num_bn_biases])
                    ptr += num_bn_biases

                    bn_running_var = torch.from_numpy(
                        weights[ptr:ptr + num_bn_biases])
                    ptr += num_bn_biases

                    # reshape loaded weights for model weights
                    bn_biases = bn_biases.view_as(bn.bias.data)
                    bn_weights = bn_weights.view_as(bn.weight.data)
                    bn_running_mean = bn_running_mean.view_as(bn.running_mean)
                    bn_running_var = bn_running_var.view_as(bn.running_var)

                    # copy data to model
                    bn.bias.data.copy_(bn_biases)
                    bn.weight.data.copy_(bn_weights)
                    bn.running_mean.copy_(bn_running_mean)
                    bn.running_var.copy_(bn_running_var)

                else:  # if no batch norm just load biases for conv layer
                    num_biases = conv.bias.numel()  # number of elements in biases

                    conv_biases = torch.from_numpy(
                        weights[ptr: ptr + num_biases])
                    ptr += num_biases

                    conv_biases = conv_biases.view_as(conv.bias.data)

                    conv.bias.data.copy_(conv_biases)

                # load conv layer's weights
                num_weights = conv.weight.numel()

                conv_weights = torch.from_numpy(
                    weights[ptr: ptr + num_weights])
                ptr = ptr + num_weights

                conv_weights = conv_weights.view_as(conv.weight.data)
                conv.weight.data.copy_(conv_weights)

    def forward(self, x, CUDA):
        modules = self.blocks[1:]  # first element of blocks is net
        outputs = {}  # store outputs for route layers

        write = 0
        for i, module in enumerate(modules):
            module_type = (module["type"])

            if (module_type == "convolutional" or module_type == "upsample" or module_type == "maxpool"):
                x = self.module_list[i](x)

            elif (module_type == "route"):
                layers = module["layers"]
                layers = [int(a) for a in layers]  # cast whole list to ints

                if (layers[0] > 0):
                    layers[0] = layers[0] - i  # i is the current layer

                if len(layers) == 1:
                    x = outputs[i + layers[0]]

                else:
                    if (layers[1] > 0):
                        layers[1] = layers[1] - i

                    map1 = outputs[i+layers[0]]
                    map2 = outputs[i + layers[1]]

                    x = torch.cat((map1, map2), 1)  # concatenate along depth

            elif (module_type == "shortcut"):
                from_ = int(module["from"])
                x = outputs[i-1] + outputs[i+from_]

            elif (module_type == "yolo"):
                anchors = self.module_list[i][0].anchors
                inp_dim = int(self.net_info["height"])
                num_classes = int(module["classes"])

                # transform
                x = x.data
                x = predict_transform(x, inp_dim, anchors, num_classes, CUDA)
                if not write:  # if detection collecter is not yet initialised
                    detections = x
                    write = 1
                else:
                    detections = torch.cat((detections, x), 1)

            outputs[i] = x

        return (detections)
