#Pytorch-TorchVision已实现了ResNet https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
#This is the Jermy_Lu's Version
#原文链接 https://arxiv.org/pdf/1512.03385.pdf

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List

def conv3x3(in_channels: int, out_channels: int, stride=1, padding=1, bias=False):
    """3x3 convolution with padding=1即等长卷积"""
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=padding, bias=bias)

def conv1x1(in_channels: int, out_channels: int, stride=1, bias=False):
    """1x1 convolution"""
    return nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=bias)

# def conv_bn_relu_block(in_channels: int, out_channels: int, kernel_size: int, stride: int,\
#     padding: int, bias=False, activation=True):
#     layers = [
#         nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias),
#         nn.BatchNorm2d(out_channels)
#     ]
#     if activation:#True
#         layers.append(nn.ReLU(inplace=True))
#     return nn.Sequential(*layers)

class BasicBlock(nn.Module):
    """
    basic building block for ResNet-18, ResNet-34
    """
    mul = 1
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride=1,
        padding=1,
        bias=False,
        if_maxpooling=False
    ):
        super(BasicBlock, self).__init__()
        self.conv_1 = conv3x3(in_channels, out_channels)
        self.bn1 = nn.BatchNorm2d(out_channels)
        # self.relu = nn.ReLU(inplace=True)
        self.conv_2 = conv3x3(out_channels, out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        if if_maxpooling:#True
            self.maxpooling = nn.MaxPool2d(kernel_size=2, stride=2)
        else:
            self.maxpooling = None

        self.conv_f = nn.Sequential()
        if in_channels != out_channels:
            self.conv_f = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        #x.shape: (batch_size, in_c, h, w)
        out = self.conv_1(x)#out.shape: (batch_size, out_c, h, w)
        out = self.bn1(out)
        # out = self.relu(out)
        out = F.relu(out)
        out = self.conv_2(out)#out.shape: (batch_size, out_c, h, w)
        out = self.bn2(out)
        #当in_c与out_c不相等时，需要用Conv2d将其相等化再相加
        out += self.conv_f(x)
        # out = self.relu(out)
        out = F.relu(out)
        if self.maxpooling:#not None
            out = self.maxpooling(out)
        return out


class Bottleneck(nn.Module):
    mul = 4
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride=1,
        padding=1,
        bias=False,
        if_maxpooling=False
    ):
        super(Bottleneck, self).__init__()
        self.conv1 = conv1x1(in_channels, out_channels)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = conv3x3(out_channels, out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = conv1x1(out_channels, 4*out_channels)
        self.bn3 = nn.BatchNorm2d(4*out_channels)
        
        if if_maxpooling:#True
            self.maxpooling = nn.MaxPool2d(kernel_size=2, stride=2)
        else:
            self.maxpooling = None

        self.conv_f = nn.Sequential()
        if in_channels != 4*out_channels:
            self.conv_f = nn.Sequential(
                nn.Conv2d(in_channels, 4*out_channels, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(4*out_channels)
            )

    def forward(self, x):
        #x.shape: (batch_size, in_c, h, w)
        out = self.conv1(x)#out.shape: (batch_size, out_c, h, w)
        out = self.bn1(out)
        out = F.relu(out)
        out = self.conv2(out)#out.shape: (batch_size, out_c, h, w)
        outx = self.bn2(out)
        out = F.relu(out)
        out = self.conv3(out)#out.shape: (batch_size, 4*out_C, h, w)
        out = self.bn3(out)
        #当In_c与4*out_c不相等时，需要先将Conv2d将其相等化再相加
        # x = self.conv_f(x)
        # out += x
        out += self.conv_f(x)
        out = F.relu(out)
        if self.maxpooling:#not None
            out = self.maxpooling(out)
        return out

# bottle = BasicBlock(3, 64)
# input = torch.randn(32, 3, 100, 100)
# output = bottle(input)
# print(output.size())#torch.Size([32, 64, 100, 100])

# bottle = Bottleneck(64, 64)
# input = torch.randn(32, 64, 100, 100)
# output = bottle(input)
# print(output.size())#torch.Size([32, 256, 100, 100])

def build_conv_layer(in_channels: int ,out_channels: int, block: nn.Module, num_blocks: int):
        #if_maxpooling_list = [False]*(num_blocks-1) + [True]
        if_maxpooling_list = [True] + [False]*(num_blocks-1)
        layers = []
        for flag in if_maxpooling_list:
            layers.append(block(in_channels, out_channels, if_maxpooling=flag))
            in_channels = out_channels * block.mul#更新in_channels
        return nn.Sequential(*layers)

# test_basicBlock = build_conv_layer(3, 64, BasicBlock, num_blocks=2)
# print(test_basicBlock)
# input = torch.randn(32, 3, 112, 112)
# output = test_basicBlock(input)
# print(output.size())#torch.Size([32, 64, 56, 56])

# test_bottleNeck = build_conv_layer(3, 64, Bottleneck, num_blocks=3)
# print(test_bottleNeck)
# input = torch.randn(32, 3, 112, 112)
# output = test_bottleNeck(input)
# print(output.size())#torch.Size([32, 256, 56, 56])

class ResNet(nn.Module):
    def __init__(self, block: nn.Module, num_layers_list: List, num_classes=1000):
        super(ResNet, self).__init__()
        self.in_channels = 64
        #论文中为7x7的卷积核，stride为2
        #为了方便起见，使用3x3的卷积核，stride为1
        #以下为第1个卷积层
        #注：以下注释部分输入输出大小均基于ResNet18和ResNet34
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)#input.shape: (3, 224, 224); output.shape: (64, 224, 224)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpooling = nn.MaxPool2d(kernel_size=2, stride=2)#1/2池化; (64, 224, 224)-->(64, 112, 112)
        
        #以下为第2~5个卷积层
        self.conv_layer2 = build_conv_layer(in_channels=64, out_channels=64, block=block, num_blocks=num_layers_list[0])#(64, 112, 112)-->(64, 56, 56)
        self.conv_layer3 = build_conv_layer(in_channels=64*block.mul, out_channels=128, block=block, num_blocks=num_layers_list[1])#(64, 56, 56)-->(128, 28, 28)
        self.conv_layer4 = build_conv_layer(in_channels=128*block.mul, out_channels=256, block=block, num_blocks=num_layers_list[2])#(128, 28, 28)-->(256, 14, 14)
        self.conv_layer5 = build_conv_layer(in_channels=256*block.mul, out_channels=512, block=block, num_blocks=num_layers_list[3])#(256, 14, 14)-->(512, 7, 7)
        self.fc = nn.Linear(512*block.mul, num_classes)

    def forward(self, x):
        #x.shape: (batch_size, 3, 224, 224)
        out = self.conv1(x)#out.shape: (batch_size, 64, 224, 224)
        out = self.bn1(out)
        out = self.maxpooling(out)#out: (batch_size, 64, 112, 112)
        out = self.conv_layer2(out)
        out = self.conv_layer3(out)
        out = self.conv_layer4(out)
        out = self.conv_layer5(out)
        out = F.adaptive_avg_pool2d(out, output_size=(1, 1))#out: (batch_size, *, 1, 1)
        #out = F.avg_pool2d(7, 7)#kernel_size=(7, 7), stride=(7, 7)
        out = out.view(out.size()[0], -1)#out: (batch_size, *)
        #out = torch.squeeze(out)#out: (batch_size, *)
        out = self.fc(out)#out: (batch_size, 1000)
        out_p = F.softmax(out, dim=1)
        return out_p


def ResNet_builder(num_layers: int):
    num_layers_dict = {
        18: (BasicBlock, [2, 2, 2, 2]),
        34: (BasicBlock, [3, 4, 6, 3]),
        50: (Bottleneck, [3, 4, 6, 3]),
        101: (Bottleneck, [3, 4, 23, 3]),
        152: (Bottleneck, [3, 8, 36, 3])
    }
    if num_layers not in num_layers_dict:
        raise KeyError
    block, num_layers_list = num_layers_dict[num_layers]
    resnet_model = ResNet(block=block, num_layers_list=num_layers_list)
    return resnet_model

# testing
# for num in [18, 34, 50, 101, 152]:
#     resnet_test = ResNet_builder(num_layers=num)
#     input = torch.randn(32, 3, 224, 224)
#     output = resnet_test(input)
#     print("ResNet_%d:" % num, output.size())
# ResNet_18: torch.Size([32, 1000])
# ResNet_34: torch.Size([32, 1000])
# ResNet_50: torch.Size([32, 1000])
# ResNet_101: torch.Size([32, 1000])
# ResNet_152: torch.Size([32, 1000])
