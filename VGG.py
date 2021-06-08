#https://arxiv.org/pdf/1409.1556.pdf
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List

#VGG中卷积为等长卷积，卷积核大小为3
def base_vgg_block(in_channels, out_channels, conv_nums):
    """
    Args：
        in_channels: 输入通道数
        out_channels: 输出通道数
        conv_nums: 等长卷积的数量
    """
    net = [nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(num_features=out_channels), nn.ReLU(inplace=True)]
    for i in range(conv_nums):
        net.append(nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1))
        net.append(nn.BatchNorm2d(num_features=out_channels))
        net.append(nn.ReLU(inplace=True))

    net.append(nn.MaxPool2d(kernel_size=2, stride=2))
    return nn.Sequential(*net)

# #testing
# vgg_b = base_vgg_block(64, 128, 3)
# print(vgg_b)
# input = torch.randn(32, 64, 112, 112)
# output = vgg_b(input)
# print(output.size())#池化之前：torch.Size([32, 128, 112, 112]); 池化之后: torch.Size([32, 128, 56, 56])

class BaseVGG(nn.Module):
    def __init__(self, conv_nums_list: List, num_classes=1000):
        super(BaseVGG, self).__init__()
        self.conv3_64 = base_vgg_block(3, 64, conv_nums=conv_nums_list[0])
        self.conv64_128 = base_vgg_block(64, 128, conv_nums=conv_nums_list[1])
        self.conv128_256 = base_vgg_block(128, 256, conv_nums=conv_nums_list[2])
        self.conv256_512 = base_vgg_block(256, 512, conv_nums=conv_nums_list[3])
        self.conv512_512 = base_vgg_block(512, 512, conv_nums=conv_nums_list[4])#img_output: 7*7*512
        self.fc_1 = nn.Linear(7*7*512, 4096)
        self.bn_1 = nn.BatchNorm1d(4096)#每个全连接层后接一个BN
        self.fc_2 = nn.Linear(4096, 4096)
        self.bn_2 = nn.BatchNorm1d(4096)
        self.output = nn.Linear(4096, num_classes)

    def forward(self, x):
        #建议x.shape: (batch_size, 3, 224, 224)
        x = self.conv3_64(x)#x.shape: (batch_size, 64, 112, 112)
        x = self.conv64_128(x)#x.shape: (batch_size, 128, 56, 56)
        x = self.conv128_256(x)#x.shape: (batch_size, 256, 28, 28)
        x = self.conv256_512(x)#x.shape: (batch_size, 512, 14, 14)
        x = self.conv512_512(x)#x.shape: (batch_size, 512, 7, 7)
        x = x.view(x.size()[0], -1)#x.shape: (batch_size, 7*7*512)
        x = self.fc_1(x)#x.shape: (batch_size, 4096)
        x = self.bn_1(x)
        x = self.fc_2(x)#x.shape: (batch_size, 4096)
        x = self.bn_2(x)
        x = F.relu(x)#使用RELU函数增加非线性能力
        output = self.output(x)#output.shape: (batch_size, num_classes)
        output_p = F.softmax(output, dim=1)
        return output_p

      
def VGG_bulilder(num_layers):
    num_layers_dict = {
        11: [0, 0, 1, 1, 1],
        13: [1, 1, 1, 1, 1],
        16: [1, 1, 2, 2, 2],
        19: [1, 1, 3, 3, 3]
    }
    if num_layers not in num_layers_dict:
        raise KeyError
    num_layers_list = num_layers_dict[num_layers]
    vgg_model = BaseVGG(conv_nums_list=num_layers_list)
    return vgg_model

vgg_19 = VGG_bulilder(num_layers=19)
print(vgg_19)
input = torch.randn(32, 3, 224, 224)#32: batch_size, 3: channels, 224*224:width*height
output_p = vgg_19(input)
print(output_p.size())
