#Description of base idea: Simulate the way humans look at the images:
#   combining the overall perspective and the local perspective.
#Base idea_1: using convolution and self-attention for image detection; 
#   using the convolution extracts the local feature;
#   using the self-attention extracts the entire feature
#Base idea_2: avgpooling should be used before appling self-attention, in order to save memory. 

#Note: 假设输入为224*224，批处理进行query*key时，[batch_size, 3,  224*224, 1] * [batch_size, 3, 1, 224*224]
#   那么，结果矩阵大小为[batch_size, 3, 224*224, 224*224]，则计算机应分配batch_size*3*224*224*224*224个单精度浮点数的内存大小。
#   一个单精度浮点数占4Byte，因此当batch_size为1时，参数占据的内存已有28GB左右。
#Note2: 为了节省内存空间，在对输入进行self-attention之前，先进行resize或进行avgpooling等操作。
#   例如，当输入由224*224-->112*112时，当batch_size为1时，参数个数有1*3*112*112*112*112个，其占据内存大小为1.76GB左右，小了16倍。
#   要是输入为224*224-->56*56，则占据内存大小仅为0.11GB即100MB左右。

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
from typing import List

#VGG中卷积为等长卷积，卷积核大小为3
def base_vgg_block(in_channels, out_channels, conv_nums, if_maxpooling=True):
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

    if if_maxpooling:#True
        net.append(nn.MaxPool2d(kernel_size=2, stride=2))
    return nn.Sequential(*net)


class Config(object):
    query_dim = 1
    key_dim = 1
    value_dim = 1
    num_classes = 10


class VGG_SelfAttention(nn.Module):
    def __init__(self, config: Config, conv_nums_list: List):
        super(VGG_SelfAttention, self).__init__()
        #Convolution: extract the local features
        self.conv3_64 = base_vgg_block(3, 64, conv_nums=conv_nums_list[0])
        self.conv64_128 = base_vgg_block(64, 128, conv_nums=conv_nums_list[1])
        self.conv128_256 = base_vgg_block(128, 256, conv_nums=conv_nums_list[2])
        self.conv256_512 = base_vgg_block(256, 512, conv_nums=conv_nums_list[3])
        self.conv512_512 = base_vgg_block(512, 512, conv_nums=conv_nums_list[4])#img_output: 7*7*512
        
        #Self-Attention Mechanism: extract the global features
        #a pixel is a basic element
        self.query = nn.Linear(1, config.query_dim)
        self.key = nn.Linear(1, config.key_dim)
        self.value = nn.Linear(1, config.value_dim)

        #Full-connection layer
        self.fc_1 = nn.Linear(7*7*512+3*56*56, 4096)
        self.bn_1 = nn.BatchNorm1d(4096)#每个全连接层后接一个BN
        self.fc_2 = nn.Linear(4096, 4096)
        self.bn_2 = nn.BatchNorm1d(4096)
        self.output = nn.Linear(4096, config.num_classes)

    def forward(self, x):
        #x.shape: (batch_size, 3, 224, 224)
        #Convolution: extract the local feature
        conv_x = self.conv3_64(x)#conv_x.shape: (batch_size, 64, 112, 112)
        conv_x = self.conv64_128(conv_x)#conv_x.shape: (batch_size, 128, 56, 56)
        conv_x = self.conv128_256(conv_x)#conv_x.shape: (batch_size, 256, 24, 24)
        conv_x = self.conv256_512(conv_x)#conv_x.shape: (batch_size, 512, 14, 14)
        conv_x = self.conv512_512(conv_x)#conv_x.shape: (batch_size, 512, 7, 7)
        conv_x = conv_x.view(conv_x.size()[0], -1)#conv_x.shape: (batch_size, 7*7*512)

        #Self-Attention Mechanism: extract the global features
        #a pixel is a basic element
        #为了节省内存资源，先resize
        avg_x = F.adaptive_avg_pool2d(x, output_size=(56, 56))#x.shape: (batch_size, 3, 56, 56)
        avg_x = torch.reshape(avg_x, (avg_x.size()[0], avg_x.size()[1], -1, 1))#x.shape: (batch_size, 3, 56*56, 1)
        query_x = self.query(avg_x)#query_x.shape: (batch_size, 3, 56*56, 1)
        key_x = self.query(avg_x)#key_x.shape: (batch_size, 3, 56*56, 1)
        value_x = self.query(avg_x)#value_X.shape: (batch_size, 3, 56*56, 1)
        attention_scores = torch.matmul(query_x, key_x.transpose(2, 3))#(batch_size, 3, 56*56, 56*56)
        attention_p = F.softmax(attention_scores, dim=-1)
        value_x = torch.matmul(attention_p, value_x)#(batch_size, 3, 56*56, 1)
        value_x = value_x.view(value_x.size()[0], -1)#(batch_size, 3*56*56)

        #Put them together
        final_x = torch.cat((conv_x, value_x), 1)#(batch_size, 7*7*512+3*56*56)

        #Full connection
        out = self.fc_1(final_x)#(batch_size, 4096)
        out = self.bn_1(out)
        out = self.fc_2(out)#(batch_size, 4096)
        out = self.bn_2(out)
        out = F.relu(out)
        out = self.output(out)#(batch_size, num_classes)
        out_p = F.softmax(out, dim=1)
        return out_p, final_x


#Testing
config = Config()
vgg16 = [1, 1, 2, 2, 2]
model = VGG_SelfAttention(config=config, conv_nums_list=vgg16)
# print(model)
# VGG_SelfAttention(
#   (conv3_64): Sequential(
#     (0): Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#     (1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#     (2): ReLU(inplace=True)
#     (3): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#     (4): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#     (5): ReLU(inplace=True)
#     (6): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
#   )
#   (conv64_128): Sequential(
#     (0): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#     (1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#     (2): ReLU(inplace=True)
#     (3): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#     (4): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#     (5): ReLU(inplace=True)
#     (6): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
#   )
#   (conv128_256): Sequential(
#     (0): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#     (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#     (2): ReLU(inplace=True)
#     (3): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#     (4): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#     (5): ReLU(inplace=True)
#     (6): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#     (7): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#     (8): ReLU(inplace=True)
#     (9): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
#   )
#   (conv256_512): Sequential(
#     (0): Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#     (1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#     (2): ReLU(inplace=True)
#     (3): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#     (4): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#     (5): ReLU(inplace=True)
#     (6): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#     (7): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#     (8): ReLU(inplace=True)
#     (9): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
#   )
#   (conv512_512): Sequential(
#     (0): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#     (1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#     (2): ReLU(inplace=True)
#     (3): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#     (4): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#     (5): ReLU(inplace=True)
#     (6): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#     (7): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#     (8): ReLU(inplace=True)
#     (9): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
#   )
#   (query): Linear(in_features=1, out_features=1, bias=True)
#   (key): Linear(in_features=1, out_features=1, bias=True)
#   (value): Linear(in_features=1, out_features=1, bias=True)
#   (fc_1): Linear(in_features=34496, out_features=4096, bias=True)
#   (bn_1): BatchNorm1d(4096, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#   (fc_2): Linear(in_features=4096, out_features=4096, bias=True)
#   (bn_2): BatchNorm1d(4096, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#   (output): Linear(in_features=4096, out_features=10, bias=True)
# )
input = torch.randn(32, 3, 224, 224)
output_p, final_f = model(input)
print(output_p.size())#(32, 10)
print(final_f.size())#(32, 34496=7*7*512+3*56*56) 
summary(model, (3, 224, 224))
# Total params: 172,861,520
# Trainable params: 172,861,520
# Non-trainable params: 0
# ----------------------------------------------------------------
# Input size (MB): 0.57
# Forward/backward pass size (MB): 322.09
# Params size (MB): 659.41
# Estimated Total Size (MB): 982.08