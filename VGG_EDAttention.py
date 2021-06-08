#EDAttention is ABB of Encoder-Decoder Attention
#Description of base idea: 借鉴自transformer
#   combining the whole image and partition, 
#   and using the convolution result as query to get its value, its value is the input of next convolution
#   and cat the whole feature and convolution feature as final feature.
#Base idea_1: using convolution and Encoder-Decoder-attention for image detection; 
#   using the convolution extracts the local feature, and then take this local feature as query to get its value,
#   this value instead of local feature is the input of next convolution block.
#Base idea_2: using the self-attention extracts the entire feature
#   avgpooling should be used before appling self-attention, in order to save memory. 

#Note: 假设输入为224*224，批处理进行query*key时，[batch_size, 3,  224*224, 1] * [batch_size, 3, 1, 224*224]
#   那么，结果矩阵大小为[batch_size, 3, 224*224, 224*224]，则计算机应分配batch_size*3*224*224*224*224个单精度浮点数的内存大小。
#   单精度浮点数占4Byte，因此当batch_size为1时，参数占据的内存已有28GB左右。
#Note2: 为了节省内存空间，在对输入进行self-attention之前，先进行resize或进行avgpooling等操作。
#   例如，当输入由224*224-->112*112时，当batch_size为1时，参数个数有1*3*112*112*112*112个，其占据内存大小为1.76GB左右，小了16倍。

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
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

class VGG_EDAttention(nn.Module):
    def __init__(self, config: Config, conv_nums_list: List):
        super(VGG_EDAttention, self).__init__()
        #Convolution: extract the local features
        self.conv3_64 = base_vgg_block(3, 64, conv_nums=conv_nums_list[0])
        self.conv64_128 = base_vgg_block(64, 128, conv_nums=conv_nums_list[1])
        self.conv128_256 = base_vgg_block(128, 256, conv_nums=conv_nums_list[2])
        self.conv256_512 = base_vgg_block(256, 512, conv_nums=conv_nums_list[3])#img_output: 7*7*512
        #self.conv512_512 = base_vgg_block(512, 512, conv_nums=conv_nums_list[4], if_maxpooling=False)#img_output: 7*7*512
        
        #Self-Attention Mechanism: extract the global features
        #a pixel is a basic element
        self.query = nn.Linear(1, config.query_dim)
        self.key = nn.Linear(1, config.key_dim)
        self.value = nn.Linear(1, config.value_dim)
        #Encoder-Decoder Mechanism
        self.conv3_64_att = nn.Linear(1, config.query_dim)
        self.conv64_128_att = nn.Linear(1, config.query_dim)
        self.conv128_256_att = nn.Linear(1, config.query_dim)
        self.conv256_512_att = nn.Linear(1, config.query_dim)
        #self.conv512_512_att = nn.Linear(1, config.query_dim)

        #Full-connection layer
        self.fc_1 = nn.Linear(7*7*512+3*56*56, 4096)
        self.bn_1 = nn.BatchNorm1d(4096)#每个全连接层后接一个BN
        self.fc_2 = nn.Linear(4096, 4096)
        self.bn_2 = nn.BatchNorm1d(4096)
        self.output = nn.Linear(4096, config.num_classes)

    def attention_calculate(self, query: Tensor, key: Tensor, value: Tensor):
        """Args:
            query: (batch_size, channels, length_q, dim_q)
            key: (batch_size, 3, length_k, dim_k=dim_q)
            value: (batch_size, 3, length_k, dim_v)
        """
        #https://pytorch.org/docs/stable/generated/torch.matmul.html#torch.matmul
        channels_q, channels_k, channels_v = query.size()[1], key.size()[1], value.size()[1]
        assert channels_k == channels_v
        if channels_q == channels_k:
            attention_scores = torch.matmul(query, key.transpose(2, 3))#(batch_size, channels, length_q, length_k)
            attention_p = F.softmax(attention_scores, dim=-1)
            final_value = torch.matmul(attention_p, value)#(batch_size, channels, length_q, dim_v)
            return final_value
        
        else:
            #avg key on the channels
            key = torch.mean(key, dim=1)#key.shape: (batch_size, length, dim_k)
            batch_q ,channels_q = query.size()[0], query.size(1)
            length_q, length_k = query.size()[2], key.size()[1]
            # print(query.size(), key.size(), value.size())
            print(batch_q*channels_q*length_q*length_k)
            attention_scores = torch.zeros(batch_q, channels_q, length_q, length_k)#shape: (batch_size, channels, length_q, length_k)
            i = 0
            #in the each batch, do the broadcast mechanism
            for q, k in zip(query, key):
                #q.shape: (channels, length_q, dim_q)
                #k.shape: (length_k, dim_k)
                #broadcasted matrix k
                attention_scores[i] = torch.matmul(q, k.transpose(0, 1))
                i += 1
            attention_p = F.softmax(attention_scores, dim=-1)#shape: (batch_size, channels, length_q, length_k)
            
            #avg value on the channels
            value = torch.mean(value, dim=1)#value.shape: (batch_size, length_k, dim_v)
            batch_f, channels_f = attention_p.size()[0], attention_p.size()[1]
            length_q, dim_v = attention_p.size()[2], value.size()[2]
            final_value = torch.zeros(batch_f, channels_f, length_q, dim_v)
            i = 0
            #in the each batch, do the broadcast mechanism
            for a_p, v in zip(attention_p, value):
                #a_p.shape: (channels, length_q, length_k)
                #value.shape: (length_k, dim_v)
                final_value[i] = torch.matmul(a_p, v)#(channels, length_q, dim_v)
                i += 1

            return final_value

    def attention_in_conv(self, query: Tensor, key: Tensor, value: Tensor, m: nn.Linear):
        pass
        """Args:
            query: (batch_size, channels_q, dim, dim)
            key: (batch_size, channels_k, length_k, 1)
            value: (batch_size, channels_v, length_v, 1)
            m: Linear layer
        """
        dim = query.size()[-1]#dim: 卷积之后的image的特征图的长宽（相同）
        query = query.view(query.size()[0], query.size()[1], -1, 1)#(batch_size, channels_q, dim*dim, 1)
        #print(query.size(), key.size(), value.size())
        query = m(query)
        query = self.attention_calculate(query=query, key=key, value=value)
        query = query.view(query.size()[0], query.size()[1], dim, dim)#(batch_size, channels_q, dim, dim)
        return query

    def forward(self, x):
        #x.shape: (batch_size, 3, 224, 224)
        #Self-Attention Mechanism: extract the global features
        #a pixel is a basic element
        #为了节省内存资源，先resize
        conv_x = F.adaptive_avg_pool2d(x, output_size=(112, 112))#x.shape: (batch_size, 3, 112, 112)
        self_attention_x = F.adaptive_avg_pool2d(x, output_size=(56, 56))#x.shape: (batch_size, 3, 56, 56)#为了节省内存
        self_attention_x = self_attention_x.view(self_attention_x.size()[0], self_attention_x.size()[1], -1, 1)#its shape: (batch_size, 3, 56*56, 1)

        #self-attention, value_att是利用self-attention机制对整张图片的编码
        query_att = self.query(self_attention_x)
        key_att = self.key(self_attention_x)
        value_att = self.value(self_attention_x)
        value_att = self.attention_calculate(query=query_att, key=key_att, value=value_att)#its shape: (batch_size, 3, 56*56, 1)
    
        #在卷积层中应用EDAttention, 卷积层用于 extract the local feature
        conv_x = self.conv3_64(conv_x)#conv_x.shape: (batch_size, 64, 56, 56)
        conv_x = self.attention_in_conv(query=conv_x, key=key_att, value=value_att, m=self.conv3_64_att)
        conv_x = self.conv64_128(conv_x)#(batch_size, 128, 28, 28)
        conv_x = self.attention_in_conv(query=conv_x, key=key_att, value=value_att, m=self.conv64_128_att)
        conv_x = self.conv128_256(conv_x)#(batch_size, 256, 14, 14)
        conv_x = self.attention_in_conv(query=conv_x, key=key_att, value=value_att, m=self.conv128_256_att)
        conv_x = self.conv256_512(conv_x)#(batch_size, 512, 7, 7)
        conv_x = self.attention_in_conv(query=conv_x, key=key_att, value=value_att, m=self.conv256_512_att)
        #flatten conv_x
        conv_x = conv_x.view(conv_x.size()[0], -1)#(batch_size, 512*7*7)
        #flatten value_att
        value_att = value_att.view(value_att.size()[0], -1)#(batch_size, 3*56*56)
        #Put them together
        final_x = torch.cat((value_att, conv_x), 1)#(batch_size, 3*56*56+512*7*7)
        # print(conv_x.size(), value_att.size(), final_x.size())

        #Full connection
        out = self.fc_1(final_x)#(batch_size, 4096)
        out = self.bn_1(out)
        out = self.fc_2(out)#(batch_size, 4096)
        out = self.bn_2(out)
        out = F.relu(out)
        out = self.output(out)#(batch_size, num_classes)
        out_p = F.softmax(out, dim=1)
        return out_p, final_x

# #Testing
config = Config()
vgg16 = [1, 1, 2, 2]
model = VGG_EDAttention(config=config, conv_nums_list=vgg16)
# summary(model, (3, 224, 224))
input = torch.randn(2, 3, 224, 224)
output_p, final_f = model(input)
print(output_p.size())#torch.Size([2, 10])
print(final_f.size())#torch.Size([2, 34496])