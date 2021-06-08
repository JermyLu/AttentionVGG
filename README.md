# AttentionVGG
在VGG中增加了Attention机制，分别为Self-attention和Encoder-Decoder-attention
1. VGG_SelfAttention.py
其只在VGG的基础上增加了一个self-attention特征提取模块，模型结构如图self-attention-VGG所示。
![image](https://github.com/JermyLu/AttentionVGG/blob/main/self-attention-VGG.png)
2. VGG_EDAttention.py
其在VGG的基础上不仅增加了一个self-attention特征提取模块，还将VGG原来的卷积层获得卷积特征作为query，与self-attention编码的全局特征进行了Encoder-Decoder-attention处理，模型结构如图Encoder-Decoder-Attention所示。
![image](https://github.com/JermyLu/AttentionVGG/blob/main/Encoder-Decoder-Attention.png)
