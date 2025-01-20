#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2025/1/15 上午11:18
# @Author  : Mr Zhang
# @File    : CF-ATT.py
# @Software: PyCharm


import torch
import torch.nn.functional as F
import math
import numpy as np
import torch.nn as nn
from TCN import TemporalConvNet

# class PositionalEncoding(torch.nn.Module):
#     def __init__(self, d_model, max_len=17520):
#         super(PositionalEncoding, self).__init__()
#         pe = torch.zeros(max_len, d_model)
#         position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
#         div_term = torch.exp(torch.arange(0, d_model, dtype=torch.float) * (-math.log(10000.0) / d_model))
#         pe[:, 0::2] = torch.sin(position * div_term)
#         pe[:, 1::2] = torch.cos(position * div_term)
#         pe = pe.unsqueeze(0).transpose(0, 1)
#         self.register_buffer('pe', pe)
#
#     def forward(self, x):
#         return x + self.pe[:x.size(1), :]

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=17520):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        '''
        x: [seq_len, batch_size, d_model]
        '''
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class CAM(torch.nn.Module):
    def __init__(self, embed_dim, r=4):
        super(CAM, self).__init__()  # batch_size, embed_dim, sequence_length
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        self.pointwise_conv1 = nn.Conv1d(embed_dim, embed_dim // r, 1)
        self.bn1 = nn.BatchNorm1d(embed_dim // r)
        self.relu = nn.ReLU()
        self.pointwise_conv2 = nn.Conv1d(embed_dim // r, embed_dim, 1)
        self.bn2 = nn.BatchNorm1d(embed_dim)

        self.pointwise_conv3 = nn.Conv1d(embed_dim, embed_dim // r, 1)
        self.bn3 = nn.BatchNorm1d(embed_dim //r)
        self.pointwise_conv4 = nn.Conv1d(embed_dim // r, embed_dim, 1)
        self.bn4 = nn.BatchNorm1d(embed_dim)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x_avg = self.global_avg_pool(x)
        x_avg = self.pointwise_conv1(x_avg)
        x_avg = self.bn1(x_avg)
        x_avg = self.relu(x_avg)
        x_avg = self.pointwise_conv2(x_avg)
        x_avg = self.bn2(x_avg)

        x_conv = self.pointwise_conv3(x)
        x_conv = self.bn3(x_conv)
        x_conv = self.relu(x_conv)
        x_conv = self.pointwise_conv4(x_conv)
        x_conv = self.bn4(x_conv)

        x_out = x_avg + x_conv
        x_out = self.sigmoid(x_out)
        x_out = x_out * x

        return x_out


class DifferenceAttention(torch.nn.Module):
    def __init__(self, embed_dim = 64):
        super(DifferenceAttention, self).__init__()
        self.tcn = TemporalConvNet(num_inputs=10, num_channels=[16, 32, 64], kernel_size=3, dropout=0.2)

        self.pos_encoder = PositionalEncoding(d_model=embed_dim)

        self.query_linear = nn.Conv1d(embed_dim, embed_dim, 1)
        self.key_linear = nn.Conv1d(embed_dim, embed_dim, 1)
        self.value_linear_1 = nn.Conv1d(embed_dim, embed_dim, 1)
        self.value_linear_2 = nn.Conv1d(embed_dim, embed_dim, 1)
        self.CAM = CAM(embed_dim)

        self.linear1 = nn.Conv1d(embed_dim, 32, 1)
        self.linear2 = nn.Conv1d(32, 16, 1)
        self.linear3 = nn.Conv1d(16, 1, 1)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        tcn_in = x.permute(0, 2, 1)
        tcn_out = self.tcn(tcn_in)
        pos_in = tcn_out.permute(0, 2, 1)
        encoded_x = self.pos_encoder(pos_in)  # 8 17520 64
        encoded_x = encoded_x.permute(0, 2, 1)  # 8 64 17520

        q = self.query_linear(encoded_x)
        k = self.key_linear(encoded_x)
        v1 = self.value_linear_1(encoded_x)
        v2 = self.value_linear_1(encoded_x)
        attention_scores = torch.matmul(q, k.transpose(-1, -2))
        attention_scores = F.softmax(attention_scores, dim=-1)
        diff_attention_output_1 = torch.matmul(attention_scores, v1)
        diff_attention_output_2 = torch.matmul(attention_scores, v2)
        cam_output = self.CAM((diff_attention_output_1+diff_attention_output_2)/2)
        merge1 = cam_output * encoded_x
        merge2 = 1 - cam_output * encoded_x
        TAFF_output = (merge1 + merge2) / 2

        x_out = self.linear1(TAFF_output)
        x_out = self.linear2(x_out)
        x_out = self.linear3(x_out)

        x_out = self.sigmoid(x_out)

        return x_out


if __name__ == '__main__':
    # 输入数据
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size, seq_len, embed_dim = 8, 17520, 10
    x = torch.randn(batch_size, seq_len, embed_dim).float().to(device)

    # 位置编码
    # pos_encoder = PositionalEncoding(embed_dim)
    # pos_encoder = PositionalEncoding(d_model=embed_dim).cuda()
    # encoded_x = pos_encoder(x)
    # 差分注意力模块
    diff_attention = DifferenceAttention(64).cuda()
    output = diff_attention(x)

    print(output.shape)  # 应该输出：torch.Size([8, 64, 17520])




