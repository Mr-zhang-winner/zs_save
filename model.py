import unittest

import torch
import torch.nn as nn
import os
import torch.nn.functional as F

from utils.DiffAtt import DiffAtt
from utils.TCN import TemporalConvNet
from utils.TrainUtils import calculate_res, process_cas, generate_pseudo_labels_zs
import math
import time

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


class CSEN(nn.Module):
    def __init__(self):
        super(CSEN, self).__init__()
        self.tcn = TemporalConvNet(num_inputs=10, num_channels=[16, 32, 64], kernel_size=3, dropout=0.2)

        self.conv1 = nn.Conv1d(
            in_channels=64,
            out_channels=128,
            kernel_size=7,
            padding_mode="replicate",
            padding=3,
        )
        self.conv2 = nn.Conv1d(
            in_channels=128, out_channels=128, kernel_size=5, padding=2
        )
        self.conv3 = nn.Conv1d(
            in_channels=128, out_channels=128, kernel_size=3, padding=1
        )

        self.conv_16 = nn.Conv1d(in_channels=16, out_channels=8, kernel_size=3, padding=1)
        self.conv_1 = nn.Conv1d(in_channels=8, out_channels=1, kernel_size=1, padding=0)
        # 全连接层
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)

        self.sigmoid = nn.Sigmoid()
        self.leak_relu = nn.LeakyReLU()

    def forward(self, x):
        """
        :param x: (b,t,c)
        :return: p_class(b,), cas(b,t)
        """
        out = x.permute(0, 2, 1)
        out = self.tcn(out)
        out = self.conv1(out)
        out = self.leak_relu(out)
        out = self.conv2(out)
        out = self.leak_relu(out)
        out = self.conv3(out)
        out = self.leak_relu(out)
        out = out.permute(0, 2, 1)
        out = self.fc1(out)
        out = self.leak_relu(out)
        out = self.fc2(out)
        out = self.leak_relu(out)
        out = self.fc3(out)
        out = out.permute(0, 2, 1)
        cas = self.sigmoid(out)

        # out = self.conv_16(out)
        # out = self.leak_relu(out)
        # out = self.conv_1(out)
        # cas = self.sigmoid(out)
        pseudo_label = process_cas(cas)
        cas, p_class = calculate_res(cas)

        return pseudo_label, p_class, cas


class T_UNet(nn.Module):
    """
    unet的
    in_channels: 输入通道数 64
    out_classes: 输出通道数 1
    """
    def __init__(self, out_channels=1):
        super(T_UNet, self).__init__()
        self.tcn = TemporalConvNet(num_inputs=10, num_channels=[16, 32, 64], kernel_size=3, dropout=0.2)

        # self.inc = ConvBReLu1d(64, 16)
        self.SA = nn.MultiheadAttention(64, 1)
        self.down1 = Down1d(64, 128)
        self.down2 = Down1d(128, 256)
        self.up1 = Up1d(256, 128)
        self.up2 = Up1d(128, 64)
        # self.outc1 = OutConv(64, 32)
        # self.outc2 = OutConv(32, 16)
        self.outc3 = OutConv(64, out_channels)

    def forward(self, x, point_label, label):
        x_in = x.permute(0, 2, 1)  # x: 8 17520 10  x_in:8 10 17520
        tcn_out = self.tcn(x_in)
        # inc_out = self.inc(tcn_out)  # 降到32

        # SA_in = tcn_out.permute(2, 0, 1)  # SA_in: 17520 8 64
        # SA_out, _ = self.SA(SA_in, SA_in, SA_in)
        # x1 = SA_out.permute(1, 2, 0)  # x1: 8 64 17520

        x1 = tcn_out
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.up1(x3, x2)
        x5 = self.up2(x4, x1)
        out = self.outc3(x5)
        # out = self.outc2(out)
        # out = self.outc3(out)
        # print('tcn_in', x_in.shape)
        # print('tcn_out', tcn_out.shape)
        # print('SA_in', SA_in.shape)
        # print('SA_out', SA_out.shape)
        # print('down1_out', x2.shape)
        # print('down2_out', x3.shape)
        # print('up1_out', x4.shape)
        # print('up2_out', x5.shape)
        # print('outc', out.shape)   #8 1 17520

        cas = F.sigmoid(out)
        pseudo_label = process_cas(cas, point_label)  # 最终 NMS 结果  8 17520 1
        cas, p_class = calculate_res(cas, label)
        return pseudo_label, p_class, cas



class ConvBReLu1d(nn.Module):
    """(convolution => [BN] => ReLU)"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down1d(nn.Module):
    """Downscaling with maxpool then double conv"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool1d(2),
            ConvBReLu1d(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up1d(nn.Module):
    """Upscaling then double conv"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # 使用反卷积进行上采样
        self.up = nn.ConvTranspose1d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = ConvBReLu1d(in_channels, out_channels)

    def forward(self, x_1, x_2):
        x_1 = self.up(x_1)
        x = torch.cat([x_2, x_1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, in_channels // 2, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels // 2, out_channels, kernel_size=1)

    def forward(self, x):
        conv1_out = self.conv1(x)
        return self.conv2(conv1_out)




class RNN(nn.Module):
    def __init__(self):
        super(RNN, self).__init__()
        self.tcn = TemporalConvNet(num_inputs=10, num_channels=[16, 32, 64], kernel_size=3, dropout=0.2)
        self.SA = nn.MultiheadAttention(64, 1)

        self.gru = nn.GRU(64, hidden_size=128, num_layers=2, batch_first=True)
        self.conv_16 = nn.Conv1d(in_channels=128, out_channels=16, kernel_size=3, padding=1)
        self.conv_1 = nn.Conv1d(in_channels=16, out_channels=1, kernel_size=1, padding=0)
        # 全连接层
        self.fc0 = nn.Linear(64, 128)
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)

        self.leak_relu = nn.LeakyReLU()
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        :param x: (b,t,c)
        :return: p_class(b,), cas(b,t)
        """
        # print('rnn in', x.shape)
        h0 = torch.randn(2, x.size(0), 128).to(x.device)
        out = x.permute(0, 2, 1)  # 8 10 17520
        out = self.tcn(out)  # 8 64 17520
        # print('tcn out', out.shape)
        SA_in = out.permute(2, 0, 1) # 17520 8 64
        SA_out, _ = self.SA(SA_in, SA_in, SA_in)
        out = SA_out.permute(1, 0, 2)  # 8 17520 64
        # out = out.permute(0, 2, 1)  # 8 17520 64
        out, _ = self.gru(out, h0)  # 8 17520 128
        # out = self.fc0(out)
        # time1 = time.time()
        out = self.fc1(out)
        out = self.leak_relu(out)
        out = self.fc2(out)
        out = self.leak_relu(out)
        out = self.fc3(out)
        # print('rnn out',out.shape)
        out = out.permute(0, 2, 1)  # 8 1 17520
        cas = self.sigmoid(out)
        # print('cas', cas.shape)

        # post-processing
        nms_map = process_cas(cas)  # 最终 NMS 结果  8 17520 1
        # out = self.conv_16(out)
        # out = self.leak_relu(out)  # （batch_size, t, 16）
        # out = self.conv_1(out)
        # cas = self.sigmoid(out)
        cas, p_class = calculate_res(cas)
        # print('zuihou',cas.shape, p_class.shape)

        return nms_map, p_class, cas


class DAtt(nn.Module):
    def __init__(self):
        super(DAtt, self).__init__()
        self.tcn = TemporalConvNet(num_inputs=10, num_channels=[16, 32, 64], kernel_size=3, dropout=0.2)
        self.down = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=1, stride=10)

        self.diff_att = DiffAtt()

        self.up = nn.ConvTranspose1d(in_channels=512, out_channels=128, kernel_size=10, stride=10)

        self.conv_16 = nn.Conv1d(in_channels=128, out_channels=16, kernel_size=3, padding=1)
        self.conv_1 = nn.Conv1d(in_channels=16, out_channels=1, kernel_size=1, padding=0)

        self.leak_relu = nn.LeakyReLU()
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        :param x: (b,t,c)
        :return: p_class(b,), cas(b,t)
        """
        out = x.permute(0, 2, 1)
        out = self.tcn(out)
        out = self.down(out)
        out = self.diff_att(out)  # (b,t,c)
        out = out.permute(0, 2, 1)  # (b,c,t)
        out = self.up(out)
        out = self.conv_16(out)
        out = self.leak_relu(out)  # （batch_size, 16, t）
        out = self.conv_1(out)
        cas = self.sigmoid(out)
        cas, p_class = calculate_res(cas)
        return p_class, cas


########################################################################################################################
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


class DAtt_ZS(torch.nn.Module):
    def __init__(self, embed_dim=64):
        super(DAtt_ZS, self).__init__()
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

    def forward(self, x, point_label, label):

        tcn_in = x.permute(0, 2, 1)
        tcn_out = self.tcn(tcn_in)
        pos_in = tcn_out.permute(0, 2, 1)
        encoded_x = self.pos_encoder(pos_in)  # 8 17520 64
        encoded_x = encoded_x.permute(0, 2, 1)  # 8 64 17520

        q = self.query_linear(encoded_x)
        k = self.key_linear(encoded_x)
        v1 = self.value_linear_1(encoded_x)
        v2 = self.value_linear_2(encoded_x)
        attention_scores = torch.matmul(q, k.transpose(-1, -2))
        attention_scores = F.softmax(attention_scores, dim=-1)
        diff_attention_output_1 = torch.matmul(attention_scores, v1)
        diff_attention_output_2 = torch.matmul(attention_scores, v2)
        cam_output = self.CAM((diff_attention_output_1+diff_attention_output_2)/2)
        merge1 = cam_output * diff_attention_output_1
        merge2 = 1 - cam_output * diff_attention_output_2
        TAFF_output = (merge1 + merge2) / 2

        x_out = self.linear1(TAFF_output)
        x_out = self.linear2(x_out)
        x_out = self.linear3(x_out)

        cas = self.sigmoid(x_out)
        # pseudo_labels = process_cas(cas, point_label)  # 最终 NMS 结果  8 17520 1
        pseudo_labels = generate_pseudo_labels_zs(cas, point_label)
        cas, p_class = calculate_res(cas, label)

        return pseudo_labels, p_class, cas



class TestDAtt(unittest.TestCase):

    def setUp(self):
        """在每个测试方法执行前调用，用于设置环境"""
        self.batch_size, self.seq_len, self.num_inputs = 8, 17520, 10  # 测试参数
        self.model = DAtt().cuda()  # 实例化DAtt模型

    def test_forward(self):
        """测试DAtt类的forward方法"""
        x = torch.randn(self.batch_size, self.seq_len, self.num_inputs).cuda()  # 生成随机数据作为输入
        p_class, cas = self.model(x)  # 对输入数据进行前向传播

        # 测试p_class和cas的形状是否符合预期
        self.assertTrue(p_class.shape[0] == self.batch_size, "p_class batch size mismatch")
        # 假设calculate_res返回的cas形状与输入形状相同
        self.assertTrue(cas.shape == x.shape[:2], "cas shape mismatch")


class TestRNN(unittest.TestCase):

    def test_forward_pass(self):
        batch_size = 16
        time_steps = 17520
        input_size = 10

        model = RNN()
        x = torch.rand(batch_size, time_steps, input_size)

        p_class, cas = model(x)
        print(p_class.shape, cas.shape)
        self.assertEqual(p_class.shape, torch.Size([batch_size, ]))
        self.assertEqual(cas.shape, torch.Size([batch_size, time_steps, ]))


class TestCSEN(unittest.TestCase):

    def test_forward_pass(self):
        batch_size = 16
        time_steps = 17520
        input_size = 10

        model = CSEN()
        x = torch.rand(batch_size, time_steps, input_size)

        p_class, cas = model(x)

        self.assertEqual(p_class.shape, torch.Size([batch_size]))
        self.assertEqual(cas.shape, torch.Size([batch_size, time_steps]))


class TestTUNet(unittest.TestCase):
    def test_forward_pass(self):
        batch_size = 8
        channels = 10
        length = 17520
        out_channels = 1

        model = T_UNet(out_channels=out_channels)  # B C L
        x = torch.randn(batch_size, length, channels)  # B C L
        pseudo_label, outputs, cas = model(x)
        print(pseudo_label.shape)


class TestDATT_ZS(unittest.TestCase):
    def test_forward_pass(self):
        batch_size = 8
        channels = 10
        length = 17520
        out_channels = 1

        model = DAtt_ZS().cuda()
        x = torch.randn(batch_size, length, channels).float().cuda()  # B C L


        output = model(x)

        # pseudo_label, outputs, cas = model(x)
        print(output.shape)


