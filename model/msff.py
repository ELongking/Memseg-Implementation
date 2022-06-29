# re-implemented via official implementation of coordinate attention, for more details you can check at
# https://github.com/Andrew-Qibin/CoordAttention

import torch
import torch.nn as nn
import math
import torch.nn.functional as F


def sp_attention(f):
    return f.mean(dim=0)


class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6


class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)


class CoordAtt(nn.Module):
    def __init__(self, in_channel, out_channel, reduction=32):
        super(CoordAtt, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        mip = max(8, in_channel // reduction)

        self.conv1 = nn.Conv2d(in_channel, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = h_swish()

        self.conv_h = nn.Conv2d(mip, out_channel, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, out_channel, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        identity = x

        n, c, h, w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)

        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)

        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()

        out = identity * a_w * a_h

        return out


class MSFFBlock(nn.Module):
    def __init__(self, in_channel):
        super(MSFFBlock, self).__init__()
        self.conv = nn.Conv2d(in_channel, in_channel, kernel_size=3, stride=1, padding=1)
        self.attn = CoordAtt(in_channel, in_channel)
        self.conn = nn.Sequential(nn.Conv2d(in_channel, in_channel // 2, kernel_size=3, stride=1, padding=1),
                                  nn.Conv2d(in_channel // 2, in_channel // 2, kernel_size=3, stride=1, padding=1))

    def forward(self, x):
        x_conv = self.conv(x)
        x_att = self.attn(x)
        x_concat = x_conv * x_att
        x_k = self.conn(x_concat)
        return x_k

    
class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat((torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1)


class SpatialAttentionBlock(nn.Module):
    def __init__(self):
        super(SpatialAttentionBlock, self).__init__()
        self.pool = ChannelPool()
        self.conv = nn.Sequential(nn.Conv2d(2, 1, kernel_size=7, stride=1, padding=3),
                                  nn.BatchNorm2d(1),
                                  nn.Sigmoid()
                                  )

    def forward(self, x):
        x_pool = self.pool(x)
        x_att = self.conv(x_pool)

        x_f = x * x_att
        return x_f


class MSFF(nn.Module):
    def __init__(self):
        super(MSFF, self).__init__()
        self.blk1 = MSFFBlock(512)
        self.blk2 = MSFFBlock(256)
        self.blk3 = MSFFBlock(128)

        self.sa1 = SpatialAttentionBlock()
        self.sa2 = SpatialAttentionBlock()
        self.sa3 = SpatialAttentionBlock()

        self.upconv32 = nn.Sequential(nn.Upsample(scale_factor=0.5),
                                      nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1))
        self.upconv21 = nn.Sequential(nn.Upsample(scale_factor=0.5),
                                      nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1))

    def forward(self, x):
        f3, f2, f1 = x
        f1_k = self.blk1(f1)
        f2_k = self.blk2(f2)
        f3_k = self.blk3(f3)

        f3_f = f3_k
        f2_f = f2_k + self.upconv32(f3_f)
        f1_f = f1_k + self.upconv21(f2_f)

        f1_out = self.sa1(f1_f)  # torch.Size([1, 256, 16, 16])
        f2_out = self.sa2(f2_f)  # torch.Size([1, 128, 32, 32])
        f3_out = self.sa3(f3_f)  # torch.Size([1, 64, 64, 64])

        return f1_out, f2_out, f3_out
