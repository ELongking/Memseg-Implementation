import torch
import torch.nn as nn

from model import *
from utils import *


class UpConvBlock(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(UpConvBlock, self).__init__()
        self.blk = nn.Sequential(nn.Upsample(scale_factor=2),
                                 nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=1, padding=0),
                                 nn.BatchNorm2d(out_channel),
                                 nn.ReLU()
                                 )

    def forward(self, x):
        return self.blk(x)


class Memseg(nn.Module):
    def __init__(self, memory_path):
        super(Memseg, self).__init__()
        self.memory_path = memory_path
        self.bank = memory_module.MemoryBank(self.memory_path)
        self.msff = msff.MSFF()

        self.encoder = resnet.resnet18(pretrained=False)
        self.addition_conv = nn.Conv2d(64, 48, kernel_size=1, stride=1, padding=0)

        self.upconv0 = UpConvBlock(512, 256)
        self.upconv1 = UpConvBlock(512, 128)
        self.upconv2 = UpConvBlock(256, 64)
        self.upconv3 = UpConvBlock(128, 48)
        self.upconv4 = UpConvBlock(96, 48)

        self.final_conv = nn.Conv2d(48, 2, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x_simulated = self.bank.process(x)
        n, c, h, w = x_simulated[0].shape

        x_att_1, x_att_2, x_att_3 = self.msff(x_simulated)
        _, x = self.encoder(x)
        x_identity = x[0]
        x_identity = self.addition_conv(x_identity)  # torch.Size([1, 48, 128, 128])
        x = x[-1]

        x_up0 = self.upconv0(x)

        x = torch.cat([x_up0, x_att_1], dim=1)  # (1,512,16,16)

        x_up1 = self.upconv1(x)
        x = torch.cat([x_up1, x_att_2], dim=1)  # (1,256,32,32)

        x_up2 = self.upconv2(x)
        x = torch.cat([x_up2, x_att_3], dim=1)  # (1,128,64,64)

        x_up3 = self.upconv3(x)
        x = torch.cat([x_up3, x_identity], dim=1)  # (1,96,128,128)

        x_up4 = self.upconv4(x)  # (1,48,256,256)

        x_predict = self.final_conv(x_up4)  # torch.Size([1, 2, 256, 256])
        return x_predict


if __name__ == '__main__':
    image = torch.randn(10, 3, 256, 256)
    mD = r'G:/Dataset/mvtec_anomaly_detection/mvtec_anomaly_detection/capsule/train/good/'
    model = Memseg(mD)
    model(image)
