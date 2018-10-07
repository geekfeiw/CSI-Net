import scipy.io as sio
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
# from model import locNN
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import matplotlib.pyplot as plt
import math


# 3x3 Convolution
def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3,
                     stride=stride, padding=1,  bias=False)




# Bottleneck module
class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out



# Residual Block
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = conv3x3(in_channels, out_channels, stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(out_channels, out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out

# ResNet Module
class ResNet(nn.Module):
    def __init__(self, block, layers):
        super(ResNet, self).__init__()

       # self.bn1 = nn.BatchNorm2d(52)
       #  self.generation = nn.Sequential(
       #      # 30*1*1 -> 256*2*2
       #      nn.ConvTranspose2d(30, 256, kernel_size=4, stride=2, padding=1, bias=False),
       #      nn.BatchNorm2d(256),
       #      nn.ReLU(),
       #      # 256*2*2 -> 128*4*4
       #      nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=False),
       #      nn.BatchNorm2d(128),
       #      nn.ReLU(),
       #
       #      # 128*4*4 -> 64*8*8
       #      nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias=False),
       #      nn.BatchNorm2d(64),
       #      nn.ReLU(),
       #      # 64*8*8 -> 32*16*16
       #      nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1, bias=False),
       #      nn.BatchNorm2d(32),
       #      nn.ReLU(),
       #      # 32*16*16 -> 16*32*32
       #      nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1, bias=False),
       #      nn.BatchNorm2d(16),
       #      nn.ReLU(),
       #
       #      # 16*32*32 ->64 8*64*64
       #      nn.ConvTranspose2d(16, 3, kernel_size=4, stride=2, padding=1, bias=False),
       #      nn.BatchNorm2d(3),
       #      nn.ReLU(),
       #
       #  )

        self.generation = nn.Sequential(
            # 30*1*1 -> 256*2*2
            nn.ConvTranspose2d(30, 384, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(384),
            nn.ReLU(),
            # 256*2*2 -> 128*4*4
            nn.ConvTranspose2d(384, 192, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(192),
            nn.ReLU(),

            # 128*4*4 -> 64*7*7
            nn.ConvTranspose2d(192, 96, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(96),
            nn.ReLU(),

            # 7 -> 14
            nn.ConvTranspose2d(96, 48, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(),

            # 14 -> 28
            nn.ConvTranspose2d(48, 24, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(24),
            nn.ReLU(),

            # 28 -> 56
            nn.ConvTranspose2d(24, 12, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(12),
            nn.ReLU(),

            # 56 -> 112
            nn.ConvTranspose2d(12, 6, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(6),
            nn.ReLU(),

            # 112 -> 224c
            nn.ConvTranspose2d(6, 6, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(6),
            nn.ReLU(),
        )

        self.conv1 = nn.Sequential(
            nn.Conv2d(6, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        self.in_channels = 64
        self.layer1 = self.make_layer(block, 64, layers[0]) #8*64*64 -> 16*32*32
        self.layer2 = self.make_layer(block, 128, layers[1], 2) #16*32*32 -> 32*16*16
        self.layer3 = self.make_layer(block, 256, layers[2], 2) #32*16*16 -> 64*8*8
        self.layer4 = self.make_layer(block, 512, layers[3], 2) #64*8*8 -> 128*4*4


        self.humanid = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(),

            nn.Conv2d(512, 256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AvgPool2d(2),
        )

        self.fcc = nn.Linear(256, 30)

        self.biometrics = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(),

            nn.Conv2d(512, 256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AvgPool2d(2),
        )

        self.fcr = nn.Linear(256, 4)

        # self.humanid = nn.Sequential(
        #     nn.Conv2d(256, 128, kernel_size=3, stride=2, padding=1, bias=False),
        #     nn.BatchNorm2d(128),
        #     nn.ReLU(),
        #
        #     nn.Conv2d(128, 64, kernel_size=4, stride=2, padding=1, bias=False),
        #     nn.BatchNorm2d(64),
        #     nn.ReLU(),
        #     nn.AvgPool2d(2),
        # )
        #
        # self.fcc = nn.Linear(64, 30)
        #
        # self.biometrics = nn.Sequential(
        #     nn.Conv2d(256, 128, kernel_size=3, stride=2, padding=1, bias=False),
        #     nn.BatchNorm2d(128),
        #     nn.ReLU(),
        #
        #     nn.Conv2d(128, 64, kernel_size=4, stride=2, padding=1, bias=False),
        #     nn.BatchNorm2d(64),
        #     nn.ReLU(),
        #     nn.AvgPool2d(2),
        # )
        #
        # self.fcr = nn.Linear(64, 4)



        # self.avgpool = nn.AvgPool2d(7) #128*4*4 -> 128*1*1
        # # self.maxpool = nn.MaxPool2d(4) #128*4*4 -> 128*1*1
        # self.dp = nn.Dropout()
        #
        # #fully-connected layers
        # self.fcc1 = nn.Linear(512, 512) #128 -> 64
        # self.fcc2 = nn.Linear(512, 128) #64 -> 48
        # self.fcc3 = nn.Linear(128, 30) #48 -> 30
        # #
        # self.fcr1 = nn.Linear(512, 256)
        # self.fcr2 = nn.Linear(256, 128)
        # self.fcr3 = nn.Linear(128, 32)
        # self.fcr4 = nn.Linear(32, 4)

    def make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if (stride != 1) or (self.in_channels != out_channels):
            downsample = nn.Sequential(
                conv3x3(self.in_channels, out_channels, stride=stride),
                nn.BatchNorm2d(out_channels))
        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels
        for i in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        out_generation = self.generation(x)
        out = self.conv1(out_generation)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.maxpool(out)
#
#
        outres1 = self.layer1(out)
        outres2 = self.layer2(outres1)
        outres3 = self.layer3(outres2)
        outres4 = self.layer4(outres3)
        # return out
        #
        out_fc = self.humanid(outres4)
        out_fc = out_fc.squeeze()
        out1 = self.fcc(out_fc)
        # # # #
        # #
        out2 = self.biometrics(outres4)
        out2 = out2.squeeze()
        out2 = F.relu(self.fcr(out2))

        return out1, out2, out_generation, out, outres1, outres2, outres3, outres4, out_fc
#     def forward(self, x):c
#         out_generation = self.generation(x)
#         out = self.conv1(out_generation)
#         out = self.bn1(out)
#         out = self.relu(out)
#         out = self.maxpool(out)
# #
# #
#         out = self.layer1(out)
#         out = self.layer2(out)
#         out = self.layer3(out)
#         out = self.layer4(out)
#         # return out
#         #
#         out1 = self.humanid(out)
#         out1 = out1.squeeze()
#         out1 = self.fcc(out1)
#         # # # #
#         # #
#         out2 = self.biometrics(out)
#         out2 = out2.squeeze()
#         out2 = F.relu(self.fcr(out2))
#
#         return out1, out2