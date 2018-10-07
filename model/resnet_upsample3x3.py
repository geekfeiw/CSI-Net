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

# Residual Block
class ResidualBlock(nn.Module):
    expansion = 1
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

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels * 4)
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



# ResNet Module
class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes):
        super(ResNet, self).__init__()
        self.conv3to4 = nn.ConvTranspose2d(3, 3, kernel_size=4, stride=1, padding=1, bias=False)
        self.upsampling = nn.Upsample(mode='bilinear', scale_factor=56, align_corners=False)
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
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
            nn.Conv2d(512 * block.expansion, 512 * block.expansion, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(512 * block.expansion),
            nn.ReLU(),

            nn.Conv2d(512 * block.expansion, 256 * block.expansion, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256 * block.expansion),
            nn.ReLU(),
            nn.AvgPool2d(2),
        )

        self.fcc = nn.Linear(256 * block.expansion, num_classes)

        self.biometrics = nn.Sequential(
            nn.Conv2d(512 * block.expansion, 512 * block.expansion, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(512 * block.expansion),
            nn.ReLU(),

            nn.Conv2d(512 * block.expansion, 256 * block.expansion, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256 * block.expansion),
            nn.ReLU(),
            nn.AvgPool2d(2),
        )

        self.fcr = nn.Linear(256 * block.expansion, 4)

    def make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if (stride != 1) or (self.in_channels != out_channels * block.expansion):
            downsample = nn.Sequential(
                conv3x3(self.in_channels, out_channels * block.expansion, stride=stride),
                nn.BatchNorm2d(out_channels * block.expansion))
        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv3to4(x)
        out = self.upsampling(out)
        out = self.conv1(out)
        #
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        #
        # #
        # out = self.avgpool7(out)
        out1 = self.humanid(out)
        out1 = out1.squeeze()
        out1 = self.fcc(out1)
        # # # #
        # #
        out2 = self.biometrics(out)
        out2 = out2.squeeze()
        out2 = F.relu(self.fcr(out2))

        return out1, out2