import torch.nn as nn
import math

class AlexNet(nn.Module):

    def __init__(self, num_classes=1000):
        super(AlexNet, self).__init__()

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


        self.features = nn.Sequential(
            nn.Conv2d(6, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, 10),
            # nn.Dropout(),
            # nn.Linear(256 * 6 * 6, 1024),
            # nn.ReLU(inplace=True),
            # nn.Dropout(),
            # nn.Linear(1024, 512),
            # nn.ReLU(inplace=True),
            # nn.Linear(512, 2),
        )

    def forward(self, x):
        x = self.generation(x)
        x = self.features(x)
        x = x.view(x.size(0), 256 * 6 * 6)
        x = self.classifier(x)
        return x