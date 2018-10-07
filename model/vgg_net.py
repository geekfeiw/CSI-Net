import torch.nn as nn
import math



class VGG(nn.Module):

    def __init__(self, features, num_classes=1000, init_weights=False):
        super(VGG, self).__init__()

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
            nn.ConvTranspose2d(6, 3, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(3),
            nn.ReLU(),
        )

        self.features = features

        self.id = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 30),
        )

        self.bio = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4),
        )
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.generation(x)
        x = self.features(x)
        x = x.view(x.size(0), -1)
        out1 = self.id(x)
        out2 = self.bio(x)
        return out1, out2

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


cfg = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


# def vgg11(pretrained=False, **kwargs):
#     """VGG 11-layer model (configuration "A")
#     Args:
#         pretrained (bool): If True, returns a model pre-trained on ImageNet
#     """
#     if pretrained:
#         kwargs['init_weights'] = False
#     model = VGG(make_layers(cfg['A']), **kwargs)
#     if pretrained:
#         model.load_state_dict(model_zoo.load_url(model_urls['vgg11']))
#     return model
#
#
# def vgg11_bn(pretrained=False, **kwargs):
#     """VGG 11-layer model (configuration "A") with batch normalization
#     Args:
#         pretrained (bool): If True, returns a model pre-trained on ImageNet
#     """
#     if pretrained:
#         kwargs['init_weights'] = False
#     model = VGG(make_layers(cfg['A'], batch_norm=True), **kwargs)
#     if pretrained:
#         model.load_state_dict(model_zoo.load_url(model_urls['vgg11_bn']))
#     return model
#
#
# def vgg13(pretrained=False, **kwargs):
#     """VGG 13-layer model (configuration "B")
#     Args:
#         pretrained (bool): If True, returns a model pre-trained on ImageNet
#     """
#     if pretrained:
#         kwargs['init_weights'] = False
#     model = VGG(make_layers(cfg['B']), **kwargs)
#     if pretrained:
#         model.load_state_dict(model_zoo.load_url(model_urls['vgg13']))
#     return model
#
#
# def vgg13_bn(pretrained=False, **kwargs):
#     """VGG 13-layer model (configuration "B") with batch normalization
#     Args:
#         pretrained (bool): If True, returns a model pre-trained on ImageNet
#     """
#     if pretrained:
#         kwargs['init_weights'] = False
#     model = VGG(make_layers(cfg['B'], batch_norm=True), **kwargs)
#     if pretrained:
#         model.load_state_dict(model_zoo.load_url(model_urls['vgg13_bn']))
#     return model
#
#
# def vgg16(pretrained=False, **kwargs):
#     """VGG 16-layer model (configuration "D")
#     Args:
#         pretrained (bool): If True, returns a model pre-trained on ImageNet
#     """
#     if pretrained:
#         kwargs['init_weights'] = False
#     model = VGG(make_layers(cfg['D']), **kwargs)
#     if pretrained:
#         model.load_state_dict(model_zoo.load_url(model_urls['vgg16']))
#     return model
#
#
# def vgg16_bn(pretrained=False, **kwargs):
#     """VGG 16-layer model (configuration "D") with batch normalization
#     Args:
#         pretrained (bool): If True, returns a model pre-trained on ImageNet
#     """
#     if pretrained:
#         kwargs['init_weights'] = False
#     model = VGG(make_layers(cfg['D'], batch_norm=True), **kwargs)
#     if pretrained:
#         model.load_state_dict(model_zoo.load_url(model_urls['vgg16_bn']))
#     return model
#
#
# def vgg19(pretrained=False, **kwargs):
#     """VGG 19-layer model (configuration "E")
#     Args:
#         pretrained (bool): If True, returns a model pre-trained on ImageNet
#     """
#     if pretrained:
#         kwargs['init_weights'] = False
#     model = VGG(make_layers(cfg['E']), **kwargs)
#     if pretrained:
#         model.load_state_dict(model_zoo.load_url(model_urls['vgg19']))
#     return model
#
#
# def vgg19_bn(pretrained=False, **kwargs):
#     """VGG 19-layer model (configuration 'E') with batch normalization
#     Args:
#         pretrained (bool): If True, returns a model pre-trained on ImageNet
#     """
#     if pretrained:
#         kwargs['init_weights'] = False
#     model = VGG(make_layers(cfg['E'], batch_norm=True), **kwargs)
#     if pretrained:
#         model.load_state_dict(model_zoo.load_url(model_urls['vgg19_bn']))
#     return model