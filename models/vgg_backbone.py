import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torch.utils.model_zoo as model_zoo

class VGGBackbone(nn.Module):
    def __init__(self, pretrained=True):
        super(VGGBackbone, self).__init__()
        layers = self._make_layers()
        self.layer1 = nn.Sequential(*layers[:33])
        self.layer2 = nn.Sequential(*layers[33:43])
        if pretrained:
            self.load_pre_trained_weights()

    def load_pre_trained_weights(self):
        print('Loading Pytorch pretrained weights...')
        pretrained_dict = model_zoo.load_url('https://download.pytorch.org/models/vgg16_bn-6c64b313.pth')
        pretrained_dict = {k.replace('features.','',1) : v for k, v in pretrained_dict.items() if 'features' in k}
        self.layer1.load_state_dict({k : v for k, v in pretrained_dict.items() if int(k.split('.')[0]) < 33})
        self.layer2.load_state_dict({self._rename(k, 33) : v for k, v in pretrained_dict.items() if int(k.split('.')[0]) >= 33 and int(k.split('.')[0]) < 43})

    def _make_layers(self):
        layers = []
        in_channels = 3
        for v in [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                in_channels = v
        return layers

    def _rename(self, k, num):
        a = int(k.split('.')[0])
        return k.replace('{}.'.format(a), '{}.'.format(a-num), 1)

    def forward(self, x):
        out1 = self.layer1(x)
        out2 = self.layer2(out1)
        return out1, out2
