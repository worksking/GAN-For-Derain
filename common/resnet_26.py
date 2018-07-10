import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo
import numpy as np



def conv3x3(in_planes, out_planes, stride=1,dilattion=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=1,
                     padding=1, dilation=1, bias=True)


class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride=1, downsample = None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride=1)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out += residual
        

        return out

class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=3):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(num_classes, 64, kernel_size=3, stride=1, padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
                
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 64, layers[1])
        self.layer3 = self._make_layer(block, 64, layers[2])
        self.layer4 = self._make_layer(block, 64, layers[3])
        #self.avgpool = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)
        
        self.conv = nn.Conv2d(64, num_classes, kernel_size=1, stride=1, bias=False)
        self.bn2 = nn.BatchNorm2d(num_classes)
        self.tanh = nn.Tanh()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                #m.weight.data.normal_(0, 0.02)                
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.fill_(0)

    def _make_layer(self, block, planes, blocks, stride=1):
        layers = []
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x, x_detail):
        
        out = self.conv1(x_detail)
        out = self.bn1(out)
        out = self.relu(out)
               
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        out = self.conv(out)
        residual = self.bn2(out)        
        out = x + residual         
        return self.tanh(out)


def resnet26(**kwargs):
    model = ResNet(BasicBlock, [3, 3, 3, 3], **kwargs)
    return model