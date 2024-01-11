"""
    This is ResNet20 model.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

__all__ = ['resnet20', 'resnet32', 'resnet44', 'resnet56', 'resnet110', 'resnet1202']

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    # TODO feed the inputs
    # return nn.Conv2d(in, out, kernel, stride=,
                     # padding=, groups=, bias=, dilation=)

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    # TODO feed the inputs
    # return nn.Conv2d(in, out, kernel, stride=, bias=)


class BasicBlock(nn.Module):
    expansion = 1
    __constants__ = ['downsample']

    def __init__(self, inplanes, planes, stride=1, downsample=None, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        # TODO implement the downsample block regarding the residual connection & downsample function
        # identity =

        # out =
        # out =
        # out =
        # out =
        # out =

        # if self.downsample is not None:
            # identity =

        # out +=
        # out =

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=10, norm_layer=None, **kwargs):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 16

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU()

        # TODO build the layers according to the elements in the 'layers' list regrading the strides
        # self.layer1 = self._make_layer(block, 16, )
        # self.layer2 = self._make_layer(block, 32, , stride=)
        # self.layer3 = self._make_layer(block, 64, , stride=)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, num_classes)

    def _make_layer(self, block, planes, blocks, stride=1):
        # TODO try to understand the code below (no need to modify)
        norm_layer = self._norm_layer
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x):
        # TODO implement the resnet block
        # x =
        # x =
        # x =

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        # x =
        x = x.view(x.size(0), -1)
        # x =

        return x

def resnet20():
    # TODO how many layers are assigned for resnet20?
    # return ResNet(BasicBlock, [, , ])
    pass

def resnet32():
    return ResNet(BasicBlock, [5, 5, 5])

def resnet44():
    return ResNet(BasicBlock, [7, 7, 7])

def resnet56():
    return ResNet(BasicBlock, [9, 9, 9])

def resnet110():
    return ResNet(BasicBlock, [18, 18, 18])

def resnet1202():
    return ResNet(BasicBlock, [200, 200, 200])
