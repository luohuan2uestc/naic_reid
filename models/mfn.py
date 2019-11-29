# -*- encoding: utf-8 -*-
'''
@File    :   mfn.py    
@Contact :   whut.hexin@foxmail.com
@License :   (C)Copyright 2017-2020, HeXin

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2019/11/6 18:50   xin      1.0         None
'''

import torch
import torch.nn as nn
from torch.nn import init

from .backbones.senet import se_resnext50_32x4d
from .backbones.resnet import ResNet
from .backbones.resnet_ibn_a import resnet50_ibn_a



######################################################################
def weights_init_kaiming(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv2d') != -1:
        init.kaiming_normal(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        init.kaiming_normal(m.weight.data, a=0, mode='fan_out')
        # init.constant(m.bias.data, 0.0)
    elif classname.find('BatchNorm1d') != -1:
        init.normal(m.weight.data, 1.0, 0.02)
        init.constant(m.bias.data, 0.0)
    elif classname.find('BatchNorm2d') != -1:
        init.normal(m.weight.data, 1.0, 0.02)
        init.constant(m.bias.data, 0.0)


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        init.normal(m.weight.data, std=0.001)
        # init.constant(m.bias.data, 0.0)


# Defines the new fc layer and classification layer
# |--Linear--|--bn--|--relu--|--Linear--|
class ClassBlock(nn.Module):
    def __init__(self, j, input_dim, class_num, h, stride, dropout=False, relu=False):
        super(ClassBlock, self).__init__()
        add_block = []
        add_block1 = []
        add_block1 += [nn.BatchNorm2d(j)]
        add_block1 += [nn.ReLU(inplace=True)]
        add_block1 += [nn.Conv2d(j, input_dim, kernel_size=h, stride=stride, bias=False)]

        add_block += [nn.BatchNorm1d(input_dim)]
        # if relu:
        # add_block += [nn.LeakyReLU(0.1)]
        # if dropout:
        # add_block += [nn.Dropout(p=0.5)]
        add_block = nn.Sequential(*add_block)
        add_block.apply(weights_init_kaiming)
        add_block1 = nn.Sequential(*add_block1)
        add_block1.apply(weights_init_kaiming)

        classifier = []
        classifier += [nn.Linear(input_dim, class_num, bias=False)]
        classifier = nn.Sequential(*classifier)
        classifier.apply(weights_init_classifier)
        self.add_block1 = add_block1
        self.add_block = add_block

        self.classifier = classifier
        self.avgpool_1 = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        x = self.add_block1(x)
        x = self.avgpool_1(x)
        x = torch.squeeze(x)
        x1 = self.add_block(x)
        x2 = self.classifier(x1)
        return x, x1, x2


#
class ClassBlock1(nn.Module):
    def __init__(self, input_dim, class_num, dropout=False, relu=True, num_bottleneck=512):
        super(ClassBlock1, self).__init__()
        add_block = []
        add_block1 = []
        add_block += [nn.BatchNorm1d(input_dim)]
        if relu:
            add_block += [nn.LeakyReLU(0.1)]
        if dropout:
            add_block += [nn.Dropout(p=0.5)]
        add_block1 += [nn.Linear(input_dim, num_bottleneck, bias=False)]
        add_block1 += [nn.BatchNorm1d(num_bottleneck)]
        # if relu:
        # add_block1 += [nn.LeakyReLU(0.1)]

        add_block = nn.Sequential(*add_block)
        add_block.apply(weights_init_kaiming)
        add_block1 = nn.Sequential(*add_block1)
        add_block1.apply(weights_init_kaiming)
        classifier = []
        classifier += [nn.Linear(num_bottleneck, class_num, bias=False)]
        classifier = nn.Sequential(*classifier)
        classifier.apply(weights_init_classifier)

        self.add_block = add_block
        self.add_block1 = add_block1
        self.classifier = classifier

    def forward(self, x):
        x = self.add_block(x)
        x1 = self.add_block1(x)
        x2 = self.classifier(x1)
        return x1, x2


# Define the ResNet50-based Model
class MFN(nn.Module):

    def __init__(self, num_classes, model_path, backbone='resnet50'):
        super(MFN, self).__init__()
        if backbone == "resnet50":
            model_ft = ResNet()
        elif backbone == "resnet50_ibn_a":
            model_ft = resnet50_ibn_a(1)
        elif backbone == 'resnext50':
            model_ft = se_resnext50_32x4d(pretrained=False)
        model_ft.load_param(model_path)
        # model_ft = se_resnext50_32x4d(pretrained='imagenet')
        # avg pooling to global pooling
        model_ft.avg_pool = nn.Sequential()
        model_ft.last_linear = nn.Sequential()
        # remove the final downsample
        # self.model.fc = nn.Sequential()
        self.model = model_ft
        # self.avgpool_1 = nn.AdaptiveAvgPool2d((1,1))
        # self.avgpool_2 = nn.AdaptiveAvgPool2d((1,1))
        self.maxgpool_1 = nn.AdaptiveMaxPool2d((1, 1))
        self.maxgpool_2 = nn.AdaptiveMaxPool2d((1, 1))
        self.avggpool_1 = nn.AvgPool2d(kernel_size=3, stride=1, padding=0)
        self.avggpool_2 = nn.AvgPool2d(kernel_size=3, stride=1, padding=0)
        self.model.layer4[0].downsample[0].stride = (1, 1)
        self.model.layer4[0].conv2.stride = (1, 1)
        self.classifier = ClassBlock(2048, 512, num_classes, 1, 1)
        self.classifier1 = ClassBlock(2048, 512, num_classes, 1, 2)
        self.classifier2 = ClassBlock1(2048, num_classes)
        self.classifier3 = ClassBlock1(1024, num_classes)


    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)

        x = self.model.layer2(x)

        x0 = self.model.layer3(x)
        x = self.model.layer4(x0)
        x10, x1, x2 = self.classifier(x)
        x30, x3, x4 = self.classifier1(x)
        x_2 = self.avggpool_1(x)
        x_2 = self.maxgpool_1(x_2)

        # x_2 = self.avgpool_1(x_2)
        x_2 = torch.squeeze(x_2)
        x5, x6 = self.classifier2(x_2)
        x0 = self.avggpool_1(x0)
        x0 = self.maxgpool_2(x0)
        # x0 = self.avgpool_2(x0)
        x0 = torch.squeeze(x0)

        x7, x8 = self.classifier3(x0)
        if self.training:
            return x2, x4, x6, x8, x_2, x0, x10, x30
        else:
            return torch.cat((x1, x3, x5, x7), 1)


