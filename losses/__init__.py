# -*- encoding: utf-8 -*-
'''
@File    :   __init__.py.py    
@Contact :   whut.hexin@foxmail.com
@License :   (C)Copyright 2017-2020, HeXin

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2019/11/6 18:10   xin      1.0         None
'''

import torch.nn as nn

from .triplet import TripletLoss, CrossEntropyLabelSmooth
from .advdiv_loss import AdvDivLoss
from .center_loss import CenterLoss
from .arcloss import ArcCos


class BaseLineLoss(nn.modules.loss._Loss):
    def __init__(self, num_classes, margin=0.3):
        super(BaseLineLoss, self).__init__()
        self.num_classes = num_classes
        self.margin = margin

    def forward(self, outputs, labels):
        score, feat = outputs
        cross_entropy_loss = CrossEntropyLabelSmooth(self.num_classes)
        triplet_loss = TripletLoss(self.margin)
        Triplet_Loss = triplet_loss(feat, labels)
        CrossEntropy_Loss = cross_entropy_loss(score, labels)
        loss_sum = Triplet_Loss + CrossEntropy_Loss

        print('\rtotal loss:%.2f  Triplet_Loss:%.2f  CrossEntropy_Loss:%.2f' % (
            loss_sum.data.cpu().numpy(),
            Triplet_Loss.data.cpu().numpy(),
            CrossEntropy_Loss.data.cpu().numpy()),
              end=' ')
        return loss_sum


class MGNLoss(nn.modules.loss._Loss):
    def __init__(self, num_classes, margin=0.3):
        super(MGNLoss, self).__init__()
        self.num_classes = num_classes
        self.margin = margin

    def forward(self, outputs, labels):
        cross_entropy_loss = CrossEntropyLabelSmooth(self.num_classes)
        triplet_loss = TripletLoss(self.margin)

        Triplet_Loss = [triplet_loss(output, labels) for output in outputs[:3]]
        Triplet_Loss = sum(Triplet_Loss) / len(Triplet_Loss)

        CrossEntropy_Loss = [cross_entropy_loss(output, labels) for output in outputs[3:]]
        CrossEntropy_Loss = sum(CrossEntropy_Loss) / len(CrossEntropy_Loss)

        loss_sum = Triplet_Loss + 2 * CrossEntropy_Loss

        print('\rtotal loss:%.2f  Triplet_Loss:%.2f  CrossEntropy_Loss:%.2f' % (
            loss_sum.data.cpu().numpy(),
            Triplet_Loss.data.cpu().numpy(),
            CrossEntropy_Loss.data.cpu().numpy()),
              end=' ')
        return loss_sum


class MFNLoss(nn.modules.loss._Loss):
    def __init__(self, class_nums, margin=0.4):
        super(MFNLoss, self).__init__()
        self.class_nums = class_nums
        self.margin = margin

    def forward(self, outputs, labels):

        cross_entropy_loss = CrossEntropyLabelSmooth(num_classes=self.class_nums)
        triplet_loss = TripletLoss(self.margin)

        Triplet_Loss = [triplet_loss(output, labels) for output in outputs[4:]]
        Triplet_Loss = sum(Triplet_Loss) / len(Triplet_Loss)

        CrossEntropy_Loss = [cross_entropy_loss(output, labels) for output in outputs[:4]]
        CrossEntropy_Loss = sum(CrossEntropy_Loss) / len(CrossEntropy_Loss)


        loss_sum = Triplet_Loss + CrossEntropy_Loss
        print('\rtotal loss:%.2f  Triplet_Loss:%.2f  CrossEntropy_Loss:%.2f' % (
            loss_sum.data.cpu().numpy(),
            Triplet_Loss.data.cpu().numpy(),
            CrossEntropy_Loss.data.cpu().numpy()),
              end=' ')
        return loss_sum

class PCBLoss(nn.modules.loss._Loss):
    def __init__(self, class_nums):
        super(PCBLoss, self).__init__()
        self.class_nums = class_nums


    def forward(self, outputs, labels):

        cross_entropy_loss = CrossEntropyLabelSmooth(num_classes=self.class_nums)

        _, _, _, logits_list, _ = outputs

        CrossEntropy_Loss = [cross_entropy_loss(output, labels) for output in logits_list]
        CrossEntropy_Loss = sum(CrossEntropy_Loss) / len(CrossEntropy_Loss)

        loss_sum = CrossEntropy_Loss

        print('\rtotal loss:%.2f  CrossEntropy_Loss:%.2f' % (
            loss_sum.data.cpu().numpy(),
            CrossEntropy_Loss.data.cpu().numpy()),
              end=' ')
        return loss_sum
class MHNPCBLoss(nn.modules.loss._Loss):
    def __init__(self, class_nums, alpha=1.0):
        super(MHNPCBLoss, self).__init__()
        self.class_nums = class_nums
        self.alpha = alpha


    def forward(self, outputs, labels):

        cross_entropy_loss = CrossEntropyLabelSmooth(num_classes=self.class_nums)
        logits_list, fea = outputs
        adv_div_loss = AdvDivLoss(len(fea))

        CrossEntropy_Loss = [cross_entropy_loss(output, labels) for output in logits_list]
        CrossEntropy_Loss = sum(CrossEntropy_Loss) / len(CrossEntropy_Loss)

        AdvDiv_Loss = [adv_div_loss(output) for output in fea]
        AdvDiv_Loss = sum(AdvDiv_Loss) / len(AdvDiv_Loss)

        loss_sum = CrossEntropy_Loss + AdvDiv_Loss*self.alpha

        print('\rtotal loss:%.2f  AdvDiv_Loss:%.2f  CrossEntropy_Loss:%.2f' % (
            loss_sum.data.cpu().numpy(),
            AdvDiv_Loss.data.cpu().numpy(),
            CrossEntropy_Loss.data.cpu().numpy()),
              end=' ')
        return loss_sum


def make_loss(cfg, num_classes):
    if cfg.MODEL.NAME == "baseline":
        loss = BaseLineLoss(num_classes)
    elif cfg.MODEL.NAME == "mgn":
        loss = MGNLoss(num_classes)
    elif cfg.MODEL.NAME == "mfn":
        loss= MFNLoss(num_classes)
    elif cfg.MODEL.NAME == "pcb":
        loss = PCBLoss(num_classes)
    elif cfg.MODEL.NAME == "small_mhn_pcb":
        loss = MHNPCBLoss(num_classes)
    else:
        loss = None
    return loss

