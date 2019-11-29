# -*- encoding: utf-8 -*-
'''
@File    :   __init__.py.py    
@Contact :   whut.hexin@foxmail.com
@License :   (C)Copyright 2017-2020, HeXin

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2019/11/6 18:10   xin      1.0         None
'''

from .baseline import Baseline
from .mgn import MGN
from .mfn import MFN
from .pcb import PCB
from .small_mhn_pcb import MHN_smallPCB


def build_model(cfg, num_classes):
    if cfg.MODEL.NAME == "baseline":
        model = Baseline(num_classes, cfg.MODEL.LAST_STRIDE, cfg.MODEL.PRETRAIN_PATH,
                         backbone=cfg.MODEL.BACKBONE, use_dropout=cfg.MODEL.USE_DROPOUT)
    elif cfg.MODEL.NAME == "mgn":
        model = MGN(num_classes, cfg.MODEL.PRETRAIN_PATH, backbone=cfg.MODEL.BACKBONE)
    elif cfg.MODEL.NAME == "mfn":
        model = MFN(num_classes, cfg.MODEL.PRETRAIN_PATH, backbone=cfg.MODEL.BACKBONE)
    elif cfg.MODEL.NAME == "pcb":
        model = PCB(num_classes, cfg.MODEL.PRETRAIN_PATH, backbone=cfg.MODEL.BACKBONE)
    elif cfg.MODEL.NAME == "small_mhn_pcb":
        model = MHN_smallPCB(num_classes, cfg.MODEL.PRETRAIN_PATH, backbone=cfg.MODEL.BACKBONE)
    else:
        model = None
    return model

