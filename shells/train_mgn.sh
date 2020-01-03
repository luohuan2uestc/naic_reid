#!/usr/bin/env bash

# PRETRAIN=../weights/r50_ibn_a.pth
# DATA_DIR='/share/Dataset/PReID/pre/'
# SAVE_DIR='../work_dirs/exp3-mgn_sgd-resnet50ibnls2-384x128-bs12x4-warmup10-flip-pad10-meanstd-lbsm/' #(h, w)
# CUDA_VISIBLE_DEVICES=3,4 python train.py --config_file='configs/preid/mgn_sgd.yml' \
#     SOLVER.SYNCBN "False" \
#     MODEL.NAME "mgn" MODEL.BACKBONE "('resnet50_ibn_a')" \
#     DATASETS.DATA_PATH "('${DATA_DIR}')" DATASETS.TRAIN_PATH "train"\
#     MODEL.PRETRAIN_PATH "('${PRETRAIN}')"  \
#     OUTPUT_DIR "('${SAVE_DIR}')" 

# PRETRAIN=../weights/r50_ibn_a.pth
# DATA_DIR='/data/Dataset/PReID/pre/'
# SAVE_DIR='../work_dirs/exp3-mgn-resnet50ibnls2-384x128-bs12x4-warmup10-flip-pad10-meanstd-lbsm/' #(h, w)
# CUDA_VISIBLE_DEVICES=2,3 python train.py --config_file='configs/preid/mgn.yml' \
#     SOLVER.SYNCBN "False" \
#     MODEL.NAME "mgn" MODEL.BACKBONE "('resnet50_ibn_a')" \
#     DATASETS.DATA_PATH "('${DATA_DIR}')" DATASETS.TRAIN_PATH "train"\
#     MODEL.PRETRAIN_PATH "('${PRETRAIN}')"  \
#     OUTPUT_DIR "('${SAVE_DIR}')" 

# PRETRAIN=../weights/r50_ibn_a.pth
# DATA_DIR='/data/Dataset/PReID/pre/'
# SAVE_DIR='../work_dirs/exp3-mgn-resnet50ibnls2-384x128-bs12x4-warmup10-flip-pad10-meanstd-lbsm-center0005/' #(h, w)
# CUDA_VISIBLE_DEVICES=4,5 python train.py --config_file='configs/preid/mgn_center.yml' \
#     SOLVER.SYNCBN "False" \
#     MODEL.NAME "mgn" MODEL.BACKBONE "('resnet50_ibn_a')" \
#     DATASETS.DATA_PATH "('${DATA_DIR}')" DATASETS.TRAIN_PATH "train"\
#     MODEL.PRETRAIN_PATH "('${PRETRAIN}')"  \
#     OUTPUT_DIR "('${SAVE_DIR}')" 


# PRETRAIN=../weights/r50_ibn_a.pth
# DATA_DIR='/data/Dataset/PReID/pre/'
# SAVE_DIR='../work_dirs/exp4-mgn-resnet50ibnls2-384x128-bs12x4-warmup10-flip-pad10-meanstd-lbsm-center0005/' #(h, w)
# CUDA_VISIBLE_DEVICES=2,3 python train.py --config_file='configs/preid/mgn_center.yml' \
#     MODEL.NAME "mgn" MODEL.BACKBONE "('resnet50_ibn_a')" \
#     DATASETS.DATA_PATH "('${DATA_DIR}')" DATASETS.TRAIN_PATH "train"\
#     MODEL.PRETRAIN_PATH "('${PRETRAIN}')"  \
#     OUTPUT_DIR "('${SAVE_DIR}')" 

# PRETRAIN=../weights/r50_ibn_a.pth
# DATA_DIR='/share/Dataset/PReID/pre/'
# SAVE_DIR='../work_dirs/exp4-mgn-resnet50ibnls2-384x128-bs12x4-warmup10-flip-pad10-meanstd-lbsm-center0005-tpl12/' #(h, w)
# CUDA_VISIBLE_DEVICES=4,7 python train.py --config_file='configs/preid/mgn_center.yml' \
#     SOLVER.MARGIN "1.2" \
#     MODEL.NAME "mgn" MODEL.BACKBONE "('resnet50_ibn_a')" \
#     DATASETS.DATA_PATH "('${DATA_DIR}')" DATASETS.TRAIN_PATH "train"\
#     MODEL.PRETRAIN_PATH "('${PRETRAIN}')"  \
#     OUTPUT_DIR "('${SAVE_DIR}')" 

# PRETRAIN=../weights/resnext101_ibn_a.pth.tar
# DATA_DIR='/data/Dataset/PReID/pre/'
# SAVE_DIR='../work_dirs/exp4-mgn-resnext101ibnls1-384x128-bs12x8-warmup10-flip-pad10-meanstd-lbsm-center0005-tpl12/' #(h, w)
# CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py --config_file='configs/preid/mgn_center.yml' \
#     SOLVER.MARGIN "1.2" DATALOADER.NUM_INSTANCE "8"\
#     MODEL.NAME "mgn" MODEL.BACKBONE "('resnext101_ibn_a')" \
#     DATASETS.DATA_PATH "('${DATA_DIR}')" DATASETS.TRAIN_PATH "train"\
#     MODEL.PRETRAIN_PATH "('${PRETRAIN}')"  \
#     OUTPUT_DIR "('${SAVE_DIR}')" 

# [20191128]
# PRETRAIN=../weights/r50_ibn_a.pth
# DATA_DIR='/share/Dataset/PReID/pre/'
# SAVE_DIR='../work_dirs/exp5-mgn-resnet50ibnls2-384x128-bs12x8-warmup10-flip-pad10-meanstd-lbsm-layer3_-5/' #(h, w)
# CUDA_VISIBLE_DEVICES=2,3 python train.py --config_file='configs/preid/mgn.yml' \
#     MODEL.MGN.NUM_SHARE_LAYER3 "-5" MODEL.NAME "mgn" MODEL.BACKBONE "('resnet50_ibn_a')" \
#     SOLVER.IMS_PER_BATCH "96" DATALOADER.NUM_INSTANCE "8" \
#     DATASETS.DATA_PATH "('${DATA_DIR}')" DATASETS.TRAIN_PATH "train"\
#     MODEL.PRETRAIN_PATH "('${PRETRAIN}')"  \
#     OUTPUT_DIR "('${SAVE_DIR}')" 

# PRETRAIN=../weights/resnext101_ibn_a.pth.tar
# DATA_DIR='/share/Dataset/PReID/pre/'
# SAVE_DIR='../work_dirs/exp5-mgn-resnext101ibnls2-384x128-bs12x8-warmup10-flip-pad10-meanstd-lbsm-layer3_-5/' #(h, w)
# CUDA_VISIBLE_DEVICES=4,5,6 python train.py --config_file='configs/preid/mgn.yml' \
#     MODEL.MGN.NUM_SHARE_LAYER3 "-5" MODEL.NAME "mgn" MODEL.BACKBONE "('resnext101_ibn_a')" \
#     SOLVER.IMS_PER_BATCH "96" DATALOADER.NUM_INSTANCE "8" \
#     DATASETS.DATA_PATH "('${DATA_DIR}')" DATASETS.TRAIN_PATH "train"\
#     MODEL.PRETRAIN_PATH "('${PRETRAIN}')"  \
#     OUTPUT_DIR "('${SAVE_DIR}')" 

# [20191129]

# PRETRAIN=../weights/r50_ibn_a.pth
# DATA_DIR='/share/Dataset/PReID/pre/'
# SAVE_DIR='../work_dirs/exp6-mgn-resnet50ibnls2-384x128-bs12x8-warmup10-flip-pad10-meanstd-lbsm-layer3_-5/' #(h, w)
# CUDA_VISIBLE_DEVICES=2,3 python train.py --config_file='configs/preid/mgn.yml' \
#     MODEL.MGN.NUM_SHARE_LAYER3 "-5" MODEL.NAME "mgn" MODEL.BACKBONE "('resnet50_ibn_a')" \
#     MODEL.OPT_LEVEL "O1" SOLVER.IMS_PER_BATCH "96" DATALOADER.NUM_INSTANCE "8" \
#     DATASETS.DATA_PATH "('${DATA_DIR}')" DATASETS.TRAIN_PATH "train"\
#     MODEL.PRETRAIN_PATH "('${PRETRAIN}')"  \
#     OUTPUT_DIR "('${SAVE_DIR}')" 

# PRETRAIN=../weights/r50_ibn_a.pth
# DATA_DIR='/share/Dataset/PReID/pre/'
# SAVE_DIR='../work_dirs/exp6-mgn-resnet50ibnls2-384x128-bs12x8-warmup10-flip-pad10-meanstd-lbsm-layer3_-5-single_gpu/' #(h, w)
# CUDA_VISIBLE_DEVICES=4 python train.py --config_file='configs/preid/mgn.yml' \
#     MODEL.MGN.NUM_SHARE_LAYER3 "-5" MODEL.NAME "mgn" MODEL.BACKBONE "('resnet50_ibn_a')" \
#     MODEL.OPT_LEVEL "O1" SOLVER.IMS_PER_BATCH "96" DATALOADER.NUM_INSTANCE "8" \
#     DATASETS.DATA_PATH "('${DATA_DIR}')" DATASETS.TRAIN_PATH "train"\
#     MODEL.PRETRAIN_PATH "('${PRETRAIN}')"  \
#     OUTPUT_DIR "('${SAVE_DIR}')" 

# PRETRAIN=../weights/r50_ibn_a.pth
# DATA_DIR='/share/Dataset/PReID/pre/'
# SAVE_DIR='../work_dirs/exp6-mgn-resnet50ibnls2-384x128-bs12x4-warmup10-flip-pad10-meanstd-lbsm-layer3_-5-center0005-single_gpu/' #(h, w)
# CUDA_VISIBLE_DEVICES=5 python train.py --config_file='configs/preid/mgn_center.yml' \
#     MODEL.MGN.NUM_SHARE_LAYER3 "-5" MODEL.NAME "mgn" MODEL.BACKBONE "('resnet50_ibn_a')" \
#     MODEL.OPT_LEVEL "O1" SOLVER.IMS_PER_BATCH "96" DATALOADER.NUM_INSTANCE "8" \
#     DATASETS.DATA_PATH "('${DATA_DIR}')" DATASETS.TRAIN_PATH "train"\
#     MODEL.PRETRAIN_PATH "('${PRETRAIN}')"  \
#     OUTPUT_DIR "('${SAVE_DIR}')" 

# [20191201]
# PRETRAIN=../weights/r50_ibn_a.pth
# DATA_DIR='/share/Dataset/PReID/pre/'
# SAVE_DIR='../work_dirs/exp5-mgnbnneck-resnet50ibnls2-384x128-bs12x8-warmup10-flip-pad10-meanstd-lbsm-layer3_-5/' #(h, w)
# CUDA_VISIBLE_DEVICES=0,1 python train.py --config_file='configs/preid/mgn.yml' \
#     MODEL.MGN.NUM_SHARE_LAYER3 "-5" MODEL.NAME "mgn_bnneck" MODEL.BACKBONE "('resnet50_ibn_a')" \
#     SOLVER.IMS_PER_BATCH "96" DATALOADER.NUM_INSTANCE "8" \
#     DATASETS.DATA_PATH "('${DATA_DIR}')" DATASETS.TRAIN_PATH "train"\
#     MODEL.PRETRAIN_PATH "('${PRETRAIN}')"  \
#     OUTPUT_DIR "('${SAVE_DIR}')" 

# PRETRAIN=../weights/r50_ibn_a.pth
# DATA_DIR='/share/Dataset/PReID/pre/'
# SAVE_DIR='../work_dirs/exp5-mgnbnneck-resnet50ibnls2-384x128-bs12x8-warmup10-flip-pad10-meanstd-lbsm-layer3_-5-avg_max_mean/' #(h, w)
# CUDA_VISIBLE_DEVICES=2,3 python train.py --config_file='configs/preid/mgn.yml' \
#     MODEL.MGN.POOL_TYPE "avg_max_mean" MODEL.MGN.NUM_SHARE_LAYER3 "-5" MODEL.NAME "mgn_bnneck" MODEL.BACKBONE "('resnet50_ibn_a')" \
#     SOLVER.IMS_PER_BATCH "96" DATALOADER.NUM_INSTANCE "8" \
#     DATASETS.DATA_PATH "('${DATA_DIR}')" DATASETS.TRAIN_PATH "train"\
#     MODEL.PRETRAIN_PATH "('${PRETRAIN}')"  \
#     OUTPUT_DIR "('${SAVE_DIR}')" 

# [20191203]
# PRETRAIN=../weights/r50_ibn_a.pth
# DATA_DIR='/share/Dataset/PReID/pre/'
# SAVE_DIR='../work_dirs/exp5-mgn-resnet50ibnls2-384x128-bs12x8-warmup10-flip-pad10-meanstd-lbsm-layer3_-5-center0005/' #(h, w)
# CUDA_VISIBLE_DEVICES=0,1 python train.py --config_file='configs/preid/mgn_center.yml' \
#     MODEL.MGN.NUM_SHARE_LAYER3 "-5" MODEL.NAME "mgn" MODEL.BACKBONE "('resnet50_ibn_a')" \
#     SOLVER.IMS_PER_BATCH "96" DATALOADER.NUM_INSTANCE "8" \
#     DATASETS.DATA_PATH "('${DATA_DIR}')" DATASETS.TRAIN_PATH "train"\
#     MODEL.PRETRAIN_PATH "('${PRETRAIN}')"  \
#     OUTPUT_DIR "('${SAVE_DIR}')" 

# PRETRAIN=../weights/r50_ibn_a.pth
# DATA_DIR='/share/Dataset/PReID/pre/'
# SAVE_DIR='../work_dirs/exp5-mgn-resnet50ibnls2-384x128-bs12x8-warmup10-flip-pad10-meanstd-lbsm-layer3_-5-center0005_05/' #(h, w)
# CUDA_VISIBLE_DEVICES=2,3 python train.py --config_file='configs/preid/mgn_center.yml' \
#     SOLVER.CENTER_LOSS.ALPHA "0.5" SOLVER.CENTER_LOSS.WEIGHT "0.0005"\
#     MODEL.MGN.NUM_SHARE_LAYER3 "-5" MODEL.NAME "mgn" MODEL.BACKBONE "('resnet50_ibn_a')" \
#     SOLVER.IMS_PER_BATCH "96" DATALOADER.NUM_INSTANCE "8" \
#     DATASETS.DATA_PATH "('${DATA_DIR}')" DATASETS.TRAIN_PATH "train"\
#     MODEL.PRETRAIN_PATH "('${PRETRAIN}')"  \
#     OUTPUT_DIR "('${SAVE_DIR}')" 

# modify mgnloss manually
# PRETRAIN=../weights/r50_ibn_a.pth
# DATA_DIR='/share/Dataset/PReID/pre/'
# SAVE_DIR='../work_dirs/exp5-mgn-resnet50ibnls2-384x128-bs12x8-warmup10-flip-pad10-meanstd-lbsm-layer3_-5-drop1celoss/' #(h, w)
# CUDA_VISIBLE_DEVICES=2,3 python train.py --config_file='configs/preid/mgn.yml' \
#     MODEL.MGN.NUM_SHARE_LAYER3 "-5" MODEL.NAME "mgn" MODEL.BACKBONE "('resnet50_ibn_a')" \
#     SOLVER.IMS_PER_BATCH "96" DATALOADER.NUM_INSTANCE "8" \
#     DATASETS.DATA_PATH "('${DATA_DIR}')" DATASETS.TRAIN_PATH "train"\
#     MODEL.PRETRAIN_PATH "('${PRETRAIN}')"  \
#     OUTPUT_DIR "('${SAVE_DIR}')" 

# PRETRAIN=../weights/r50_ibn_a.pth
# DATA_DIR='/share/Dataset/PReID/pre/'
# SAVE_DIR='../work_dirs/exp5-mgn-resnet50ibnls2-384x128-bs12x8-warmup10-flip-pad10-meanstd-lbsm-layer3_-5-erase0502/' #(h, w)
# CUDA_VISIBLE_DEVICES=2,3 python train.py --config_file='configs/preid/mgn.yml' \
#     INPUT.RE_PROB "0.5" INPUT.RE_MAX_RATIO "0.2" \
#     MODEL.MGN.NUM_SHARE_LAYER3 "-5" MODEL.NAME "mgn" MODEL.BACKBONE "('resnet50_ibn_a')" \
#     SOLVER.IMS_PER_BATCH "96" DATALOADER.NUM_INSTANCE "8" \
#     DATASETS.DATA_PATH "('${DATA_DIR}')" DATASETS.TRAIN_PATH "train"\
#     MODEL.PRETRAIN_PATH "('${PRETRAIN}')"  \
#     OUTPUT_DIR "('${SAVE_DIR}')" 

# PRETRAIN=../weights/r50_ibn_a.pth
# DATA_DIR='/share/Dataset/PReID/pre/'
# SAVE_DIR='../work_dirs/exp5-mgn-resnet50ibnls2-384x128-bs12x8-warmup10-flip-pad10-meanstd-lbsm-layer3_-5-erase0502-gem_3/' #(h, w)
# CUDA_VISIBLE_DEVICES=0,1 python train.py --config_file='configs/preid/mgn.yml' \
#     INPUT.RE_PROB "0.5" INPUT.RE_MAX_RATIO "0.2" \
#     MODEL.MGN.NUM_SHARE_LAYER3 "-5" MODEL.NAME "mgn" MODEL.BACKBONE "('resnet50_ibn_a')" MODEL.MGN.POOL_TYPE "gem_3"\
#     SOLVER.IMS_PER_BATCH "96" DATALOADER.NUM_INSTANCE "8" \
#     DATASETS.DATA_PATH "('${DATA_DIR}')" DATASETS.TRAIN_PATH "train"\
#     MODEL.PRETRAIN_PATH "('${PRETRAIN}')"  \
#     OUTPUT_DIR "('${SAVE_DIR}')" 

# SAVE_DIR='../work_dirs/debug/' #(h, w)
# CUDA_VISIBLE_DEVICES=4,5 python train.py --config_file='configs/preid/mgn.yml' \
#     MODEL.MGN.NUM_SHARE_LAYER3 "-1" MODEL.NAME "mgn" MODEL.BACKBONE "('resnet50_ibn_a')" \
#     SOLVER.IMS_PER_BATCH "64" DATALOADER.NUM_INSTANCE "4" \
#     DATASETS.DATA_PATH "('${DATA_DIR}')" DATASETS.TRAIN_PATH "train"\
#     MODEL.PRETRAIN_PATH "('${PRETRAIN}')"  \
#     OUTPUT_DIR "('${SAVE_DIR}')" 

PRETRAIN=../weights/r50_ibn_a.pth
DATA_DIR='/data/Dataset/PReID/pre/'
SAVE_DIR='../work_dirs/exp5-mgn-resnet50ibnls2-384x128-bs12x8-warmup10-flip-pad10-meanstd-lbsm-layer3_-5-erase0502-allgem_3/' #(h, w)
CUDA_VISIBLE_DEVICES=1 python train.py --config_file='configs/preid/mgn.yml' \
    SOLVER.IMS_PER_BATCH "96" DATALOADER.NUM_INSTANCE "8" \
    MODEL.MGN.NUM_SHARE_LAYER3 "-5" MODEL.NAME "mgn" MODEL.BACKBONE "('resnet50_ibn_a')" MODEL.MGN.POOL_TYPE "gem_3" MODEL.MGN.PART_POOL_TYPE "gem_3"\
    DATASETS.DATA_PATH "('${DATA_DIR}')" DATASETS.TRAIN_PATH "train"\
    MODEL.PRETRAIN_PATH "('${PRETRAIN}')"  \
    OUTPUT_DIR "('${SAVE_DIR}')" 
