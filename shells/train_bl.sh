#!/usr/bin/env bash

# PRETRAIN=../weights/r50_ibn_a.pth
# DATA_DIR='/share/Dataset/PReID/pre/'
# SAVE_DIR='../work_dirs/exp4-baseline-resnet50ibnls1-256x128-bs16x4-warmup10-flip-pad10-meanstd-erase0502-lbsm/' #(h, w)
# CUDA_VISIBLE_DEVICES=6 python train.py --config_file='configs/preid/baseline.yml' \
#     MODEL.NAME "baseline" MODEL.BACKBONE "('resnet50_ibn_a')" \
#     DATASETS.DATA_PATH "('${DATA_DIR}')" DATASETS.TRAIN_PATH "train"\
#     MODEL.PRETRAIN_PATH "('${PRETRAIN}')"  \
#     OUTPUT_DIR "('${SAVE_DIR}')" 

# PRETRAIN=../weights/r50_ibn_a.pth
# DATA_DIR='/share/Dataset/PReID/pre/'
# SAVE_DIR='../work_dirs/exp4-baseline-resnet50ibnls1-256x128-bs16x4-warmup10-flip-pad10-meanstd-erase0502-lbsm-gem_3/' #(h, w)
# CUDA_VISIBLE_DEVICES=7 python train.py --config_file='configs/preid/baseline.yml' \
#     MODEL.NAME "baseline" MODEL.BACKBONE "('resnet50_ibn_a')" MODEL.BASELINE.POOL_TYPE "gem_3"\
#     DATASETS.DATA_PATH "('${DATA_DIR}')" DATASETS.TRAIN_PATH "train"\
#     MODEL.PRETRAIN_PATH "('${PRETRAIN}')"  \
#     OUTPUT_DIR "('${SAVE_DIR}')" 

# PRETRAIN=../weights/r50_ibn_a.pth
# DATA_DIR='/share/Dataset/PReID/pre/'
# SAVE_DIR='../work_dirs/exp4-baseline-resnet50ibnls1-256x128-bs16x4-warmup10-flip-pad10-meanstd-erase0502-lbsm-cat_pool/' #(h, w)
# CUDA_VISIBLE_DEVICES=6 python train.py --config_file='configs/preid/baseline.yml' \
#     MODEL.NAME "baseline" MODEL.BACKBONE "('resnet50_ibn_a')" MODEL.BASELINE.POOL_TYPE "GlobalConcatPool2d"\
#     DATASETS.DATA_PATH "('${DATA_DIR}')" DATASETS.TRAIN_PATH "train"\
#     MODEL.PRETRAIN_PATH "('${PRETRAIN}')"  \
#     OUTPUT_DIR "('${SAVE_DIR}')" 

# PRETRAIN=../weights/r50_ibn_a.pth
# DATA_DIR='/share/Dataset/PReID/pre/'
# SAVE_DIR='../work_dirs/exp4-baseline-resnet50ibnls1-256x128-bs16x4-warmup10-flip-pad10-meanstd-erase0502-lbsm-att_pool/' #(h, w)
# CUDA_VISIBLE_DEVICES=7 python train.py --config_file='configs/preid/baseline.yml' \
#     MODEL.NAME "baseline" MODEL.BACKBONE "('resnet50_ibn_a')" MODEL.BASELINE.POOL_TYPE "GlobalAttnPool2d"\
#     DATASETS.DATA_PATH "('${DATA_DIR}')" DATASETS.TRAIN_PATH "train"\
#     MODEL.PRETRAIN_PATH "('${PRETRAIN}')"  \
#     OUTPUT_DIR "('${SAVE_DIR}')" 
