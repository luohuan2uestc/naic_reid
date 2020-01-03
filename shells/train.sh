#!/usr/bin/env bash

# PRETRAIN=../weights/r50_ibn_a.pth
# DATA_DIR='/share/Dataset/PReID/pre/'
# SAVE_DIR='../work_dirs/exp2_mfn_resnet50_ibn_a/' #(h, w)
# CUDA_VISIBLE_DEVICES=4,6 python train.py --config_file='configs/preid/baseline.yml' \
#     MODEL.NAME "mfn" MODEL.BACKBONE "('resnet50_ibn_a')" \
#     DATASETS.DATA_PATH "('${DATA_DIR}')" DATASETS.TRAIN_PATH "train"\
#     MODEL.PRETRAIN_PATH "('${PRETRAIN}')"  \
#     OUTPUT_DIR "('${SAVE_DIR}')" 

# PRETRAIN=../weights/r50_ibn_a.pth
# DATA_DIR='/share/Dataset/PReID/pre/'
# SAVE_DIR='../work_dirs/exp2_mfn_resnet50_ibn_a_adam/' #(h, w)
# CUDA_VISIBLE_DEVICES=4,6 python train.py --config_file='configs/preid/baseline_adam.yml' \
#     MODEL.NAME "mfn" MODEL.BACKBONE "('resnet50_ibn_a')" \
#     DATASETS.DATA_PATH "('${DATA_DIR}')" DATASETS.TRAIN_PATH "train"\
#     MODEL.PRETRAIN_PATH "('${PRETRAIN}')"  \
#     OUTPUT_DIR "('${SAVE_DIR}')" 


# PRETRAIN=../weights/r50_ibn_a.pth
# DATA_DIR='/share/Dataset/PReID/pre/'
# SAVE_DIR='../work_dirs/exp2-mfn-resnet50ibnls1-syncbn-256x128-bs16x4-warmup10-flip-pad10-erase0504-trainVal/' #(h, w)
# CUDA_VISIBLE_DEVICES=4,6 python train.py --config_file='configs/preid/baseline_adam.yml' \
#     MODEL.NAME "mfn" MODEL.BACKBONE "('resnet50_ibn_a')" \
#     DATASETS.DATA_PATH "('${DATA_DIR}')" DATASETS.TRAIN_PATH "trainVal"\
#     MODEL.PRETRAIN_PATH "('${PRETRAIN}')"  \
#     OUTPUT_DIR "('${SAVE_DIR}')" 


# PRETRAIN=../weights/r50_ibn_a.pth
# DATA_DIR='/share/Dataset/PReID/pre/'
# SAVE_DIR='../work_dirs/exp2-mfn-resnet50ibnls1-syncbn-256x128-bs16x4-warmup10-flip-pad10-meanstd-erase0502/' #(h, w)
# CUDA_VISIBLE_DEVICES=4,6 python train.py --config_file='configs/preid/mfn.yml' \
#     MODEL.NAME "mfn" MODEL.BACKBONE "('resnet50_ibn_a')" \
#     DATASETS.DATA_PATH "('${DATA_DIR}')" DATASETS.TRAIN_PATH "train"\
#     MODEL.PRETRAIN_PATH "('${PRETRAIN}')"  \
#     OUTPUT_DIR "('${SAVE_DIR}')" 

# PRETRAIN=../weights/r50_ibn_a.pth
# DATA_DIR='/share/Dataset/PReID/pre/'
# SAVE_DIR='../work_dirs/exp2-mfn-resnet50ibnls1-syncbn-256x128-bs16x4-warmup10-flip-pad10-meanstd-erase0502-trainVal/' #(h, w)
# CUDA_VISIBLE_DEVICES=1,7 python train.py --config_file='configs/preid/mfn.yml' \
#     MODEL.NAME "mfn" MODEL.BACKBONE "('resnet50_ibn_a')" \
#     DATASETS.DATA_PATH "('${DATA_DIR}')" DATASETS.TRAIN_PATH "trainVal"\
#     MODEL.PRETRAIN_PATH "('${PRETRAIN}')"  \
#     OUTPUT_DIR "('${SAVE_DIR}')" 

# PRETRAIN=../weights/r50_ibn_a.pth
# DATA_DIR='/data/Dataset/PReID/pre/'
# SAVE_DIR='../work_dirs/exp2_debug/' #(h, w)
# CUDA_VISIBLE_DEVICES=0,1 python train.py --config_file='configs/preid/baseline_adam.yml' \
#     SOLVER.EVAL_PERIOD "1" SOLVER.IMS_PER_BATCH "128" DATALOADER.NUM_INSTANCE "8"\
#     MODEL.NAME "mfn" MODEL.BACKBONE "('resnet50_ibn_a')" \
#     DATASETS.DATA_PATH "('${DATA_DIR}')" DATASETS.TRAIN_PATH "train"\
#     MODEL.PRETRAIN_PATH "('${PRETRAIN}')"  \
#     OUTPUT_DIR "('${SAVE_DIR}')" 

# PRETRAIN=../weights/r50_ibn_a.pth
# DATA_DIR='/share/Dataset/PReID/pre/'
# SAVE_DIR='../work_dirs/exp3-mfn-resnet50ibnls1-syncbn-256x128-bs16x4-warmup10-flip-pad10-meanstd-erase0502/' #(h, w)
# CUDA_VISIBLE_DEVICES=3,4 python train.py --config_file='configs/preid/mfn.yml' \
#     MODEL.NAME "mfn" MODEL.BACKBONE "('resnet50_ibn_a')" \
#     DATASETS.DATA_PATH "('${DATA_DIR}')" DATASETS.TRAIN_PATH "train"\
#     MODEL.PRETRAIN_PATH "('${PRETRAIN}')"  \
#     OUTPUT_DIR "('${SAVE_DIR}')" 



# PRETRAIN=../weights/r50_ibn_a.pth
# DATA_DIR='/share/Dataset/PReID/pre/'
# SAVE_DIR='../work_dirs/exp3-mfn_sgd_train-resnet50ibnls1-syncbn-256x128-bs16x4-warmup10-flip-pad10-meanstd-erase0502/' #(h, w)
# CUDA_VISIBLE_DEVICES=3,4 python sgd_train.py --config_file='configs/preid/mfn_sgd.yml' \
#     MODEL.NAME "mfn" MODEL.BACKBONE "('resnet50_ibn_a')" \
#     DATASETS.DATA_PATH "('${DATA_DIR}')" DATASETS.TRAIN_PATH "train"\
#     MODEL.PRETRAIN_PATH "('${PRETRAIN}')"  \
#     OUTPUT_DIR "('${SAVE_DIR}')" 


# PRETRAIN=../weights/r50_ibn_a.pth
# DATA_DIR='/share/Dataset/PReID/pre/'
# SAVE_DIR='../work_dirs/exp3-mfn-resnet50ibnls1-256x128-bs16x4-warmup10-flip-pad10-meanstd-erase0502/' #(h, w)
# CUDA_VISIBLE_DEVICES=3,4 python train.py --config_file='configs/preid/mfn.yml' \
#     SOLVER.SYNCBN "False" \
#     MODEL.NAME "mfn" MODEL.BACKBONE "('resnet50_ibn_a')" \
#     DATASETS.DATA_PATH "('${DATA_DIR}')" DATASETS.TRAIN_PATH "train"\
#     MODEL.PRETRAIN_PATH "('${PRETRAIN}')"  \
#     OUTPUT_DIR "('${SAVE_DIR}')" 

# PRETRAIN=../weights/r50_ibn_a.pth
# DATA_DIR='/share/Dataset/PReID/pre/'
# SAVE_DIR='../work_dirs/exp3-mfn_sgd-resnet50ibnls1-syncbn-256x128-bs16x4-warmup10-flip-pad10-meanstd-erase0502/' #(h, w)
# CUDA_VISIBLE_DEVICES=2,5 python train.py --config_file='configs/preid/mfn_sgd.yml' \
#     MODEL.NAME "mfn" MODEL.BACKBONE "('resnet50_ibn_a')" \
#     DATASETS.DATA_PATH "('${DATA_DIR}')" DATASETS.TRAIN_PATH "train"\
#     MODEL.PRETRAIN_PATH "('${PRETRAIN}')"  \
#     OUTPUT_DIR "('${SAVE_DIR}')" 

# [todo]
# nolbsm tpl03
# PRETRAIN=../weights/r50_ibn_a.pth
# DATA_DIR='/share/Dataset/PReID/pre/'
# SAVE_DIR='../work_dirs/exp3-mfn-resnet50ibnls1-256x128-bs16x4-warmup10-flip-pad10-meanstd-erase0502-nolbsm-tpl03/' #(h, w)
# CUDA_VISIBLE_DEVICES=3,4 python train.py --config_file='configs/preid/mfn.yml' \
#     SOLVER.SYNCBN "False" MODEL.LABEL_SMOOTH "False" SOLVER.MARGIN "0.3"\
#     MODEL.NAME "mfn" MODEL.BACKBONE "('resnet50_ibn_a')" \
#     DATASETS.DATA_PATH "('${DATA_DIR}')" DATASETS.TRAIN_PATH "train"\
#     MODEL.PRETRAIN_PATH "('${PRETRAIN}')"  \
#     OUTPUT_DIR "('${SAVE_DIR}')" 


# PRETRAIN=../weights/resnext101_ibn_a.pth.tar
# DATA_DIR='/data/Dataset/PReID/pre/'
# SAVE_DIR='../work_dirs/exp3-mfn-resnet101ibnls1-256x128-bs16x4-warmup10-flip-pad10-meanstd-erase0502-lbsm-tpl05/' #(h, w)
# CUDA_VISIBLE_DEVICES=0,1 python train.py --config_file='configs/preid/mfn.yml' \
#     SOLVER.SYNCBN "False" \
#     MODEL.NAME "mfn" MODEL.BACKBONE "('resnext101_ibn_a')" \
#     DATASETS.DATA_PATH "('${DATA_DIR}')" DATASETS.TRAIN_PATH "train"\
#     MODEL.PRETRAIN_PATH "('${PRETRAIN}')"  \
#     OUTPUT_DIR "('${SAVE_DIR}')" 

# [20191127] fix batch size
# PRETRAIN=../weights/resnext101_ibn_a.pth.tar
# DATA_DIR='/data/Dataset/PReID/pre/'
# SAVE_DIR='../work_dirs/exp4-mfn-resnet101ibnls1-256x128-bs16x4-warmup10-flip-pad10-meanstd-erase0502-lbsm-tpl05/' #(h, w)
# CUDA_VISIBLE_DEVICES=0,1 python train.py --config_file='configs/preid/mfn.yml' \
#     SOLVER.SYNCBN "False" \
#     MODEL.NAME "mfn" MODEL.BACKBONE "('resnext101_ibn_a')" \
#     DATASETS.DATA_PATH "('${DATA_DIR}')" DATASETS.TRAIN_PATH "train"\
#     MODEL.PRETRAIN_PATH "('${PRETRAIN}')"  \
#     OUTPUT_DIR "('${SAVE_DIR}')" 



# PRETRAIN=../weights/r50_ibn_a.pth
# DATA_DIR='/share/Dataset/PReID/pre/'
# SAVE_DIR='../work_dirs/exp4-mfn-resnet50ibnls1-256x128-bs16x8-warmup10-flip-pad10-meanstd-erase0502-lbsm-tpl05-trainVal/' #(h, w)
# CUDA_VISIBLE_DEVICES=2,3 python train.py --config_file='configs/preid/mfn.yml' \
#     SOLVER.SYNCBN "False" SOLVER.IMS_PER_BATCH "128" DATALOADER.NUM_INSTANCE "8" \
#     MODEL.NAME "mfn" MODEL.BACKBONE "('resnet50_ibn_a')" \
#     DATASETS.DATA_PATH "('${DATA_DIR}')" DATASETS.TRAIN_PATH "trainVal"\
#     MODEL.PRETRAIN_PATH "('${PRETRAIN}')"  \
#     OUTPUT_DIR "('${SAVE_DIR}')" 

# PRETRAIN=../weights/resnext101_ibn_a.pth.tar
# DATA_DIR='/data/Dataset/PReID/pre/'
# SAVE_DIR='../work_dirs/exp4-mfn-resnet101ibnls1-256x128-bs16x8-warmup10-flip-pad10-meanstd-erase0502-lbsm-tpl05-trainVal/' #(h, w)
# CUDA_VISIBLE_DEVICES=0,1 python train.py --config_file='configs/preid/mfn.yml' \
#     SOLVER.SYNCBN "False" SOLVER.IMS_PER_BATCH "128" DATALOADER.NUM_INSTANCE "8"\
#     MODEL.NAME "mfn" MODEL.BACKBONE "('resnext101_ibn_a')" \
#     DATASETS.DATA_PATH "('${DATA_DIR}')" DATASETS.TRAIN_PATH "trainVal"\
#     MODEL.PRETRAIN_PATH "('${PRETRAIN}')"  \
#     OUTPUT_DIR "('${SAVE_DIR}')" 

# [todo] test

# PRETRAIN=../weights/r50_ibn_a.pth
# DATA_DIR='/data/Dataset/PReID/pre/'
# SAVE_DIR='../work_dirs/exp4-mfn-resnet50ibnls1-256x128-bs16x4-warmup10-flip-pad10-meanstd-erase0502-centerloss00005/' #(h, w)
# CUDA_VISIBLE_DEVICES=0,1 python train.py --config_file='configs/preid/mfn_center.yml' \
#     MODEL.NAME "mfn" MODEL.BACKBONE "('resnet50_ibn_a')" \
#     DATASETS.DATA_PATH "('${DATA_DIR}')" DATASETS.TRAIN_PATH "train"\
#     MODEL.PRETRAIN_PATH "('${PRETRAIN}')"  \
#     OUTPUT_DIR "('${SAVE_DIR}')" 


# PRETRAIN=../weights/r50_ibn_a.pth
# DATA_DIR='/data/Dataset/PReID/pre/'
# SAVE_DIR='../work_dirs/exp4-mfn-resnet50ibnls1-256x128-bs16x4-warmup10-flip-pad10-meanstd-erase0502/' #(h, w)
# CUDA_VISIBLE_DEVICES=4,5 python train.py --config_file='configs/preid/mfn.yml' \
#     SOLVER.SYNCBN "False" \
#     MODEL.NAME "mfn" MODEL.BACKBONE "('resnet50_ibn_a')" \
#     DATASETS.DATA_PATH "('${DATA_DIR}')" DATASETS.TRAIN_PATH "train"\
#     MODEL.PRETRAIN_PATH "('${PRETRAIN}')"  \
#     OUTPUT_DIR "('${SAVE_DIR}')" 

# PRETRAIN=../weights/r50_ibn_a.pth
# DATA_DIR='/data/Dataset/PReID/pre/'
# # SAVE_DIR='../work_dirs/exp4-mfn-resnet50ibnls1-256x128-bs16x4-warmup10-flip-pad10-meanstd-erase0502-negmixup/' #(h, w)
# SAVE_DIR='../work_dirs/exp4-mfn-debug/' #(h, w)
# CUDA_VISIBLE_DEVICES=4,5 python train.py --config_file='configs/preid/mfn_negmixup.yml' \
#     MODEL.NAME "mfn" MODEL.BACKBONE "('resnet50_ibn_a')" \
#     DATASETS.DATA_PATH "('${DATA_DIR}')" DATASETS.TRAIN_PATH "train"\
#     MODEL.PRETRAIN_PATH "('${PRETRAIN}')"  \
#     OUTPUT_DIR "('${SAVE_DIR}')" 

# PRETRAIN=../weights/r50_ibn_a.pth
# DATA_DIR='/share/Dataset/PReID/pre/'
# SAVE_DIR='../work_dirs/exp4-mfn-resnet50ibnls1-256x128-bs16x4-warmup10-flip-pad10-meanstd-erase0502-negmixup-mixce00/' #(h, w)
# # SAVE_DIR='../work_dirs/exp4-mfn-debug/' #(h, w)
# CUDA_VISIBLE_DEVICES=2,3 python train.py --config_file='configs/preid/mfn_negmixup.yml' \
#     MODEL.NAME "mfn" MODEL.BACKBONE "('resnet50_ibn_a')" SOLVER.MIXUP.CE_WEIGHT "0.0"\
#     DATASETS.DATA_PATH "('${DATA_DIR}')" DATASETS.TRAIN_PATH "train"\
#     MODEL.PRETRAIN_PATH "('${PRETRAIN}')"  \
#     OUTPUT_DIR "('${SAVE_DIR}')" 

# PRETRAIN=../weights/r50_ibn_a.pth
# DATA_DIR='/share/Dataset/PReID/pre/'
# SAVE_DIR='../work_dirs/exp4-mfn-resnet50ibnls1-256x128-bs16x4-warmup10-flip-pad10-meanstd-erase0502/' #(h, w)
# CUDA_VISIBLE_DEVICES=0,1 python train.py --config_file='configs/preid/mfn.yml' \
#     MODEL.NAME "mfn" MODEL.BACKBONE "('resnet50_ibn_a')" \
#     DATASETS.DATA_PATH "('${DATA_DIR}')" DATASETS.TRAIN_PATH "train"\
#     MODEL.PRETRAIN_PATH "('${PRETRAIN}')"  \
#     OUTPUT_DIR "('${SAVE_DIR}')" 

# PRETRAIN=../weights/r50_ibn_a.pth
# DATA_DIR='/share/Dataset/PReID/pre/'
# SAVE_DIR='../work_dirs/exp4-mfn-resnet50ibnls1-256x128-bs16x4-warmup10-flip-pad10-meanstd-erase0502-exemplar/' #(h, w)
# CUDA_VISIBLE_DEVICES=0,1 python train.py --config_file='configs/preid/mfn_exemplar.yml' \
#     DATASETS.EXEMPLAR.PATH "exemplar_valid" \
#     MODEL.NAME "mfn" MODEL.BACKBONE "('resnet50_ibn_a')" \
#     DATASETS.DATA_PATH "('${DATA_DIR}')" DATASETS.TRAIN_PATH "train"\
#     MODEL.PRETRAIN_PATH "('${PRETRAIN}')"  \
#     OUTPUT_DIR "('${SAVE_DIR}')" 

# PRETRAIN=../weights/resnet50_ibn_b.pth.tar
# DATA_DIR='/share/Dataset/PReID/pre/'
# SAVE_DIR='../work_dirs/exp4-mfn-resnet50ibn_bls1-256x128-bs16x4-warmup10-flip-pad10-meanstd-erase0502/' #(h, w)
# CUDA_VISIBLE_DEVICES=4,5 python train.py --config_file='configs/preid/mfn.yml' \
#     MODEL.NAME "mfn" MODEL.BACKBONE "('resnet50_ibn_b')" \
#     DATASETS.DATA_PATH "('${DATA_DIR}')" DATASETS.TRAIN_PATH "train"\
#     MODEL.PRETRAIN_PATH "('${PRETRAIN}')"  \
#     OUTPUT_DIR "('${SAVE_DIR}')" 

# PRETRAIN=../weights/r50_ibn_a.pth
# DATA_DIR='/share/Dataset/PReID/pre/'
# SAVE_DIR='../work_dirs/exp4-mfn-resnet50ibnls1-256x128-bs16x4-warmup10-flip-pad10-meanstd-erase0502-exemplar_dg/' #(h, w)
# CUDA_VISIBLE_DEVICES=0,1 python train.py --config_file='configs/preid/mfn_exemplar.yml' \
#     DATASETS.EXEMPLAR.PATH "exemplar_valid_dg" \
#     MODEL.NAME "mfn" MODEL.BACKBONE "('resnet50_ibn_a')" \
#     DATASETS.DATA_PATH "('${DATA_DIR}')" DATASETS.TRAIN_PATH "train"\
#     MODEL.PRETRAIN_PATH "('${PRETRAIN}')"  \
#     OUTPUT_DIR "('${SAVE_DIR}')" 

# modify code manually(different target for origin and gan)
# PRETRAIN=../weights/r50_ibn_a.pth
# DATA_DIR='/share/Dataset/PReID/pre/'
# SAVE_DIR='../work_dirs/exp4-mfn-resnet50ibnls1-256x128-bs16x4-warmup10-flip-pad10-meanstd-erase0502-exemplar_dg_diffid/' #(h, w)
# CUDA_VISIBLE_DEVICES=2,3 python train.py --config_file='configs/preid/mfn_exemplar.yml' \
#     DATASETS.EXEMPLAR.PATH "exemplar_valid_dg" \
#     MODEL.NAME "mfn" MODEL.BACKBONE "('resnet50_ibn_a')" \
#     DATASETS.DATA_PATH "('${DATA_DIR}')" DATASETS.TRAIN_PATH "train"\
#     MODEL.PRETRAIN_PATH "('${PRETRAIN}')"  \
#     OUTPUT_DIR "('${SAVE_DIR}')" 

# PRETRAIN=../weights/r50_ibn_a.pth
# DATA_DIR='/share/Dataset/PReID/pre/'
# SAVE_DIR='../work_dirs/exp4-mfn-resnet50ibnls1-256x128-bs16x4-warmup10-flip-pad10-meanstd-erase0502-exemplar_dg_diffid/' #(h, w)
# CUDA_VISIBLE_DEVICES=2,3 python train.py --config_file='configs/preid/mfn_exemplar.yml' \
#     DATASETS.EXEMPLAR.PATH "exemplar_valid_dg" \
#     MODEL.NAME "mfn" MODEL.BACKBONE "('resnet50_ibn_a')" \
#     DATASETS.DATA_PATH "('${DATA_DIR}')" DATASETS.TRAIN_PATH "train"\
#     MODEL.PRETRAIN_PATH "('${PRETRAIN}')"  \
#     OUTPUT_DIR "('${SAVE_DIR}')" 

# PRETRAIN=../weights/r50_ibn_a.pth
# WEIGTH=../work_dirs/exp4-mfn-resnet50ibnls1-256x128-bs16x4-warmup10-flip-pad10-meanstd-erase0502/mfn_epoch120.pth
# DATA_DIR='/share/Dataset/PReID/pre/'
# SAVE_DIR='../work_dirs/exp4-mfn-resnet50ibnls1-256x128-bs16x4-warmup10-flip-pad10-meanstd-erase0502-uirl/' #(h, w)
# # SAVE_DIR='../work_dirs/debug/'
# CUDA_VISIBLE_DEVICES=0,1 python train.py --config_file='configs/preid/mfn_unknown_identity.yml' \
#     MODEL.WEIGHT "${WEIGTH}" \
#     DATASETS.EXEMPLAR.PATH "exemplar_valid" \
#     MODEL.NAME "mfn" MODEL.BACKBONE "('resnet50_ibn_a')" \
#     DATASETS.DATA_PATH "('${DATA_DIR}')" DATASETS.TRAIN_PATH "train"\
#     MODEL.PRETRAIN_PATH "('${PRETRAIN}')"  \
#     OUTPUT_DIR "('${SAVE_DIR}')" 

# [20191208]
# PRETRAIN=../weights/r50_ibn_a.pth
# DATA_DIR='/share/Dataset/PReID/pre/'
# SAVE_DIR='../work_dirs/exp4-mfn-resnet50ibnls1-256x128-bs16x8-warmup10-flip-pad10-meanstd-erase0502-lbsm-tpl05-minInst2/' #(h, w)
# CUDA_VISIBLE_DEVICES=0,1 python train.py --config_file='configs/preid/mfn.yml' \
#     SOLVER.IMS_PER_BATCH "128" DATALOADER.NUM_INSTANCE "8" \
#     MODEL.NAME "mfn" MODEL.BACKBONE "('resnet50_ibn_a')" \
#     DATASETS.DATA_PATH "('${DATA_DIR}')" DATASETS.TRAIN_PATH "train2"\
#     MODEL.PRETRAIN_PATH "('${PRETRAIN}')"  \
#     OUTPUT_DIR "('${SAVE_DIR}')" 

# PRETRAIN=../weights/r50_ibn_a.pth
# DATA_DIR='/share/Dataset/PReID/pre/'
# SAVE_DIR='../work_dirs/exp4-mfn-resnet50ibnls1-256x128-bs16x8-warmup10-flip-pad10-meanstd-erase0502-lbsm-tpl05-minInst2-syncbn/' #(h, w)
# CUDA_VISIBLE_DEVICES=2,3 python train.py --config_file='configs/preid/mfn.yml' \
#     SOLVER.IMS_PER_BATCH "128" DATALOADER.NUM_INSTANCE "8" \
#     SOLVER.SYNCBN "True"  MODEL.NAME "mfn" MODEL.BACKBONE "('resnet50_ibn_a')" \
#     DATASETS.DATA_PATH "('${DATA_DIR}')" DATASETS.TRAIN_PATH "train2"\
#     MODEL.PRETRAIN_PATH "('${PRETRAIN}')"  \
#     OUTPUT_DIR "('${SAVE_DIR}')" 


# PRETRAIN=../weights/r50_ibn_a.pth
# DATA_DIR='/share/Dataset/PReID/pre/'
# SAVE_DIR='../work_dirs/exp4-mfn_sgd_train-resnet50ibnls1-syncbn-256x128-bs16x8-warmup10-flip-pad10-meanstd-erase0502-minInst2/' #(h, w)
# CUDA_VISIBLE_DEVICES=0,1 python sgd_train.py --config_file='configs/preid/mfn_sgd.yml' \
#     SOLVER.IMS_PER_BATCH "128" DATALOADER.NUM_INSTANCE "8" \
#     SOLVER.SYNCBN "True" MODEL.NAME "mfn" MODEL.BACKBONE "('resnet50_ibn_a')" \
#     DATASETS.DATA_PATH "('${DATA_DIR}')" DATASETS.TRAIN_PATH "train2"\
#     MODEL.PRETRAIN_PATH "('${PRETRAIN}')"  \
#     OUTPUT_DIR "('${SAVE_DIR}')" 


# PRETRAIN=../weights/r50_ibn_a.pth
# DATA_DIR='/share/Dataset/PReID/pre/'
# SAVE_DIR='../work_dirs/exp4-mfn-resnet50ibnls1-256x128-bs16x8-warmup10-flip-pad10-meanstd-erase0502-lbsm-tpl05-gem_3/' #(h, w)
# CUDA_VISIBLE_DEVICES=6,7 python train.py --config_file='configs/preid/mfn.yml' \
#     SOLVER.IMS_PER_BATCH "128" DATALOADER.NUM_INSTANCE "8" \
#     MODEL.NAME "mfn" MODEL.BACKBONE "('resnet50_ibn_a')" MODEL.MFN.POOL_TYPE "gem_3" \
#     DATASETS.DATA_PATH "('${DATA_DIR}')" DATASETS.TRAIN_PATH "train"\
#     MODEL.PRETRAIN_PATH "('${PRETRAIN}')"  \
#     OUTPUT_DIR "('${SAVE_DIR}')" 

# PRETRAIN=../weights/r50_ibn_a.pth
# DATA_DIR='/share/Dataset/PReID/pre/'
# SAVE_DIR='../work_dirs/exp4-mfn-resnet50ibnls1-256x128-bs16x8-warmup10-flip-pad10-meanstd-erase0502-lbsm-tpl05/' #(h, w)
# CUDA_VISIBLE_DEVICES=6,7 python train.py --config_file='configs/preid/mfn.yml' \
#     SOLVER.IMS_PER_BATCH "128" DATALOADER.NUM_INSTANCE "8" \
#     MODEL.NAME "mfn" MODEL.BACKBONE "('resnet50_ibn_a')" \
#     DATASETS.DATA_PATH "('${DATA_DIR}')" DATASETS.TRAIN_PATH "train"\
#     MODEL.PRETRAIN_PATH "('${PRETRAIN}')"  \
#     OUTPUT_DIR "('${SAVE_DIR}')" 


PRETRAIN=../weights/r50_ibn_a.pth
DATA_DIR='/share/Dataset/PReID/pre/'
SAVE_DIR='../work_dirs/debug/'
CUDA_VISIBLE_DEVICES=6,7 python train.py --config_file='configs/preid/mfn.yml' \
    SOLVER.TENSORBOARD.USE "True" \
    TEST.RANDOMPERM "5" SOLVER.EVAL_PERIOD "1" \
    SOLVER.IMS_PER_BATCH "128" DATALOADER.NUM_INSTANCE "8" \
    MODEL.NAME "mfn" MODEL.BACKBONE "('resnet50_ibn_a')" \
    DATASETS.DATA_PATH "('${DATA_DIR}')" DATASETS.TRAIN_PATH "train"\
    MODEL.PRETRAIN_PATH "('${PRETRAIN}')"  \
    OUTPUT_DIR "('${SAVE_DIR}')" 