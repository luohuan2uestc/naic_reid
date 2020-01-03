#!/usr/bin/env bash
PRETRAIN=../weights/r50_ibn_a.pth
ROOT_DIR=/share/Dataset/PReID/
DATA_DIR=${ROOT_DIR}pre/

# MODEL_DIR=../work_dirs/exp2_mfn_resnet50_ibn_a_adam/ #(h, w)
# WEIGHT=${MODEL_DIR}mfn_epoch120.pth
# SAVE_DIR=${MODEL_DIR}eval/
# CUDA_VISIBLE_DEVICES=6,7 python test.py --config_file='configs/preid/baseline_adam.yml' \
#     --rerank --k1 8 --k2 3 --lambda_value 0.8 \
#     MODEL.NAME "mfn" MODEL.BACKBONE "('resnet50_ibn_a')" \
#     DATASETS.DATA_PATH "('${DATA_DIR}')" DATASETS.TRAIN_PATH "train"\
#     MODEL.PRETRAIN_PATH "('${PRETRAIN}')"  \
#     OUTPUT_DIR "('${SAVE_DIR}')" \
#     TEST.WEIGHT "${WEIGHT}"

MODEL_DIR=../work_dirs/exp3-mfn-resnet50ibnls1-256x128-bs16x4-warmup10-flip-pad10-meanstd-erase0502/ #(h, w)
WEIGHT=${MODEL_DIR}mfn_epoch120.pth
SAVE_DIR=${MODEL_DIR}eval/
    # --rerank --k1 8 --k2 3 --lambda_value 0.8 \

CUDA_VISIBLE_DEVICES=6,7 python test.py --config_file='configs/preid/baseline_adam.yml' \
    MODEL.NAME "mfn" MODEL.BACKBONE "('resnet50_ibn_a')" \
    DATASETS.DATA_PATH "('${DATA_DIR}')" DATASETS.TRAIN_PATH "train"\
    MODEL.PRETRAIN_PATH "('${PRETRAIN}')"  \
    OUTPUT_DIR "('${SAVE_DIR}')" \
    TEST.WEIGHT "${WEIGHT}"
