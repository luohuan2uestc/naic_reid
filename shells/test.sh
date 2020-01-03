#!/usr/bin/env bash
PRETRAIN=../weights/r50_ibn_a.pth

ROOT_DIR=/share/Dataset/PReID/
# ROOT_DIR=/data/Dataset/PReID/
DATA_DIR=${ROOT_DIR}pre/
QUERY_TXT=${ROOT_DIR}test/query_a_list.txt
QUERY_DIR=${ROOT_DIR}test/query_a/
GALLERY_DIR=${ROOT_DIR}test/gallery_a/

# MODEL_DIR=../work_dirs/exp2_mfn_resnet50_ibn_a_adam/ #(h, w)
# WEIGHT=${MODEL_DIR}mfn_epoch120.pth
# SAVE_DIR=${MODEL_DIR}sub/
# CUDA_VISIBLE_DEVICES=6,7 python test.py --config_file='configs/preid/baseline_adam.yml' \
#     --rerank --k1 8 --k2 3 --lambda_value 0.8 \
#     --sub \
#     --query_txt ${QUERY_TXT} \
#     --query_dir ${QUERY_DIR}\
#     --gallery_dir ${GALLERY_DIR}\
#     MODEL.NAME "mfn" MODEL.BACKBONE "('resnet50_ibn_a')" \
#     DATASETS.DATA_PATH "('${DATA_DIR}')" DATASETS.TRAIN_PATH "train"\
#     MODEL.PRETRAIN_PATH "('${PRETRAIN}')"  \
#     OUTPUT_DIR "('${SAVE_DIR}')" \
#     TEST.WEIGHT "${WEIGHT}"

# MODEL_DIR=../work_dirs/exp2-mfn-resnet50ibnls1-256x128-bs16x4-warmup10-flip-pad10-meanstd-erase0502/ #(h, w)
# WEIGHT=${MODEL_DIR}mfn_epoch120.pth
# SAVE_DIR=${MODEL_DIR}sub/
# CUDA_VISIBLE_DEVICES=6,7 python test.py --config_file='configs/preid/mfn.yml' \
#     --rerank --k1 8 --k2 3 --lambda_value 0.8 \
#     --sub \
#     --query_txt ${QUERY_TXT} \
#     --query_dir ${QUERY_DIR}\
#     --gallery_dir ${GALLERY_DIR}\
#     MODEL.NAME "mfn" MODEL.BACKBONE "('resnet50_ibn_a')" \
#     DATASETS.DATA_PATH "('${DATA_DIR}')" DATASETS.TRAIN_PATH "train"\
#     MODEL.PRETRAIN_PATH "('${PRETRAIN}')"  \
#     OUTPUT_DIR "('${SAVE_DIR}')" \
#     TEST.WEIGHT "${WEIGHT}"

# MODEL_DIR=../work_dirs/exp2-mfn-resnet50ibnls1-256x128-bs16x4-warmup10-flip-pad10-meanstd-erase0502-trainVal/ #(h, w)
# WEIGHT=${MODEL_DIR}mfn_epoch120.pth
# SAVE_DIR=${MODEL_DIR}sub/
# CUDA_VISIBLE_DEVICES=6,7 python test.py --config_file='configs/preid/mfn.yml' \
#     --rerank --k1 8 --k2 3 --lambda_value 0.8 \
#     --sub \
#     --query_txt ${QUERY_TXT} \
#     --query_dir ${QUERY_DIR}\
#     --gallery_dir ${GALLERY_DIR}\
#     MODEL.NAME "mfn" MODEL.BACKBONE "('resnet50_ibn_a')" \
#     DATASETS.DATA_PATH "('${DATA_DIR}')" DATASETS.TRAIN_PATH "trainVal"\
#     MODEL.PRETRAIN_PATH "('${PRETRAIN}')"  \
#     OUTPUT_DIR "('${SAVE_DIR}')" \
#     TEST.WEIGHT "${WEIGHT}"

# MODEL_DIR=../work_dirs/exp3-mfn-resnet50ibnls1-256x128-bs16x4-warmup10-flip-pad10-meanstd-erase0502/ #(h, w)
# WEIGHT=${MODEL_DIR}mfn_epoch120.pth
# SAVE_DIR=${MODEL_DIR}sub/
# CUDA_VISIBLE_DEVICES=3,4 python test.py --config_file='configs/preid/mfn.yml' \
#     --rerank --k1 8 --k2 3 --lambda_value 0.8 \
#     --sub \
#     --query_txt ${QUERY_TXT} \
#     --query_dir ${QUERY_DIR}\
#     --gallery_dir ${GALLERY_DIR}\
#     SOLVER.SYNCBN "False" \
#     MODEL.NAME "mfn" MODEL.BACKBONE "('resnet50_ibn_a')" \
#     DATASETS.DATA_PATH "('${DATA_DIR}')" DATASETS.TRAIN_PATH "train"\
#     MODEL.PRETRAIN_PATH "('${PRETRAIN}')"  \
#     OUTPUT_DIR "('${SAVE_DIR}')" \
#     TEST.WEIGHT "${WEIGHT}"
    
# MODEL_DIR=../work_dirs/exp3-mfn-resnet50ibnls1-256x128-bs16x4-warmup10-flip-pad10-meanstd-erase0502-nolbsm-tpl03/ #(h, w)
# WEIGHT=${MODEL_DIR}mfn_epoch120.pth
# SAVE_DIR=${MODEL_DIR}sub/
# CUDA_VISIBLE_DEVICES=3,4 python test.py --config_file='configs/preid/mfn.yml' \
#     --rerank --k1 8 --k2 3 --lambda_value 0.8 \
#     --sub \
#     --query_txt ${QUERY_TXT} \
#     --query_dir ${QUERY_DIR}\
#     --gallery_dir ${GALLERY_DIR}\
#     SOLVER.SYNCBN "False" MODEL.LABEL_SMOOTH "False" SOLVER.MARGIN "0.3"\
#     MODEL.NAME "mfn" MODEL.BACKBONE "('resnet50_ibn_a')" \
#     DATASETS.DATA_PATH "('${DATA_DIR}')" DATASETS.TRAIN_PATH "train"\
#     MODEL.PRETRAIN_PATH "('${PRETRAIN}')"  \
#     OUTPUT_DIR "('${SAVE_DIR}')" \
#     TEST.WEIGHT "${WEIGHT}"

# PRETRAIN=../weights/resnext101_ibn_a.pth.tar
# MODEL_DIR=../work_dirs/exp3-mfn-resnet101ibnls1-256x128-bs16x4-warmup10-flip-pad10-meanstd-erase0502-lbsm-tpl05/ #(h, w)
# WEIGHT=${MODEL_DIR}mfn_epoch120.pth
# SAVE_DIR=${MODEL_DIR}eval/
# CUDA_VISIBLE_DEVICES=3,4 python test.py --config_file='configs/preid/mfn.yml' \
#     --rerank --k1 8 --k2 3 --lambda_value 0.8 \
#     MODEL.NAME "mfn" MODEL.BACKBONE "('resnext101_ibn_a')" \
#     DATASETS.DATA_PATH "('${DATA_DIR}')" DATASETS.TRAIN_PATH "train"\
#     MODEL.PRETRAIN_PATH "('${PRETRAIN}')"  \
#     OUTPUT_DIR "('${SAVE_DIR}')" \
#     TEST.WEIGHT "${WEIGHT}"

PRETRAIN=../weights/r50_ibn_a.pth
MODEL_DIR=../work_dirs/exp3-mfn-resnet50ibnls1-256x128-bs16x4-warmup10-flip-pad10-meanstd-erase0502/ #(h, w)
WEIGHT=${MODEL_DIR}mfn_epoch120.pth
SAVE_DIR=${MODEL_DIR}eval/


CUDA_VISIBLE_DEVICES=3 python test.py --config_file='configs/preid/mfn.yml' \
    --rerank --k1 8 --k2 10 --lambda_value 0.8 \
    MODEL.NAME "mfn" MODEL.BACKBONE "('resnet50_ibn_a')" \
    DATASETS.DATA_PATH "('${DATA_DIR}')" DATASETS.TRAIN_PATH "train"\
    MODEL.PRETRAIN_PATH "('${PRETRAIN}')"  \
    OUTPUT_DIR "('${SAVE_DIR}')" \
    TEST.WEIGHT "${WEIGHT}"

# MODEL_DIR=../work_dirs/exp3-mfn-resnet101ibnls1-256x128-bs16x4-warmup10-flip-pad10-meanstd-erase0502-lbsm-tpl05/ #(h, w)
# WEIGHT=${MODEL_DIR}mfn_epoch120.pth
# SAVE_DIR=${MODEL_DIR}sub/
# CUDA_VISIBLE_DEVICES=3,4 python test.py --config_file='configs/preid/mfn.yml' \
#     --rerank --k1 8 --k2 3 --lambda_value 0.8 \
#     --sub \
#     --query_txt ${QUERY_TXT} \
#     --query_dir ${QUERY_DIR}\
#     --gallery_dir ${GALLERY_DIR}\
#     SOLVER.SYNCBN "False" \
#     MODEL.NAME "mfn" MODEL.BACKBONE "('resnext101_ibn_a')" \
#     DATASETS.DATA_PATH "('${DATA_DIR}')" DATASETS.TRAIN_PATH "train"\
#     MODEL.PRETRAIN_PATH "('${PRETRAIN}')"  \
#     OUTPUT_DIR "('${SAVE_DIR}')" \
#     TEST.WEIGHT "${WEIGHT}"

# PRETRAIN=../weights/r50_ibn_a.pth
# MODEL_DIR=../work_dirs/exp4-mfn-resnet50ibnls1-256x128-bs16x8-warmup10-flip-pad10-meanstd-erase0502-lbsm-tpl05-trainVal/ #(h, w)
# WEIGHT=${MODEL_DIR}mfn_epoch120.pth
# SAVE_DIR=${MODEL_DIR}sub/
# CUDA_VISIBLE_DEVICES=3,1 python test.py --config_file='configs/preid/mfn.yml' \
#     --rerank --k1 8 --k2 3 --lambda_value 0.8 \
#     --sub \
#     --query_txt ${QUERY_TXT} \
#     --query_dir ${QUERY_DIR}\
#     --gallery_dir ${GALLERY_DIR}\
#     SOLVER.SYNCBN "False" SOLVER.IMS_PER_BATCH "128" DATALOADER.NUM_INSTANCE "8" \
#     MODEL.NAME "mfn" MODEL.BACKBONE "('resnet50_ibn_a')" \
#     DATASETS.DATA_PATH "('${DATA_DIR}')" DATASETS.TRAIN_PATH "trainVal"\
#     MODEL.PRETRAIN_PATH "('${PRETRAIN}')"  \
#     OUTPUT_DIR "('${SAVE_DIR}')" \
#     TEST.WEIGHT "${WEIGHT}"
