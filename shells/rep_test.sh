#!/usr/bin/env bash
PRETRAIN=../weights/r50_ibn_a.pth

# ROOT_DIR=/share/Dataset/PReID/
ROOT_DIR=/data/Dataset/PReID/
DATA_DIR=${ROOT_DIR}all_dataset/
# DATA_DIR=${ROOT_DIR}rep_dataset/

QUERY_DIR=${ROOT_DIR}dataset2/query_a/
GALLERY_DIR=${ROOT_DIR}dataset2/gallery_a/


# --rerank --k1 9 --k2 3 --lambda_value 0.6 \
# --rerank --k1 8 --k2 3 --lambda_value 0.8 \

# --dba --dba_k2 10 --dba_alpha -3.0 \
# --aqe --aqe_k2 7 --aqe_alpha 3.0 \
# --rerank --k1 8 --k2 3 --lambda_value 0.8 \


# # eval
# PRETRAIN=../weights/r50_ibn_a.pth
# MODEL_DIR=../rep_work_dirs/exp4-mfn-resnet50ibnls1-256x128-bs16x8-warmup10-flip-pad10-meanstd-erase0502-lbsm-tpl05-gem_3/ #(h, w)
# WEIGHT=${MODEL_DIR}mfn_epoch120.pth
# SAVE_DIR=${MODEL_DIR}eval/

# CUDA_VISIBLE_DEVICES=1 python test2.py --config_file='configs/naic/mfn.yml' \
#     --aqe --aqe_k2 7 --aqe_alpha 3.0 \
#     TEST.IMS_PER_BATCH "128" TEST.RANDOMPERM "5"\
#     MODEL.NAME "mfn" MODEL.BACKBONE "('resnet50_ibn_a')" MODEL.MFN.POOL_TYPE "gem_3" \
#     DATASETS.DATA_PATH "('${DATA_DIR}')" DATASETS.TRAIN_PATH "rep_f0_train"\
#     MODEL.PRETRAIN_PATH "('${PRETRAIN}')"  \
#     OUTPUT_DIR "('${SAVE_DIR}')" \
#     TEST.WEIGHT "${WEIGHT}"



# # --rerank --k1 9 --k2 3 --lambda_value 0.6 \
# # --dba --dba_k2 10 --dba_alpha -3.0
# # --aqe --aqe_k2 7 --aqe_alpha 3.0

# # SAVE_DIR=${MODEL_DIR}sub/
# # CUDA_VISIBLE_DEVICES=0 python test2.py --config_file='configs/naic/mfn.yml' \
# #     --sub \
# #     --aqe --aqe_k2 7 --aqe_alpha 3.0 \
# #     --query_dir ${QUERY_DIR}\
# #     --gallery_dir ${GALLERY_DIR}\
# #     MODEL.NAME "mfn" MODEL.BACKBONE "('resnet50_ibn_a')" MODEL.MFN.POOL_TYPE "gem_3" \
# #     DATASETS.DATA_PATH "('${DATA_DIR}')" DATASETS.TRAIN_PATH "rep_f0_train"\
# #     MODEL.PRETRAIN_PATH "('${PRETRAIN}')"  \
# #     OUTPUT_DIR "('${SAVE_DIR}')" \
# #     TEST.WEIGHT "${WEIGHT}"

# # [20191211]

# # eval
# PRETRAIN=../weights/r50_ibn_a.pth
# MODEL_DIR='../rep_work_dirs/exp4-mfn-resnet50ibnls1-256x128-bs16x8-warmup10-flip-pad10-meanstd-erase0502-lbsm-tpl05-nosm_allgem_3-minInst2/' #(h, w)
# WEIGHT=${MODEL_DIR}mfn_epoch120.pth
# SAVE_DIR=${MODEL_DIR}eval/

#     # --aqe --aqe_k2 7 --aqe_alpha 3.0 \
#     # --adabn --adabn_emv\
# #     --adabn \

# CUDA_VISIBLE_DEVICES=0 python test2.py --config_file='configs/naic/mfn.yml' \
#     TEST.IMS_PER_BATCH "128" TEST.RANDOMPERM "5"\
#     MODEL.NAME "mfn" MODEL.BACKBONE "('resnet50_ibn_a')" MODEL.MFN.POOL_TYPE "gem_3" MODEL.MFN.AUX_POOL_TYPE "gem_3" MODEL.MFN.AUX_SMOOTH "False"\
#     DATASETS.DATA_PATH "('${DATA_DIR}')" DATASETS.TRAIN_PATH "rep_f0_train2"\
#     MODEL.PRETRAIN_PATH "('${PRETRAIN}')"  \
#     OUTPUT_DIR "('${SAVE_DIR}')" \
#     TEST.WEIGHT "${WEIGHT}"



# # --rerank --k1 9 --k2 3 --lambda_value 0.6 \
# # --dba --dba_k2 10 --dba_alpha -3.0
# # --aqe --aqe_k2 7 --aqe_alpha 3.0
# #     --adabn \

# # SAVE_DIR=${MODEL_DIR}sub/
# # CUDA_VISIBLE_DEVICES=0 python test2.py --config_file='configs/naic/mfn.yml' \
# #     --adabn \
# #     --sub \
# #     --query_dir ${QUERY_DIR}\
# #     --gallery_dir ${GALLERY_DIR}\
# #     MODEL.NAME "mfn" MODEL.BACKBONE "('resnet50_ibn_a')" MODEL.MFN.POOL_TYPE "gem_3" MODEL.MFN.AUX_POOL_TYPE "gem_3" MODEL.MFN.AUX_SMOOTH "False"\
# #     DATASETS.DATA_PATH "('${DATA_DIR}')" DATASETS.TRAIN_PATH "rep_f0_train2"\
# #     MODEL.PRETRAIN_PATH "('${PRETRAIN}')"  \
# #     OUTPUT_DIR "('${SAVE_DIR}')" \
# #     TEST.WEIGHT "${WEIGHT}"


# [20191212]

# eval
# PRETRAIN=../weights/se_resnext50_32x4d-a260b3a4.pth
# MODEL_DIR='../rep_work_dirs/exp4-mfn-se_resnext50ls1-256x128-bs16x8-warmup10-flip-pad10-meanstd-erase0502-lbsm-tpl05-nosm_allgem_3-minInst2/' #(h, w)
# WEIGHT=${MODEL_DIR}mfn_epoch120.pth
# SAVE_DIR=${MODEL_DIR}eval/

    # --aqe --aqe_k2 7 --aqe_alpha 3.0 \
    # --adabn --adabn_emv --adabn_all\ 



# CUDA_VISIBLE_DEVICES=0 python test2.py --config_file='configs/naic/mfn.yml' \
#     --adabn --adabn_all --adabn_emv\
#     TEST.IMS_PER_BATCH "128" TEST.RANDOMPERM "5"\
#     MODEL.NAME "mfn" MODEL.BACKBONE "('se_resnext50_32x4d')" MODEL.MFN.POOL_TYPE "gem_3" MODEL.MFN.AUX_POOL_TYPE "gem_3" MODEL.MFN.AUX_SMOOTH "False"\
#     DATASETS.DATA_PATH "('${DATA_DIR}')" DATASETS.TRAIN_PATH "rep_f0_train2"\
#     MODEL.PRETRAIN_PATH "('${PRETRAIN}')"  \
#     OUTPUT_DIR "('${SAVE_DIR}')" \
#     TEST.WEIGHT "${WEIGHT}"


# SAVE_DIR=${MODEL_DIR}sub/
# CUDA_VISIBLE_DEVICES=0 python test2.py --config_file='configs/naic/mfn.yml' \
#     --adabn\
#     --sub \
#     --query_dir ${QUERY_DIR}\
#     --gallery_dir ${GALLERY_DIR}\
#     MODEL.NAME "mfn" MODEL.BACKBONE "('se_resnext50_32x4d')" MODEL.MFN.POOL_TYPE "gem_3" MODEL.MFN.AUX_POOL_TYPE "gem_3" MODEL.MFN.AUX_SMOOTH "False"\
#     DATASETS.DATA_PATH "('${DATA_DIR}')" DATASETS.TRAIN_PATH "rep_f0_train2"\
#     MODEL.PRETRAIN_PATH "('${PRETRAIN}')"  \
#     OUTPUT_DIR "('${SAVE_DIR}')" \
#     TEST.WEIGHT "${WEIGHT}"


# [20191213]

# eval
# PRETRAIN=../weights/se_resnext50_32x4d-a260b3a4.pth
# MODEL_DIR='../rep_work_dirs/exp4-mfn-se_resnext50ls1-256x128-bs16x8-warmup10-flip-pad10-meanstd-erase0502-lbsm-tpl05-nosm_allgem_3-minInst2/' #(h, w)
# WEIGHT=${MODEL_DIR}mfn_epoch120.pth
# SAVE_DIR=${MODEL_DIR}eval/

#     # --aqe --aqe_k2 7 --aqe_alpha 3.0 \
#     # --adabn --adabn_emv --adabn_all\ 


# # --adabn \
# # CUDA_VISIBLE_DEVICES=4 python test3.py --config_file='configs/naic/mfn.yml' \
# #     --adabn \
# #     TEST.IMS_PER_BATCH "128" TEST.RANDOMPERM "5"\
# #     MODEL.NAME "mfn" MODEL.BACKBONE "('se_resnext50_32x4d')" MODEL.MFN.POOL_TYPE "gem_3" MODEL.MFN.AUX_POOL_TYPE "gem_3" MODEL.MFN.AUX_SMOOTH "False"\
# #     DATASETS.DATA_PATH "('${DATA_DIR}')" DATASETS.TRAIN_PATH "rep_f0_train2"\
# #     MODEL.PRETRAIN_PATH "('${PRETRAIN}')"  \
# #     OUTPUT_DIR "('${SAVE_DIR}')" \
# #     TEST.WEIGHT "${WEIGHT}"


# SAVE_DIR=${MODEL_DIR}sub/
# #     # --adabn\
# CUDA_VISIBLE_DEVICES=4 python test3.py --config_file='configs/naic/mfn.yml' \
#     --adabn \
#     --sub \
#     --query_dir ${QUERY_DIR}\
#     --gallery_dir ${GALLERY_DIR}\
#     MODEL.NAME "mfn" MODEL.BACKBONE "('se_resnext50_32x4d')" MODEL.MFN.POOL_TYPE "gem_3" MODEL.MFN.AUX_POOL_TYPE "gem_3" MODEL.MFN.AUX_SMOOTH "False"\
#     DATASETS.DATA_PATH "('${DATA_DIR}')" DATASETS.TRAIN_PATH "rep_f0_train2"\
#     MODEL.PRETRAIN_PATH "('${PRETRAIN}')"  \
#     OUTPUT_DIR "('${SAVE_DIR}')" \
#     TEST.WEIGHT "${WEIGHT}"

# # [1215]
# DATA_DIR='/data/Dataset/PReID/all_dataset/'
# PRETRAIN=../weights/r50_ibn_a.pth
# MODEL_DIR='../rep_work_dirs/exp4-mfn-resnet50ibnls1-288x144-bs16x8-warmup10-flip-pad10-meanstd-erase0502-lbsm-tpl05-nosm_allgem_3-minInst2/' #(h, w)
# WEIGHT=${MODEL_DIR}mfn_epoch120.pth
# SAVE_DIR=${MODEL_DIR}eval/
# # CUDA_VISIBLE_DEVICES=6 python test2.py --config_file='configs/naic/mfn.yml' \
# #     --flip --adabn\
# #     TEST.IMS_PER_BATCH "128" TEST.RANDOMPERM "5"\
# #     INPUT.SIZE_TRAIN "([288,144])" INPUT.SIZE_TEST "([288,144])" \
# #     MODEL.NAME "mfn" MODEL.BACKBONE "('resnet50_ibn_a')" MODEL.MFN.POOL_TYPE "gem_3" MODEL.MFN.AUX_POOL_TYPE "gem_3" MODEL.MFN.AUX_SMOOTH "False"\
# #     DATASETS.DATA_PATH "('${DATA_DIR}')" DATASETS.TRAIN_PATH "rep_f0_train2"\
# #     MODEL.PRETRAIN_PATH "('${PRETRAIN}')"  \
# #     OUTPUT_DIR "('${SAVE_DIR}')" \
# #     TEST.WEIGHT "${WEIGHT}"

# SAVE_DIR=${MODEL_DIR}sub/
# CUDA_VISIBLE_DEVICES=6 python test2.py --config_file='configs/naic/mfn.yml' \
#     --flip --adabn \
#     --sub \
#     --query_dir ${QUERY_DIR}\
#     --gallery_dir ${GALLERY_DIR}\
#     INPUT.SIZE_TRAIN "([288,144])" INPUT.SIZE_TEST "([288,144])" \
#     MODEL.NAME "mfn" MODEL.BACKBONE "('resnet50_ibn_a')" MODEL.MFN.POOL_TYPE "gem_3" MODEL.MFN.AUX_POOL_TYPE "gem_3" MODEL.MFN.AUX_SMOOTH "False"\
#     DATASETS.DATA_PATH "('${DATA_DIR}')" DATASETS.TRAIN_PATH "rep_f0_train2"\
#     MODEL.PRETRAIN_PATH "('${PRETRAIN}')"  \
#     OUTPUT_DIR "('${SAVE_DIR}')" \
#     TEST.WEIGHT "${WEIGHT}"

# [1216]
# DATA_DIR='/data/Dataset/PReID/all_dataset/'
DATA_DIR='/data/Dataset/PReID/rep_dataset/'
PRETRAIN=../weights/r50_ibn_a.pth
MODEL_DIR='../rep_work_dirs/exp4-mfn-resnet50ibnls1-288x144-bs16x8-warmup10-flip-pad10-meanstd-erase0502-lbsm-tpl05-nosm_allgem_3-trainVal2/' #(h, w)
WEIGHT=${MODEL_DIR}mfn_epoch120.pth
# SAVE_DIR=${MODEL_DIR}eval/
# CUDA_VISIBLE_DEVICES=1 python test2.py --config_file='configs/naic/mfn.yml' \
#     --flip --adabn\
#     TEST.IMS_PER_BATCH "128" TEST.RANDOMPERM "5"\
#     INPUT.SIZE_TRAIN "([288,144])" INPUT.SIZE_TEST "([288,144])" \
#     MODEL.NAME "mfn" MODEL.BACKBONE "('resnet50_ibn_a')" MODEL.MFN.POOL_TYPE "gem_3" MODEL.MFN.AUX_POOL_TYPE "gem_3" MODEL.MFN.AUX_SMOOTH "False"\
#     DATASETS.DATA_PATH "('${DATA_DIR}')" DATASETS.TRAIN_PATH "rep_trainVal2"\
#     MODEL.PRETRAIN_PATH "('${PRETRAIN}')"  \
#     OUTPUT_DIR "('${SAVE_DIR}')" \
#     TEST.WEIGHT "${WEIGHT}"
# SAVE_DIR=${MODEL_DIR}sub/
# CUDA_VISIBLE_DEVICES=1 python test2.py --config_file='configs/naic/mfn.yml' \
#     --flip --adabn \
#     --sub \
#     --query_dir ${QUERY_DIR}\
#     --gallery_dir ${GALLERY_DIR}\
#     INPUT.SIZE_TRAIN "([288,144])" INPUT.SIZE_TEST "([288,144])" \
#     MODEL.NAME "mfn" MODEL.BACKBONE "('resnet50_ibn_a')" MODEL.MFN.POOL_TYPE "gem_3" MODEL.MFN.AUX_POOL_TYPE "gem_3" MODEL.MFN.AUX_SMOOTH "False"\
#     DATASETS.DATA_PATH "('${DATA_DIR}')" DATASETS.TRAIN_PATH "rep_trainVal2"\
#     MODEL.PRETRAIN_PATH "('${PRETRAIN}')"  \
#     OUTPUT_DIR "('${SAVE_DIR}')" \
#     TEST.WEIGHT "${WEIGHT}"


# DATA_DIR='/data/Dataset/PReID/all_dataset/'
# MODEL_DIR='../rep_work_dirs/exp4-mfn-resnet50ibnls1-256x128-bs16x8-warmup10-flip-pad10-meanstd-erase0502-lbsm-tpl05-nosm_allgem-minInst2/' #(h, w)
# WEIGHT=${MODEL_DIR}mfn_epoch120.pth
# CUDA_VISIBLE_DEVICES=1 python test2.py --config_file='configs/naic/mfn.yml' \
#     TEST.IMS_PER_BATCH "128" TEST.RANDOMPERM "5"\
#     MODEL.NAME "mfn" MODEL.BACKBONE "('resnet50_ibn_a')" MODEL.MFN.POOL_TYPE "gem" MODEL.MFN.AUX_POOL_TYPE "gem" MODEL.MFN.AUX_SMOOTH "False"\
#     DATASETS.DATA_PATH "('${DATA_DIR}')" DATASETS.TRAIN_PATH "rep_f0_train2"\
#     MODEL.PRETRAIN_PATH "('${PRETRAIN}')"  \
#     OUTPUT_DIR "('${SAVE_DIR}')" \
#     TEST.WEIGHT "${WEIGHT}"

PRETRAIN=../weights/resnet101_ibn_a.pth.tar
DATA_DIR='/data/Dataset/PReID/rep_dataset/'
MODEL_DIR='../rep_work_dirs/exp4-mfn-resnet101ibnls1_sestn-288x144-bs16x8-warmup10-flip-pad10-meanstd-erase0502-lbsm-tpl05-nosm_allavg-cj-minInst2/' #(h, w)
WEIGHT=${MODEL_DIR}mfn_epoch94.pth
# SAVE_DIR=${MODEL_DIR}eval/
# CUDA_VISIBLE_DEVICES=0 python test2.py --config_file='configs/naic/mfn.yml' \
#     TEST.IMS_PER_BATCH "128" TEST.RANDOMPERM "5"\
#     INPUT.SIZE_TRAIN "([288,144])" INPUT.SIZE_TEST "([288,144])" \
#     MODEL.NAME "mfn" MODEL.BACKBONE "('resnet101_ibn_a')" MODEL.MFN.POOL_TYPE "avg" MODEL.MFN.AUX_POOL_TYPE "avg" MODEL.MFN.AUX_SMOOTH "False" MODEL.MFN.USE_SESTN "True"\
#     DATASETS.DATA_PATH "('${DATA_DIR}')" DATASETS.TRAIN_PATH "rep_f0_train2"\
#     MODEL.PRETRAIN_PATH "('${PRETRAIN}')"  \
#     OUTPUT_DIR "('${SAVE_DIR}')" \
#     TEST.WEIGHT "${WEIGHT}"
# SAVE_DIR=${MODEL_DIR}sub/
# CUDA_VISIBLE_DEVICES=0 python test2.py --config_file='configs/naic/mfn.yml' \
#     --aqe --aqe_k2 7 --aqe_alpha 3.0 \
#     --flip  \
#     --sub --save_fname "origin_e94_" \
#     --query_dir ${QUERY_DIR}\
#     --gallery_dir ${GALLERY_DIR}\
#     INPUT.SIZE_TRAIN "([288,144])" INPUT.SIZE_TEST "([288,144])" \
#     MODEL.NAME "mfn" MODEL.BACKBONE "('resnet101_ibn_a')" MODEL.MFN.POOL_TYPE "avg" MODEL.MFN.AUX_POOL_TYPE "avg" MODEL.MFN.AUX_SMOOTH "False" MODEL.MFN.USE_SESTN "True"\
#     DATASETS.DATA_PATH "('${DATA_DIR}')" DATASETS.TRAIN_PATH "rep_f0_train2"\
#     MODEL.PRETRAIN_PATH "('${PRETRAIN}')"  \
#     OUTPUT_DIR "('${SAVE_DIR}')" \
#     TEST.WEIGHT "${WEIGHT}"
