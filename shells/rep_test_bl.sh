#!/usr/bin/env bash
PRETRAIN=../weights/r50_ibn_a.pth

ROOT_DIR=/data/Dataset/PReID/
# ROOT_DIR=/data/Dataset/PReID/
# DATA_DIR=${ROOT_DIR}all_dataset/
DATA_DIR=${ROOT_DIR}rep_dataset/

QUERY_DIR=${ROOT_DIR}dataset2/query_a/
GALLERY_DIR=${ROOT_DIR}dataset2/gallery_a/

# eval
PRETRAIN=../weights/r50_ibn_a.pth
MODEL_DIR=../rep_work_dirs/exp4-baseline-resnet50ibnls1-256x128-bs16x4-warmup10-flip-pad10-meanstd-erase0502-lbsm-gem_3-minInst2/ #(h, w)
WEIGHT=${MODEL_DIR}baseline_epoch120.pth
SAVE_DIR=${MODEL_DIR}eval/

# --rerank --k1 9 --k2 3 --lambda_value 0.6 \
# --rerank --k1 8 --k2 3 --lambda_value 0.8 \

# --dba --dba_k2 10 --dba_alpha -3.0 \
# --aqe --aqe_k2 7 --aqe_alpha 3.0 \
# --rerank --k1 8 --k2 3 --lambda_value 0.8 \
    # --aqe --aqe_k2 7 --aqe_alpha 3.0 \
    # --adabn --adabn_emv\
    # --adabn \
# CUDA_VISIBLE_DEVICES=6 python test2.py --config_file='configs/naic/baseline.yml' \
#     TEST.IMS_PER_BATCH "128" TEST.RANDOMPERM "5"\
#     MODEL.NAME "baseline" MODEL.BACKBONE "('resnet50_ibn_a')" MODEL.BASELINE.POOL_TYPE "gem_3"\
#     DATASETS.DATA_PATH "('${DATA_DIR}')" DATASETS.TRAIN_PATH "rep_f0_train2"\
#     MODEL.PRETRAIN_PATH "('${PRETRAIN}')"  \
#     OUTPUT_DIR "('${SAVE_DIR}')" \
#     TEST.WEIGHT "${WEIGHT}"



# --rerank --k1 9 --k2 3 --lambda_value 0.6 \
# --dba --dba_k2 10 --dba_alpha -3.0
# --aqe --aqe_k2 7 --aqe_alpha 3.0

# SAVE_DIR=${MODEL_DIR}sub/
# CUDA_VISIBLE_DEVICES=0 python test2.py --config_file='configs/naic/baseline.yml' \
#     --sub \
#     --aqe --aqe_k2 7 --aqe_alpha 3.0 \
#     --query_dir ${QUERY_DIR}\
#     --gallery_dir ${GALLERY_DIR}\
    # MODEL.NAME "baseline" MODEL.BACKBONE "('resnet50_ibn_a')" MODEL.BASELINE.POOL_TYPE "gem"\
    # DATASETS.DATA_PATH "('${DATA_DIR}')" DATASETS.TRAIN_PATH "rep_f0_train2"\
    # MODEL.PRETRAIN_PATH "('${PRETRAIN}')"  \
    # OUTPUT_DIR "('${SAVE_DIR}')" \
    # TEST.WEIGHT "${WEIGHT}"

# [20191220]
DATA_DIR=${ROOT_DIR}rep_dataset/
PRETRAIN=../weights/r50_ibn_a.pth
MODEL_DIR=../rep_work_dirs/exp4-cosinebaseline-resnet50ibnls1-384x192-bs16x6-warmup10-flip-pad10-meanstd-erase0502-nolbsm-gem-cosface30_035_10033-use_bnbias-minInst2/ #(h, w)
WEIGHT=${MODEL_DIR}cosine_baseline_epoch90.pth
SAVE_DIR=${MODEL_DIR}eval/
# CUDA_VISIBLE_DEVICES=4 python test2.py --config_file='configs/naic/arcface_baseline.yml' \
#     --adabn \
#     TEST.IMS_PER_BATCH "128" TEST.RANDOMPERM "5"\
#     MODEL.BASELINE.S "30.0" MODEL.BASELINE.M "0.35" MODEL.BASELINE.USE_BNBAIS "True"\
#     MODEL.BASELINE.COSINE_LOSS_TYPE 'CosFace' \
#     INPUT.SIZE_TRAIN "([384,192])" INPUT.SIZE_TEST "([384,192])" \
#     MODEL.NAME "cosine_baseline" MODEL.BACKBONE "('resnet50_ibn_a')" MODEL.BASELINE.POOL_TYPE "gem"\
#     DATASETS.DATA_PATH "('${DATA_DIR}')" DATASETS.TRAIN_PATH "rep_f0_train2"\
#     MODEL.PRETRAIN_PATH "('${PRETRAIN}')"  \
#     OUTPUT_DIR "('${SAVE_DIR}')" \
#     TEST.WEIGHT "${WEIGHT}"

# --rerank --k1 9 --k2 3 --lambda_value 0.6 \
# --dba --dba_k2 10 --dba_alpha -3.0
# --aqe --aqe_k2 7 --aqe_alpha 3.0

# SAVE_DIR=${MODEL_DIR}sub/
# CUDA_VISIBLE_DEVICES=4 python test2.py --config_file='configs/naic/arcface_baseline.yml' \
#     --sub \
#     --flip --adabn\
#     --query_dir ${QUERY_DIR}\
#     --gallery_dir ${GALLERY_DIR}\
#     MODEL.BASELINE.S "30.0" MODEL.BASELINE.M "0.35" MODEL.BASELINE.USE_BNBAIS "True"\
#     MODEL.BASELINE.COSINE_LOSS_TYPE 'CosFace' \
#     INPUT.SIZE_TRAIN "([384,192])" INPUT.SIZE_TEST "([384,192])" \
#     MODEL.NAME "cosine_baseline" MODEL.BACKBONE "('resnet50_ibn_a')" MODEL.BASELINE.POOL_TYPE "gem"\
#     DATASETS.DATA_PATH "('${DATA_DIR}')" DATASETS.TRAIN_PATH "rep_f0_train2"\
#     MODEL.PRETRAIN_PATH "('${PRETRAIN}')"  \
#     OUTPUT_DIR "('${SAVE_DIR}')" \
#     TEST.WEIGHT "${WEIGHT}"


DATA_DIR=${ROOT_DIR}rep_dataset/
PRETRAIN=../weights/r50_ibn_a.pth
MODEL_DIR=../rep_work_dirs/exp4-cosinebaseline-resnet50ibnls1-384x192-bs16x6-warmup10-flip-pad10-meanstd-erase0502-nolbsm-gem-cosface30_035_10033-use_bnbias-sepnorm-minInst2/ #(h, w)
WEIGHT=${MODEL_DIR}cosine_baseline_epoch90.pth
# SAVE_DIR=${MODEL_DIR}eval/
#     # INPUT.SEPNORM.USE "True" \
# # CUDA_VISIBLE_DEVICES=4 python test2.py --config_file='configs/naic/arcface_baseline.yml' \

# CUDA_VISIBLE_DEVICES=6 python sepnorm_test.py --config_file='configs/naic/arcface_baseline.yml' \
#     --adabn \
#     TEST.IMS_PER_BATCH "128" TEST.RANDOMPERM "5"\
#     MODEL.BASELINE.COSINE_LOSS_TYPE 'CosFace'  MODEL.BASELINE.S "30.0" MODEL.BASELINE.M "0.35" \
#     INPUT.SEPNORM.USE "True" \
#     INPUT.SEPNORM.PIXEL_MEAN "([[0.1164,0.1567,0.1796],[0.4556,0.5367,0.6493]])" \
#     INPUT.SEPNORM.PIXEL_STD "([[0.1467,0.1497,0.1930],[0.0750,0.0613,0.1040]])" \
#     INPUT.SIZE_TRAIN "([384,192])" INPUT.SIZE_TEST "([384,192])" \
#     MODEL.BASELINE.USE_BNBAIS "True" \
#     MODEL.NAME "cosine_baseline" MODEL.BACKBONE "('resnet50_ibn_a')" MODEL.BASELINE.POOL_TYPE "gem" \
#     DATASETS.DATA_PATH "('${DATA_DIR}')" DATASETS.TRAIN_PATH "rep_f0_train2"\
#     MODEL.PRETRAIN_PATH "('${PRETRAIN}')"  \
#     OUTPUT_DIR "('${SAVE_DIR}')" \
#     TEST.WEIGHT "${WEIGHT}"

# SAVE_DIR=${MODEL_DIR}sub/
# CUDA_VISIBLE_DEVICES=5 python sepnorm_test.py --config_file='configs/naic/arcface_baseline.yml' \
#     --sub \
#     --flip --adabn \
#     --query_dir ${QUERY_DIR}\
#     --gallery_dir ${GALLERY_DIR}\
#     TEST.IMS_PER_BATCH "128" TEST.RANDOMPERM "5"\
#     MODEL.BASELINE.COSINE_LOSS_TYPE 'CosFace'  MODEL.BASELINE.S "30.0" MODEL.BASELINE.M "0.35" \
#     INPUT.SEPNORM.USE "True" \
#     INPUT.SEPNORM.TEST_PIXEL_MEAN "([[0.1164,0.1567,0.1796],[0.5410,0.4631,0.5808]])" \
#     INPUT.SEPNORM.TEST_PIXEL_STD "([[0.1467,0.1497,0.1930],[0.0444,0.0821,0.0981]])" \
#     INPUT.SIZE_TRAIN "([384,192])" INPUT.SIZE_TEST "([384,192])" \
#     MODEL.BASELINE.USE_BNBAIS "True" \
#     MODEL.NAME "cosine_baseline" MODEL.BACKBONE "('resnet50_ibn_a')" MODEL.BASELINE.POOL_TYPE "gem" \
#     DATASETS.DATA_PATH "('${DATA_DIR}')" DATASETS.TRAIN_PATH "rep_f0_train2"\
#     MODEL.PRETRAIN_PATH "('${PRETRAIN}')"  \
#     OUTPUT_DIR "('${SAVE_DIR}')" \
#     TEST.WEIGHT "${WEIGHT}"



# [20191220]
# DATA_DIR=${ROOT_DIR}rep_dataset/
# PRETRAIN=../weights/r50_ibn_a.pth
# MODEL_DIR=../rep_work_dirs/exp4-cosinebaseline-resnet50ibnls1-384x192-bs16x6-warmup10-flip-pad10-meanstd-erase0502-nolbsm-gem-cosface30_035_10033-use_bnbias-cj-minInst2/ #(h, w)
# WEIGHT=${MODEL_DIR}cosine_baseline_epoch90.pth
# SAVE_DIR=${MODEL_DIR}eval/

# CUDA_VISIBLE_DEVICES=4 python test2.py --config_file='configs/naic/arcface_baseline.yml' \
#     --flip \
#     --aqe --aqe_k2 7 --aqe_alpha 3.0 \
#     TEST.IMS_PER_BATCH "128" TEST.RANDOMPERM "5"\
#     MODEL.BASELINE.S "30.0" MODEL.BASELINE.M "0.35" MODEL.BASELINE.USE_BNBAIS "True"\
#     MODEL.BASELINE.COSINE_LOSS_TYPE 'CosFace' \
#     INPUT.SIZE_TRAIN "([384,192])" INPUT.SIZE_TEST "([384,192])" \
#     MODEL.NAME "cosine_baseline" MODEL.BACKBONE "('resnet50_ibn_a')" MODEL.BASELINE.POOL_TYPE "gem"\
#     DATASETS.DATA_PATH "('${DATA_DIR}')" DATASETS.TRAIN_PATH "rep_f0_train2"\
#     MODEL.PRETRAIN_PATH "('${PRETRAIN}')"  \
#     OUTPUT_DIR "('${SAVE_DIR}')" \
#     TEST.WEIGHT "${WEIGHT}"

# # --rerank --k1 9 --k2 3 --lambda_value 0.6 \
# # --dba --dba_k2 10 --dba_alpha -3.0

# SAVE_DIR=${MODEL_DIR}sub/
# CUDA_VISIBLE_DEVICES=4 python test2.py --config_file='configs/naic/arcface_baseline.yml' \
#     --sub \
#     --flip --adabn\
#     --aqe --aqe_k2 7 --aqe_alpha 3.0 \
#     --query_dir ${QUERY_DIR}\
#     --gallery_dir ${GALLERY_DIR}\
#     MODEL.BASELINE.S "30.0" MODEL.BASELINE.M "0.35" MODEL.BASELINE.USE_BNBAIS "True"\
#     MODEL.BASELINE.COSINE_LOSS_TYPE 'CosFace' \
#     INPUT.SIZE_TRAIN "([384,192])" INPUT.SIZE_TEST "([384,192])" \
#     MODEL.NAME "cosine_baseline" MODEL.BACKBONE "('resnet50_ibn_a')" MODEL.BASELINE.POOL_TYPE "gem"\
#     DATASETS.DATA_PATH "('${DATA_DIR}')" DATASETS.TRAIN_PATH "rep_f0_train2"\
#     MODEL.PRETRAIN_PATH "('${PRETRAIN}')"  \
#     OUTPUT_DIR "('${SAVE_DIR}')" \
#     TEST.WEIGHT "${WEIGHT}"



# # [20191221]
# DATA_DIR=${ROOT_DIR}rep_dataset/
# PRETRAIN=../weights/r50_ibn_a.pth
# MODEL_DIR=../rep_work_dirs/exp4-cosinebaseline-resnet50ibnls1-384x192-bs16x6-warmup10-flip-pad10-meanstd-erase0502-nolbsm-avg-cosface30_035_10033-cj-minInst2/ #(h, w)
# WEIGHT=${MODEL_DIR}cosine_baseline_epoch90.pth
# SAVE_DIR=${MODEL_DIR}eval/
# # # 
# # CUDA_VISIBLE_DEVICES=4 python test2.py --config_file='configs/naic/arcface_baseline.yml' \
# #     --flip \
# #     TEST.IMS_PER_BATCH "128" TEST.RANDOMPERM "5" \
# #     MODEL.BASELINE.S "30.0" MODEL.BASELINE.M "0.35" \
# #     MODEL.BASELINE.COSINE_LOSS_TYPE 'CosFace' \
# #     INPUT.SIZE_TRAIN "([384,192])" INPUT.SIZE_TEST "([384,192])" \
# #     MODEL.NAME "cosine_baseline" MODEL.BACKBONE "('resnet50_ibn_a')" MODEL.BASELINE.POOL_TYPE "avg"\
# #     DATASETS.DATA_PATH "('${DATA_DIR}')" DATASETS.TRAIN_PATH "rep_f0_train2"\
# #     MODEL.PRETRAIN_PATH "('${PRETRAIN}')"  \
# #     OUTPUT_DIR "('${SAVE_DIR}')" \
# #     TEST.WEIGHT "${WEIGHT}"

# # --rerank --k1 9 --k2 3 --lambda_value 0.6 \
# # --dba --dba_k2 10 --dba_alpha -3.0

# SAVE_DIR=${MODEL_DIR}sub/
# CUDA_VISIBLE_DEVICES=4 python test2.py --config_file='configs/naic/arcface_baseline.yml' \
#     --sub \
#     --flip \
#     --aqe --aqe_k2 7 --aqe_alpha 3.0 \
#     --query_dir ${QUERY_DIR}\
#     --gallery_dir ${GALLERY_DIR}\
#     MODEL.BASELINE.S "30.0" MODEL.BASELINE.M "0.35" \
#     MODEL.BASELINE.COSINE_LOSS_TYPE 'CosFace' \
#     INPUT.SIZE_TRAIN "([384,192])" INPUT.SIZE_TEST "([384,192])" \
#     MODEL.NAME "cosine_baseline" MODEL.BACKBONE "('resnet50_ibn_a')" MODEL.BASELINE.POOL_TYPE "avg"\
#     DATASETS.DATA_PATH "('${DATA_DIR}')" DATASETS.TRAIN_PATH "rep_f0_train2"\
#     MODEL.PRETRAIN_PATH "('${PRETRAIN}')"  \
#     OUTPUT_DIR "('${SAVE_DIR}')" \
#     TEST.WEIGHT "${WEIGHT}"


# [20191221]
DATA_DIR=${ROOT_DIR}rep_dataset/
PRETRAIN=../weights/resnet101_ibn_a.pth.tar
MODEL_DIR=../rep_work_dirs/exp4-cosinebaseline-resnet101ibnls1-384x192-bs16x6-warmup10-flip-pad10-meanstd-erase0502-nolbsm-avg-cosface30_035_10033-cj-trainVal2/ #(h, w)
WEIGHT=${MODEL_DIR}cosine_baseline_epoch90.pth
SAVE_DIR=${MODEL_DIR}eval/
# 
#   --flip \
#     --aqe --aqe_k2 7 --aqe_alpha 3.0 \
# CUDA_VISIBLE_DEVICES=2 python test2.py --config_file='configs/naic/arcface_baseline.yml' \
#     TEST.IMS_PER_BATCH "128" TEST.RANDOMPERM "5" \
#     MODEL.BASELINE.S "30.0" MODEL.BASELINE.M "0.35" \
#     MODEL.BASELINE.COSINE_LOSS_TYPE 'CosFace' \
#     INPUT.SIZE_TRAIN "([384,192])" INPUT.SIZE_TEST "([384,192])" \
#     MODEL.NAME "cosine_baseline" MODEL.BACKBONE "('resnet101_ibn_a')" MODEL.BASELINE.POOL_TYPE "avg"\
#     DATASETS.DATA_PATH "('${DATA_DIR}')" DATASETS.TRAIN_PATH "rep_trainVal2"\
#     MODEL.PRETRAIN_PATH "('${PRETRAIN}')"  \
#     OUTPUT_DIR "('${SAVE_DIR}')" \
#     TEST.WEIGHT "${WEIGHT}"


# --rerank --k1 9 --k2 3 --lambda_value 0.6 \
# --dba --dba_k2 10 --dba_alpha -3.0

# SAVE_DIR=${MODEL_DIR}sub/
# CUDA_VISIBLE_DEVICES=2 python test2.py --config_file='configs/naic/arcface_baseline.yml' \
#     --sub \
#     --flip \
#     --aqe --aqe_k2 7 --aqe_alpha 3.0 \
#     --query_dir ${QUERY_DIR}\
#     --gallery_dir ${GALLERY_DIR}\
#     MODEL.BASELINE.S "30.0" MODEL.BASELINE.M "0.35" \
#     MODEL.BASELINE.COSINE_LOSS_TYPE 'CosFace' \
#     INPUT.SIZE_TRAIN "([384,192])" INPUT.SIZE_TEST "([384,192])" \
#     MODEL.NAME "cosine_baseline" MODEL.BACKBONE "('resnet101_ibn_a')" MODEL.BASELINE.POOL_TYPE "avg"\
#     DATASETS.DATA_PATH "('${DATA_DIR}')" DATASETS.TRAIN_PATH "rep_trainVal2"\
#     MODEL.PRETRAIN_PATH "('${PRETRAIN}')"  \
#     OUTPUT_DIR "('${SAVE_DIR}')" \
#     TEST.WEIGHT "${WEIGHT}"


# [20191221]
DATA_DIR=${ROOT_DIR}rep_dataset/
PRETRAIN=../weights/resnet101_ibn_a.pth.tar
MODEL_DIR=../rep_work_dirs/exp4-cosinebaseline-resnet101ibnls1-384x192-bs16x6-warmup10-flip-pad10-meanstd-erase0502-nolbsm-avg-cosface30_035_10033-cj05-trainVal2/ #(h, w)
WEIGHT=${MODEL_DIR}cosine_baseline_epoch90.pth
SAVE_DIR=${MODEL_DIR}eval/
# 
#   --flip \
#     --aqe --aqe_k2 7 --aqe_alpha 3.0 \
# CUDA_VISIBLE_DEVICES=0 python test2.py --config_file='configs/naic/arcface_baseline.yml' \
#     --flip \
#     --aqe --aqe_k2 7 --aqe_alpha 3.0 \
#     TEST.IMS_PER_BATCH "128" TEST.RANDOMPERM "0" \
#     MODEL.BASELINE.S "30.0" MODEL.BASELINE.M "0.35" \
#     MODEL.BASELINE.COSINE_LOSS_TYPE 'CosFace' \
#     INPUT.SIZE_TRAIN "([384,192])" INPUT.SIZE_TEST "([384,192])" \
#     MODEL.NAME "cosine_baseline" MODEL.BACKBONE "('resnet101_ibn_a')" MODEL.BASELINE.POOL_TYPE "avg"\
#     DATASETS.DATA_PATH "('${DATA_DIR}')" DATASETS.TRAIN_PATH "rep_trainVal2"\
#     MODEL.PRETRAIN_PATH "('${PRETRAIN}')"  \
#     OUTPUT_DIR "('${SAVE_DIR}')" \
#     TEST.WEIGHT "${WEIGHT}"


# --rerank --k1 9 --k2 3 --lambda_value 0.6 \
# --dba --dba_k2 10 --dba_alpha -3.0

# SAVE_DIR=${MODEL_DIR}sub/
# CUDA_VISIBLE_DEVICES=0 python test2.py --config_file='configs/naic/arcface_baseline.yml' \
#     --sub \
#     --aqe --aqe_k2 7 --aqe_alpha 3.0 \
#     --query_dir ${QUERY_DIR}\
#     --gallery_dir ${GALLERY_DIR}\
#     MODEL.BASELINE.S "30.0" MODEL.BASELINE.M "0.35" \
#     MODEL.BASELINE.COSINE_LOSS_TYPE 'CosFace' \
#     INPUT.SIZE_TRAIN "([384,192])" INPUT.SIZE_TEST "([384,192])" \
#     MODEL.NAME "cosine_baseline" MODEL.BACKBONE "('resnet101_ibn_a')" MODEL.BASELINE.POOL_TYPE "avg"\
#     DATASETS.DATA_PATH "('${DATA_DIR}')" DATASETS.TRAIN_PATH "rep_trainVal2"\
#     MODEL.PRETRAIN_PATH "('${PRETRAIN}')"  \
#     OUTPUT_DIR "('${SAVE_DIR}')" \
#     TEST.WEIGHT "${WEIGHT}"

# [20191228]
DATA_DIR=${ROOT_DIR}rep_dataset/
PRETRAIN=../weights/resnet101_ibn_a.pth.tar
MODEL_DIR=../rep_work_dirs/exp4-cosinebaseline-resnet101ibnls1-384x192-bs16x6-warmup10-flip-pad10-meanstd-erase0502-nolbsm-avg-arcface30_035_10033-cj05-trainVal2/ #(h, w)
# WEIGHT=${MODEL_DIR}cosine_baseline_epoch90.pth
WEIGHT=${MODEL_DIR}cosine_baseline_epoch80.pth

SAVE_DIR=${MODEL_DIR}eval/
# 
#   --flip \
#     --aqe --aqe_k2 7 --aqe_alpha 3.0 \
# CUDA_VISIBLE_DEVICES=5 python test2.py --config_file='configs/naic/arcface_baseline.yml' \
#     TEST.IMS_PER_BATCH "128" TEST.RANDOMPERM "5" \
#     MODEL.BASELINE.S "30.0" MODEL.BASELINE.M "0.35" \
#     MODEL.BASELINE.COSINE_LOSS_TYPE 'ArcCos' \
#     INPUT.SIZE_TRAIN "([384,192])" INPUT.SIZE_TEST "([384,192])" \
#     MODEL.NAME "cosine_baseline" MODEL.BACKBONE "('resnet101_ibn_a')" MODEL.BASELINE.POOL_TYPE "avg"\
#     DATASETS.DATA_PATH "('${DATA_DIR}')" DATASETS.TRAIN_PATH "rep_trainVal2"\
#     MODEL.PRETRAIN_PATH "('${PRETRAIN}')"  \
#     OUTPUT_DIR "('${SAVE_DIR}')" \
#     TEST.WEIGHT "${WEIGHT}"


# --rerank --k1 9 --k2 3 --lambda_value 0.6 \
# --dba --dba_k2 10 --dba_alpha -3.0

# SAVE_DIR=${MODEL_DIR}sub/
# CUDA_VISIBLE_DEVICES=6 python test2.py --config_file='configs/naic/arcface_baseline.yml' \
#     --save_fname 'origin_tpl03_e80_'\
#     --sub \
#     --flip \
#     --aqe --aqe_k2 7 --aqe_alpha 3.0 \
#     --query_dir ${QUERY_DIR}\
#     --gallery_dir ${GALLERY_DIR}\
#     MODEL.BASELINE.S "30.0" MODEL.BASELINE.M "0.35" \
#     MODEL.BASELINE.COSINE_LOSS_TYPE 'ArcCos' \
#     INPUT.SIZE_TRAIN "([384,192])" INPUT.SIZE_TEST "([384,192])" \
#     MODEL.NAME "cosine_baseline" MODEL.BACKBONE "('resnet101_ibn_a')" MODEL.BASELINE.POOL_TYPE "avg"\
#     DATASETS.DATA_PATH "('${DATA_DIR}')" DATASETS.TRAIN_PATH "rep_trainVal2"\
#     MODEL.PRETRAIN_PATH "('${PRETRAIN}')"  \
#     OUTPUT_DIR "('${SAVE_DIR}')" \
#     TEST.WEIGHT "${WEIGHT}"

# [20191230]
DATA_DIR=${ROOT_DIR}rep_dataset/
PRETRAIN=../weights/resnet101_ibn_a.pth.tar
MODEL_DIR=../rep_work_dirs/exp4-cosinebaseline-resnet101ibnls1-384x192-bs16x6-warmup10-flip-pad10-meanstd-erase0502-nolbsm-avg-arcface30_035_10033-cj05-trainVal2-finetune_tpl05/ #(h, w)
# WEIGHT=${MODEL_DIR}cosine_baseline_epoch16.pth
WEIGHT=${MODEL_DIR}cosine_baseline_epoch14.pth

SAVE_DIR=${MODEL_DIR}eval/
# 
#   --flip \
#     --aqe --aqe_k2 7 --aqe_alpha 3.0 \
# CUDA_VISIBLE_DEVICES=2 python test2.py --config_file='configs/naic/arcface_baseline.yml' \
#     TEST.IMS_PER_BATCH "128" TEST.RANDOMPERM "5" \
#     MODEL.BASELINE.S "30.0" MODEL.BASELINE.M "0.35" \
#     MODEL.BASELINE.COSINE_LOSS_TYPE 'ArcCos' \
#     INPUT.SIZE_TRAIN "([384,192])" INPUT.SIZE_TEST "([384,192])" \
#     MODEL.NAME "cosine_baseline" MODEL.BACKBONE "('resnet101_ibn_a')" MODEL.BASELINE.POOL_TYPE "avg"\
#     DATASETS.DATA_PATH "('${DATA_DIR}')" DATASETS.TRAIN_PATH "rep_trainVal2"\
#     MODEL.PRETRAIN_PATH "('${PRETRAIN}')"  \
#     OUTPUT_DIR "('${SAVE_DIR}')" \
#     TEST.WEIGHT "${WEIGHT}"


# --rerank --k1 9 --k2 3 --lambda_value 0.6 \
# --dba --dba_k2 10 --dba_alpha -3.0

# SAVE_DIR=${MODEL_DIR}sub/
# CUDA_VISIBLE_DEVICES=2 python test2.py --config_file='configs/naic/arcface_baseline.yml' \
#     --save_fname 'finetune_tpl05_e14_'\
#     --sub \
#     --flip \
#     --aqe --aqe_k2 7 --aqe_alpha 3.0 \
#     --query_dir ${QUERY_DIR}\
#     --gallery_dir ${GALLERY_DIR}\
#     MODEL.BASELINE.S "30.0" MODEL.BASELINE.M "0.35" \
#     MODEL.BASELINE.COSINE_LOSS_TYPE 'ArcCos' \
#     INPUT.SIZE_TRAIN "([384,192])" INPUT.SIZE_TEST "([384,192])" \
#     MODEL.NAME "cosine_baseline" MODEL.BACKBONE "('resnet101_ibn_a')" MODEL.BASELINE.POOL_TYPE "avg"\
#     DATASETS.DATA_PATH "('${DATA_DIR}')" DATASETS.TRAIN_PATH "rep_trainVal2"\
#     MODEL.PRETRAIN_PATH "('${PRETRAIN}')"  \
#     OUTPUT_DIR "('${SAVE_DIR}')" \
#     TEST.WEIGHT "${WEIGHT}"
# [20191228]
# DATA_DIR=${ROOT_DIR}rep_dataset/
# PRETRAIN=../weights/resnet101_ibn_a.pth.tar
# MODEL_DIR=../rep_work_dirs/exp5-cosinebaseline-resnet101ibnls1-384x192-bs16x6-warmup10-flip-pad10-meanstd-erase0502-nolbsm-avgmax-arcface30_035_10033-cj05-trainVal2/ #(h, w)
# WEIGHT=${MODEL_DIR}cosine_baseline_epoch90.pth
# SAVE_DIR=${MODEL_DIR}eval/
# # 
# #   --flip \
# #     --aqe --aqe_k2 7 --aqe_alpha 3.0 \
# CUDA_VISIBLE_DEVICES=6 python test2.py --config_file='configs/naic/arcface_baseline.yml' \
#     TEST.IMS_PER_BATCH "128" TEST.RANDOMPERM "5" \
#     MODEL.BASELINE.S "30.0" MODEL.BASELINE.M "0.35" \
#     MODEL.BASELINE.COSINE_LOSS_TYPE 'ArcCos' \
#     INPUT.SIZE_TRAIN "([384,192])" INPUT.SIZE_TEST "([384,192])" \
#     MODEL.NAME "cosine_baseline" MODEL.BACKBONE "('resnet101_ibn_a')" MODEL.BASELINE.POOL_TYPE "GlobalAvgMaxPool2d"\
#     DATASETS.DATA_PATH "('${DATA_DIR}')" DATASETS.TRAIN_PATH "rep_trainVal2"\
#     MODEL.PRETRAIN_PATH "('${PRETRAIN}')"  \
#     OUTPUT_DIR "('${SAVE_DIR}')" \
#     TEST.WEIGHT "${WEIGHT}"


# --rerank --k1 9 --k2 3 --lambda_value 0.6 \
# --dba --dba_k2 10 --dba_alpha -3.0

# SAVE_DIR=${MODEL_DIR}sub/
# CUDA_VISIBLE_DEVICES=6 python test2.py --config_file='configs/naic/arcface_baseline.yml' \
#     --sub \
#     --flip \
#     --aqe --aqe_k2 7 --aqe_alpha 3.0 \
#     --query_dir ${QUERY_DIR}\
#     --gallery_dir ${GALLERY_DIR}\
#     MODEL.BASELINE.S "30.0" MODEL.BASELINE.M "0.35" \
#     MODEL.BASELINE.COSINE_LOSS_TYPE 'ArcCos' \
#     INPUT.SIZE_TRAIN "([384,192])" INPUT.SIZE_TEST "([384,192])" \
#     MODEL.NAME "cosine_baseline" MODEL.BACKBONE "('resnet101_ibn_a')" MODEL.BASELINE.POOL_TYPE "GlobalAvgMaxPool2d"\
#     DATASETS.DATA_PATH "('${DATA_DIR}')" DATASETS.TRAIN_PATH "rep_trainVal2"\
#     MODEL.PRETRAIN_PATH "('${PRETRAIN}')"  \
#     OUTPUT_DIR "('${SAVE_DIR}')" \
#     TEST.WEIGHT "${WEIGHT}"

# [20191228]
DATA_DIR=${ROOT_DIR}rep_dataset/
PRETRAIN=../weights/resnet101_ibn_a.pth.tar
MODEL_DIR=../rep_work_dirs/exp4-cosinebaseline-resnet101ibnls1-384x192-bs16x6-warmup10-flip-pad10-meanstd-erase0502-nolbsm-avg-cosface30_035_10033-cj05-trainVal2-pseudo/ #(h, w)
WEIGHT=${MODEL_DIR}cosine_baseline_epoch10.pth
SAVE_DIR=${MODEL_DIR}eval/
# # 
# #   --flip \
# #     --aqe --aqe_k2 7 --aqe_alpha 3.0 \
# CUDA_VISIBLE_DEVICES=6 python test2.py --config_file='configs/naic/arcface_baseline.yml' \
#     TEST.IMS_PER_BATCH "128" TEST.RANDOMPERM "5" \
#     MODEL.BASELINE.S "30.0" MODEL.BASELINE.M "0.35" \
#     MODEL.BASELINE.COSINE_LOSS_TYPE 'CosFace' \
#     INPUT.SIZE_TRAIN "([384,192])" INPUT.SIZE_TEST "([384,192])" \
#     MODEL.NAME "cosine_baseline" MODEL.BACKBONE "('resnet101_ibn_a')" MODEL.BASELINE.POOL_TYPE "avg"\
#     DATASETS.DATA_PATH "('${DATA_DIR}')" DATASETS.TRAIN_PATH "/data/Dataset/PReID/pseudo_aqe_eps02_2_dataset/rep_trainVal2"\
#     MODEL.PRETRAIN_PATH "('${PRETRAIN}')"  \
#     OUTPUT_DIR "('${SAVE_DIR}')" \
#     TEST.WEIGHT "${WEIGHT}"



# SAVE_DIR=${MODEL_DIR}sub/
# CUDA_VISIBLE_DEVICES=6 python test2.py --config_file='configs/naic/arcface_baseline.yml' \
#     --sub \
#     --flip \
#     --aqe --aqe_k2 7 --aqe_alpha 3.0 \
#     --query_dir ${QUERY_DIR}\
#     --gallery_dir ${GALLERY_DIR}\
#     MODEL.BASELINE.S "30.0" MODEL.BASELINE.M "0.35" \
#     MODEL.BASELINE.COSINE_LOSS_TYPE 'CosFace' \
#     INPUT.SIZE_TRAIN "([384,192])" INPUT.SIZE_TEST "([384,192])" \
#     MODEL.NAME "cosine_baseline" MODEL.BACKBONE "('resnet101_ibn_a')" MODEL.BASELINE.POOL_TYPE "avg"\
#     DATASETS.DATA_PATH "('${DATA_DIR}')" DATASETS.TRAIN_PATH "/data/Dataset/PReID/pseudo_aqe_eps02_2_dataset/rep_trainVal2"\
#     MODEL.PRETRAIN_PATH "('${PRETRAIN}')"  \
#     OUTPUT_DIR "('${SAVE_DIR}')" \
#     TEST.WEIGHT "${WEIGHT}"

DATA_DIR=${ROOT_DIR}rep_dataset/
PRETRAIN=../weights/r50_ibn_a.pth
MODEL_DIR=../rep_work_dirs/exp4-cosinebaseline-resnet50ibnls1_sestn-384x192-bs16x6-warmup10-flip-pad10-meanstd-erase0502-nolbsm-avg-cosface30_035_10033-cj-minInst2/ #(h, w)
WEIGHT=${MODEL_DIR}cosine_baseline_epoch90.pth
SAVE_DIR=${MODEL_DIR}eval/
# # 
    # 

# CUDA_VISIBLE_DEVICES=6 python test2.py --config_file='configs/naic/arcface_baseline.yml' \
#     --aqe --aqe_k2 7 --aqe_alpha 3.0 \
#     --rerank --k1 7 --k2 2 --lambda_value 0.6 \
#     TEST.IMS_PER_BATCH "128" TEST.RANDOMPERM "5" \
#     MODEL.BASELINE.S "30.0" MODEL.BASELINE.M "0.35" \
#     MODEL.BASELINE.COSINE_LOSS_TYPE 'CosFace' \
#     INPUT.SIZE_TRAIN "([384,192])" INPUT.SIZE_TEST "([384,192])" \
#     MODEL.NAME "cosine_baseline" MODEL.BACKBONE "('resnet50_ibn_a')" MODEL.BASELINE.POOL_TYPE "avg" MODEL.BASELINE.USE_SESTN "True"\
#     DATASETS.DATA_PATH "('${DATA_DIR}')" DATASETS.TRAIN_PATH "rep_f0_train2"\
#     MODEL.PRETRAIN_PATH "('${PRETRAIN}')"  \
#     OUTPUT_DIR "('${SAVE_DIR}')" \
#     TEST.WEIGHT "${WEIGHT}"

# # --rerank --k1 9 --k2 3 --lambda_value 0.6 \
# # --dba --dba_k2 10 --dba_alpha -3.0

# SAVE_DIR=${MODEL_DIR}sub/
# CUDA_VISIBLE_DEVICES=6 python test2.py --config_file='configs/naic/arcface_baseline.yml' \
#     --sub \
#     --flip \
#     --aqe --aqe_k2 7 --aqe_alpha 3.0 \
#     --query_dir ${QUERY_DIR}\
#     --gallery_dir ${GALLERY_DIR}\
#     MODEL.BASELINE.S "30.0" MODEL.BASELINE.M "0.35" \
#     MODEL.BASELINE.COSINE_LOSS_TYPE 'CosFace' \
#     INPUT.SIZE_TRAIN "([384,192])" INPUT.SIZE_TEST "([384,192])" \
#     MODEL.NAME "cosine_baseline" MODEL.BACKBONE "('resnet50_ibn_a')" MODEL.BASELINE.POOL_TYPE "avg" MODEL.BASELINE.USE_SESTN "True"\
#     DATASETS.DATA_PATH "('${DATA_DIR}')" DATASETS.TRAIN_PATH "rep_f0_train2"\
#     MODEL.PRETRAIN_PATH "('${PRETRAIN}')"  \
#     OUTPUT_DIR "('${SAVE_DIR}')" \
#     TEST.WEIGHT "${WEIGHT}"

# [20191229]
DATA_DIR=${ROOT_DIR}rep_dataset/
PRETRAIN=../weights/resnet101_ibn_a.pth.tar
MODEL_DIR=../rep_work_dirs/exp4-cosinebaseline-resnet101ibnls1-288x144-bs16x8-warmup10-flip-pad10-meanstd-erase0504-lbsm-avg-arcface30_035_10033-cj05-trainVal/ #(h, w)
WEIGHT=${MODEL_DIR}cosine_baseline_epoch90.pth
SAVE_DIR=${MODEL_DIR}eval/
# 
#   --flip \
#     --aqe --aqe_k2 7 --aqe_alpha 3.0 \
# CUDA_VISIBLE_DEVICES=3 python test2.py --config_file='configs/naic/arcface_baseline.yml' \
#     TEST.IMS_PER_BATCH "128" TEST.RANDOMPERM "5" \
#     MODEL.BASELINE.S "30.0" MODEL.BASELINE.M "0.35" \
#     MODEL.BASELINE.COSINE_LOSS_TYPE 'ArcCos' \
#     INPUT.SIZE_TRAIN "([288,144])" INPUT.SIZE_TEST "([288,144])" \
#     MODEL.NAME "cosine_baseline" MODEL.BACKBONE "('resnet101_ibn_a')" MODEL.BASELINE.POOL_TYPE "avg"\
#     DATASETS.DATA_PATH "('${DATA_DIR}')" DATASETS.TRAIN_PATH "rep_trainVal"\
#     MODEL.PRETRAIN_PATH "('${PRETRAIN}')"  \
#     OUTPUT_DIR "('${SAVE_DIR}')" \
#     TEST.WEIGHT "${WEIGHT}"


# --rerank --k1 9 --k2 3 --lambda_value 0.6 \
# --dba --dba_k2 10 --dba_alpha -3.0

# SAVE_DIR=${MODEL_DIR}sub/
# CUDA_VISIBLE_DEVICES=3 python test2.py --config_file='configs/naic/arcface_baseline.yml' \
#     --sub \
#     --flip \
#     --aqe --aqe_k2 7 --aqe_alpha 3.0 \
#     --query_dir ${QUERY_DIR}\
#     --gallery_dir ${GALLERY_DIR}\
#     MODEL.BASELINE.S "30.0" MODEL.BASELINE.M "0.35" \
#     MODEL.BASELINE.COSINE_LOSS_TYPE 'ArcCos' \
#     INPUT.SIZE_TRAIN "([288,144])" INPUT.SIZE_TEST "([288,144])" \
#     MODEL.NAME "cosine_baseline" MODEL.BACKBONE "('resnet101_ibn_a')" MODEL.BASELINE.POOL_TYPE "avg"\
#     DATASETS.DATA_PATH "('${DATA_DIR}')" DATASETS.TRAIN_PATH "rep_trainVal"\
#     MODEL.PRETRAIN_PATH "('${PRETRAIN}')"  \
#     OUTPUT_DIR "('${SAVE_DIR}')" \
#     TEST.WEIGHT "${WEIGHT}"

# DATA_DIR=${ROOT_DIR}rep_dataset/
# PRETRAIN=../weights/resnet101_ibn_a.pth.tar
# MODEL_DIR=../rep_work_dirs/exp4-cosinebaseline-resnet101ibnls1-288x144-bs16x8-warmup10-flip-pad10-nomeanstd-erase0504-lbsm-avg-arcface30_035_10033-cj05-trainVal/ #(h, w)
# WEIGHT=${MODEL_DIR}cosine_baseline_epoch90.pth
# SAVE_DIR=${MODEL_DIR}eval/
# # 
# #   --flip \
# #     --aqe --aqe_k2 7 --aqe_alpha 3.0 \
# CUDA_VISIBLE_DEVICES=0 python test2.py --config_file='configs/naic/arcface_baseline.yml' \
#     TEST.IMS_PER_BATCH "128" TEST.RANDOMPERM "5" \
#     INPUT.PIXEL_MEAN "[0.485, 0.456, 0.406]" INPUT.PIXEL_STD "[0.229, 0.224, 0.225]" \
#     MODEL.BASELINE.S "30.0" MODEL.BASELINE.M "0.35" \
#     MODEL.BASELINE.COSINE_LOSS_TYPE 'ArcCos' \
#     INPUT.SIZE_TRAIN "([288,144])" INPUT.SIZE_TEST "([288,144])" \
#     MODEL.NAME "cosine_baseline" MODEL.BACKBONE "('resnet101_ibn_a')" MODEL.BASELINE.POOL_TYPE "avg"\
#     DATASETS.DATA_PATH "('${DATA_DIR}')" DATASETS.TRAIN_PATH "rep_trainVal"\
#     MODEL.PRETRAIN_PATH "('${PRETRAIN}')"  \
#     OUTPUT_DIR "('${SAVE_DIR}')" \
#     TEST.WEIGHT "${WEIGHT}"


# --rerank --k1 9 --k2 3 --lambda_value 0.6 \
# --dba --dba_k2 10 --dba_alpha -3.0

# SAVE_DIR=${MODEL_DIR}sub/
# CUDA_VISIBLE_DEVICES=0 python test2.py --config_file='configs/naic/arcface_baseline.yml' \
#     --sub \
#     --flip \
#     --aqe --aqe_k2 7 --aqe_alpha 3.0 \
#     --query_dir ${QUERY_DIR}\
#     --gallery_dir ${GALLERY_DIR}\
#     INPUT.PIXEL_MEAN "[0.485, 0.456, 0.406]" INPUT.PIXEL_STD "[0.229, 0.224, 0.225]" \
#     MODEL.BASELINE.S "30.0" MODEL.BASELINE.M "0.35" \
#     MODEL.BASELINE.COSINE_LOSS_TYPE 'ArcCos' \
#     INPUT.SIZE_TRAIN "([288,144])" INPUT.SIZE_TEST "([288,144])" \
#     MODEL.NAME "cosine_baseline" MODEL.BACKBONE "('resnet101_ibn_a')" MODEL.BASELINE.POOL_TYPE "avg"\
#     DATASETS.DATA_PATH "('${DATA_DIR}')" DATASETS.TRAIN_PATH "rep_trainVal"\
#     MODEL.PRETRAIN_PATH "('${PRETRAIN}')"  \
#     OUTPUT_DIR "('${SAVE_DIR}')" \
#     TEST.WEIGHT "${WEIGHT}"

# [20191228]
DATA_DIR=${ROOT_DIR}rep_dataset/
PRETRAIN=../weights/resnet101_ibn_a.pth.tar
MODEL_DIR=../rep_work_dirs/exp4-cosinebaseline-resnet101ibnls1-384x192-bs16x6-warmup10-flip-pad10-meanstd-erase0502-nolbsm-avg-arcface30_035_10033-cj05-trainVal2-pseudo/ #(h, w)
WEIGHT=${MODEL_DIR}cosine_baseline_epoch16.pth
SAVE_DIR=${MODEL_DIR}eval/
# # 
# #   --flip \
# #     --aqe --aqe_k2 7 --aqe_alpha 3.0 \
# CUDA_VISIBLE_DEVICES=6 python test2.py --config_file='configs/naic/arcface_baseline.yml' \
#     TEST.IMS_PER_BATCH "128" TEST.RANDOMPERM "5" \
#     MODEL.BASELINE.S "30.0" MODEL.BASELINE.M "0.35" \
#     MODEL.BASELINE.COSINE_LOSS_TYPE 'ArcCos' \
#     INPUT.SIZE_TRAIN "([384,192])" INPUT.SIZE_TEST "([384,192])" \
#     MODEL.NAME "cosine_baseline" MODEL.BACKBONE "('resnet101_ibn_a')" MODEL.BASELINE.POOL_TYPE "avg"\
#     DATASETS.DATA_PATH "('${DATA_DIR}')" DATASETS.TRAIN_PATH "/data/Dataset/PReID/pseudo_auto_flip_eps055_2_dataset/rep_trainVal2"\
#     MODEL.PRETRAIN_PATH "('${PRETRAIN}')"  \
#     OUTPUT_DIR "('${SAVE_DIR}')" \
#     TEST.WEIGHT "${WEIGHT}"




# SAVE_DIR=${MODEL_DIR}sub/
# CUDA_VISIBLE_DEVICES=6 python test2.py --config_file='configs/naic/arcface_baseline.yml' \
#     --sub \
#     --flip \
#     --aqe --aqe_k2 7 --aqe_alpha 3.0 \
#     --query_dir ${QUERY_DIR}\
#     --gallery_dir ${GALLERY_DIR}\
#     MODEL.BASELINE.S "30.0" MODEL.BASELINE.M "0.35" \
#     MODEL.BASELINE.COSINE_LOSS_TYPE 'ArcCos' \
#     INPUT.SIZE_TRAIN "([384,192])" INPUT.SIZE_TEST "([384,192])" \
#     MODEL.NAME "cosine_baseline" MODEL.BACKBONE "('resnet101_ibn_a')" MODEL.BASELINE.POOL_TYPE "avg"\
#     DATASETS.DATA_PATH "('${DATA_DIR}')" DATASETS.TRAIN_PATH "/data/Dataset/PReID/pseudo_auto_flip_eps055_2_dataset/rep_trainVal2"\
#     MODEL.PRETRAIN_PATH "('${PRETRAIN}')"  \
#     OUTPUT_DIR "('${SAVE_DIR}')" \
#     TEST.WEIGHT "${WEIGHT}"

# [20191228]
DATA_DIR=${ROOT_DIR}rep_dataset/
PRETRAIN=../weights/resnet101_ibn_a.pth.tar
MODEL_DIR=../rep_work_dirs/exp4-cosinebaseline-resnet101ibnls1-384x192-bs16x6-warmup10-flip-pad10-meanstd-erase0502-nolbsm-avg-arcface30_035_10033-cj05-trainVal2-pseudo_smlbsm/ #(h, w)
WEIGHT=${MODEL_DIR}cosine_baseline_epoch16.pth
SAVE_DIR=${MODEL_DIR}eval/
# # 
# #   --flip \
# #     --aqe --aqe_k2 7 --aqe_alpha 3.0 \
# CUDA_VISIBLE_DEVICES=6 python test2.py --config_file='configs/naic/arcface_baseline.yml' \
#     TEST.IMS_PER_BATCH "128" TEST.RANDOMPERM "5" \
#     MODEL.BASELINE.S "30.0" MODEL.BASELINE.M "0.35" \
#     MODEL.BASELINE.COSINE_LOSS_TYPE 'ArcCos' \
#     INPUT.SIZE_TRAIN "([384,192])" INPUT.SIZE_TEST "([384,192])" \
#     MODEL.NAME "cosine_baseline" MODEL.BACKBONE "('resnet101_ibn_a')" MODEL.BASELINE.POOL_TYPE "avg"\
#     DATASETS.DATA_PATH "('${DATA_DIR}')" DATASETS.TRAIN_PATH "/data/Dataset/PReID/pseudo_auto_flip_eps055_2_dataset/rep_trainVal2"\
#     MODEL.PRETRAIN_PATH "('${PRETRAIN}')"  \
#     OUTPUT_DIR "('${SAVE_DIR}')" \
#     TEST.WEIGHT "${WEIGHT}"




# SAVE_DIR=${MODEL_DIR}sub/
# CUDA_VISIBLE_DEVICES=6 python test2.py --config_file='configs/naic/arcface_baseline.yml' \
#     --sub \
#     --flip \
#     --aqe --aqe_k2 7 --aqe_alpha 3.0 \
#     --query_dir ${QUERY_DIR}\
#     --gallery_dir ${GALLERY_DIR}\
#     MODEL.BASELINE.S "30.0" MODEL.BASELINE.M "0.35" \
#     MODEL.BASELINE.COSINE_LOSS_TYPE 'ArcCos' \
#     INPUT.SIZE_TRAIN "([384,192])" INPUT.SIZE_TEST "([384,192])" \
#     MODEL.NAME "cosine_baseline" MODEL.BACKBONE "('resnet101_ibn_a')" MODEL.BASELINE.POOL_TYPE "avg"\
#     DATASETS.DATA_PATH "('${DATA_DIR}')" DATASETS.TRAIN_PATH "/data/Dataset/PReID/pseudo_auto_flip_eps055_2_dataset/rep_trainVal2"\
#     MODEL.PRETRAIN_PATH "('${PRETRAIN}')"  \
#     OUTPUT_DIR "('${SAVE_DIR}')" \
#     TEST.WEIGHT "${WEIGHT}"

# [20191229]
DATA_DIR=${ROOT_DIR}rep_dataset/
PRETRAIN=../weights/resnet101_ibn_a.pth.tar
MODEL_DIR=../rep_work_dirs/exp4-cosinebaseline-resnet101ibnls1-384x192-bs16x6-warmup10-flip-pad10-nomeanstd-erase0504-nolbsm-avg-arcface30_035_10033-cj05-trainVal2/ #(h, w)
WEIGHT=${MODEL_DIR}cosine_baseline_epoch90.pth
SAVE_DIR=${MODEL_DIR}eval/
# 
#   --flip \
#     --aqe --aqe_k2 7 --aqe_alpha 3.0 \
# CUDA_VISIBLE_DEVICES=6 python test2.py --config_file='configs/naic/arcface_baseline.yml' \
#     TEST.IMS_PER_BATCH "128" TEST.RANDOMPERM "5" \
#     INPUT.PIXEL_MEAN "[0.485, 0.456, 0.406]" INPUT.PIXEL_STD "[0.229, 0.224, 0.225]" \
#     MODEL.BASELINE.S "30.0" MODEL.BASELINE.M "0.35" \
#     MODEL.BASELINE.COSINE_LOSS_TYPE 'ArcCos' \
#     INPUT.SIZE_TRAIN "([384,192])" INPUT.SIZE_TEST "([384,192])" \
#     MODEL.NAME "cosine_baseline" MODEL.BACKBONE "('resnet101_ibn_a')" MODEL.BASELINE.POOL_TYPE "avg"\
#     DATASETS.DATA_PATH "('${DATA_DIR}')" DATASETS.TRAIN_PATH "rep_trainVal2"\
#     MODEL.PRETRAIN_PATH "('${PRETRAIN}')"  \
#     OUTPUT_DIR "('${SAVE_DIR}')" \
#     TEST.WEIGHT "${WEIGHT}"


# --rerank --k1 9 --k2 3 --lambda_value 0.6 \
# --dba --dba_k2 10 --dba_alpha -3.0

# SAVE_DIR=${MODEL_DIR}sub/
# CUDA_VISIBLE_DEVICES=6 python test2.py --config_file='configs/naic/arcface_baseline.yml' \
#     --sub \
#     --flip \
#     --aqe --aqe_k2 7 --aqe_alpha 3.0 \
#     --query_dir ${QUERY_DIR}\
#     --gallery_dir ${GALLERY_DIR}\
#     INPUT.PIXEL_MEAN "[0.485, 0.456, 0.406]" INPUT.PIXEL_STD "[0.229, 0.224, 0.225]" \
#     MODEL.BASELINE.S "30.0" MODEL.BASELINE.M "0.35" \
#     MODEL.BASELINE.COSINE_LOSS_TYPE 'ArcCos' \
#     INPUT.SIZE_TRAIN "([384,192])" INPUT.SIZE_TEST "([384,192])" \
#     MODEL.NAME "cosine_baseline" MODEL.BACKBONE "('resnet101_ibn_a')" MODEL.BASELINE.POOL_TYPE "avg"\
#     DATASETS.DATA_PATH "('${DATA_DIR}')" DATASETS.TRAIN_PATH "rep_trainVal2"\
#     MODEL.PRETRAIN_PATH "('${PRETRAIN}')"  \
#     OUTPUT_DIR "('${SAVE_DIR}')" \
#     TEST.WEIGHT "${WEIGHT}"

# [20191230]
DATA_DIR=${ROOT_DIR}rep_dataset/
PRETRAIN=../weights/resnet101_ibn_a.pth.tar
MODEL_DIR=../rep_work_dirs/exp4-cosinebaseline-resnet101ibnls1-384x192-bs16x6-warmup10-flip-pad10-meanstd-erase0502-nolbsm-avg-arcface30_035_10033-cj05-trainVal2-pseudo_resume/ #(h, w)
WEIGHT=${MODEL_DIR}cosine_baseline_epoch60.pth
SAVE_DIR=${MODEL_DIR}eval/
# # 
# #   --flip \
# #     --aqe --aqe_k2 7 --aqe_alpha 3.0 \
# CUDA_VISIBLE_DEVICES=6 python test2.py --config_file='configs/naic/arcface_baseline.yml' \
#     TEST.IMS_PER_BATCH "128" TEST.RANDOMPERM "5" \
#     MODEL.BASELINE.S "30.0" MODEL.BASELINE.M "0.35" \
#     MODEL.BASELINE.COSINE_LOSS_TYPE 'ArcCos' \
#     INPUT.SIZE_TRAIN "([384,192])" INPUT.SIZE_TEST "([384,192])" \
#     MODEL.NAME "cosine_baseline" MODEL.BACKBONE "('resnet101_ibn_a')" MODEL.BASELINE.POOL_TYPE "avg"\
#     DATASETS.DATA_PATH "('${DATA_DIR}')" DATASETS.TRAIN_PATH "/data/Dataset/PReID/pseudo_auto_flip_eps055_2_dataset/rep_trainVal2"\
#     MODEL.PRETRAIN_PATH "('${PRETRAIN}')"  \
#     OUTPUT_DIR "('${SAVE_DIR}')" \
#     TEST.WEIGHT "${WEIGHT}"




# SAVE_DIR=${MODEL_DIR}sub/
# CUDA_VISIBLE_DEVICES=6 python test2.py --config_file='configs/naic/arcface_baseline.yml' \
#     --sub --save_fname "pseudo_resume_e60_" \
#     --flip \
#     --aqe --aqe_k2 7 --aqe_alpha 3.0 \
#     --query_dir ${QUERY_DIR}\
#     --gallery_dir ${GALLERY_DIR}\
#     MODEL.BASELINE.S "30.0" MODEL.BASELINE.M "0.35" \
#     MODEL.BASELINE.COSINE_LOSS_TYPE 'ArcCos' \
#     INPUT.SIZE_TRAIN "([384,192])" INPUT.SIZE_TEST "([384,192])" \
#     MODEL.NAME "cosine_baseline" MODEL.BACKBONE "('resnet101_ibn_a')" MODEL.BASELINE.POOL_TYPE "avg"\
#     DATASETS.DATA_PATH "('${DATA_DIR}')" DATASETS.TRAIN_PATH "/data/Dataset/PReID/pseudo_auto_flip_eps055_2_dataset/rep_trainVal2"\
#     MODEL.PRETRAIN_PATH "('${PRETRAIN}')"  \
#     OUTPUT_DIR "('${SAVE_DIR}')" \
#     TEST.WEIGHT "${WEIGHT}"

# [20191230]
DATA_DIR=${ROOT_DIR}rep_dataset/
PRETRAIN=../weights/resnet101_ibn_a.pth.tar
MODEL_DIR=../rep_work_dirs/exp4-cosinebaseline-resnet101ibnls1-384x192-bs16x6-warmup10-flip-pad10-meanstd-erase0502-nolbsm-avg-arcface30_035_10033-cj05-trainVal2-pseudo_resume_smlbsm/ #(h, w)
# WEIGHT=${MODEL_DIR}cosine_baseline_epoch60.pth
WEIGHT=${MODEL_DIR}cosine_baseline_epoch50.pth
SAVE_DIR=${MODEL_DIR}eval/
# # 
# #   --flip \
# # #     --aqe --aqe_k2 7 --aqe_alpha 3.0 \
# CUDA_VISIBLE_DEVICES=6 python test2.py --config_file='configs/naic/arcface_baseline.yml' \
#     TEST.IMS_PER_BATCH "128" TEST.RANDOMPERM "5" \
#     MODEL.USE_COS "False" MODEL.BASELINE.COSINE_LOSS_TYPE ''  MODEL.BASELINE.TPL_WEIGHT "1.0" MODEL.BASELINE.CE_WEIGHT "1.0" MODEL.LABEL_SMOOTH "True" \
#     INPUT.SIZE_TRAIN "([384,192])" INPUT.SIZE_TEST "([384,192])" \
#     MODEL.NAME "cosine_baseline" MODEL.BACKBONE "('resnet101_ibn_a')" MODEL.BASELINE.POOL_TYPE "avg"\
#     DATASETS.DATA_PATH "('${DATA_DIR}')" DATASETS.TRAIN_PATH "/data/Dataset/PReID/pseudo_auto_flip_eps055_2_dataset/rep_trainVal2"\
#     MODEL.PRETRAIN_PATH "('${PRETRAIN}')"  \
#     OUTPUT_DIR "('${SAVE_DIR}')" \
#     TEST.WEIGHT "${WEIGHT}"

# SAVE_DIR=${MODEL_DIR}sub/
# CUDA_VISIBLE_DEVICES=6 python test2.py --config_file='configs/naic/arcface_baseline.yml' \
#     --sub --save_fname "pseudo_resume_smlbsm_e50_" \
#     --flip \
#     --aqe --aqe_k2 7 --aqe_alpha 3.0 \
#     --query_dir ${QUERY_DIR}\
#     --gallery_dir ${GALLERY_DIR}\
#     MODEL.USE_COS "False" MODEL.BASELINE.COSINE_LOSS_TYPE ''  MODEL.BASELINE.TPL_WEIGHT "1.0" MODEL.BASELINE.CE_WEIGHT "1.0" MODEL.LABEL_SMOOTH "True" \
#     INPUT.SIZE_TRAIN "([384,192])" INPUT.SIZE_TEST "([384,192])" \
#     MODEL.NAME "cosine_baseline" MODEL.BACKBONE "('resnet101_ibn_a')" MODEL.BASELINE.POOL_TYPE "avg"\
#     DATASETS.DATA_PATH "('${DATA_DIR}')" DATASETS.TRAIN_PATH "/data/Dataset/PReID/pseudo_auto_flip_eps055_2_dataset/rep_trainVal2"\
#     MODEL.PRETRAIN_PATH "('${PRETRAIN}')"  \
#     OUTPUT_DIR "('${SAVE_DIR}')" \
#     TEST.WEIGHT "${WEIGHT}"

# # [pseudo]

# DATA_DIR=${ROOT_DIR}rep_dataset/
# PRETRAIN=../weights/resnet101_ibn_a.pth.tar
# MODEL_DIR=../rep_work_dirs/exp4-cosinebaseline-resnet101ibnls1-384x192-bs16x6-warmup10-flip-pad10-meanstd-erase0502-nolbsm-avg-cosface30_035_10033-cj05-trainVal2/ #(h, w)
# WEIGHT=${MODEL_DIR}cosine_baseline_epoch90.pth
# SAVE_DIR=${MODEL_DIR}eval/

# # --aqe --aqe_k2 7 --aqe_alpha 3.0 \
# #     --pseudo  --pseudo_visual --pseudo_eps 0.35 --pseudo_minpoints 2 --pseudo_maxpoints 100 --pseudo_savepath '../rep_work_dirs/pseudo_1/'\
 
# SAVE_DIR=${MODEL_DIR}sub/
# CUDA_VISIBLE_DEVICES=4 python test2.py --config_file='configs/naic/arcface_baseline.yml' \
#     --sub \
#     --flip \
#     --aqe --aqe_k2 7 --aqe_alpha 3.0 \
#     --pseudo  --pseudo_eps 0.20 --pseudo_minpoints 2 --pseudo_maxpoints 100 --pseudo_savepath '../rep_work_dirs/pseudo_aqe_eps02_2/'\
#     --query_dir ${QUERY_DIR}\
#     --gallery_dir ${GALLERY_DIR}\
#     MODEL.BASELINE.S "30.0" MODEL.BASELINE.M "0.35" \
#     MODEL.BASELINE.COSINE_LOSS_TYPE 'CosFace' \
#     INPUT.SIZE_TRAIN "([384,192])" INPUT.SIZE_TEST "([384,192])" \
#     MODEL.NAME "cosine_baseline" MODEL.BACKBONE "('resnet101_ibn_a')" MODEL.BASELINE.POOL_TYPE "avg"\
#     DATASETS.DATA_PATH "('${DATA_DIR}')" DATASETS.TRAIN_PATH "rep_trainVal2"\
#     MODEL.PRETRAIN_PATH "('${PRETRAIN}')"  \
#     OUTPUT_DIR "('${SAVE_DIR}')" \
#     TEST.WEIGHT "${WEIGHT}"


# [pseudo1229]

# DATA_DIR=${ROOT_DIR}rep_dataset/
# PRETRAIN=../weights/resnet101_ibn_a.pth.tar
# MODEL_DIR=../rep_work_dirs/exp4-cosinebaseline-resnet101ibnls1-384x192-bs16x6-warmup10-flip-pad10-meanstd-erase0502-nolbsm-avg-arcface30_035_10033-cj05-trainVal2/ #(h, w)
# WEIGHT=${MODEL_DIR}cosine_baseline_epoch90.pth
# SAVE_DIR=${MODEL_DIR}eval/

# # --aqe --aqe_k2 7 --aqe_alpha 3.0 \
# #     --pseudo  --pseudo_visual --pseudo_eps 0.35 --pseudo_minpoints 2 --pseudo_maxpoints 100 --pseudo_savepath '../rep_work_dirs/pseudo_1/'\
 
# SAVE_DIR=${MODEL_DIR}sub/
# CUDA_VISIBLE_DEVICES=3 python test2.py --config_file='configs/naic/arcface_baseline.yml' \
#     --sub \
#     --flip \
#     --pseudo --pseudo_algorithm "auto" --pseudo_eps 0.55 --pseudo_minpoints 2 --pseudo_maxpoints 100 --pseudo_savepath '../rep_work_dirs/pseudo_auto_flip_eps055_2/'\
#     --query_dir ${QUERY_DIR}\
#     --gallery_dir ${GALLERY_DIR}\
#     MODEL.BASELINE.S "30.0" MODEL.BASELINE.M "0.35" \
#     MODEL.BASELINE.COSINE_LOSS_TYPE 'ArcCos' \
#     INPUT.SIZE_TRAIN "([384,192])" INPUT.SIZE_TEST "([384,192])" \
#     MODEL.NAME "cosine_baseline" MODEL.BACKBONE "('resnet101_ibn_a')" MODEL.BASELINE.POOL_TYPE "avg"\
#     DATASETS.DATA_PATH "('${DATA_DIR}')" DATASETS.TRAIN_PATH "rep_trainVal2"\
#     MODEL.PRETRAIN_PATH "('${PRETRAIN}')"  \
#     OUTPUT_DIR "('${SAVE_DIR}')" \
#     TEST.WEIGHT "${WEIGHT}"




# [debug]
DATA_DIR=${ROOT_DIR}rep_dataset/
PRETRAIN=../weights/resnet101_ibn_a.pth.tar
MODEL_DIR=../rep_work_dirs/exp4-cosinebaseline-resnet101ibnls1-384x192-bs16x6-warmup10-flip-pad10-meanstd-erase0502-nolbsm-avg-cosface30_035_10033-cj-minInst2/ #(h, w)
WEIGHT=${MODEL_DIR}cosine_baseline_epoch90.pth
SAVE_DIR=${MODEL_DIR}eval/
# 
#   --flip \
#     --aqe --aqe_k2 7 --aqe_alpha 3.0 \
    # --aqe --aqe_k2 7 --aqe_alpha 3.0 \

# CUDA_VISIBLE_DEVICES=4 python test2.py --config_file='configs/naic/arcface_baseline.yml' \
#     --pseudo --pseudo_eps 0.55 --pseudo_minpoints 3 --pseudo_maxpoints 50 \
#     TEST.IMS_PER_BATCH "128" TEST.RANDOMPERM "5" \
#     MODEL.BASELINE.S "30.0" MODEL.BASELINE.M "0.35" \
#     MODEL.BASELINE.COSINE_LOSS_TYPE 'CosFace' \
#     INPUT.SIZE_TRAIN "([384,192])" INPUT.SIZE_TEST "([384,192])" \
#     MODEL.NAME "cosine_baseline" MODEL.BACKBONE "('resnet101_ibn_a')" MODEL.BASELINE.POOL_TYPE "avg"\
#     DATASETS.DATA_PATH "('${DATA_DIR}')" DATASETS.TRAIN_PATH "rep_f0_train2"\
#     MODEL.PRETRAIN_PATH "('${PRETRAIN}')"  \
#     OUTPUT_DIR "('${SAVE_DIR}')" \
#     TEST.WEIGHT "${WEIGHT}"

#     

# # --rerank --k1 9 --k2 3 --lambda_value 0.6 \
# # --dba --dba_k2 10 --dba_alpha -3.0
    # --pseudo  --pseudo_eps 0.5 --pseudo_minpoints 2 --pseudo_maxpoints 100 --pseudo_savepath '../rep_work_dirs/pseudo_aqe/'\

#   --aqe --aqe_k2 7 --aqe_alpha 3.0 \
#     # --flip \
# #     
# SAVE_DIR=${MODEL_DIR}sub/
# CUDA_VISIBLE_DEVICES=6 python test2.py --config_file='configs/naic/arcface_baseline.yml' \
#     --sub --flip\
#     --pseudo  --pseudo_visual --pseudo_eps 0.55 --pseudo_minpoints 2 --pseudo_maxpoints 100 --pseudo_savepath '../rep_work_dirs/pseudo_debug2/'\
#     --query_dir ${QUERY_DIR}\
#     --gallery_dir ${GALLERY_DIR}\
#     MODEL.BASELINE.S "30.0" MODEL.BASELINE.M "0.35" \
#     MODEL.BASELINE.COSINE_LOSS_TYPE 'CosFace' \
#     INPUT.SIZE_TRAIN "([384,192])" INPUT.SIZE_TEST "([384,192])" \
#     MODEL.NAME "cosine_baseline" MODEL.BACKBONE "('resnet101_ibn_a')" MODEL.BASELINE.POOL_TYPE "avg"\
#     DATASETS.DATA_PATH "('${DATA_DIR}')" DATASETS.TRAIN_PATH "rep_f0_train2"\
#     MODEL.PRETRAIN_PATH "('${PRETRAIN}')"  \
#     OUTPUT_DIR "('${SAVE_DIR}')" \
#     TEST.WEIGHT "${WEIGHT}"



# [20191231] testb pseudo

QUERY_DIR=${ROOT_DIR}dataset2/rep_B/query_b/
GALLERY_DIR=${ROOT_DIR}dataset2/rep_B/gallery_b/
DATA_DIR=${ROOT_DIR}rep_dataset/
# PRETRAIN=../weights/resnet101_ibn_a.pth.tar
# MODEL_DIR=../rep_work_dirs/exp4-cosinebaseline-resnet101ibnls1-384x192-bs16x6-warmup10-flip-pad10-meanstd-erase0502-nolbsm-avg-arcface30_035_10033-cj05-trainVal2/ #(h, w)
# WEIGHT=${MODEL_DIR}cosine_baseline_epoch90.pth
# SAVE_DIR=${MODEL_DIR}eval/

# # --aqe --aqe_k2 7 --aqe_alpha 3.0 \
# #     --pseudo  --pseudo_visual --pseudo_eps 0.35 --pseudo_minpoints 2 --pseudo_maxpoints 100 --pseudo_savepath '../rep_work_dirs/pseudo_1/'\
#     # --sub --save_fname 'testb_e90_'\
# #  --flip \
# SAVE_DIR=${MODEL_DIR}sub/
# CUDA_VISIBLE_DEVICES=2 python test2.py --config_file='configs/naic/arcface_baseline.yml' \
#     --sub \
#     --pseudo --pseudo_hist --pseudo_visual --pseudo_algorithm "auto" --pseudo_eps 0.55 --pseudo_minpoints 3 --pseudo_maxpoints 100 --pseudo_savepath '../rep_work_dirs/testb_pseudo_hist_065_080/'\
#     --query_dir ${QUERY_DIR}\
#     --gallery_dir ${GALLERY_DIR}\
#     MODEL.BASELINE.S "30.0" MODEL.BASELINE.M "0.35" \
#     MODEL.BASELINE.COSINE_LOSS_TYPE 'ArcCos' \
#     INPUT.SIZE_TRAIN "([384,192])" INPUT.SIZE_TEST "([384,192])" \
#     MODEL.NAME "cosine_baseline" MODEL.BACKBONE "('resnet101_ibn_a')" MODEL.BASELINE.POOL_TYPE "avg"\
#     DATASETS.DATA_PATH "('${DATA_DIR}')" DATASETS.TRAIN_PATH "rep_trainVal2"\
#     MODEL.PRETRAIN_PATH "('${PRETRAIN}')"  \
#     OUTPUT_DIR "('${SAVE_DIR}')" \
#     TEST.WEIGHT "${WEIGHT}"

DATA_DIR=${ROOT_DIR}rep_dataset/
PRETRAIN=../weights/resnet101_ibn_a.pth.tar
MODEL_DIR=../rep_work_dirs/exp4-cosinebaseline-resnet101ibnls1-384x192-bs16x6-warmup10-flip-pad10-meanstd-erase0502-nolbsm-avg-arcface30_035_10033-cj05-trainVal2-testb_pseudo_retrain_smlbsm/ #(h, w)
WEIGHT=${MODEL_DIR}cosine_baseline_epoch60.pth
SAVE_DIR=${MODEL_DIR}eval/
# # 
# #   --flip \
# # #     --aqe --aqe_k2 7 --aqe_alpha 3.0 \
# CUDA_VISIBLE_DEVICES=6 python test2.py --config_file='configs/naic/arcface_baseline.yml' \
#     TEST.IMS_PER_BATCH "128" TEST.RANDOMPERM "5" \
#     MODEL.USE_COS "False" MODEL.BASELINE.COSINE_LOSS_TYPE ''  MODEL.BASELINE.TPL_WEIGHT "1.0" MODEL.BASELINE.CE_WEIGHT "1.0" MODEL.LABEL_SMOOTH "True" \
#     INPUT.SIZE_TRAIN "([384,192])" INPUT.SIZE_TEST "([384,192])" \
#     MODEL.NAME "cosine_baseline" MODEL.BACKBONE "('resnet101_ibn_a')" MODEL.BASELINE.POOL_TYPE "avg"\
#     DATASETS.DATA_PATH "('${DATA_DIR}')" DATASETS.TRAIN_PATH "/data/Dataset/PReID/testb_pseudo_hist_065_080_dataset/rep_trainVal2"\
#     MODEL.PRETRAIN_PATH "('${PRETRAIN}')"  \
#     OUTPUT_DIR "('${SAVE_DIR}')" \
#     TEST.WEIGHT "${WEIGHT}"

# SAVE_DIR=${MODEL_DIR}sub/
# CUDA_VISIBLE_DEVICES=6 python test2.py --config_file='configs/naic/arcface_baseline.yml' \
#     --sub --save_fname "testb_pseudo_retrain_smlbsm_e60_" \
#     --flip \
#     --aqe --aqe_k2 7 --aqe_alpha 3.0 \
#     --query_dir ${QUERY_DIR}\
#     --gallery_dir ${GALLERY_DIR}\
#     MODEL.USE_COS "False" MODEL.BASELINE.COSINE_LOSS_TYPE ''  MODEL.BASELINE.TPL_WEIGHT "1.0" MODEL.BASELINE.CE_WEIGHT "1.0" MODEL.LABEL_SMOOTH "True" \
#     INPUT.SIZE_TRAIN "([384,192])" INPUT.SIZE_TEST "([384,192])" \
#     MODEL.NAME "cosine_baseline" MODEL.BACKBONE "('resnet101_ibn_a')" MODEL.BASELINE.POOL_TYPE "avg"\
#     DATASETS.DATA_PATH "('${DATA_DIR}')" DATASETS.TRAIN_PATH "/data/Dataset/PReID/testb_pseudo_hist_065_080_dataset/rep_trainVal2"\
#     MODEL.PRETRAIN_PATH "('${PRETRAIN}')"  \
#     OUTPUT_DIR "('${SAVE_DIR}')" \
#     TEST.WEIGHT "${WEIGHT}"


DATA_DIR=${ROOT_DIR}rep_dataset/
PRETRAIN=../weights/resnet101_ibn_a.pth.tar
MODEL_DIR=../rep_work_dirs/exp4-cosinebaseline-resnet101ibnls1-384x192-bs16x6-warmup10-flip-pad10-meanstd-erase0502-nolbsm-avg-arcface30_035_10033-cj05-trainVal2-testb_pseudo_retrain_arface/ #(h, w)
WEIGHT=${MODEL_DIR}cosine_baseline_epoch60.pth
SAVE_DIR=${MODEL_DIR}eval/
# # 
# #   --flip \
# # #     --aqe --aqe_k2 7 --aqe_alpha 3.0 \
# CUDA_VISIBLE_DEVICES=2 python test2.py --config_file='configs/naic/arcface_baseline.yml' \
#     TEST.IMS_PER_BATCH "128" TEST.RANDOMPERM "5" \
#     MODEL.BASELINE.TPL_WEIGHT "1.0" MODEL.BASELINE.CE_WEIGHT "0.33" MODEL.LABEL_SMOOTH "False" MODEL.BASELINE.S "30.0" MODEL.BASELINE.M "0.35" MODEL.BASELINE.COSINE_LOSS_TYPE 'ArcCos' \
#     INPUT.SIZE_TRAIN "([384,192])" INPUT.SIZE_TEST "([384,192])" \
#     MODEL.NAME "cosine_baseline" MODEL.BACKBONE "('resnet101_ibn_a')" MODEL.BASELINE.POOL_TYPE "avg"\
#     DATASETS.DATA_PATH "('${DATA_DIR}')" DATASETS.TRAIN_PATH "/data/Dataset/PReID/testb_pseudo_hist_065_080_dataset/rep_trainVal2"\
#     MODEL.PRETRAIN_PATH "('${PRETRAIN}')"  \
#     OUTPUT_DIR "('${SAVE_DIR}')" \
#     TEST.WEIGHT "${WEIGHT}"

SAVE_DIR=${MODEL_DIR}sub/
# CUDA_VISIBLE_DEVICES=2 python test2.py --config_file='configs/naic/arcface_baseline.yml' \
#     --sub --save_fname "testb_pseudo_retrain_arcface_e60_" \
#     --flip \
#     --aqe --aqe_k2 7 --aqe_alpha 3.0 \
#     --query_dir ${QUERY_DIR}\
#     --gallery_dir ${GALLERY_DIR}\
#     MODEL.BASELINE.TPL_WEIGHT "1.0" MODEL.BASELINE.CE_WEIGHT "0.33" MODEL.LABEL_SMOOTH "False" MODEL.BASELINE.S "30.0" MODEL.BASELINE.M "0.35" MODEL.BASELINE.COSINE_LOSS_TYPE 'ArcCos' \
#     INPUT.SIZE_TRAIN "([384,192])" INPUT.SIZE_TEST "([384,192])" \
#     MODEL.NAME "cosine_baseline" MODEL.BACKBONE "('resnet101_ibn_a')" MODEL.BASELINE.POOL_TYPE "avg"\
#     DATASETS.DATA_PATH "('${DATA_DIR}')" DATASETS.TRAIN_PATH "/data/Dataset/PReID/testb_pseudo_hist_065_080_dataset/rep_trainVal2"\
#     MODEL.PRETRAIN_PATH "('${PRETRAIN}')"  \
#     OUTPUT_DIR "('${SAVE_DIR}')" \
#     TEST.WEIGHT "${WEIGHT}"



DATA_DIR=${ROOT_DIR}rep_dataset/
PRETRAIN=../weights/resnet101_ibn_a.pth.tar
MODEL_DIR=../rep_work_dirs/exp4-cosinebaseline-resnet101ibnls1-384x192-bs16x6-warmup10-flip-pad10-meanstd-erase0502-nolbsm-avg-arcface30_035_10033-cj05-trainVal2-testb_pseudo_retrain_arface_e66/ #(h, w)
WEIGHT=${MODEL_DIR}cosine_baseline_epoch66.pth
SAVE_DIR=${MODEL_DIR}eval/
# # 
# #   --flip \
# # #     --aqe --aqe_k2 7 --aqe_alpha 3.0 \
# CUDA_VISIBLE_DEVICES=2 python test2.py --config_file='configs/naic/arcface_baseline.yml' \
#     TEST.IMS_PER_BATCH "128" TEST.RANDOMPERM "5" \
#     MODEL.BASELINE.TPL_WEIGHT "1.0" MODEL.BASELINE.CE_WEIGHT "0.33" MODEL.LABEL_SMOOTH "False" MODEL.BASELINE.S "30.0" MODEL.BASELINE.M "0.35" MODEL.BASELINE.COSINE_LOSS_TYPE 'ArcCos' \
#     INPUT.SIZE_TRAIN "([384,192])" INPUT.SIZE_TEST "([384,192])" \
#     MODEL.NAME "cosine_baseline" MODEL.BACKBONE "('resnet101_ibn_a')" MODEL.BASELINE.POOL_TYPE "avg"\
#     DATASETS.DATA_PATH "('${DATA_DIR}')" DATASETS.TRAIN_PATH "/data/Dataset/PReID/testb_pseudo_hist_065_080_dataset/rep_trainVal2"\
#     MODEL.PRETRAIN_PATH "('${PRETRAIN}')"  \
#     OUTPUT_DIR "('${SAVE_DIR}')" \
#     TEST.WEIGHT "${WEIGHT}"

# WEIGHT=${MODEL_DIR}cosine_baseline_epoch66.pth
# SAVE_DIR=${MODEL_DIR}sub/
# CUDA_VISIBLE_DEVICES=2 python test2.py --config_file='configs/naic/arcface_baseline.yml' \
#     --sub --save_fname "testb_pseudo_retrain_arcface_e66_" \
#     --flip \
#     --aqe --aqe_k2 7 --aqe_alpha 3.0 \
#     --query_dir ${QUERY_DIR}\
#     --gallery_dir ${GALLERY_DIR}\
#     MODEL.BASELINE.TPL_WEIGHT "1.0" MODEL.BASELINE.CE_WEIGHT "0.33" MODEL.LABEL_SMOOTH "False" MODEL.BASELINE.S "30.0" MODEL.BASELINE.M "0.35" MODEL.BASELINE.COSINE_LOSS_TYPE 'ArcCos' \
#     INPUT.SIZE_TRAIN "([384,192])" INPUT.SIZE_TEST "([384,192])" \
#     MODEL.NAME "cosine_baseline" MODEL.BACKBONE "('resnet101_ibn_a')" MODEL.BASELINE.POOL_TYPE "avg"\
#     DATASETS.DATA_PATH "('${DATA_DIR}')" DATASETS.TRAIN_PATH "/data/Dataset/PReID/testb_pseudo_hist_065_080_dataset/rep_trainVal2"\
#     MODEL.PRETRAIN_PATH "('${PRETRAIN}')"  \
#     OUTPUT_DIR "('${SAVE_DIR}')" \
#     TEST.WEIGHT "${WEIGHT}"


# WEIGHT=${MODEL_DIR}cosine_baseline_epoch62.pth
# SAVE_DIR=${MODEL_DIR}sub/
# CUDA_VISIBLE_DEVICES=3 python test2.py --config_file='configs/naic/arcface_baseline.yml' \
#     --sub --save_fname "testb_pseudo_retrain_arcface_e62_" \
#     --flip \
#     --aqe --aqe_k2 7 --aqe_alpha 3.0 \
#     --query_dir ${QUERY_DIR}\
#     --gallery_dir ${GALLERY_DIR}\
#     MODEL.BASELINE.TPL_WEIGHT "1.0" MODEL.BASELINE.CE_WEIGHT "0.33" MODEL.LABEL_SMOOTH "False" MODEL.BASELINE.S "30.0" MODEL.BASELINE.M "0.35" MODEL.BASELINE.COSINE_LOSS_TYPE 'ArcCos' \
#     INPUT.SIZE_TRAIN "([384,192])" INPUT.SIZE_TEST "([384,192])" \
#     MODEL.NAME "cosine_baseline" MODEL.BACKBONE "('resnet101_ibn_a')" MODEL.BASELINE.POOL_TYPE "avg"\
#     DATASETS.DATA_PATH "('${DATA_DIR}')" DATASETS.TRAIN_PATH "/data/Dataset/PReID/testb_pseudo_hist_065_080_dataset/rep_trainVal2"\
#     MODEL.PRETRAIN_PATH "('${PRETRAIN}')"  \
#     OUTPUT_DIR "('${SAVE_DIR}')" \
#     TEST.WEIGHT "${WEIGHT}"

DATA_DIR=${ROOT_DIR}rep_dataset/
PRETRAIN=../weights/resnet101_ibn_a.pth.tar
MODEL_DIR=../rep_work_dirs/exp4-cosinebaseline-resnet101ibnls1-384x192-bs16x6-warmup10-flip-pad10-meanstd-erase0502-nolbsm-avg-arcface30_035_10033-cj05-trainVal2/ #(h, w)
WEIGHT=${MODEL_DIR}cosine_baseline_epoch90.pth

SAVE_DIR=${MODEL_DIR}eval/
# 
#   --flip \
#     --aqe --aqe_k2 7 --aqe_alpha 3.0 \
# CUDA_VISIBLE_DEVICES=5 python test2.py --config_file='configs/naic/arcface_baseline.yml' \
#     TEST.IMS_PER_BATCH "128" TEST.RANDOMPERM "5" \
#     MODEL.BASELINE.S "30.0" MODEL.BASELINE.M "0.35" \
#     MODEL.BASELINE.COSINE_LOSS_TYPE 'ArcCos' \
#     INPUT.SIZE_TRAIN "([384,192])" INPUT.SIZE_TEST "([384,192])" \
#     MODEL.NAME "cosine_baseline" MODEL.BACKBONE "('resnet101_ibn_a')" MODEL.BASELINE.POOL_TYPE "avg"\
#     DATASETS.DATA_PATH "('${DATA_DIR}')" DATASETS.TRAIN_PATH "rep_trainVal2"\
#     MODEL.PRETRAIN_PATH "('${PRETRAIN}')"  \
#     OUTPUT_DIR "('${SAVE_DIR}')" \
#     TEST.WEIGHT "${WEIGHT}"


# --rerank --k1 9 --k2 3 --lambda_value 0.6 \
# --dba --dba_k2 10 --dba_alpha -3.0

# WEIGHT=${MODEL_DIR}cosine_baseline_epoch90.pth
# SAVE_DIR=${MODEL_DIR}sub/
# CUDA_VISIBLE_DEVICES=5 python test2.py --config_file='configs/naic/arcface_baseline.yml' \
#     --save_fname 'testb_origin_tpl03_e90_'\
#     --sub \
#     --flip \
#     --aqe --aqe_k2 7 --aqe_alpha 3.0 \
#     --query_dir ${QUERY_DIR}\
#     --gallery_dir ${GALLERY_DIR}\
#     MODEL.BASELINE.S "30.0" MODEL.BASELINE.M "0.35" \
#     MODEL.BASELINE.COSINE_LOSS_TYPE 'ArcCos' \
#     INPUT.SIZE_TRAIN "([384,192])" INPUT.SIZE_TEST "([384,192])" \
#     MODEL.NAME "cosine_baseline" MODEL.BACKBONE "('resnet101_ibn_a')" MODEL.BASELINE.POOL_TYPE "avg"\
#     DATASETS.DATA_PATH "('${DATA_DIR}')" DATASETS.TRAIN_PATH "rep_trainVal2"\
#     MODEL.PRETRAIN_PATH "('${PRETRAIN}')"  \
#     OUTPUT_DIR "('${SAVE_DIR}')" \
#     TEST.WEIGHT "${WEIGHT}"

# WEIGHT=${MODEL_DIR}cosine_baseline_epoch84.pth
# SAVE_DIR=${MODEL_DIR}sub/
# CUDA_VISIBLE_DEVICES=7 python test2.py --config_file='configs/naic/arcface_baseline.yml' \
#     --save_fname 'testb_origin_tpl03_e84_'\
#     --sub \
#     --flip \
#     --aqe --aqe_k2 7 --aqe_alpha 3.0 \
#     --query_dir ${QUERY_DIR}\
#     --gallery_dir ${GALLERY_DIR}\
#     MODEL.BASELINE.S "30.0" MODEL.BASELINE.M "0.35" \
#     MODEL.BASELINE.COSINE_LOSS_TYPE 'ArcCos' \
#     INPUT.SIZE_TRAIN "([384,192])" INPUT.SIZE_TEST "([384,192])" \
#     MODEL.NAME "cosine_baseline" MODEL.BACKBONE "('resnet101_ibn_a')" MODEL.BASELINE.POOL_TYPE "avg"\
#     DATASETS.DATA_PATH "('${DATA_DIR}')" DATASETS.TRAIN_PATH "rep_trainVal2"\
#     MODEL.PRETRAIN_PATH "('${PRETRAIN}')"  \
#     OUTPUT_DIR "('${SAVE_DIR}')" \
#     TEST.WEIGHT "${WEIGHT}"


DATA_DIR=${ROOT_DIR}rep_dataset/
PRETRAIN=../weights/resnet101_ibn_a.pth.tar
MODEL_DIR='../rep_work_dirs/exp4-cosinebaseline-resnet101ibnls1-384x192-bs16x6-warmup10-flip-pad10-meanstd-erase0502-nolbsm-avg-arcface30_035_10033-cj05-trainVal2-testb_pseudo_retrain_arcface-fp16/' #(h, w)
WEIGHT=${MODEL_DIR}cosine_baseline_epoch66.pth
SAVE_DIR=${MODEL_DIR}eval/
# # 
# #   --flip \
# # #     --aqe --aqe_k2 7 --aqe_alpha 3.0 \
# CUDA_VISIBLE_DEVICES=2 python test2.py --config_file='configs/naic/arcface_baseline.yml' \
#     TEST.IMS_PER_BATCH "128" TEST.RANDOMPERM "5" \
#     MODEL.BASELINE.TPL_WEIGHT "1.0" MODEL.BASELINE.CE_WEIGHT "0.33" MODEL.LABEL_SMOOTH "False" MODEL.BASELINE.S "30.0" MODEL.BASELINE.M "0.35" MODEL.BASELINE.COSINE_LOSS_TYPE 'ArcCos' \
#     INPUT.SIZE_TRAIN "([384,192])" INPUT.SIZE_TEST "([384,192])" \
#     MODEL.NAME "cosine_baseline" MODEL.BACKBONE "('resnet101_ibn_a')" MODEL.BASELINE.POOL_TYPE "avg"\
#     DATASETS.DATA_PATH "('${DATA_DIR}')" DATASETS.TRAIN_PATH "/data/Dataset/PReID/testb_pseudo_hist_065_080_dataset/rep_trainVal2"\
#     MODEL.PRETRAIN_PATH "('${PRETRAIN}')"  \
#     OUTPUT_DIR "('${SAVE_DIR}')" \
#     TEST.WEIGHT "${WEIGHT}"

# WEIGHT=${MODEL_DIR}cosine_baseline_epoch66.pth
# SAVE_DIR=${MODEL_DIR}sub/
# CUDA_VISIBLE_DEVICES=2 python test2.py --config_file='configs/naic/arcface_baseline.yml' \
#     --sub --save_fname "testb_pseudo_retrain_arcface_e66_" \
#     --flip \
#     --aqe --aqe_k2 7 --aqe_alpha 3.0 \
#     --query_dir ${QUERY_DIR}\
#     --gallery_dir ${GALLERY_DIR}\
#     MODEL.BASELINE.TPL_WEIGHT "1.0" MODEL.BASELINE.CE_WEIGHT "0.33" MODEL.LABEL_SMOOTH "False" MODEL.BASELINE.S "30.0" MODEL.BASELINE.M "0.35" MODEL.BASELINE.COSINE_LOSS_TYPE 'ArcCos' \
#     INPUT.SIZE_TRAIN "([384,192])" INPUT.SIZE_TEST "([384,192])" \
#     MODEL.NAME "cosine_baseline" MODEL.BACKBONE "('resnet101_ibn_a')" MODEL.BASELINE.POOL_TYPE "avg"\
#     DATASETS.DATA_PATH "('${DATA_DIR}')" DATASETS.TRAIN_PATH "/data/Dataset/PReID/testb_pseudo_hist_065_080_dataset/rep_trainVal2"\
#     MODEL.PRETRAIN_PATH "('${PRETRAIN}')"  \
#     OUTPUT_DIR "('${SAVE_DIR}')" \
#     TEST.WEIGHT "${WEIGHT}"





DATA_DIR=${ROOT_DIR}rep_dataset/
PRETRAIN=../weights/resnet101_ibn_a.pth.tar
MODEL_DIR='../rep_work_dirs/exp4-cosinebaseline-resnet101ibnls1-384x192-bs16x6-warmup10-flip-pad10-meanstd-erase0502-nolbsm-avg-arcface30_035_10033-cj05-trainVal2-testb_pseudo_retrain_arcface_cj05-fp16/' #(h, w)
WEIGHT=${MODEL_DIR}cosine_baseline_epoch66.pth
SAVE_DIR=${MODEL_DIR}eval/
# # # 
# # #   --flip \
# # # #     --aqe --aqe_k2 7 --aqe_alpha 3.0 \
# CUDA_VISIBLE_DEVICES=3 python test2.py --config_file='configs/naic/arcface_baseline.yml' \
#     TEST.IMS_PER_BATCH "128" TEST.RANDOMPERM "5" \
#     MODEL.BASELINE.TPL_WEIGHT "1.0" MODEL.BASELINE.CE_WEIGHT "0.33" MODEL.LABEL_SMOOTH "False" MODEL.BASELINE.S "30.0" MODEL.BASELINE.M "0.35" MODEL.BASELINE.COSINE_LOSS_TYPE 'ArcCos' \
#     INPUT.SIZE_TRAIN "([384,192])" INPUT.SIZE_TEST "([384,192])" \
#     MODEL.NAME "cosine_baseline" MODEL.BACKBONE "('resnet101_ibn_a')" MODEL.BASELINE.POOL_TYPE "avg"\
#     DATASETS.DATA_PATH "('${DATA_DIR}')" DATASETS.TRAIN_PATH "/data/Dataset/PReID/testb_pseudo_hist_065_080_dataset/rep_trainVal2"\
#     MODEL.PRETRAIN_PATH "('${PRETRAIN}')"  \
#     OUTPUT_DIR "('${SAVE_DIR}')" \
#     TEST.WEIGHT "${WEIGHT}"

# WEIGHT=${MODEL_DIR}cosine_baseline_epoch66.pth
# SAVE_DIR=${MODEL_DIR}sub/
# CUDA_VISIBLE_DEVICES=3 python test2.py --config_file='configs/naic/arcface_baseline.yml' \
#     --sub --save_fname "testb_pseudo_retrain_arcface_e66_" \
#     --flip \
#     --aqe --aqe_k2 7 --aqe_alpha 3.0 \
#     --query_dir ${QUERY_DIR}\
#     --gallery_dir ${GALLERY_DIR}\
#     MODEL.BASELINE.TPL_WEIGHT "1.0" MODEL.BASELINE.CE_WEIGHT "0.33" MODEL.LABEL_SMOOTH "False" MODEL.BASELINE.S "30.0" MODEL.BASELINE.M "0.35" MODEL.BASELINE.COSINE_LOSS_TYPE 'ArcCos' \
#     INPUT.SIZE_TRAIN "([384,192])" INPUT.SIZE_TEST "([384,192])" \
#     MODEL.NAME "cosine_baseline" MODEL.BACKBONE "('resnet101_ibn_a')" MODEL.BASELINE.POOL_TYPE "avg"\
#     DATASETS.DATA_PATH "('${DATA_DIR}')" DATASETS.TRAIN_PATH "/data/Dataset/PReID/testb_pseudo_hist_065_080_dataset/rep_trainVal2"\
#     MODEL.PRETRAIN_PATH "('${PRETRAIN}')"  \
#     OUTPUT_DIR "('${SAVE_DIR}')" \
#     TEST.WEIGHT "${WEIGHT}"

