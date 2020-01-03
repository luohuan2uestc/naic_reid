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




# [20191211]

# # eval
# PRETRAIN=../weights/r50_ibn_a.pth
# MODEL_DIR='../rep_work_dirs/exp5-mgn-resnet50ibnls2-384x128-bs12x8-warmup10-flip-pad10-meanstd-lbsm-layer3_-5-erase0502-allgem_3-minInst2/' #(h, w)
# WEIGHT=${MODEL_DIR}mgn_epoch120.pth
# SAVE_DIR=${MODEL_DIR}eval/

#     # --aqe --aqe_k2 7 --aqe_alpha 3.0 \
#     # --adabn --adabn_emv\
# #     --adabn \

# # CUDA_VISIBLE_DEVICES=5 python test2.py --config_file='configs/naic/mgn.yml' \
# #     --flip --adabn\
# #     TEST.IMS_PER_BATCH "128" TEST.RANDOMPERM "5"\
# #     MODEL.MGN.NUM_SHARE_LAYER3 "-5" MODEL.NAME "mgn" MODEL.BACKBONE "('resnet50_ibn_a')" MODEL.MGN.POOL_TYPE "gem_3" MODEL.MGN.PART_POOL_TYPE "gem_3"\
# #     DATASETS.DATA_PATH "('${DATA_DIR}')" DATASETS.TRAIN_PATH "rep_f0_train2"\
# #     MODEL.PRETRAIN_PATH "('${PRETRAIN}')"  \
# #     OUTPUT_DIR "('${SAVE_DIR}')" \
# #     TEST.WEIGHT "${WEIGHT}"




# # --rerank --k1 9 --k2 3 --lambda_value 0.6 \
# # --dba --dba_k2 10 --dba_alpha -3.0
# # --aqe --aqe_k2 7 --aqe_alpha 3.0
# #     --adabn \

# SAVE_DIR=${MODEL_DIR}sub/
# CUDA_VISIBLE_DEVICES=0 python test2.py --config_file='configs/naic/mgn.yml' \
#     --aqe --aqe_k2 7 --aqe_alpha 3.0 \
#     --flip --adabn \
#     --sub \
#     --query_dir ${QUERY_DIR}\
#     --gallery_dir ${GALLERY_DIR}\
#     MODEL.MGN.NUM_SHARE_LAYER3 "-5" MODEL.NAME "mgn" MODEL.BACKBONE "('resnet50_ibn_a')" MODEL.MGN.POOL_TYPE "gem_3" MODEL.MGN.PART_POOL_TYPE "gem_3"\
#     DATASETS.DATA_PATH "('${DATA_DIR}')" DATASETS.TRAIN_PATH "rep_f0_train2"\
#     MODEL.PRETRAIN_PATH "('${PRETRAIN}')"  \
#     OUTPUT_DIR "('${SAVE_DIR}')" \
#     TEST.WEIGHT "${WEIGHT}"

# [20191216]
# eval
# DATA_DIR=${ROOT_DIR}rep_dataset/
# PRETRAIN=../weights/r50_ibn_a.pth
# MODEL_DIR='../rep_work_dirs/exp5-mgn-resnet50ibnls2-384x128-bs12x8-warmup10-flip-pad10-meanstd-lbsm-layer3_-5-erase0502-allgem_3-trainVal2/' #(h, w)
# WEIGHT=${MODEL_DIR}mgn_epoch120.pth
# SAVE_DIR=${MODEL_DIR}eval/

    # --aqe --aqe_k2 7 --aqe_alpha 3.0 \
    # --adabn --adabn_emv\
#     --adabn \
    # --flip --adabn\
# CUDA_VISIBLE_DEVICES=1 python test2.py --config_file='configs/naic/mgn.yml' \
#     --flip --adabn\
#     TEST.IMS_PER_BATCH "128" TEST.RANDOMPERM "5"\
#     MODEL.MGN.NUM_SHARE_LAYER3 "-5" MODEL.NAME "mgn" MODEL.BACKBONE "('resnet50_ibn_a')" MODEL.MGN.POOL_TYPE "gem_3" MODEL.MGN.PART_POOL_TYPE "gem_3"\
#     DATASETS.DATA_PATH "('${DATA_DIR}')" DATASETS.TRAIN_PATH "rep_trainVal2"\
#     MODEL.PRETRAIN_PATH "('${PRETRAIN}')"  \
#     OUTPUT_DIR "('${SAVE_DIR}')" \
#     TEST.WEIGHT "${WEIGHT}"



# --rerank --k1 9 --k2 3 --lambda_value 0.6 \
# --dba --dba_k2 10 --dba_alpha -3.0
# --aqe --aqe_k2 7 --aqe_alpha 3.0
#     --adabn \

# SAVE_DIR=${MODEL_DIR}sub/
# CUDA_VISIBLE_DEVICES=1 python test2.py --config_file='configs/naic/mgn.yml' \
#     --flip --adabn \
#     --sub \
#     --query_dir ${QUERY_DIR}\
#     --gallery_dir ${GALLERY_DIR}\
#     MODEL.MGN.NUM_SHARE_LAYER3 "-5" MODEL.NAME "mgn" MODEL.BACKBONE "('resnet50_ibn_a')" MODEL.MGN.POOL_TYPE "gem_3" MODEL.MGN.PART_POOL_TYPE "gem_3"\
#     DATASETS.DATA_PATH "('${DATA_DIR}')" DATASETS.TRAIN_PATH "rep_trainVal2"\
#     MODEL.PRETRAIN_PATH "('${PRETRAIN}')"  \
#     OUTPUT_DIR "('${SAVE_DIR}')" \
#     TEST.WEIGHT "${WEIGHT}"


# part test
# [20191216]

# # eval
# PRETRAIN=../weights/r50_ibn_a.pth
# MODEL_DIR='../rep_work_dirs/exp5-mgn-resnet50ibnls2-384x128-bs12x8-warmup10-flip-pad10-meanstd-lbsm-layer3_-5-erase0502-allgem_3-minInst2/' #(h, w)
# WEIGHT=${MODEL_DIR}mgn_epoch120.pth
# SAVE_DIR=${MODEL_DIR}eval/

# #     # --aqe --aqe_k2 7 --aqe_alpha 3.0 \
# #     # --adabn --adabn_emv\
# # #     --adabn \
# #     # --flip \
# # INPUT.SIZE_TRAIN "([432,144])" INPUT.SIZE_TEST "([432,144])" \

# CUDA_VISIBLE_DEVICES=7 python mgn_test.py --config_file='configs/naic/mgn.yml' \
#     INPUT.SIZE_TRAIN "([270,90])" INPUT.SIZE_TEST "([270,90])" \
#     TEST.IMS_PER_BATCH "128" TEST.RANDOMPERM "5"\
#     MODEL.MGN.NUM_SHARE_LAYER3 "-5" MODEL.NAME "mgn" MODEL.BACKBONE "('resnet50_ibn_a')" MODEL.MGN.POOL_TYPE "gem_3" MODEL.MGN.PART_POOL_TYPE "gem_3"\
#     DATASETS.DATA_PATH "('${DATA_DIR}')" DATASETS.TRAIN_PATH "rep_f0_train2"\
#     MODEL.PRETRAIN_PATH "('${PRETRAIN}')"  \
#     OUTPUT_DIR "('${SAVE_DIR}')" \
#     TEST.WEIGHT "${WEIGHT}"

# CUDA_VISIBLE_DEVICES=7 python mgn_test.py --config_file='configs/naic/mgn.yml' \
#     --flip \
#     --sub \
#     --query_dir ${QUERY_DIR} \
#     --gallery_dir ${GALLERY_DIR} \
#     MODEL.MGN.NUM_SHARE_LAYER3 "-5" MODEL.NAME "mgn" MODEL.BACKBONE "('resnet50_ibn_a')" MODEL.MGN.POOL_TYPE "gem_3" MODEL.MGN.PART_POOL_TYPE "gem_3"\
#     DATASETS.DATA_PATH "('${DATA_DIR}')" DATASETS.TRAIN_PATH "rep_f0_train2"\
#     MODEL.PRETRAIN_PATH "('${PRETRAIN}')"  \
#     OUTPUT_DIR "('${SAVE_DIR}')" \
#     TEST.WEIGHT "${WEIGHT}"
# part test
# [20191216]

# eval
# DATA_DIR=${ROOT_DIR}rep_dataset/
# PRETRAIN=../weights/r50_ibn_a.pth
# MODEL_DIR='../rep_work_dirs/exp5-mgn-resnet50ibnls2-384x128-bs12x8-warmup10-flip-pad10-meanstd-lbsm-layer3_-5-erase0502-allgem_3-trainVal2/' #(h, w)
# WEIGHT=${MODEL_DIR}mgn_epoch120.pth
# SAVE_DIR=${MODEL_DIR}eval/

# CUDA_VISIBLE_DEVICES=7 python mgn_test.py --config_file='configs/naic/mgn.yml' \
#     --flip \
#     TEST.IMS_PER_BATCH "128" TEST.RANDOMPERM "5"\
#     MODEL.MGN.NUM_SHARE_LAYER3 "-5" MODEL.NAME "mgn" MODEL.BACKBONE "('resnet50_ibn_a')" MODEL.MGN.POOL_TYPE "gem_3" MODEL.MGN.PART_POOL_TYPE "gem_3"\
#     DATASETS.DATA_PATH "('${DATA_DIR}')" DATASETS.TRAIN_PATH "rep_trainVal2"\
#     MODEL.PRETRAIN_PATH "('${PRETRAIN}')"  \
#     OUTPUT_DIR "('${SAVE_DIR}')" \
#     TEST.WEIGHT "${WEIGHT}"

# # part test
# # [20191217]

# # # eval
DATA_DIR=${ROOT_DIR}rep_dataset/
PRETRAIN=../weights/r50_ibn_a.pth
MODEL_DIR='../rep_work_dirs/exp5-mgn-resnet50ibnls2-384x144-bs12x8-warmup10-flip-pad10-meanstd-lbsm-layer3_-5-erase0502-allgem_3-minInst2/' #(h, w)
WEIGHT=${MODEL_DIR}mgn_epoch110.pth
SAVE_DIR=${MODEL_DIR}eval/

# #     # --aqe --aqe_k2 7 --aqe_alpha 3.0 \
# #     # --adabn --adabn_emv\
# # #     --adabn \
# #     # --flip \
# # INPUT.SIZE_TRAIN "([432,144])" INPUT.SIZE_TEST "([432,144])" \

# # CUDA_VISIBLE_DEVICES=7 python test2.py --config_file='configs/naic/mgn.yml' \
# CUDA_VISIBLE_DEVICES=4 python mgn_test.py --config_file='configs/naic/mgn.yml' \
#     --flip \
#     --aqe --aqe_k2 7 --aqe_alpha 3.0 \
#     INPUT.SIZE_TRAIN "([384,144])" INPUT.SIZE_TEST "([384,144])" \
#     MODEL.MGN.NUM_SHARE_LAYER3 "-5" MODEL.NAME "mgn" MODEL.BACKBONE "('resnet50_ibn_a')" MODEL.MGN.POOL_TYPE "gem_3" MODEL.MGN.PART_POOL_TYPE "gem_3"\
#     DATASETS.DATA_PATH "('${DATA_DIR}')" DATASETS.TRAIN_PATH "rep_f0_train2"\
#     MODEL.PRETRAIN_PATH "('${PRETRAIN}')"  \
#     OUTPUT_DIR "('${SAVE_DIR}')" \
#     TEST.WEIGHT "${WEIGHT}"

# CUDA_VISIBLE_DEVICES=0 python mgn_test.py --config_file='configs/naic/mgn.yml' \
#     --aqe --aqe_k2 7 --aqe_alpha 3.0 \
#     --flip --adabn \
#     --sub --save_fname "origin_e110_" \
#     --query_dir ${QUERY_DIR} \
#     --gallery_dir ${GALLERY_DIR} \
#     INPUT.SIZE_TRAIN "([384,144])" INPUT.SIZE_TEST "([384,144])" \
#     MODEL.MGN.NUM_SHARE_LAYER3 "-5" MODEL.NAME "mgn" MODEL.BACKBONE "('resnet50_ibn_a')" MODEL.MGN.POOL_TYPE "gem_3" MODEL.MGN.PART_POOL_TYPE "gem_3"\
#     DATASETS.DATA_PATH "('${DATA_DIR}')" DATASETS.TRAIN_PATH "rep_f0_train2"\
#     MODEL.PRETRAIN_PATH "('${PRETRAIN}')"  \
#     OUTPUT_DIR "('${SAVE_DIR}')" \
#     TEST.WEIGHT "${WEIGHT}"

# part test
# [20191219]

# # eval
DATA_DIR=${ROOT_DIR}rep_dataset/
PRETRAIN=../weights/r50_ibn_a.pth
MODEL_DIR='../rep_work_dirs/exp5-cosinemgn2d-resnet50ibnls2-384x128-bs12x8-warmup10-flip-pad10-meanstd-lbsm-layer3_-5-erase0502-allgem_3-minInst2/' #(h, w)
WEIGHT=${MODEL_DIR}cosinemgn2d_epoch90.pth
SAVE_DIR=${MODEL_DIR}eval/

#     # --aqe --aqe_k2 7 --aqe_alpha 3.0 \
#     # --adabn --adabn_emv\
# #     --adabn \
#     # --flip \
    # --flip \
    # --aqe --aqe_k2 7 --aqe_alpha 3.0 \
# CUDA_VISIBLE_DEVICES=2 python test2.py --config_file='configs/naic/cosinemgn.yml' \

# CUDA_VISIBLE_DEVICES=5 python mgn_test.py --config_file='configs/naic/cosinemgn.yml' \
#     MODEL.NAME "cosinemgn2d" MODEL.BACKBONE "('resnet50_ibn_a')" MODEL.COSINEMGN.NUM_SHARE_LAYER3 "-5" MODEL.COSINEMGN.POOL_TYPE "avg" MODEL.COSINEMGN.PART_POOL_TYPE "gem_3"\
#     DATASETS.DATA_PATH "('${DATA_DIR}')" DATASETS.TRAIN_PATH "rep_f0_train2"\
#     MODEL.PRETRAIN_PATH "('${PRETRAIN}')"  \
#     OUTPUT_DIR "('${SAVE_DIR}')" \
#     TEST.WEIGHT "${WEIGHT}"

# CUDA_VISIBLE_DEVICES=2 python mgn_test.py --config_file='configs/naic/cosinemgn.yml' \
#     --aqe --aqe_k2 7 --aqe_alpha 3.0 \
#     --flip --adabn \
#     --sub \
#     --query_dir ${QUERY_DIR} \
#     --gallery_dir ${GALLERY_DIR} \
#     MODEL.NAME "cosinemgn2d" MODEL.BACKBONE "('resnet50_ibn_a')" MODEL.COSINEMGN.NUM_SHARE_LAYER3 "-5" MODEL.COSINEMGN.POOL_TYPE "avg" MODEL.COSINEMGN.PART_POOL_TYPE "gem_3"\
#     DATASETS.DATA_PATH "('${DATA_DIR}')" DATASETS.TRAIN_PATH "rep_f0_train2"\
#     MODEL.PRETRAIN_PATH "('${PRETRAIN}')"  \
#     OUTPUT_DIR "('${SAVE_DIR}')" \
#     TEST.WEIGHT "${WEIGHT}"


# # eval
# DATA_DIR=${ROOT_DIR}rep_dataset/
# PRETRAIN=../weights/r50_ibn_a.pth
# MODEL_DIR='../rep_work_dirs/exp5-cosinemgn2d-resnet50ibnls2-384x128-bs12x8-warmup10-flip-pad10-meanstd-lbsm-layer3_-5-erase0502-allgem_3-cb2d_late-minInst2/' #(h, w)
# WEIGHT=${MODEL_DIR}cosinemgn2d_epoch90.pth
# SAVE_DIR=${MODEL_DIR}eval/

#     # --aqe --aqe_k2 7 --aqe_alpha 3.0 \
#     # --adabn --adabn_emv\
# #     --adabn \
#     # --flip \
    # --flip \
    # --aqe --aqe_k2 7 --aqe_alpha 3.0 \

# CUDA_VISIBLE_DEVICES=6 python test2.py --config_file='configs/naic/cosinemgn.yml' \
# CUDA_VISIBLE_DEVICES=6 python mgn_test.py --config_file='configs/naic/cosinemgn.yml' \
#     --flip \
#     --aqe --aqe_k2 7 --aqe_alpha 3.0 \
    # MODEL.NAME "cosinemgn2d" MODEL.BACKBONE "('resnet50_ibn_a')" MODEL.COSINEMGN.NUM_SHARE_LAYER3 "-5" MODEL.COSINEMGN.POOL_TYPE "avg" MODEL.COSINEMGN.PART_POOL_TYPE "gem_3" MODEL.COSINEMGN.CB2D_BNFIRST "False"\
    # DATASETS.DATA_PATH "('${DATA_DIR}')" DATASETS.TRAIN_PATH "rep_f0_train2"\
    # MODEL.PRETRAIN_PATH "('${PRETRAIN}')"  \
    # OUTPUT_DIR "('${SAVE_DIR}')" \
    # TEST.WEIGHT "${WEIGHT}"

# CUDA_VISIBLE_DEVICES=6 python mgn_test.py --config_file='configs/naic/cosinemgn.yml' \
#     --aqe --aqe_k2 7 --aqe_alpha 3.0 \
#     --flip --adabn \
#     --sub \
#     --query_dir ${QUERY_DIR} \
#     --gallery_dir ${GALLERY_DIR} \
#     MODEL.NAME "cosinemgn2d" MODEL.BACKBONE "('resnet50_ibn_a')" MODEL.COSINEMGN.NUM_SHARE_LAYER3 "-5" MODEL.COSINEMGN.POOL_TYPE "avg" MODEL.COSINEMGN.PART_POOL_TYPE "gem_3" MODEL.COSINEMGN.CB2D_BNFIRST "False"\
#     DATASETS.DATA_PATH "('${DATA_DIR}')" DATASETS.TRAIN_PATH "rep_f0_train2"\
#     MODEL.PRETRAIN_PATH "('${PRETRAIN}')"  \
#     OUTPUT_DIR "('${SAVE_DIR}')" \
#     TEST.WEIGHT "${WEIGHT}"

PRETRAIN=../weights/r50_ibn_a.pth
DATA_DIR='/data/Dataset/PReID/rep_dataset/'
MODEL_DIR='../rep_work_dirs/exp5-mgn-resnet50ibnls2-384x144-bs16x6-warmup10-flip-pad10-meanstd-lbsm-layer3_-5-erase0502-allgem_3-sepnorm-minInst2/' #(h, w)
WEIGHT=${MODEL_DIR}mgn_epoch100.pth
SAVE_DIR=${MODEL_DIR}eval/
# CUDA_VISIBLE_DEVICES=0 python sepnorm_test.py --config_file='configs/naic/mgn.yml' \

# CUDA_VISIBLE_DEVICES=0 python mgn_sepnorm_test.py --config_file='configs/naic/mgn.yml' \
#     --aqe --aqe_k2 7 --aqe_alpha 3.0 \
#     --flip \
#     INPUT.SEPNORM.USE "True" \
#     INPUT.SEPNORM.PIXEL_MEAN "([[0.1164,0.1567,0.1796],[0.4556,0.5367,0.6493]])" \
#     INPUT.SEPNORM.PIXEL_STD "([[0.1467,0.1497,0.1930],[0.0750,0.0613,0.1040]])" \
#     INPUT.SIZE_TRAIN "([384,144])" INPUT.SIZE_TEST "([384,144])" \
#     MODEL.MGN.NUM_SHARE_LAYER3 "-5" MODEL.NAME "mgn" MODEL.BACKBONE "('resnet50_ibn_a')" MODEL.MGN.POOL_TYPE "gem_3" MODEL.MGN.PART_POOL_TYPE "gem_3"\
#     DATASETS.DATA_PATH "('${DATA_DIR}')" DATASETS.TRAIN_PATH "rep_f0_train2"\
#     MODEL.PRETRAIN_PATH "('${PRETRAIN}')"  \
#     OUTPUT_DIR "('${SAVE_DIR}')" \
#     TEST.WEIGHT "${WEIGHT}"

# SAVE_DIR=${MODEL_DIR}sub/
# CUDA_VISIBLE_DEVICES=2 python mgn_sepnorm_test.py --config_file='configs/naic/mgn.yml' \
#     --aqe --aqe_k2 7 --aqe_alpha 3.0 \
#     --flip --adabn \
#     --sub \
#     --query_dir ${QUERY_DIR} \
#     --gallery_dir ${GALLERY_DIR} \
#     INPUT.SEPNORM.USE "True" \
#     INPUT.SEPNORM.TEST_PIXEL_MEAN "([[0.1164,0.1567,0.1796],[0.5410,0.4631,0.5808]])" \
#     INPUT.SEPNORM.TEST_PIXEL_STD "([[0.1467,0.1497,0.1930],[0.0444,0.0821,0.0981]])" \
#     INPUT.SIZE_TRAIN "([384,144])" INPUT.SIZE_TEST "([384,144])" \
#     MODEL.MGN.NUM_SHARE_LAYER3 "-5" MODEL.NAME "mgn" MODEL.BACKBONE "('resnet50_ibn_a')" MODEL.MGN.POOL_TYPE "gem_3" MODEL.MGN.PART_POOL_TYPE "gem_3"\
#     DATASETS.DATA_PATH "('${DATA_DIR}')" DATASETS.TRAIN_PATH "rep_f0_train2"\
#     MODEL.PRETRAIN_PATH "('${PRETRAIN}')"  \
#     OUTPUT_DIR "('${SAVE_DIR}')" \
#     TEST.WEIGHT "${WEIGHT}"


# PRETRAIN=../weights/r50_ibn_a.pth
# DATA_DIR='/data/Dataset/PReID/rep_dataset/'
# MODEL_DIR='../rep_work_dirs/exp5-mgn-resnet50ibnls2-384x144-bs16x6-warmup10-flip-pad10-meanstd-lbsm-layer3_-5-erase0502-allgem_3-sepnorm-minInst2/' #(h, w)
# WEIGHT=${MODEL_DIR}mgn_epoch120.pth
# SAVE_DIR=${MODEL_DIR}eval/
# CUDA_VISIBLE_DEVICES=0 python sepnorm_test.py --config_file='configs/naic/mgn.yml' \

# CUDA_VISIBLE_DEVICES=0 python mgn_sepnorm_test.py --config_file='configs/naic/mgn.yml' \
#     --aqe --aqe_k2 7 --aqe_alpha 3.0 \
#     --flip \
#     INPUT.SEPNORM.USE "True" \
#     INPUT.SEPNORM.PIXEL_MEAN "([[0.1164,0.1567,0.1796],[0.4556,0.5367,0.6493]])" \
#     INPUT.SEPNORM.PIXEL_STD "([[0.1467,0.1497,0.1930],[0.0750,0.0613,0.1040]])" \
#     INPUT.SIZE_TRAIN "([384,144])" INPUT.SIZE_TEST "([384,144])" \
#     MODEL.MGN.NUM_SHARE_LAYER3 "-5" MODEL.NAME "mgn" MODEL.BACKBONE "('resnet50_ibn_a')" MODEL.MGN.POOL_TYPE "gem_3" MODEL.MGN.PART_POOL_TYPE "gem_3"\
#     DATASETS.DATA_PATH "('${DATA_DIR}')" DATASETS.TRAIN_PATH "rep_f0_train2"\
#     MODEL.PRETRAIN_PATH "('${PRETRAIN}')"  \
#     OUTPUT_DIR "('${SAVE_DIR}')" \
#     TEST.WEIGHT "${WEIGHT}"

# SAVE_DIR=${MODEL_DIR}sub/
# CUDA_VISIBLE_DEVICES=2 python mgn_sepnorm_test.py --config_file='configs/naic/mgn.yml' \
#     --aqe --aqe_k2 7 --aqe_alpha 3.0 \
#     --flip --adabn \
#     --sub \
#     --query_dir ${QUERY_DIR} \
#     --gallery_dir ${GALLERY_DIR} \
#     INPUT.SEPNORM.USE "True" \
#     INPUT.SEPNORM.TEST_PIXEL_MEAN "([[0.1164,0.1567,0.1796],[0.5410,0.4631,0.5808]])" \
#     INPUT.SEPNORM.TEST_PIXEL_STD "([[0.1467,0.1497,0.1930],[0.0444,0.0821,0.0981]])" \
#     INPUT.SIZE_TRAIN "([384,144])" INPUT.SIZE_TEST "([384,144])" \
#     MODEL.MGN.NUM_SHARE_LAYER3 "-5" MODEL.NAME "mgn" MODEL.BACKBONE "('resnet50_ibn_a')" MODEL.MGN.POOL_TYPE "gem_3" MODEL.MGN.PART_POOL_TYPE "gem_3"\
#     DATASETS.DATA_PATH "('${DATA_DIR}')" DATASETS.TRAIN_PATH "rep_f0_train2"\
#     MODEL.PRETRAIN_PATH "('${PRETRAIN}')"  \
#     OUTPUT_DIR "('${SAVE_DIR}')" \
#     TEST.WEIGHT "${WEIGHT}"

# eval
# [20191223]
DATA_DIR=${ROOT_DIR}rep_dataset/
PRETRAIN=../weights/r50_ibn_a.pth
MODEL_DIR='../rep_work_dirs/exp5-cosinemgn2d-resnet50ibnls2-384x144-bs16x6-warmup10-flip-pad10-meanstd-lbsm-layer3_-5-erase0502-allgem_3-cb2d_late-allbnbias-minInst2-e140/' #(h, w)
WEIGHT=${MODEL_DIR}cosinemgn2d_epoch140.pth
SAVE_DIR=${MODEL_DIR}eval/

    # --aqe --aqe_k2 7 --aqe_alpha 3.0 \
    # --adabn --adabn_emv\
#     --adabn \
    # --flip \
    # --flip \
    # --aqe --aqe_k2 7 --aqe_alpha 3.0 \

# CUDA_VISIBLE_DEVICES=2 python test2.py --config_file='configs/naic/cosinemgn.yml' \

# CUDA_VISIBLE_DEVICES=2 python mgn_test.py --config_file='configs/naic/cosinemgn.yml' \
#     --flip \
#     --aqe --aqe_k2 7 --aqe_alpha 3.0 \
#     TEST.IMS_PER_BATCH "128" TEST.RANDOMPERM "5"\
#     INPUT.SIZE_TRAIN "([384,144])" INPUT.SIZE_TEST "([384,144])" \
#     MODEL.NAME "cosinemgn2d" MODEL.BACKBONE "('resnet50_ibn_a')" MODEL.COSINEMGN.NUM_SHARE_LAYER3 "-5" \
#     MODEL.COSINEMGN.POOL_TYPE "gem_3" MODEL.COSINEMGN.PART_POOL_TYPE "gem_3" MODEL.COSINEMGN.CB2D_BNFIRST "False" MODEL.COSINEMGN.GUSE_BNBAIS "True" MODEL.COSINEMGN.PUSE_BNBAIS "True"\
#     DATASETS.DATA_PATH "('${DATA_DIR}')" DATASETS.TRAIN_PATH "rep_f0_train2"\
#     MODEL.PRETRAIN_PATH "('${PRETRAIN}')"  \
#     OUTPUT_DIR "('${SAVE_DIR}')" \
#     TEST.WEIGHT "${WEIGHT}"


# CUDA_VISIBLE_DEVICES=2 python mgn_test.py --config_file='configs/naic/cosinemgn.yml' \
#     --aqe --aqe_k2 7 --aqe_alpha 3.0 \
#     --flip --adabn \
#     --sub \
#     --query_dir ${QUERY_DIR} \
#     --gallery_dir ${GALLERY_DIR} \
#     INPUT.SIZE_TRAIN "([384,144])" INPUT.SIZE_TEST "([384,144])" \
#     MODEL.NAME "cosinemgn2d" MODEL.BACKBONE "('resnet50_ibn_a')" MODEL.COSINEMGN.NUM_SHARE_LAYER3 "-5" \
#     MODEL.COSINEMGN.POOL_TYPE "gem_3" MODEL.COSINEMGN.PART_POOL_TYPE "gem_3" MODEL.COSINEMGN.CB2D_BNFIRST "False" MODEL.COSINEMGN.GUSE_BNBAIS "True" MODEL.COSINEMGN.PUSE_BNBAIS "True"\
#     DATASETS.DATA_PATH "('${DATA_DIR}')" DATASETS.TRAIN_PATH "rep_f0_train2"\
#     MODEL.PRETRAIN_PATH "('${PRETRAIN}')"  \
#     OUTPUT_DIR "('${SAVE_DIR}')" \
#     TEST.WEIGHT "${WEIGHT}"




PRETRAIN=../weights/r50_ibn_a.pth
DATA_DIR='/data/Dataset/PReID/rep_dataset/'
MODEL_DIR='../rep_work_dirs/exp5-cosinemgn2d-resnet50ibnls2-384x144-bs16x6-warmup10-flip-pad10-meanstd-lbsm-layer3_-5-erase0502-allgem_3-cb2d_late-allbnbias-sepnorm-minInst2-e140/' #(h, w)
WEIGHT=${MODEL_DIR}cosinemgn2d_epoch140.pth
SAVE_DIR=${MODEL_DIR}eval/
# --aqe --aqe_k2 7 --aqe_alpha 3.0 \
#     --flip \

# CUDA_VISIBLE_DEVICES=3 python sepnorm_test.py --config_file='configs/naic/cosinemgn.yml' \
# CUDA_VISIBLE_DEVICES=3 python mgn_sepnorm_test.py --config_file='configs/naic/cosinemgn.yml' \
#     --flip \
#     --aqe --aqe_k2 7 --aqe_alpha 3.0 \
#     INPUT.SEPNORM.USE "True" \
#     INPUT.SEPNORM.PIXEL_MEAN "([[0.1164,0.1567,0.1796],[0.4556,0.5367,0.6493]])" \
#     INPUT.SEPNORM.PIXEL_STD "([[0.1467,0.1497,0.1930],[0.0750,0.0613,0.1040]])" \
#     INPUT.SIZE_TRAIN "([384,144])" INPUT.SIZE_TEST "([384,144])" \
#     MODEL.NAME "cosinemgn2d" MODEL.BACKBONE "('resnet50_ibn_a')" MODEL.COSINEMGN.NUM_SHARE_LAYER3 "-5" \
#     MODEL.COSINEMGN.POOL_TYPE "gem_3" MODEL.COSINEMGN.PART_POOL_TYPE "gem_3" MODEL.COSINEMGN.CB2D_BNFIRST "False" MODEL.COSINEMGN.GUSE_BNBAIS "True" MODEL.COSINEMGN.PUSE_BNBAIS "True"\
#     DATASETS.DATA_PATH "('${DATA_DIR}')" DATASETS.TRAIN_PATH "rep_f0_train2"\
#     MODEL.PRETRAIN_PATH "('${PRETRAIN}')"  \
#     OUTPUT_DIR "('${SAVE_DIR}')" \
#     TEST.WEIGHT "${WEIGHT}"

# SAVE_DIR=${MODEL_DIR}sub/
# CUDA_VISIBLE_DEVICES=3 python mgn_sepnorm_test.py --config_file='configs/naic/cosinemgn.yml' \
#     --aqe --aqe_k2 7 --aqe_alpha 3.0 \
#     --flip --adabn --adabn_com\
#     --sub \
#     --query_dir ${QUERY_DIR} \
#     --gallery_dir ${GALLERY_DIR} \
#     INPUT.SEPNORM.USE "True" \
#     INPUT.SEPNORM.TEST_PIXEL_MEAN "([[0.1164,0.1567,0.1796],[0.5410,0.4631,0.5808]])" \
#     INPUT.SEPNORM.TEST_PIXEL_STD "([[0.1467,0.1497,0.1930],[0.0444,0.0821,0.0981]])" \
#     INPUT.SIZE_TRAIN "([384,144])" INPUT.SIZE_TEST "([384,144])" \
#     MODEL.NAME "cosinemgn2d" MODEL.BACKBONE "('resnet50_ibn_a')" MODEL.COSINEMGN.NUM_SHARE_LAYER3 "-5" \
#     MODEL.COSINEMGN.POOL_TYPE "gem_3" MODEL.COSINEMGN.PART_POOL_TYPE "gem_3" MODEL.COSINEMGN.CB2D_BNFIRST "False" MODEL.COSINEMGN.GUSE_BNBAIS "True" MODEL.COSINEMGN.PUSE_BNBAIS "True"\
#     DATASETS.DATA_PATH "('${DATA_DIR}')" DATASETS.TRAIN_PATH "rep_f0_train2"\
#     MODEL.PRETRAIN_PATH "('${PRETRAIN}')"  \
#     OUTPUT_DIR "('${SAVE_DIR}')" \
#     TEST.WEIGHT "${WEIGHT}"


# # # eval
DATA_DIR=${ROOT_DIR}rep_dataset/
PRETRAIN=../weights/r50_ibn_a.pth
MODEL_DIR='../rep_work_dirs/exp5-mgn-resnet50ibnls2-384x144-bs12x8-warmup10-flip-pad10-meanstd-lbsm-layer3_-5-erase0502-allgem_3-cj-minInst2/' #(h, w)
WEIGHT=${MODEL_DIR}mgn_epoch10.pth
SAVE_DIR=${MODEL_DIR}eval/

#     # --aqe --aqe_k2 7 --aqe_alpha 3.0 \
#     # --adabn --adabn_emv\
# #     --adabn \
#     # --flip \
# INPUT.SIZE_TRAIN "([432,144])" INPUT.SIZE_TEST "([432,144])" \
# CUDA_VISIBLE_DEVICES=7 python test2.py --config_file='configs/naic/mgn.yml' \
    # --flip \
    # --aqe --aqe_k2 7 --aqe_alpha 3.0 \
# CUDA_VISIBLE_DEVICES=7 python mgn_test.py --config_file='configs/naic/mgn.yml' \
#     --flip \
#     INPUT.SIZE_TRAIN "([384,144])" INPUT.SIZE_TEST "([384,144])" \
#     MODEL.MGN.NUM_SHARE_LAYER3 "-5" MODEL.NAME "mgn" MODEL.BACKBONE "('resnet50_ibn_a')" MODEL.MGN.POOL_TYPE "gem_3" MODEL.MGN.PART_POOL_TYPE "gem_3"\
#     DATASETS.DATA_PATH "('${DATA_DIR}')" DATASETS.TRAIN_PATH "rep_f0_train2"\
#     MODEL.PRETRAIN_PATH "('${PRETRAIN}')"  \
#     OUTPUT_DIR "('${SAVE_DIR}')" \
#     TEST.WEIGHT "${WEIGHT}"

CUDA_VISIBLE_DEVICES=4 python mgn_test.py --config_file='configs/naic/mgn.yml' \
    --aqe --aqe_k2 7 --aqe_alpha 3.0 \
    --flip \
    --sub --save_fname "origin_e120_" \
    --query_dir ${QUERY_DIR} \
    --gallery_dir ${GALLERY_DIR} \
    INPUT.SIZE_TRAIN "([384,144])" INPUT.SIZE_TEST "([384,144])" \
    MODEL.MGN.NUM_SHARE_LAYER3 "-5" MODEL.NAME "mgn" MODEL.BACKBONE "('resnet50_ibn_a')" MODEL.MGN.POOL_TYPE "gem_3" MODEL.MGN.PART_POOL_TYPE "gem_3"\
    DATASETS.DATA_PATH "('${DATA_DIR}')" DATASETS.TRAIN_PATH "rep_f0_train2"\
    MODEL.PRETRAIN_PATH "('${PRETRAIN}')"  \
    OUTPUT_DIR "('${SAVE_DIR}')" \
    TEST.WEIGHT "${WEIGHT}"

# # [20191227]
# DATA_DIR=${ROOT_DIR}rep_dataset/
# PRETRAIN=../weights/resnet101_ibn_a.pth.tar
# MODEL_DIR='../rep_work_dirs/exp5-mgn-resnet101ibnls1-384x144-bs16x6-warmup10-flip-pad10-meanstd-lbsm-layer3_-5-erase0502-allgem_3-cj05-trainVal2/' #(h, w)
# WEIGHT=${MODEL_DIR}mgn_epoch90.pth
# SAVE_DIR=${MODEL_DIR}eval/

# #     # --aqe --aqe_k2 7 --aqe_alpha 3.0 \
# #     # --adabn --adabn_emv\
# # #     --adabn \
# #     # --flip \
# # INPUT.SIZE_TRAIN "([432,144])" INPUT.SIZE_TEST "([432,144])" \
# # CUDA_VISIBLE_DEVICES=4 python test2.py --config_file='configs/naic/mgn.yml' \
#     # --flip \
#     # --aqe --aqe_k2 7 --aqe_alpha 3.0 \
# #     --flip \
# # CUDA_VISIBLE_DEVICES=4 python test2.py --config_file='configs/naic/mgn.yml' \
# # CUDA_VISIBLE_DEVICES=4 python mgn_test.py --config_file='configs/naic/mgn.yml' \
# #     --flip \
# #     --aqe --aqe_k2 7 --aqe_alpha 3.0 \
# #     INPUT.SIZE_TRAIN "([384,144])" INPUT.SIZE_TEST "([384,144])" \
# #     MODEL.MGN.NUM_SHARE_LAYER3 "-5" MODEL.NAME "mgn" MODEL.BACKBONE "('resnet101_ibn_a')" MODEL.MGN.POOL_TYPE "gem_3" MODEL.MGN.PART_POOL_TYPE "gem_3"\
# #     DATASETS.DATA_PATH "('${DATA_DIR}')" DATASETS.TRAIN_PATH "rep_trainVal2"\
# #     MODEL.PRETRAIN_PATH "('${PRETRAIN}')"  \
# #     OUTPUT_DIR "('${SAVE_DIR}')" \
# #     TEST.WEIGHT "${WEIGHT}"

# CUDA_VISIBLE_DEVICES=4 python mgn_test.py --config_file='configs/naic/mgn.yml' \
#     --aqe --aqe_k2 7 --aqe_alpha 3.0 \
#     --flip \
#     --sub \
#     --query_dir ${QUERY_DIR} \
#     --gallery_dir ${GALLERY_DIR} \
#     INPUT.SIZE_TRAIN "([384,144])" INPUT.SIZE_TEST "([384,144])" \
#     MODEL.MGN.NUM_SHARE_LAYER3 "-5" MODEL.NAME "mgn" MODEL.BACKBONE "('resnet101_ibn_a')" MODEL.MGN.POOL_TYPE "gem_3" MODEL.MGN.PART_POOL_TYPE "gem_3"\
#     DATASETS.DATA_PATH "('${DATA_DIR}')" DATASETS.TRAIN_PATH "rep_trainVal2"\
#     MODEL.PRETRAIN_PATH "('${PRETRAIN}')"  \
#     OUTPUT_DIR "('${SAVE_DIR}')" \
#     TEST.WEIGHT "${WEIGHT}"

# [20191228]
DATA_DIR=${ROOT_DIR}rep_dataset/
PRETRAIN=../weights/resnet101_ibn_a.pth.tar
MODEL_DIR='../rep_work_dirs/exp5-mgn-resnet101ibnls1-384x144-bs12x8-warmup10-flip-pad10-meanstd-lbsm-layer3_-5-erase0502-allgem_3-cj-trainVal2/' #(h, w)
WEIGHT=${MODEL_DIR}mgn_epoch90.pth
SAVE_DIR=${MODEL_DIR}eval/

#     # --aqe --aqe_k2 7 --aqe_alpha 3.0 \
#     # --adabn --adabn_emv\
# #     --adabn \
#     # --flip \
# INPUT.SIZE_TRAIN "([432,144])" INPUT.SIZE_TEST "([432,144])" \
# CUDA_VISIBLE_DEVICES=4 python test2.py --config_file='configs/naic/mgn.yml' \
    # --flip \
    
#     --flip \
# CUDA_VISIBLE_DEVICES=2 python test2.py --config_file='configs/naic/mgn.yml' \
# CUDA_VISIBLE_DEVICES=2 python mgn_test.py --config_file='configs/naic/mgn.yml' \
#     --aqe --aqe_k2 7 --aqe_alpha 3.0 \
#     INPUT.SIZE_TRAIN "([384,144])" INPUT.SIZE_TEST "([384,144])" \
#     MODEL.MGN.NUM_SHARE_LAYER3 "-5" MODEL.NAME "mgn" MODEL.BACKBONE "('resnet101_ibn_a')" MODEL.MGN.POOL_TYPE "gem_3" MODEL.MGN.PART_POOL_TYPE "gem_3"\
#     DATASETS.DATA_PATH "('${DATA_DIR}')" DATASETS.TRAIN_PATH "rep_trainVal2"\
#     MODEL.PRETRAIN_PATH "('${PRETRAIN}')"  \
#     OUTPUT_DIR "('${SAVE_DIR}')" \
#     TEST.WEIGHT "${WEIGHT}"

# CUDA_VISIBLE_DEVICES=2 python mgn_test.py --config_file='configs/naic/mgn.yml' \
#     --aqe --aqe_k2 7 --aqe_alpha 3.0 \
#     --flip \
#     --sub \
#     --query_dir ${QUERY_DIR} \
#     --gallery_dir ${GALLERY_DIR} \
#     INPUT.SIZE_TRAIN "([384,144])" INPUT.SIZE_TEST "([384,144])" \
#     MODEL.MGN.NUM_SHARE_LAYER3 "-5" MODEL.NAME "mgn" MODEL.BACKBONE "('resnet101_ibn_a')" MODEL.MGN.POOL_TYPE "gem_3" MODEL.MGN.PART_POOL_TYPE "gem_3"\
#     DATASETS.DATA_PATH "('${DATA_DIR}')" DATASETS.TRAIN_PATH "rep_trainVal2"\
#     MODEL.PRETRAIN_PATH "('${PRETRAIN}')"  \
#     OUTPUT_DIR "('${SAVE_DIR}')" \
#     TEST.WEIGHT "${WEIGHT}"

# [debug]
DATA_DIR=${ROOT_DIR}rep_dataset/
PRETRAIN=../weights/r50_ibn_a.pth
MODEL_DIR='../rep_work_dirs/exp5-mgn-resnet50ibnls2-384x144-bs12x8-warmup10-flip-pad10-meanstd-lbsm-layer3_-5-erase0502-allgem_3-cj-minInst2/' #(h, w)
WEIGHT=${MODEL_DIR}mgn_epoch120.pth
SAVE_DIR=${MODEL_DIR}eval/

#     # --aqe --aqe_k2 7 --aqe_alpha 3.0 \
#     # --adabn --adabn_emv\
# #     --adabn \
#     # --flip \
# INPUT.SIZE_TRAIN "([432,144])" INPUT.SIZE_TEST "([432,144])" \
# CUDA_VISIBLE_DEVICES=2 python test2.py --config_file='configs/naic/mgn.yml' \
    # --flip \
    # 
    # INPUT.SIZE_TRAIN "([384,144])" INPUT.SIZE_TEST "([384,144])" \

# CUDA_VISIBLE_DEVICES=4 python mgn_test.py --config_file='configs/naic/mgn.yml' \
#     --flip \
#     --aqe --aqe_k2 7 --aqe_alpha 3.0 \
#     INPUT.SIZE_TRAIN "([384,144])" INPUT.SIZE_TEST "([384,144])" \
#     MODEL.MGN.NUM_SHARE_LAYER3 "-5" MODEL.NAME "mgn" MODEL.BACKBONE "('resnet50_ibn_a')" MODEL.MGN.POOL_TYPE "gem_3" MODEL.MGN.PART_POOL_TYPE "gem_3"\
#     DATASETS.DATA_PATH "('${DATA_DIR}')" DATASETS.TRAIN_PATH "rep_f0_train2"\
#     MODEL.PRETRAIN_PATH "('${PRETRAIN}')"  \
#     OUTPUT_DIR "('${SAVE_DIR}')" \
#     TEST.WEIGHT "${WEIGHT}"

# [20191231]
# # # eval
DATA_DIR=${ROOT_DIR}rep_dataset/
PRETRAIN=../weights/r50_ibn_a.pth
MODEL_DIR='../rep_work_dirs/exp5-mgn-resnet50ibnls2-384x144-bs12x8-warmup10-flip-pad10-meanstd-lbsm-layer3_-5-erase0502-allgem_3-minInst2-pseudo_finetune/' #(h, w)
WEIGHT=${MODEL_DIR}mgn_epoch12.pth
SAVE_DIR=${MODEL_DIR}eval/

# #     # --aqe --aqe_k2 7 --aqe_alpha 3.0 \
# #     # --adabn --adabn_emv\
# # #     --adabn \
# #     # --flip \
# CUDA_VISIBLE_DEVICES=0 python test2.py --config_file='configs/naic/mgn.yml' \
# CUDA_VISIBLE_DEVICES=5 python mgn_test.py --config_file='configs/naic/mgn.yml' \
#     --flip \
#     --aqe --aqe_k2 7 --aqe_alpha 3.0 \
#     INPUT.SIZE_TRAIN "([384,144])" INPUT.SIZE_TEST "([384,144])" \
#     MODEL.MGN.NUM_SHARE_LAYER3 "-5" MODEL.NAME "mgn" MODEL.BACKBONE "('resnet50_ibn_a')" MODEL.MGN.POOL_TYPE "gem_3" MODEL.MGN.PART_POOL_TYPE "gem_3"\
#     DATASETS.DATA_PATH "('${DATA_DIR}')" DATASETS.TRAIN_PATH "/data/Dataset/PReID/pseudo_mgn_eps045_2_dataset/rep_trainVal2"\
#     MODEL.PRETRAIN_PATH "('${PRETRAIN}')"  \
#     OUTPUT_DIR "('${SAVE_DIR}')" \
#     TEST.WEIGHT "${WEIGHT}"

# CUDA_VISIBLE_DEVICES=0 python mgn_test.py --config_file='configs/naic/mgn.yml' \
#     --aqe --aqe_k2 7 --aqe_alpha 3.0 \
#     --flip --adabn \
#     --sub --save_fname "pseudo_finetune_e12_" \
#     --query_dir ${QUERY_DIR} \
#     --gallery_dir ${GALLERY_DIR} \
#     INPUT.SIZE_TRAIN "([384,144])" INPUT.SIZE_TEST "([384,144])" \
#     MODEL.MGN.NUM_SHARE_LAYER3 "-5" MODEL.NAME "mgn" MODEL.BACKBONE "('resnet50_ibn_a')" MODEL.MGN.POOL_TYPE "gem_3" MODEL.MGN.PART_POOL_TYPE "gem_3"\
#     DATASETS.DATA_PATH "('${DATA_DIR}')" DATASETS.TRAIN_PATH "/data/Dataset/PReID/pseudo_mgn_eps045_2_dataset/rep_trainVal2"\
#     MODEL.PRETRAIN_PATH "('${PRETRAIN}')"  \
#     OUTPUT_DIR "('${SAVE_DIR}')" \
#     TEST.WEIGHT "${WEIGHT}"

# # [pseudo]
# # # # eval
# DATA_DIR=${ROOT_DIR}rep_dataset/
# PRETRAIN=../weights/r50_ibn_a.pth
# MODEL_DIR='../rep_work_dirs/exp5-mgn-resnet50ibnls2-384x144-bs12x8-warmup10-flip-pad10-meanstd-lbsm-layer3_-5-erase0502-allgem_3-minInst2/' #(h, w)
# WEIGHT=${MODEL_DIR}mgn_epoch102.pth
# SAVE_DIR=${MODEL_DIR}eval/

# SAVE_DIR=${MODEL_DIR}sub/
# CUDA_VISIBLE_DEVICES=0 python mgn_test.py --config_file='configs/naic/mgn.yml' \
#     --pseudo  --pseudo_visual --pseudo_eps 0.45 --pseudo_minpoints 2 --pseudo_maxpoints 100 --pseudo_savepath '../rep_work_dirs/pseudo_mgn_eps045_2/'\
#     --flip --adabn \
#     --sub \
#     --query_dir ${QUERY_DIR} \
#     --gallery_dir ${GALLERY_DIR} \
#     INPUT.SIZE_TRAIN "([384,144])" INPUT.SIZE_TEST "([384,144])" \
#     MODEL.MGN.NUM_SHARE_LAYER3 "-5" MODEL.NAME "mgn" MODEL.BACKBONE "('resnet50_ibn_a')" MODEL.MGN.POOL_TYPE "gem_3" MODEL.MGN.PART_POOL_TYPE "gem_3"\
#     DATASETS.DATA_PATH "('${DATA_DIR}')" DATASETS.TRAIN_PATH "rep_f0_train2"\
#     MODEL.PRETRAIN_PATH "('${PRETRAIN}')"  \
#     OUTPUT_DIR "('${SAVE_DIR}')" \
#     TEST.WEIGHT "${WEIGHT}"
