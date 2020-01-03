#!/usr/bin/env bash

# PRETRAIN=../weights/r50_ibn_a.pth
# DATA_DIR='/data/Dataset/PReID/all_dataset/'
# SAVE_DIR='../rep_work_dirs/exp5-mgn-resnet50ibnls2-384x128-bs12x8-warmup10-flip-pad10-meanstd-lbsm-layer3_-5-erase0502-gem_3-minInst2/' #(h, w)
# CUDA_VISIBLE_DEVICES=7 python train.py --config_file='configs/naic/mgn.yml' \
#     SOLVER.IMS_PER_BATCH "96" DATALOADER.NUM_INSTANCE "8" \
#     MODEL.MGN.NUM_SHARE_LAYER3 "-5" MODEL.NAME "mgn" MODEL.BACKBONE "('resnet50_ibn_a')" MODEL.MGN.POOL_TYPE "gem_3"\
#     DATASETS.DATA_PATH "('${DATA_DIR}')" DATASETS.TRAIN_PATH "rep_f0_train2"\
#     MODEL.PRETRAIN_PATH "('${PRETRAIN}')"  \
#     OUTPUT_DIR "('${SAVE_DIR}')" 

# PRETRAIN=../weights/r50_ibn_a.pth
# DATA_DIR='/data/Dataset/PReID/all_dataset/'
# SAVE_DIR='../rep_work_dirs/exp5-mgn-resnet50ibnls2-384x128-bs12x8-warmup10-flip-pad10-meanstd-lbsm-layer3_-5-erase0502-allgem_3-minInst2/' #(h, w)
# CUDA_VISIBLE_DEVICES=2 python train.py --config_file='configs/naic/mgn.yml' \
#     SOLVER.IMS_PER_BATCH "96" DATALOADER.NUM_INSTANCE "8" \
#     MODEL.MGN.NUM_SHARE_LAYER3 "-5" MODEL.NAME "mgn" MODEL.BACKBONE "('resnet50_ibn_a')" MODEL.MGN.POOL_TYPE "gem_3" MODEL.MGN.PART_POOL_TYPE "gem_3"\
#     DATASETS.DATA_PATH "('${DATA_DIR}')" DATASETS.TRAIN_PATH "rep_f0_train2"\
#     MODEL.PRETRAIN_PATH "('${PRETRAIN}')"  \
#     OUTPUT_DIR "('${SAVE_DIR}')" 



# PRETRAIN=../weights/r50_ibn_a.pth
# DATA_DIR='/data/Dataset/PReID/rep_dataset/'
# SAVE_DIR='../rep_work_dirs/exp5-mgn-resnet50ibnls2-384x128-bs12x8-warmup10-flip-pad10-meanstd-lbsm-layer3_-5-erase0502-gem_3-nobnbias_minInst2/' #(h, w)
# CUDA_VISIBLE_DEVICES=7 python train.py --config_file='configs/naic/mgn.yml' \
#     SOLVER.IMS_PER_BATCH "96" DATALOADER.NUM_INSTANCE "8" \
#     MODEL.MGN.NUM_SHARE_LAYER3 "-5" MODEL.NAME "mgn" MODEL.BACKBONE "('resnet50_ibn_a')" MODEL.MGN.POOL_TYPE "gem_3" MODEL.MGN.USE_BNBAIS "False"\
#     DATASETS.DATA_PATH "('${DATA_DIR}')" DATASETS.TRAIN_PATH "rep_f0_train2"\
#     MODEL.PRETRAIN_PATH "('${PRETRAIN}')"  \
#     OUTPUT_DIR "('${SAVE_DIR}')" 

# PRETRAIN=../weights/r50_ibn_a.pth
# DATA_DIR='/data/Dataset/PReID/rep_dataset/'
# SAVE_DIR='../rep_work_dirs/exp5-mgn-resnet50ibnls2-384x128-bs12x8-warmup10-flip-pad10-meanstd-lbsm-layer3_-5-erase0502-allgem_3-trainVal2/' #(h, w)
# CUDA_VISIBLE_DEVICES=2 python train.py --config_file='configs/naic/mgn.yml' \
#     SOLVER.IMS_PER_BATCH "96" DATALOADER.NUM_INSTANCE "8" \
#     MODEL.MGN.NUM_SHARE_LAYER3 "-5" MODEL.NAME "mgn" MODEL.BACKBONE "('resnet50_ibn_a')" MODEL.MGN.POOL_TYPE "gem_3" MODEL.MGN.PART_POOL_TYPE "gem_3"\
#     DATASETS.DATA_PATH "('${DATA_DIR}')" DATASETS.TRAIN_PATH "rep_trainVal2"\
#     MODEL.PRETRAIN_PATH "('${PRETRAIN}')"  \
#     OUTPUT_DIR "('${SAVE_DIR}')" 

# PRETRAIN=../weights/r50_ibn_a.pth
# DATA_DIR='/data/Dataset/PReID/rep_dataset/'
# SAVE_DIR='../rep_work_dirs/exp5-mgn-resnet50ibnls2-384x144-bs12x8-warmup10-flip-pad10-meanstd-lbsm-layer3_-5-erase0502-allgem_3-minInst2/' #(h, w)
# CUDA_VISIBLE_DEVICES=2 python train.py --config_file='configs/naic/mgn.yml' \
#     SOLVER.IMS_PER_BATCH "96" DATALOADER.NUM_INSTANCE "8" \
#     INPUT.SIZE_TRAIN "([384,144])" INPUT.SIZE_TEST "([384,144])" \
#     MODEL.MGN.NUM_SHARE_LAYER3 "-5" MODEL.NAME "mgn" MODEL.BACKBONE "('resnet50_ibn_a')" MODEL.MGN.POOL_TYPE "gem_3" MODEL.MGN.PART_POOL_TYPE "gem_3"\
#     DATASETS.DATA_PATH "('${DATA_DIR}')" DATASETS.TRAIN_PATH "rep_f0_train2"\
#     MODEL.PRETRAIN_PATH "('${PRETRAIN}')"  \
#     OUTPUT_DIR "('${SAVE_DIR}')" 

# PRETRAIN=../weights/r50_ibn_a.pth
# DATA_DIR='/data/Dataset/PReID/rep_dataset/'
# SAVE_DIR='../rep_work_dirs/exp5-mgn-resnet50ibnls2-432x144-bs12x8-warmup10-flip-pad10-meanstd-lbsm-layer3_-5-erase0502-allgem_3-minInst2/' #(h, w)
# CUDA_VISIBLE_DEVICES=1 python train.py --config_file='configs/naic/mgn.yml' \
#     SOLVER.IMS_PER_BATCH "96" DATALOADER.NUM_INSTANCE "8" \
#     INPUT.SIZE_TRAIN "([432,144])" INPUT.SIZE_TEST "([432,144])" \
#     MODEL.MGN.NUM_SHARE_LAYER3 "-5" MODEL.NAME "mgn" MODEL.BACKBONE "('resnet50_ibn_a')" MODEL.MGN.POOL_TYPE "gem_3" MODEL.MGN.PART_POOL_TYPE "gem_3"\
#     DATASETS.DATA_PATH "('${DATA_DIR}')" DATASETS.TRAIN_PATH "rep_f0_train2"\
#     MODEL.PRETRAIN_PATH "('${PRETRAIN}')"  \
#     OUTPUT_DIR "('${SAVE_DIR}')" 

# PRETRAIN=../weights/r50_ibn_a.pth
# DATA_DIR='/data/Dataset/PReID/rep_dataset/'
# SAVE_DIR='../rep_work_dirs/exp5-mgn-resnet50ibnls2-384x128-bs12x8-warmup10-flip-pad10-meanstd-lbsm-layer3_-5-erase0502-allgem_3-center000505-minInst2/' #(h, w)
# CUDA_VISIBLE_DEVICES=6 python train.py --config_file='configs/naic/mgn_center.yml' \
#     SOLVER.IMS_PER_BATCH "96" DATALOADER.NUM_INSTANCE "8" \
#     MODEL.MGN.NUM_SHARE_LAYER3 "-5" MODEL.NAME "mgn" MODEL.BACKBONE "('resnet50_ibn_a')" MODEL.MGN.POOL_TYPE "gem_3" MODEL.MGN.PART_POOL_TYPE "gem_3"\
#     DATASETS.DATA_PATH "('${DATA_DIR}')" DATASETS.TRAIN_PATH "rep_f0_train2"\
#     MODEL.PRETRAIN_PATH "('${PRETRAIN}')"  \
#     OUTPUT_DIR "('${SAVE_DIR}')" 


# PRETRAIN=../weights/r50_ibn_a.pth
# DATA_DIR='/data/Dataset/PReID/all_dataset/'
# SAVE_DIR='../rep_work_dirs/exp5-cosinemgn-resnet50ibnls2-384x128-bs12x8-warmup10-flip-pad10-meanstd-lbsm-layer3_-5-erase0502-allgem_3-minInst2/' #(h, w)
# # SAVE_DIR='../rep_work_dirs/debug/' #(h, w)

# CUDA_VISIBLE_DEVICES=0 python train.py --config_file='configs/naic/cosinemgn.yml' \
#     SOLVER.IMS_PER_BATCH "96" DATALOADER.NUM_INSTANCE "8" \
#     MODEL.NAME "cosinemgn" MODEL.BACKBONE "('resnet50_ibn_a')" MODEL.COSINEMGN.NUM_SHARE_LAYER3 "-5" MODEL.COSINEMGN.POOL_TYPE "gem_3" MODEL.COSINEMGN.PART_POOL_TYPE "gem_3"\
#     DATASETS.DATA_PATH "('${DATA_DIR}')" DATASETS.TRAIN_PATH "rep_f0_train2"\
#     MODEL.PRETRAIN_PATH "('${PRETRAIN}')"  \
#     OUTPUT_DIR "('${SAVE_DIR}')" 



# PRETRAIN=../weights/r50_ibn_a.pth
# DATA_DIR='/data/Dataset/PReID/rep_dataset/'
# SAVE_DIR='../rep_work_dirs/exp5-cosinemgn2d-resnet50ibnls2-384x128-bs12x8-warmup10-flip-pad10-meanstd-lbsm-layer3_-5-erase0502-allgem_3-minInst2/' #(h, w)
# SAVE_DIR='../rep_work_dirs/exp5-cosinemgn2d-resnet50ibnls2-384x128-bs12x8-warmup10-flip-pad10-meanstd-lbsm-layer3_-5-erase0502-avggem_3-minInst2/' #(h, w)

# SAVE_DIR='../rep_work_dirs/debug/' #(h, w)

# CUDA_VISIBLE_DEVICES=0 python train.py --config_file='configs/naic/cosinemgn.yml' \
#     SOLVER.IMS_PER_BATCH "96" DATALOADER.NUM_INSTANCE "8" \
#     MODEL.NAME "cosinemgn2d" MODEL.BACKBONE "('resnet50_ibn_a')" MODEL.COSINEMGN.NUM_SHARE_LAYER3 "-5" MODEL.COSINEMGN.POOL_TYPE "avg" MODEL.COSINEMGN.PART_POOL_TYPE "gem_3"\
#     DATASETS.DATA_PATH "('${DATA_DIR}')" DATASETS.TRAIN_PATH "rep_f0_train2"\
#     MODEL.PRETRAIN_PATH "('${PRETRAIN}')"  \
#     OUTPUT_DIR "('${SAVE_DIR}')" 

# PRETRAIN=../weights/r50_ibn_a.pth
# DATA_DIR='/data/Dataset/PReID/rep_dataset/'
# SAVE_DIR='../rep_work_dirs/exp5-cosinemgn2d-resnet50ibnls2-384x128-bs12x8-warmup10-flip-pad10-meanstd-lbsm-layer3_-5-erase0502-allgem_3-cb2d_late-minInst2/' #(h, w)
# SAVE_DIR='../rep_work_dirs/exp5-cosinemgn2d-resnet50ibnls2-384x128-bs12x8-warmup10-flip-pad10-meanstd-lbsm-layer3_-5-erase0502-avggem_3-cb2d_late-minInst2/' #(h, w)
# # SAVE_DIR='../rep_work_dirs/debug/' #(h, w)

# CUDA_VISIBLE_DEVICES=3 python train.py --config_file='configs/naic/cosinemgn.yml' \
#     SOLVER.IMS_PER_BATCH "96" DATALOADER.NUM_INSTANCE "8" \
#     MODEL.NAME "cosinemgn2d" MODEL.BACKBONE "('resnet50_ibn_a')" MODEL.COSINEMGN.NUM_SHARE_LAYER3 "-5" MODEL.COSINEMGN.POOL_TYPE "avg" MODEL.COSINEMGN.PART_POOL_TYPE "gem_3" MODEL.COSINEMGN.CB2D_BNFIRST "False"\
#     DATASETS.DATA_PATH "('${DATA_DIR}')" DATASETS.TRAIN_PATH "rep_f0_train2"\
#     MODEL.PRETRAIN_PATH "('${PRETRAIN}')"  \
#     OUTPUT_DIR "('${SAVE_DIR}')" 

# PRETRAIN=../weights/r50_ibn_a.pth
# DATA_DIR='/data/Dataset/PReID/rep_dataset/'
# SAVE_DIR='../rep_work_dirs/exp5-cosinemgn-resnet50ibnls2-384x128-bs12x8-warmup10-flip-pad10-meanstd-lbsm-layer3_-5-erase0502-allgem_3-gcos3035_033-minInst2/' #(h, w)
# # SAVE_DIR='../rep_work_dirs/debug/' #(h, w)

# CUDA_VISIBLE_DEVICES=4 python train.py --config_file='configs/naic/cosinemgn.yml' \
#     SOLVER.IMS_PER_BATCH "96" DATALOADER.NUM_INSTANCE "8" \
#     MODEL.USE_COS "True" MODEL.COSINEMGN.GCE_WEIGHT "0.333" MODEL.COSINEMGN.GCOSINE_LOSS_TYPE 'CosFace' MODEL.COSINEMGN.S "30.0" MODEL.COSINEMGN.M "0.35" \
#     MODEL.NAME "cosinemgn" MODEL.BACKBONE "('resnet50_ibn_a')" MODEL.COSINEMGN.NUM_SHARE_LAYER3 "-5" MODEL.COSINEMGN.POOL_TYPE "gem_3" MODEL.COSINEMGN.PART_POOL_TYPE "gem_3"\
#     DATASETS.DATA_PATH "('${DATA_DIR}')" DATASETS.TRAIN_PATH "rep_f0_train2"\
#     MODEL.PRETRAIN_PATH "('${PRETRAIN}')"  \
#     OUTPUT_DIR "('${SAVE_DIR}')" 

# PRETRAIN=../weights/r50_ibn_a.pth
# DATA_DIR='/data/Dataset/PReID/rep_dataset/'
# SAVE_DIR='../rep_work_dirs/exp5-cosinemgn2d-resnet50ibnls2-384x128-bs12x8-warmup10-flip-pad10-meanstd-lbsm-layer3_-5-erase0502-fix_allgem_3-cb2d_late-minInst2/' #(h, w)
# # SAVE_DIR='../rep_work_dirs/debug/' #(h, w)

# CUDA_VISIBLE_DEVICES=0 python train.py --config_file='configs/naic/cosinemgn.yml' \
#     SOLVER.IMS_PER_BATCH "96" DATALOADER.NUM_INSTANCE "8" \
#     MODEL.NAME "cosinemgn2d" MODEL.BACKBONE "('resnet50_ibn_a')" MODEL.COSINEMGN.NUM_SHARE_LAYER3 "-5" MODEL.COSINEMGN.POOL_TYPE "gem_3" MODEL.COSINEMGN.PART_POOL_TYPE "gem_3" MODEL.COSINEMGN.CB2D_BNFIRST "False" \
#     DATASETS.DATA_PATH "('${DATA_DIR}')" DATASETS.TRAIN_PATH "rep_f0_train2"\
#     MODEL.PRETRAIN_PATH "('${PRETRAIN}')"  \
#     OUTPUT_DIR "('${SAVE_DIR}')" 

# PRETRAIN=../weights/r50_ibn_a.pth
# DATA_DIR='/data/Dataset/PReID/rep_dataset/'
# SAVE_DIR='../rep_work_dirs/exp5-cosinemgn2d-resnet50ibnls2-384x128-bs12x8-warmup10-flip-pad10-meanstd-lbsm-layer3_-5-erase0502-allavg-cb2d_late-minInst2/' #(h, w)
# # SAVE_DIR='../rep_work_dirs/debug/' #(h, w)

# CUDA_VISIBLE_DEVICES=2 python train.py --config_file='configs/naic/cosinemgn.yml' \
#     SOLVER.IMS_PER_BATCH "96" DATALOADER.NUM_INSTANCE "8" \
#     MODEL.NAME "cosinemgn2d" MODEL.BACKBONE "('resnet50_ibn_a')" MODEL.COSINEMGN.NUM_SHARE_LAYER3 "-5" MODEL.COSINEMGN.POOL_TYPE "avg" MODEL.COSINEMGN.PART_POOL_TYPE "avg" MODEL.COSINEMGN.CB2D_BNFIRST "False" \
#     DATASETS.DATA_PATH "('${DATA_DIR}')" DATASETS.TRAIN_PATH "rep_f0_train2"\
#     MODEL.PRETRAIN_PATH "('${PRETRAIN}')"  \
#     OUTPUT_DIR "('${SAVE_DIR}')" 

# PRETRAIN=../weights/r50_ibn_a.pth
# DATA_DIR='/data/Dataset/PReID/rep_dataset/'
# SAVE_DIR='../rep_work_dirs/exp5-cosinemgn2d-resnet50ibnls2-384x144-bs12x8-warmup10-flip-pad10-meanstd-lbsm-layer3_-5-erase0502-allgem_3-cb2d_late-allbnbias-minInst2/' #(h, w)
# # SAVE_DIR='../rep_work_dirs/debug/' #(h, w)

# CUDA_VISIBLE_DEVICES=6 python train.py --config_file='configs/naic/cosinemgn.yml' \
#     SOLVER.IMS_PER_BATCH "96" DATALOADER.NUM_INSTANCE "8" \
#     INPUT.SIZE_TRAIN "([384,144])" INPUT.SIZE_TEST "([384,144])" \
#     MODEL.NAME "cosinemgn2d" MODEL.BACKBONE "('resnet50_ibn_a')" MODEL.COSINEMGN.NUM_SHARE_LAYER3 "-5" \
#     MODEL.COSINEMGN.POOL_TYPE "gem_3" MODEL.COSINEMGN.PART_POOL_TYPE "gem_3" MODEL.COSINEMGN.CB2D_BNFIRST "False" MODEL.COSINEMGN.GUSE_BNBAIS "True" MODEL.COSINEMGN.PUSE_BNBAIS "True"\
#     DATASETS.DATA_PATH "('${DATA_DIR}')" DATASETS.TRAIN_PATH "rep_f0_train2"\
#     MODEL.PRETRAIN_PATH "('${PRETRAIN}')"  \
#     OUTPUT_DIR "('${SAVE_DIR}')" 

# PRETRAIN=../weights/r50_ibn_a.pth
# DATA_DIR='/data/Dataset/PReID/rep_dataset/'
# SAVE_DIR='../rep_work_dirs/exp5-cosinemgn2d-resnet50ibnls2-384x144-bs12x8-warmup10-flip-pad10-meanstd-lbsm-layer3_-5-erase0502-avgsgem_3-cb2d_late-allbnbias-minInst2/' #(h, w)
# # SAVE_DIR='../rep_work_dirs/debug/' #(h, w)

# CUDA_VISIBLE_DEVICES=3 python train.py --config_file='configs/naic/cosinemgn.yml' \
#     SOLVER.IMS_PER_BATCH "96" DATALOADER.NUM_INSTANCE "8" \
#     INPUT.SIZE_TRAIN "([384,144])" INPUT.SIZE_TEST "([384,144])" \
#     MODEL.NAME "cosinemgn2d" MODEL.BACKBONE "('resnet50_ibn_a')" MODEL.COSINEMGN.NUM_SHARE_LAYER3 "-5" \
#     MODEL.COSINEMGN.POOL_TYPE "avg" MODEL.COSINEMGN.PART_POOL_TYPE "gem_3" MODEL.COSINEMGN.CB2D_BNFIRST "False" MODEL.COSINEMGN.GUSE_BNBAIS "True" MODEL.COSINEMGN.PUSE_BNBAIS "True"\
#     DATASETS.DATA_PATH "('${DATA_DIR}')" DATASETS.TRAIN_PATH "rep_f0_train2"\
#     MODEL.PRETRAIN_PATH "('${PRETRAIN}')"  \
#     OUTPUT_DIR "('${SAVE_DIR}')" 


# PRETRAIN=../weights/r50_ibn_a.pth
# DATA_DIR='/data/Dataset/PReID/rep_dataset/'
# SAVE_DIR='../rep_work_dirs/exp5-cosinemgn2d-resnet50ibnls2-384x144-bs16x6-warmup10-flip-pad10-meanstd-lbsm-layer3_-5-erase0502-allgem_3-cb2d_late-allbnbias-minInst2/' #(h, w)
# # SAVE_DIR='../rep_work_dirs/debug/' #(h, w)

# CUDA_VISIBLE_DEVICES=0 python train.py --config_file='configs/naic/cosinemgn.yml' \
#     SOLVER.IMS_PER_BATCH "96" DATALOADER.NUM_INSTANCE "6" \
#     INPUT.SIZE_TRAIN "([384,144])" INPUT.SIZE_TEST "([384,144])" \
#     MODEL.NAME "cosinemgn2d" MODEL.BACKBONE "('resnet50_ibn_a')" MODEL.COSINEMGN.NUM_SHARE_LAYER3 "-5" \
#     MODEL.COSINEMGN.POOL_TYPE "gem_3" MODEL.COSINEMGN.PART_POOL_TYPE "gem_3" MODEL.COSINEMGN.CB2D_BNFIRST "False" MODEL.COSINEMGN.GUSE_BNBAIS "True" MODEL.COSINEMGN.PUSE_BNBAIS "True"\
#     DATASETS.DATA_PATH "('${DATA_DIR}')" DATASETS.TRAIN_PATH "rep_f0_train2"\
#     MODEL.PRETRAIN_PATH "('${PRETRAIN}')"  \
#     OUTPUT_DIR "('${SAVE_DIR}')" 

# PRETRAIN=../weights/r50_ibn_a.pth
# DATA_DIR='/data/Dataset/PReID/rep_dataset/'
# SAVE_DIR='../rep_work_dirs/exp5-cosinemgn2d-resnet50ibnls2-384x144-bs16x6-warmup10-flip-pad10-meanstd-lbsm-layer3_-5-erase0502-allgem_3-cb2d_late-allbnbias-gcos3035_033-minInst2/' #(h, w)
# # SAVE_DIR='../rep_work_dirs/debug/' #(h, w)

# CUDA_VISIBLE_DEVICES=2 python train.py --config_file='configs/naic/cosinemgn.yml' \
#     SOLVER.IMS_PER_BATCH "96" DATALOADER.NUM_INSTANCE "6" \
#     MODEL.USE_COS "True" MODEL.COSINEMGN.GCE_WEIGHT "0.333" MODEL.COSINEMGN.GCOSINE_LOSS_TYPE 'CosFace' MODEL.COSINEMGN.S "30.0" MODEL.COSINEMGN.M "0.35" \
#     INPUT.SIZE_TRAIN "([384,144])" INPUT.SIZE_TEST "([384,144])" \
#     MODEL.NAME "cosinemgn2d" MODEL.BACKBONE "('resnet50_ibn_a')" MODEL.COSINEMGN.NUM_SHARE_LAYER3 "-5" \
#     MODEL.COSINEMGN.POOL_TYPE "gem_3" MODEL.COSINEMGN.PART_POOL_TYPE "gem_3" MODEL.COSINEMGN.CB2D_BNFIRST "False" MODEL.COSINEMGN.GUSE_BNBAIS "True" MODEL.COSINEMGN.PUSE_BNBAIS "True"\
#     DATASETS.DATA_PATH "('${DATA_DIR}')" DATASETS.TRAIN_PATH "rep_f0_train2"\
#     MODEL.PRETRAIN_PATH "('${PRETRAIN}')"  \
#     OUTPUT_DIR "('${SAVE_DIR}')" 
# [todo] resnext101_ibn_a about 45min/2epoch 
#

# PRETRAIN=../weights/r50_ibn_a.pth
# DATA_DIR='/data/Dataset/PReID/rep_dataset/'
# SAVE_DIR='../rep_work_dirs/exp5-mgn-resnet50ibnls2-384x144-bs16x6-warmup10-flip-pad10-meanstd-lbsm-layer3_-5-erase0502-allgem_3-minInst2/' #(h, w)
# CUDA_VISIBLE_DEVICES=3 python train.py --config_file='configs/naic/mgn.yml' \
#     SOLVER.IMS_PER_BATCH "96" DATALOADER.NUM_INSTANCE "6" \
#     INPUT.SIZE_TRAIN "([384,144])" INPUT.SIZE_TEST "([384,144])" \
#     MODEL.MGN.NUM_SHARE_LAYER3 "-5" MODEL.NAME "mgn" MODEL.BACKBONE "('resnet50_ibn_a')" MODEL.MGN.POOL_TYPE "gem_3" MODEL.MGN.PART_POOL_TYPE "gem_3"\
#     DATASETS.DATA_PATH "('${DATA_DIR}')" DATASETS.TRAIN_PATH "rep_f0_train2"\
#     MODEL.PRETRAIN_PATH "('${PRETRAIN}')"  \
#     OUTPUT_DIR "('${SAVE_DIR}')" 

# PRETRAIN=../weights/r50_ibn_a.pth
# DATA_DIR='/data/Dataset/PReID/rep_dataset/'
# SAVE_DIR='../rep_work_dirs/exp5-mgn-resnet50ibnls2-384x144-bs16x6-warmup10-flip-pad10-meanstd-lbsm-layer3_-5-erase0502-allgem_3-sepnorm-minInst2/' #(h, w)
# CUDA_VISIBLE_DEVICES=4 python train.py --config_file='configs/naic/mgn.yml' \
#     SOLVER.IMS_PER_BATCH "96" DATALOADER.NUM_INSTANCE "6" \
#     INPUT.SEPNORM.USE "True" \
#     INPUT.SEPNORM.PIXEL_MEAN "([[0.1164,0.1567,0.1796],[0.4556,0.5367,0.6493]])" \
#     INPUT.SEPNORM.PIXEL_STD "([[0.1467,0.1497,0.1930],[0.0750,0.0613,0.1040]])" \
#     INPUT.SIZE_TRAIN "([384,144])" INPUT.SIZE_TEST "([384,144])" \
#     MODEL.MGN.NUM_SHARE_LAYER3 "-5" MODEL.NAME "mgn" MODEL.BACKBONE "('resnet50_ibn_a')" MODEL.MGN.POOL_TYPE "gem_3" MODEL.MGN.PART_POOL_TYPE "gem_3"\
#     DATASETS.DATA_PATH "('${DATA_DIR}')" DATASETS.TRAIN_PATH "rep_f0_train2"\
#     MODEL.PRETRAIN_PATH "('${PRETRAIN}')"  \
#     OUTPUT_DIR "('${SAVE_DIR}')" 

# PRETRAIN=../weights/r50_ibn_a.pth
# DATA_DIR='/data/Dataset/PReID/rep_dataset/'
# SAVE_DIR='../rep_work_dirs/exp5-cosinemgn2d-resnet50ibnls2-384x144-bs16x6-warmup10-flip-pad10-meanstd-lbsm-layer3_-5-erase0502-allgem_3-cb2d_late-allbnbias-sepnorm-minInst2/' #(h, w)
# # SAVE_DIR='../rep_work_dirs/debug/' #(h, w)

# CUDA_VISIBLE_DEVICES=5 python train.py --config_file='configs/naic/cosinemgn.yml' \
#     SOLVER.IMS_PER_BATCH "96" DATALOADER.NUM_INSTANCE "6" \
#     INPUT.SEPNORM.USE "True" \
#     INPUT.SEPNORM.PIXEL_MEAN "([[0.1164,0.1567,0.1796],[0.4556,0.5367,0.6493]])" \
#     INPUT.SEPNORM.PIXEL_STD "([[0.1467,0.1497,0.1930],[0.0750,0.0613,0.1040]])" \
#     INPUT.SIZE_TRAIN "([384,144])" INPUT.SIZE_TEST "([384,144])" \
#     MODEL.NAME "cosinemgn2d" MODEL.BACKBONE "('resnet50_ibn_a')" MODEL.COSINEMGN.NUM_SHARE_LAYER3 "-5" \
#     MODEL.COSINEMGN.POOL_TYPE "gem_3" MODEL.COSINEMGN.PART_POOL_TYPE "gem_3" MODEL.COSINEMGN.CB2D_BNFIRST "False" MODEL.COSINEMGN.GUSE_BNBAIS "True" MODEL.COSINEMGN.PUSE_BNBAIS "True"\
#     DATASETS.DATA_PATH "('${DATA_DIR}')" DATASETS.TRAIN_PATH "rep_f0_train2"\
#     MODEL.PRETRAIN_PATH "('${PRETRAIN}')"  \
#     OUTPUT_DIR "('${SAVE_DIR}')" 

# PRETRAIN=../weights/r50_ibn_a.pth
# DATA_DIR='/data/Dataset/PReID/rep_dataset/'
# SAVE_DIR='../rep_work_dirs/exp5-cosinemgn2d-resnet50ibnls2-384x144-bs16x6-warmup10-flip-pad10-meanstd-lbsm-layer3_-5-erase0502-allgem_3-cb2d_late-allbnbias-minInst2-e140/' #(h, w)
# # SAVE_DIR='../rep_work_dirs/debug/' #(h, w)

# CUDA_VISIBLE_DEVICES=7 python train.py --config_file='configs/naic/cosinemgn.yml' \
#     SOLVER.IMS_PER_BATCH "96" DATALOADER.NUM_INSTANCE "6"  SOLVER.STEPS "[60, 90]" SOLVER.MAX_EPOCHS "140"\
#     INPUT.SIZE_TRAIN "([384,144])" INPUT.SIZE_TEST "([384,144])" \
#     MODEL.NAME "cosinemgn2d" MODEL.BACKBONE "('resnet50_ibn_a')" MODEL.COSINEMGN.NUM_SHARE_LAYER3 "-5" \
#     MODEL.COSINEMGN.POOL_TYPE "gem_3" MODEL.COSINEMGN.PART_POOL_TYPE "gem_3" MODEL.COSINEMGN.CB2D_BNFIRST "False" MODEL.COSINEMGN.GUSE_BNBAIS "True" MODEL.COSINEMGN.PUSE_BNBAIS "True"\
#     DATASETS.DATA_PATH "('${DATA_DIR}')" DATASETS.TRAIN_PATH "rep_f0_train2"\
#     MODEL.PRETRAIN_PATH "('${PRETRAIN}')"  \
#     OUTPUT_DIR "('${SAVE_DIR}')" 

# PRETRAIN=../weights/r50_ibn_a.pth
# DATA_DIR='/data/Dataset/PReID/rep_dataset/'
# SAVE_DIR='../rep_work_dirs/exp5-cosinemgn2d-resnet50ibnls2-384x144-bs16x6-warmup10-flip-pad10-meanstd-lbsm-layer3_-5-erase0502-allgem_3-cb2d_late-allbnbias-sepnorm-minInst2-e140/' #(h, w)
# # SAVE_DIR='../rep_work_dirs/debug/' #(h, w)

# CUDA_VISIBLE_DEVICES=5 python train.py --config_file='configs/naic/cosinemgn.yml' \
#     SOLVER.IMS_PER_BATCH "96" DATALOADER.NUM_INSTANCE "6"  SOLVER.STEPS "[60, 90]" SOLVER.MAX_EPOCHS "140"\
#     INPUT.SEPNORM.USE "True" \
#     INPUT.SEPNORM.PIXEL_MEAN "([[0.1164,0.1567,0.1796],[0.4556,0.5367,0.6493]])" \
#     INPUT.SEPNORM.PIXEL_STD "([[0.1467,0.1497,0.1930],[0.0750,0.0613,0.1040]])" \
#     INPUT.SIZE_TRAIN "([384,144])" INPUT.SIZE_TEST "([384,144])" \
#     MODEL.NAME "cosinemgn2d" MODEL.BACKBONE "('resnet50_ibn_a')" MODEL.COSINEMGN.NUM_SHARE_LAYER3 "-5" \
#     MODEL.COSINEMGN.POOL_TYPE "gem_3" MODEL.COSINEMGN.PART_POOL_TYPE "gem_3" MODEL.COSINEMGN.CB2D_BNFIRST "False" MODEL.COSINEMGN.GUSE_BNBAIS "True" MODEL.COSINEMGN.PUSE_BNBAIS "True"\
#     DATASETS.DATA_PATH "('${DATA_DIR}')" DATASETS.TRAIN_PATH "rep_f0_train2"\
#     MODEL.PRETRAIN_PATH "('${PRETRAIN}')"  \
#     OUTPUT_DIR "('${SAVE_DIR}')" 

# PRETRAIN=../weights/r50_ibn_a.pth
# DATA_DIR='/data/Dataset/PReID/rep_dataset/'
# SAVE_DIR='../rep_work_dirs/exp5-cosinemgn2d-resnet50ibnls2-384x144-bs16x6-warmup10-flip-pad10-meanstd-lbsm-layer3_-5-erase0502-allgem_3-cb2d_late-allbnbias-trainVal-e140/' #(h, w)
# # SAVE_DIR='../rep_work_dirs/debug/' #(h, w)

# CUDA_VISIBLE_DEVICES=0 python train.py --config_file='configs/naic/cosinemgn.yml' \
#     SOLVER.IMS_PER_BATCH "96" DATALOADER.NUM_INSTANCE "6"  SOLVER.STEPS "[60, 90]" SOLVER.MAX_EPOCHS "140"\
#     INPUT.SIZE_TRAIN "([384,144])" INPUT.SIZE_TEST "([384,144])" \
#     MODEL.NAME "cosinemgn2d" MODEL.BACKBONE "('resnet50_ibn_a')" MODEL.COSINEMGN.NUM_SHARE_LAYER3 "-5" \
#     MODEL.COSINEMGN.POOL_TYPE "gem_3" MODEL.COSINEMGN.PART_POOL_TYPE "gem_3" MODEL.COSINEMGN.CB2D_BNFIRST "False" MODEL.COSINEMGN.GUSE_BNBAIS "True" MODEL.COSINEMGN.PUSE_BNBAIS "True"\
#     DATASETS.DATA_PATH "('${DATA_DIR}')" DATASETS.TRAIN_PATH "rep_trainVal"\
#     MODEL.PRETRAIN_PATH "('${PRETRAIN}')"  \
#     OUTPUT_DIR "('${SAVE_DIR}')" 

# PRETRAIN=../weights/r50_ibn_a.pth
# DATA_DIR='/data/Dataset/PReID/rep_dataset/'
# SAVE_DIR='../rep_work_dirs/exp5-mgn-resnet50ibnls2-384x144-bs12x8-warmup10-flip-pad10-meanstd-lbsm-layer3_-5-erase0502-allgem_3-trainVal/' #(h, w)
# CUDA_VISIBLE_DEVICES=4 python train.py --config_file='configs/naic/mgn.yml' \
#     SOLVER.IMS_PER_BATCH "96" DATALOADER.NUM_INSTANCE "8" \
#     INPUT.SIZE_TRAIN "([384,144])" INPUT.SIZE_TEST "([384,144])" \
#     MODEL.MGN.NUM_SHARE_LAYER3 "-5" MODEL.NAME "mgn" MODEL.BACKBONE "('resnet50_ibn_a')" MODEL.MGN.POOL_TYPE "gem_3" MODEL.MGN.PART_POOL_TYPE "gem_3"\
#     DATASETS.DATA_PATH "('${DATA_DIR}')" DATASETS.TRAIN_PATH "rep_trainVal"\
#     MODEL.PRETRAIN_PATH "('${PRETRAIN}')"  \
#     OUTPUT_DIR "('${SAVE_DIR}')" 

# PRETRAIN=../weights/r50_ibn_a.pth
# DATA_DIR='/data/Dataset/PReID/rep_dataset/'
# SAVE_DIR='../rep_work_dirs/exp5-mgn-resnet50ibnls2-384x144-bs12x8-warmup10-flip-pad10-meanstd-lbsm-layer3_-5-erase0502-allgem_3-cj-minInst2/' #(h, w)
# CUDA_VISIBLE_DEVICES=2 python train.py --config_file='configs/naic/mgn.yml' \
#     INPUT.USE_CJ "True" INPUT.CJ_BRIGHNESS "0.5" INPUT.CJ_CONTRAST "0.5" INPUT.CJ_SATURATION "0.0" INPUT.CJ_HUE "0.5" \
#     SOLVER.IMS_PER_BATCH "96" DATALOADER.NUM_INSTANCE "8" \
#     INPUT.SIZE_TRAIN "([384,144])" INPUT.SIZE_TEST "([384,144])" \
#     MODEL.MGN.NUM_SHARE_LAYER3 "-5" MODEL.NAME "mgn" MODEL.BACKBONE "('resnet50_ibn_a')" MODEL.MGN.POOL_TYPE "gem_3" MODEL.MGN.PART_POOL_TYPE "gem_3"\
#     DATASETS.DATA_PATH "('${DATA_DIR}')" DATASETS.TRAIN_PATH "rep_f0_train2"\
#     MODEL.PRETRAIN_PATH "('${PRETRAIN}')"  \
#     OUTPUT_DIR "('${SAVE_DIR}')" 

# PRETRAIN=../weights/r50_ibn_a.pth
# DATA_DIR='/data/Dataset/PReID/rep_dataset/'
# SAVE_DIR='../rep_work_dirs/exp5-mgn-resnet50ibnls2-384x144-bs12x8-warmup10-flip-pad10-meanstd-lbsm-layer3_-5-erase0502-allgem_3-cj-trainVal/' #(h, w)
# CUDA_VISIBLE_DEVICES=3 python train.py --config_file='configs/naic/mgn.yml' \
#     INPUT.USE_CJ "True" INPUT.CJ_BRIGHNESS "0.5" INPUT.CJ_CONTRAST "0.5" INPUT.CJ_SATURATION "0.0" INPUT.CJ_HUE "0.5" \
#     SOLVER.IMS_PER_BATCH "96" DATALOADER.NUM_INSTANCE "8" \
#     INPUT.SIZE_TRAIN "([384,144])" INPUT.SIZE_TEST "([384,144])" \
#     MODEL.MGN.NUM_SHARE_LAYER3 "-5" MODEL.NAME "mgn" MODEL.BACKBONE "('resnet50_ibn_a')" MODEL.MGN.POOL_TYPE "gem_3" MODEL.MGN.PART_POOL_TYPE "gem_3"\
#     DATASETS.DATA_PATH "('${DATA_DIR}')" DATASETS.TRAIN_PATH "rep_trainVal"\
#     MODEL.PRETRAIN_PATH "('${PRETRAIN}')"  \
#     OUTPUT_DIR "('${SAVE_DIR}')" 

# [todo] follow baseline
# PRETRAIN=../weights/r50_ibn_a.pth
# DATA_DIR='/data/Dataset/PReID/rep_dataset/'
# SAVE_DIR='../rep_work_dirs/exp5-cosinemgn2d-resnet50ibnls2-384x144-bs16x6-warmup10-flip-pad10-meanstd-lbsm-layer3_-5-erase0502-avggem_3-cb2d_late-cj-minInst2/' #(h, w)
# # SAVE_DIR='../rep_work_dirs/debug/' #(h, w)

# CUDA_VISIBLE_DEVICES=7 python train.py --config_file='configs/naic/cosinemgn.yml' \
#     SOLVER.IMS_PER_BATCH "96" DATALOADER.NUM_INSTANCE "6" \
#     INPUT.USE_CJ "True" INPUT.CJ_BRIGHNESS "0.5" INPUT.CJ_CONTRAST "0.5" INPUT.CJ_SATURATION "0.0" INPUT.CJ_HUE "0.5" \
#     INPUT.SIZE_TRAIN "([384,144])" INPUT.SIZE_TEST "([384,144])" \
#     MODEL.NAME "cosinemgn2d" MODEL.BACKBONE "('resnet50_ibn_a')" MODEL.COSINEMGN.NUM_SHARE_LAYER3 "-5" MODEL.COSINEMGN.POOL_TYPE "avg" MODEL.COSINEMGN.PART_POOL_TYPE "gem_3" MODEL.COSINEMGN.CB2D_BNFIRST "False"\
#     MODEL.COSINEMGN.PUSE_BNBAIS "False" MODEL.COSINEMGN.GUSE_BNBAIS "False" \
#     DATASETS.DATA_PATH "('${DATA_DIR}')" DATASETS.TRAIN_PATH "rep_f0_train2"\
#     MODEL.PRETRAIN_PATH "('${PRETRAIN}')"  \
#     OUTPUT_DIR "('${SAVE_DIR}')" 

# PRETRAIN=../weights/r50_ibn_a.pth
# DATA_DIR='/data/Dataset/PReID/rep_dataset/'
# SAVE_DIR='../rep_work_dirs/exp5-cosinemgn2d-resnet50ibnls2-384x144-bs16x6-warmup10-flip-pad10-meanstd-lbsm-layer3_-5-erase0502-avggem_3-cb2d_late-cj-minInst2/' #(h, w)
# # SAVE_DIR='../rep_work_dirs/debug/' #(h, w)

# CUDA_VISIBLE_DEVICES=7 python train.py --config_file='configs/naic/cosinemgn.yml' \
#     MODEL.COSINEMGN.TPL_WEIGHT "1.0" MODEL.COSINEMGN.PCE_WEIGHT "2.0" MODEL.COSINEMGN.GCE_WEIGHT "1.0" \
#     SOLVER.IMS_PER_BATCH "96" DATALOADER.NUM_INSTANCE "6" \
#     INPUT.USE_CJ "True" INPUT.CJ_BRIGHNESS "0.5" INPUT.CJ_CONTRAST "0.5" INPUT.CJ_SATURATION "0.0" INPUT.CJ_HUE "0.5" \
#     INPUT.SIZE_TRAIN "([384,144])" INPUT.SIZE_TEST "([384,144])" \
#     MODEL.NAME "cosinemgn2d" MODEL.BACKBONE "('resnet50_ibn_a')" MODEL.COSINEMGN.NUM_SHARE_LAYER3 "-5" MODEL.COSINEMGN.POOL_TYPE "avg" MODEL.COSINEMGN.PART_POOL_TYPE "gem_3" MODEL.COSINEMGN.CB2D_BNFIRST "False"\
#     MODEL.COSINEMGN.PUSE_BNBAIS "False" MODEL.COSINEMGN.GUSE_BNBAIS "False" \
#     DATASETS.DATA_PATH "('${DATA_DIR}')" DATASETS.TRAIN_PATH "rep_f0_train2"\
#     MODEL.PRETRAIN_PATH "('${PRETRAIN}')"  \
#     OUTPUT_DIR "('${SAVE_DIR}')" 

# PRETRAIN=../weights/resnet101_ibn_a.pth.tar
# DATA_DIR='/data/Dataset/PReID/rep_dataset/'
# SAVE_DIR='../rep_work_dirs/exp5-mgn-resnet101ibnls1-384x144-bs16x6-warmup10-flip-pad10-meanstd-lbsm-layer3_-5-erase0502-allgem_3-cj-minInst2/' #(h, w)
# CUDA_VISIBLE_DEVICES=2 python train.py --config_file='configs/naic/mgn.yml' \
#     INPUT.USE_CJ "True" INPUT.CJ_BRIGHNESS "0.5" INPUT.CJ_CONTRAST "0.5" INPUT.CJ_SATURATION "0.0" INPUT.CJ_HUE "0.5" \
#     SOLVER.IMS_PER_BATCH "96" DATALOADER.NUM_INSTANCE "6" SOLVER.STEPS "[40, 70]" SOLVER.MAX_EPOCHS "90" SOLVER.START_SAVE_EPOCH "75" \
#     INPUT.SIZE_TRAIN "([384,144])" INPUT.SIZE_TEST "([384,144])" \
#     MODEL.MGN.NUM_SHARE_LAYER3 "-5" MODEL.NAME "mgn" MODEL.BACKBONE "('resnet101_ibn_a')" MODEL.MGN.POOL_TYPE "gem_3" MODEL.MGN.PART_POOL_TYPE "gem_3"\
#     DATASETS.DATA_PATH "('${DATA_DIR}')" DATASETS.TRAIN_PATH "rep_f0_train2"\
#     MODEL.PRETRAIN_PATH "('${PRETRAIN}')"  \
#     OUTPUT_DIR "('${SAVE_DIR}')" 

# PRETRAIN=../weights/resnet101_ibn_a.pth.tar
# DATA_DIR='/data/Dataset/PReID/rep_dataset/'
# SAVE_DIR='../rep_work_dirs/exp5-mgn-resnet101ibnls1-384x144-bs16x6-warmup10-flip-pad10-meanstd-lbsm-layer3_-5-erase0502-allgem_3-cj-trainVal2/' #(h, w)
# CUDA_VISIBLE_DEVICES=7 python train.py --config_file='configs/naic/mgn.yml' \
#     INPUT.USE_CJ "True" INPUT.CJ_BRIGHNESS "0.5" INPUT.CJ_CONTRAST "0.5" INPUT.CJ_SATURATION "0.0" INPUT.CJ_HUE "0.5" \
#     SOLVER.IMS_PER_BATCH "96" DATALOADER.NUM_INSTANCE "6" SOLVER.STEPS "[40, 70]" SOLVER.MAX_EPOCHS "90" SOLVER.START_SAVE_EPOCH "75" \
#     INPUT.SIZE_TRAIN "([384,144])" INPUT.SIZE_TEST "([384,144])" \
#     MODEL.MGN.NUM_SHARE_LAYER3 "-5" MODEL.NAME "mgn" MODEL.BACKBONE "('resnet101_ibn_a')" MODEL.MGN.POOL_TYPE "gem_3" MODEL.MGN.PART_POOL_TYPE "gem_3"\
#     DATASETS.DATA_PATH "('${DATA_DIR}')" DATASETS.TRAIN_PATH "rep_trainVal2"\
#     MODEL.PRETRAIN_PATH "('${PRETRAIN}')"  \
#     OUTPUT_DIR "('${SAVE_DIR}')" 

# PRETRAIN=../weights/resnet101_ibn_a.pth.tar
# DATA_DIR='/data/Dataset/PReID/rep_dataset/'
# SAVE_DIR='../rep_work_dirs/exp5-mgn-resnet101ibnls1-384x144-bs16x6-warmup10-flip-pad10-meanstd-lbsm-layer3_-5-erase0502-allgem_3-cj05-trainVal2/' #(h, w)
# CUDA_VISIBLE_DEVICES=6 python train.py --config_file='configs/naic/mgn.yml' \
#     INPUT.USE_CJ "True" INPUT.CJ_PROB "0.5" INPUT.CJ_BRIGHNESS "0.5" INPUT.CJ_CONTRAST "0.5" INPUT.CJ_SATURATION "0.0" INPUT.CJ_HUE "0.5" \
#     SOLVER.IMS_PER_BATCH "96" DATALOADER.NUM_INSTANCE "6" SOLVER.STEPS "[40, 70]" SOLVER.MAX_EPOCHS "90" SOLVER.START_SAVE_EPOCH "75" \
#     INPUT.SIZE_TRAIN "([384,144])" INPUT.SIZE_TEST "([384,144])" \
#     MODEL.MGN.NUM_SHARE_LAYER3 "-5" MODEL.NAME "mgn" MODEL.BACKBONE "('resnet101_ibn_a')" MODEL.MGN.POOL_TYPE "gem_3" MODEL.MGN.PART_POOL_TYPE "gem_3"\
#     DATASETS.DATA_PATH "('${DATA_DIR}')" DATASETS.TRAIN_PATH "rep_trainVal2"\
#     MODEL.PRETRAIN_PATH "('${PRETRAIN}')"  \
#     OUTPUT_DIR "('${SAVE_DIR}')" 

# PRETRAIN=../weights/resnet101_ibn_a.pth.tar
# DATA_DIR='/data/Dataset/PReID/rep_dataset/'
# SAVE_DIR='../rep_work_dirs/exp5-mgn-resnet101ibnls1-384x144-bs12x8-warmup10-flip-pad10-meanstd-lbsm-layer3_-5-erase0502-allgem_3-cj-trainVal2/' #(h, w)
# CUDA_VISIBLE_DEVICES=2 python train.py --config_file='configs/naic/mgn.yml' \
#     INPUT.USE_CJ "True" INPUT.CJ_BRIGHNESS "0.5" INPUT.CJ_CONTRAST "0.5" INPUT.CJ_SATURATION "0.0" INPUT.CJ_HUE "0.5" \
#     SOLVER.IMS_PER_BATCH "96" DATALOADER.NUM_INSTANCE "8" SOLVER.STEPS "[40, 70]" SOLVER.MAX_EPOCHS "90" SOLVER.START_SAVE_EPOCH "75" \
#     INPUT.SIZE_TRAIN "([384,144])" INPUT.SIZE_TEST "([384,144])" \
#     MODEL.MGN.NUM_SHARE_LAYER3 "-5" MODEL.NAME "mgn" MODEL.BACKBONE "('resnet101_ibn_a')" MODEL.MGN.POOL_TYPE "gem_3" MODEL.MGN.PART_POOL_TYPE "gem_3"\
#     DATASETS.DATA_PATH "('${DATA_DIR}')" DATASETS.TRAIN_PATH "rep_trainVal2"\
#     MODEL.PRETRAIN_PATH "('${PRETRAIN}')"  \
#     OUTPUT_DIR "('${SAVE_DIR}')" 

# PRETRAIN=../weights/resnet101_ibn_a.pth.tar
# DATA_DIR='/data/Dataset/PReID/rep_dataset/'
# SAVE_DIR='../rep_work_dirs/exp5-mgn-resnet101ibnls1-384x144-bs12x8-warmup10-flip-pad10-meanstd-lbsm-layer3_-5-erase0502-allgem_3-cj05-trainVal2/' #(h, w)
# CUDA_VISIBLE_DEVICES=3 python train.py --config_file='configs/naic/mgn.yml' \
#     INPUT.USE_CJ "True" INPUT.CJ_PROB "0.5" INPUT.CJ_BRIGHNESS "0.5" INPUT.CJ_CONTRAST "0.5" INPUT.CJ_SATURATION "0.0" INPUT.CJ_HUE "0.5" \
#     SOLVER.IMS_PER_BATCH "96" DATALOADER.NUM_INSTANCE "8" SOLVER.STEPS "[40, 70]" SOLVER.MAX_EPOCHS "90" SOLVER.START_SAVE_EPOCH "75" \
#     INPUT.SIZE_TRAIN "([384,144])" INPUT.SIZE_TEST "([384,144])" \
#     MODEL.MGN.NUM_SHARE_LAYER3 "-5" MODEL.NAME "mgn" MODEL.BACKBONE "('resnet101_ibn_a')" MODEL.MGN.POOL_TYPE "gem_3" MODEL.MGN.PART_POOL_TYPE "gem_3"\
#     DATASETS.DATA_PATH "('${DATA_DIR}')" DATASETS.TRAIN_PATH "rep_trainVal2"\
#     MODEL.PRETRAIN_PATH "('${PRETRAIN}')"  \
#     OUTPUT_DIR "('${SAVE_DIR}')" 

# PRETRAIN=../weights/resnet101_ibn_a.pth.tar
# DATA_DIR='/data/Dataset/PReID/rep_dataset/'
# SAVE_DIR='../rep_work_dirs/exp5-cosinemgn2d-resnet101ibnls2-384x144-bs12x8-warmup10-flip-pad10-meanstd-lbsm-layer3_-5-erase0502-avggem_3-cb2d_late-cj05-arcface-minInst2/' #(h, w)
# # SAVE_DIR='../rep_work_dirs/debug/' #(h, w)

# CUDA_VISIBLE_DEVICES=7 python train.py --config_file='configs/naic/cosinemgn.yml' \
#     SOLVER.STEPS "[40, 65]" SOLVER.MAX_EPOCHS "90" SOLVER.START_SAVE_EPOCH "70" \
#     MODEL.COSINEMGN.TPL_WEIGHT "1.0" MODEL.COSINEMGN.PCE_WEIGHT "2.0" MODEL.COSINEMGN.GCE_WEIGHT "0.333" \
#     SOLVER.IMS_PER_BATCH "96" DATALOADER.NUM_INSTANCE "8" \
#     INPUT.USE_CJ "True" INPUT.CJ_PROB "0.5" INPUT.CJ_BRIGHNESS "0.5" INPUT.CJ_CONTRAST "0.5" INPUT.CJ_SATURATION "0.0" INPUT.CJ_HUE "0.5" \
#     MODEL.USE_COS "True" MODEL.COSINEMGN.GCOSINE_LOSS_TYPE 'ArcCos' MODEL.COSINEMGN.S "30.0" MODEL.COSINEMGN.M "0.35" \
#     INPUT.SIZE_TRAIN "([384,144])" INPUT.SIZE_TEST "([384,144])" \
#     MODEL.NAME "cosinemgn2d" MODEL.BACKBONE "('resnet101_ibn_a')" MODEL.COSINEMGN.NUM_SHARE_LAYER3 "-5" MODEL.COSINEMGN.POOL_TYPE "avg" MODEL.COSINEMGN.PART_POOL_TYPE "gem_3" MODEL.COSINEMGN.CB2D_BNFIRST "False"\
#     MODEL.COSINEMGN.PUSE_BNBAIS "False" MODEL.COSINEMGN.GUSE_BNBAIS "False" \
#     DATASETS.DATA_PATH "('${DATA_DIR}')" DATASETS.TRAIN_PATH "rep_f0_train2"\
#     MODEL.PRETRAIN_PATH "('${PRETRAIN}')"  \
#     OUTPUT_DIR "('${SAVE_DIR}')" 

# testb
# PRETRAIN=../weights/r50_ibn_a.pth
# DATA_DIR='/data/Dataset/PReID/rep_dataset/'
# SAVE_DIR='../rep_work_dirs/exp5-mgn-resnet50ibnls2-384x144-bs12x8-warmup10-flip-pad10-meanstd-lbsm-layer3_-5-erase0502-allgem_3-minInst2_st2/' #(h, w)
# CUDA_VISIBLE_DEVICES=2 python train.py --config_file='configs/naic/mgn.yml' \
#     SOLVER.STEPS "[40, 70]" SOLVER.MAX_EPOCHS "80" SOLVER.START_SAVE_EPOCH "70" \
#     SOLVER.IMS_PER_BATCH "96" DATALOADER.NUM_INSTANCE "8" \
#     INPUT.SIZE_TRAIN "([384,144])" INPUT.SIZE_TEST "([384,144])" \
#     MODEL.MGN.NUM_SHARE_LAYER3 "-5" MODEL.NAME "mgn" MODEL.BACKBONE "('resnet50_ibn_a')" MODEL.MGN.POOL_TYPE "gem_3" MODEL.MGN.PART_POOL_TYPE "gem_3"\
#     DATASETS.DATA_PATH "('${DATA_DIR}')" DATASETS.TRAIN_PATH "rep_f0_train2"\
#     MODEL.PRETRAIN_PATH "('${PRETRAIN}')"  \
#     OUTPUT_DIR "('${SAVE_DIR}')" 

# PRETRAIN=../weights/r50_ibn_a.pth
# DATA_DIR='/data/Dataset/PReID/rep_dataset/'
# SAVE_DIR='../rep_work_dirs/exp5-mgn-resnet50ibnls2-384x144-bs12x8-warmup10-flip-pad10-meanstd-lbsm-layer3_-5-erase0502-allgem_3-minInst2-pseudo_finetune/' #(h, w)
# CUDA_VISIBLE_DEVICES=7 python train.py --config_file='configs/naic/mgn.yml' \
#     MODEL.WEIGHT "../rep_work_dirs/exp5-mgn-resnet50ibnls2-384x144-bs12x8-warmup10-flip-pad10-meanstd-lbsm-layer3_-5-erase0502-allgem_3-minInst2/mgn_epoch102.pth" \
#     SOLVER.BASE_LR '1e-4' SOLVER.WARMUP_EPOCH "4" SOLVER.STEPS "[8, 10]" SOLVER.MAX_EPOCHS "12" SOLVER.START_SAVE_EPOCH "7" SOLVER.EVAL_PERIOD "2"\
#     SOLVER.IMS_PER_BATCH "96" DATALOADER.NUM_INSTANCE "8" \
#     INPUT.SIZE_TRAIN "([384,144])" INPUT.SIZE_TEST "([384,144])" \
#     MODEL.MGN.NUM_SHARE_LAYER3 "-5" MODEL.NAME "mgn" MODEL.BACKBONE "('resnet50_ibn_a')" MODEL.MGN.POOL_TYPE "gem_3" MODEL.MGN.PART_POOL_TYPE "gem_3"\
#     DATASETS.DATA_PATH "('${DATA_DIR}')" DATASETS.TRAIN_PATH "/data/Dataset/PReID/pseudo_mgn_eps045_2_dataset/rep_trainVal2" \
#     MODEL.PRETRAIN_PATH "('${PRETRAIN}')"  \
#     OUTPUT_DIR "('${SAVE_DIR}')" 

# PRETRAIN=../weights/r50_ibn_a.pth
# DATA_DIR='/data/Dataset/PReID/rep_dataset/'
# SAVE_DIR='../rep_work_dirs/exp5-mgn-resnet50ibnls2-384x144-bs12x8-warmup10-flip-pad10-meanstd-lbsm-layer3_-5-erase0502-allgem_3-minInst2-pseudo_finetune_fixbackbone/' #(h, w)
# CUDA_VISIBLE_DEVICES=0 python train.py --config_file='configs/naic/mgn.yml' \
#     MODEL.WEIGHT "../rep_work_dirs/exp5-mgn-resnet50ibnls2-384x144-bs12x8-warmup10-flip-pad10-meanstd-lbsm-layer3_-5-erase0502-allgem_3-minInst2/mgn_epoch102.pth" \
#     SOLVER.BASE_LR '1e-4' SOLVER.WARMUP_EPOCH "4" SOLVER.STEPS "[8, 10]" SOLVER.MAX_EPOCHS "12" SOLVER.START_SAVE_EPOCH "7" SOLVER.EVAL_PERIOD "2" SOLVER.FIX_BACKBONE "True"\
#     SOLVER.IMS_PER_BATCH "96" DATALOADER.NUM_INSTANCE "8" \
#     INPUT.SIZE_TRAIN "([384,144])" INPUT.SIZE_TEST "([384,144])" \
#     MODEL.MGN.NUM_SHARE_LAYER3 "-5" MODEL.NAME "mgn" MODEL.BACKBONE "('resnet50_ibn_a')" MODEL.MGN.POOL_TYPE "gem_3" MODEL.MGN.PART_POOL_TYPE "gem_3"\
#     DATASETS.DATA_PATH "('${DATA_DIR}')" DATASETS.TRAIN_PATH "/data/Dataset/PReID/pseudo_mgn_eps045_2_dataset/rep_trainVal2" \
#     MODEL.PRETRAIN_PATH "('${PRETRAIN}')"  \
#     OUTPUT_DIR "('${SAVE_DIR}')" 

# PRETRAIN=../weights/r50_ibn_a.pth
# DATA_DIR='/data/Dataset/PReID/rep_dataset/'
# SAVE_DIR='../rep_work_dirs/exp5-mgn-resnet50ibnls2-384x144-bs12x8-warmup10-flip-pad10-meanstd-lbsm-layer3_-5-erase0502-allgem_3-minInst2-pseudo_finetune_fixbackbone_st2/' #(h, w)
# CUDA_VISIBLE_DEVICES=7 python train.py --config_file='configs/naic/mgn.yml' \
#     MODEL.WEIGHT "../rep_work_dirs/exp5-mgn-resnet50ibnls2-384x144-bs12x8-warmup10-flip-pad10-meanstd-lbsm-layer3_-5-erase0502-allgem_3-minInst2/mgn_epoch102.pth" \
#     SOLVER.BASE_LR '1e-4' SOLVER.WARMUP_EPOCH "4" SOLVER.STEPS "[10, 16]" SOLVER.MAX_EPOCHS "20" SOLVER.START_SAVE_EPOCH "7" SOLVER.EVAL_PERIOD "2" SOLVER.FIX_BACKBONE "True"\
#     SOLVER.IMS_PER_BATCH "96" DATALOADER.NUM_INSTANCE "8" \
#     INPUT.SIZE_TRAIN "([384,144])" INPUT.SIZE_TEST "([384,144])" \
#     MODEL.MGN.NUM_SHARE_LAYER3 "-5" MODEL.NAME "mgn" MODEL.BACKBONE "('resnet50_ibn_a')" MODEL.MGN.POOL_TYPE "gem_3" MODEL.MGN.PART_POOL_TYPE "gem_3"\
#     DATASETS.DATA_PATH "('${DATA_DIR}')" DATASETS.TRAIN_PATH "/data/Dataset/PReID/pseudo_mgn_eps045_2_dataset/rep_trainVal2" \
#     MODEL.PRETRAIN_PATH "('${PRETRAIN}')"  \
#     OUTPUT_DIR "('${SAVE_DIR}')" 


# PRETRAIN=../weights/r50_ibn_a.pth
# DATA_DIR='/data/Dataset/PReID/rep_dataset/'
# SAVE_DIR='../rep_work_dirs/exp5-mgn-resnet50ibnls2-384x144-bs12x8-warmup10-flip-pad10-meanstd-lbsm-layer3_-5-erase0502-allgem_3-minInst2-pseudohigh_retrain/' #(h, w)
# CUDA_VISIBLE_DEVICES=7 python train.py --config_file='configs/naic/mgn.yml' \
#     SOLVER.BASE_LR '3e-4' SOLVER.WARMUP_EPOCH "5" SOLVER.STEPS "[25, 45]" SOLVER.MAX_EPOCHS "60" SOLVER.START_SAVE_EPOCH "40" SOLVER.EVAL_PERIOD "2"\
#     SOLVER.IMS_PER_BATCH "96" DATALOADER.NUM_INSTANCE "8" \
#     INPUT.SIZE_TRAIN "([384,144])" INPUT.SIZE_TEST "([384,144])" \
#     MODEL.MGN.NUM_SHARE_LAYER3 "-5" MODEL.NAME "mgn" MODEL.BACKBONE "('resnet50_ibn_a')" MODEL.MGN.POOL_TYPE "gem_3" MODEL.MGN.PART_POOL_TYPE "gem_3"\
#     DATASETS.DATA_PATH "('${DATA_DIR}')" DATASETS.TRAIN_PATH "/data/Dataset/PReID/pseudo_auto_flip_eps055_2_dataset/rep_trainVal2" \
#     MODEL.PRETRAIN_PATH "('${PRETRAIN}')"  \
#     OUTPUT_DIR "('${SAVE_DIR}')" 