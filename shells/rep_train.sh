#!/usr/bin/env bash
# PRETRAIN=../weights/r50_ibn_a.pth
# DATA_DIR='/share/Dataset/PReID/all_dataset/'
# SAVE_DIR='../rep_work_dirs/exp4-mfn-resnet50ibnls1-256x128-bs16x8-warmup10-flip-pad10-meanstd-erase0502-lbsm-tpl05-gem_3/' #(h, w)
# CUDA_VISIBLE_DEVICES=6,7 python train.py --config_file='configs/naic/mfn.yml' \
#     SOLVER.IMS_PER_BATCH "128" DATALOADER.NUM_INSTANCE "8" \
#     MODEL.NAME "mfn" MODEL.BACKBONE "('resnet50_ibn_a')" MODEL.MFN.POOL_TYPE "gem_3" \
#     DATASETS.DATA_PATH "('${DATA_DIR}')" DATASETS.TRAIN_PATH "rep_f0_train"\
#     MODEL.PRETRAIN_PATH "('${PRETRAIN}')"  \
#     OUTPUT_DIR "('${SAVE_DIR}')" 

# PRETRAIN=../weights/r50_ibn_a.pth
# DATA_DIR='/share/Dataset/PReID/all_dataset/'
# SAVE_DIR='../rep_work_dirs/exp4-mfn-resnet50ibnls1-256x128-bs16x8-warmup10-flip-pad10-meanstd-erase0502-lbsm-tpl05-gem_3-pre/' #(h, w)
# CUDA_VISIBLE_DEVICES=0,1 python train.py --config_file='configs/naic/mfn.yml' \
#     SOLVER.IMS_PER_BATCH "128" DATALOADER.NUM_INSTANCE "8" \
#     MODEL.NAME "mfn" MODEL.BACKBONE "('resnet50_ibn_a')" MODEL.MFN.POOL_TYPE "gem_3" \
#     DATASETS.DATA_PATH "('${DATA_DIR}')" DATASETS.TRAIN_PATH "all_f0_train"\
#     MODEL.PRETRAIN_PATH "('${PRETRAIN}')"  \
#     OUTPUT_DIR "('${SAVE_DIR}')" 

# PRETRAIN=../weights/r50_ibn_a.pth
# DATA_DIR='/data/Dataset/PReID/all_dataset/'
# SAVE_DIR='../rep_work_dirs/exp4-mfn-resnet50ibnls1-256x128-bs16x8-warmup10-flip-pad10-meanstd-erase0502-lbsm-tpl05-gem_3-minInst2/' #(h, w)
# CUDA_VISIBLE_DEVICES=4 python train.py --config_file='configs/naic/mfn.yml' \
#     SOLVER.IMS_PER_BATCH "128" DATALOADER.NUM_INSTANCE "8" \
#     MODEL.NAME "mfn" MODEL.BACKBONE "('resnet50_ibn_a')" MODEL.MFN.POOL_TYPE "gem_3" \
#     DATASETS.DATA_PATH "('${DATA_DIR}')" DATASETS.TRAIN_PATH "rep_f0_train2"\
#     MODEL.PRETRAIN_PATH "('${PRETRAIN}')"  \
#     OUTPUT_DIR "('${SAVE_DIR}')" 

# PRETRAIN=../weights/r50_ibn_a.pth
# DATA_DIR='/data/Dataset/PReID/all_dataset/'
# SAVE_DIR='../rep_work_dirs/exp4-mfn-resnet50ibnls1-256x128-bs16x8-warmup10-flip-pad10-meanstd-erase0502-lbsm-tpl05-allgem_3-minInst2/' #(h, w)
# CUDA_VISIBLE_DEVICES=5 python train.py --config_file='configs/naic/mfn.yml' \
#     SOLVER.IMS_PER_BATCH "128" DATALOADER.NUM_INSTANCE "8" \
#     MODEL.NAME "mfn" MODEL.BACKBONE "('resnet50_ibn_a')" MODEL.MFN.POOL_TYPE "gem_3" MODEL.MFN.AUX_POOL_TYPE "gem_3"\
#     DATASETS.DATA_PATH "('${DATA_DIR}')" DATASETS.TRAIN_PATH "rep_f0_train2"\
#     MODEL.PRETRAIN_PATH "('${PRETRAIN}')"  \
#     OUTPUT_DIR "('${SAVE_DIR}')" 

# PRETRAIN=../weights/r50_ibn_a.pth
# DATA_DIR='/data/Dataset/PReID/all_dataset/'
# SAVE_DIR='../rep_work_dirs/exp4-mfn-resnet50ibnls1-256x128-bs16x8-warmup10-flip-pad10-meanstd-erase0502-lbsm-tpl05-allgem_3-minInst4/' #(h, w)
# CUDA_VISIBLE_DEVICES=5 python train.py --config_file='configs/naic/mfn.yml' \
#     SOLVER.IMS_PER_BATCH "128" DATALOADER.NUM_INSTANCE "8" \
#     MODEL.NAME "mfn" MODEL.BACKBONE "('resnet50_ibn_a')" MODEL.MFN.POOL_TYPE "gem_3" MODEL.MFN.AUX_POOL_TYPE "gem_3"\
#     DATASETS.DATA_PATH "('${DATA_DIR}')" DATASETS.TRAIN_PATH "rep_f0_train4"\
#     MODEL.PRETRAIN_PATH "('${PRETRAIN}')"  \
#     OUTPUT_DIR "('${SAVE_DIR}')" 

# PRETRAIN=../weights/r50_ibn_a.pth
# DATA_DIR='/data/Dataset/PReID/all_dataset/'
# SAVE_DIR='../rep_work_dirs/exp4-mfn-resnet50ibnls1-256x128-bs16x8-warmup10-flip-pad10-meanstd-erase0502-lbsm-tpl05-nosm_allgem_3-minInst2/' #(h, w)
# CUDA_VISIBLE_DEVICES=4 python train.py --config_file='configs/naic/mfn.yml' \
#     SOLVER.IMS_PER_BATCH "128" DATALOADER.NUM_INSTANCE "8" \
#     MODEL.NAME "mfn" MODEL.BACKBONE "('resnet50_ibn_a')" MODEL.MFN.POOL_TYPE "gem_3" MODEL.MFN.AUX_POOL_TYPE "gem_3" MODEL.MFN.AUX_SMOOTH "False"\
#     DATASETS.DATA_PATH "('${DATA_DIR}')" DATASETS.TRAIN_PATH "rep_f0_train2"\
#     MODEL.PRETRAIN_PATH "('${PRETRAIN}')"  \
#     OUTPUT_DIR "('${SAVE_DIR}')" 

# PRETRAIN=../weights/r50_ibn_a.pth
# DATA_DIR='/data/Dataset/PReID/all_dataset/'
# SAVE_DIR='../rep_work_dirs/exp4-mfn-resnet50ibnls1-256x128-bs16x4-warmup10-flip-pad10-meanstd-erase0502-lbsm-tpl05-nosm_allgem_3-minInst2/' #(h, w)
# CUDA_VISIBLE_DEVICES=4 python train.py --config_file='configs/naic/mfn.yml' \
#     SOLVER.IMS_PER_BATCH "64" DATALOADER.NUM_INSTANCE "4" \
#     MODEL.NAME "mfn" MODEL.BACKBONE "('resnet50_ibn_a')" MODEL.MFN.POOL_TYPE "gem_3" MODEL.MFN.AUX_POOL_TYPE "gem_3" MODEL.MFN.AUX_SMOOTH "False"\
#     DATASETS.DATA_PATH "('${DATA_DIR}')" DATASETS.TRAIN_PATH "rep_f0_train2"\
#     MODEL.PRETRAIN_PATH "('${PRETRAIN}')"  \
#     OUTPUT_DIR "('${SAVE_DIR}')" 

# PRETRAIN=../weights/r50_ibn_a.pth
# DATA_DIR='/data/Dataset/PReID/all_dataset/'
# SAVE_DIR='../rep_work_dirs/exp4-mfn-resnet50ibnls1-256x128-bs16x8-warmup10-flip-pad10-meanstd-erase0502-lbsm-tpl05-nosm_allgem-minInst2/' #(h, w)
# CUDA_VISIBLE_DEVICES=5 python train.py --config_file='configs/naic/mfn.yml' \
#     SOLVER.IMS_PER_BATCH "128" DATALOADER.NUM_INSTANCE "8" \
#     MODEL.NAME "mfn" MODEL.BACKBONE "('resnet50_ibn_a')" MODEL.MFN.POOL_TYPE "gem" MODEL.MFN.AUX_POOL_TYPE "gem" MODEL.MFN.AUX_SMOOTH "False"\
#     DATASETS.DATA_PATH "('${DATA_DIR}')" DATASETS.TRAIN_PATH "rep_f0_train2"\
#     MODEL.PRETRAIN_PATH "('${PRETRAIN}')"  \
#     OUTPUT_DIR "('${SAVE_DIR}')" 

# PRETRAIN=../weights/se_resnext50_32x4d-a260b3a4.pth
# DATA_DIR='/data/Dataset/PReID/all_dataset/'
# SAVE_DIR='../rep_work_dirs/exp4-mfn-se_resnext50ls1-256x128-bs16x8-warmup10-flip-pad10-meanstd-erase0502-lbsm-tpl05-nosm_allgem_3-minInst2/' #(h, w)
# CUDA_VISIBLE_DEVICES=4 python train.py --config_file='configs/naic/mfn.yml' \
#     SOLVER.IMS_PER_BATCH "128" DATALOADER.NUM_INSTANCE "8" \
#     MODEL.NAME "mfn" MODEL.BACKBONE "('se_resnext50_32x4d')" MODEL.MFN.POOL_TYPE "gem_3" MODEL.MFN.AUX_POOL_TYPE "gem_3" MODEL.MFN.AUX_SMOOTH "False"\
#     DATASETS.DATA_PATH "('${DATA_DIR}')" DATASETS.TRAIN_PATH "rep_f0_train2"\
#     MODEL.PRETRAIN_PATH "('${PRETRAIN}')"  \
#     OUTPUT_DIR "('${SAVE_DIR}')" 

# PRETRAIN=../weights/r50_ibn_a.pth
# DATA_DIR='/data/Dataset/PReID/all_dataset/'
# SAVE_DIR='../rep_work_dirs/exp4-mfn-resnet50ibnls1-288x144-bs16x8-warmup10-flip-pad10-meanstd-erase0502-lbsm-tpl05-nosm_allgem_3-minInst2/' #(h, w)
# CUDA_VISIBLE_DEVICES=5 python train.py --config_file='configs/naic/mfn.yml' \
#     SOLVER.IMS_PER_BATCH "128" DATALOADER.NUM_INSTANCE "8" \
#     INPUT.SIZE_TRAIN "([288,144])" INPUT.SIZE_TEST "([288,144])" \
#     MODEL.NAME "mfn" MODEL.BACKBONE "('resnet50_ibn_a')" MODEL.MFN.POOL_TYPE "gem_3" MODEL.MFN.AUX_POOL_TYPE "gem_3" MODEL.MFN.AUX_SMOOTH "False"\
#     DATASETS.DATA_PATH "('${DATA_DIR}')" DATASETS.TRAIN_PATH "rep_f0_train2"\
#     MODEL.PRETRAIN_PATH "('${PRETRAIN}')"  \
#     OUTPUT_DIR "('${SAVE_DIR}')" 

# PRETRAIN=../weights/resnext101_ibn_a.pth.tar
# DATA_DIR='/data/Dataset/PReID/all_dataset/'
# SAVE_DIR='../rep_work_dirs/exp4-mfn-resnext101ibnls1-256x128-bs16x8-warmup10-flip-pad10-meanstd-erase0502-lbsm-tpl05-nosm_allgem_3-minInst2/' #(h, w)
# CUDA_VISIBLE_DEVICES=7 python train.py --config_file='configs/naic/mfn.yml' \
#     SOLVER.IMS_PER_BATCH "128" DATALOADER.NUM_INSTANCE "8" \
#     MODEL.NAME "mfn" MODEL.BACKBONE "('resnext101_ibn_a')" MODEL.MFN.POOL_TYPE "gem_3" MODEL.MFN.AUX_POOL_TYPE "gem_3" MODEL.MFN.AUX_SMOOTH "False"\
#     DATASETS.DATA_PATH "('${DATA_DIR}')" DATASETS.TRAIN_PATH "rep_f0_train2"\
#     MODEL.PRETRAIN_PATH "('${PRETRAIN}')"  \
#     OUTPUT_DIR "('${SAVE_DIR}')" 


# PRETRAIN=../weights/se_resnext50_32x4d-a260b3a4.pth
# DATA_DIR='/data/Dataset/PReID/all_dataset/'
# SAVE_DIR='../rep_work_dirs/exp4-mfn-se_resnext50ls1-288x144-bs16x8-warmup10-flip-pad10-meanstd-erase0502-lbsm-tpl05-nosm_allgem_3-minInst2/' #(h, w)
# CUDA_VISIBLE_DEVICES=4 python train.py --config_file='configs/naic/mfn.yml' \
#     SOLVER.IMS_PER_BATCH "128" DATALOADER.NUM_INSTANCE "8" \
#     INPUT.SIZE_TRAIN "([288,144])" INPUT.SIZE_TEST "([288,144])" \
#     MODEL.NAME "mfn" MODEL.BACKBONE "('se_resnext50_32x4d')" MODEL.MFN.POOL_TYPE "gem_3" MODEL.MFN.AUX_POOL_TYPE "gem_3" MODEL.MFN.AUX_SMOOTH "False"\
#     DATASETS.DATA_PATH "('${DATA_DIR}')" DATASETS.TRAIN_PATH "rep_f0_train2"\
#     MODEL.PRETRAIN_PATH "('${PRETRAIN}')"  \
#     OUTPUT_DIR "('${SAVE_DIR}')" 

# PRETRAIN=../weights/r50_ibn_a.pth
# DATA_DIR='/share/Dataset/PReID/pre/'
# SAVE_DIR='../work_dirs/debug/'
# CUDA_VISIBLE_DEVICES=6,7 python train.py --config_file='configs/preid/mfn.yml' \
#     SOLVER.TENSORBOARD.USE "True" \
#     TEST.RANDOMPERM "5" SOLVER.EVAL_PERIOD "1" \
#     SOLVER.IMS_PER_BATCH "128" DATALOADER.NUM_INSTANCE "8" \
#     MODEL.NAME "mfn" MODEL.BACKBONE "('resnet50_ibn_a')" \
#     DATASETS.DATA_PATH "('${DATA_DIR}')" DATASETS.TRAIN_PATH "train"\
#     MODEL.PRETRAIN_PATH "('${PRETRAIN}')"  \
#     OUTPUT_DIR "('${SAVE_DIR}')" 

# [1214]

# PRETRAIN=../weights/r50_ibn_a.pth
# DATA_DIR='/data/Dataset/PReID/rep_dataset/'
# SAVE_DIR='../rep_work_dirs/exp4-mfn-resnet50ibnls1-288x144-bs16x8-warmup10-flip-pad10-meanstd-erase0502-lbsm-tpl05-nosm_allgem_3-hl10-minInst2/' #(h, w)
# # SAVE_DIR='../rep_work_dirs/debug/' #(h, w)

# CUDA_VISIBLE_DEVICES=5 python train.py --config_file='configs/naic/mfn.yml' \
#     SOLVER.IMS_PER_BATCH "128" DATALOADER.NUM_INSTANCE "8" \
#     INPUT.SIZE_TRAIN "([288,144])" INPUT.SIZE_TEST "([288,144])" DATASETS.HIST_LABEL.USE "True" DATASETS.HIST_LABEL.LOSS_WEIGHT "1.0"\
#     MODEL.NAME "mfn" MODEL.BACKBONE "('resnet50_ibn_a')" MODEL.MFN.POOL_TYPE "gem_3" MODEL.MFN.AUX_POOL_TYPE "gem_3" MODEL.MFN.AUX_SMOOTH "False"\
#     DATASETS.DATA_PATH "('${DATA_DIR}')" DATASETS.TRAIN_PATH "rep_f0_train2"\
#     MODEL.PRETRAIN_PATH "('${PRETRAIN}')"  \
#     OUTPUT_DIR "('${SAVE_DIR}')" 


# PRETRAIN=../weights/r50_ibn_a.pth
# DATA_DIR='/data/Dataset/PReID/rep_dataset/'
# SAVE_DIR='../rep_work_dirs/exp4-mfn-resnet50ibnls1-288x144-bs16x8-warmup10-flip-pad10-meanstd-erase0502-lbsm-tpl05-nosm_allgem_3-extra_tpl-minInst2/' #(h, w)
# CUDA_VISIBLE_DEVICES=4 python train.py --config_file='configs/naic/mfn.yml' \
#     SOLVER.IMS_PER_BATCH "128" DATALOADER.NUM_INSTANCE "8" \
#     INPUT.SIZE_TRAIN "([288,144])" INPUT.SIZE_TEST "([288,144])" \
#     MODEL.NAME "mfn" MODEL.BACKBONE "('resnet50_ibn_a')" MODEL.MFN.POOL_TYPE "gem_3" MODEL.MFN.AUX_POOL_TYPE "gem_3" MODEL.MFN.AUX_SMOOTH "False" MODEL.MFN.USE_EXTRA_TRIPLET "True"\
#     DATASETS.DATA_PATH "('${DATA_DIR}')" DATASETS.TRAIN_PATH "rep_f0_train2"\
#     MODEL.PRETRAIN_PATH "('${PRETRAIN}')"  \
#     OUTPUT_DIR "('${SAVE_DIR}')" 

# PRETRAIN=../weights/r50_ibn_a.pth
# DATA_DIR='/data/Dataset/PReID/rep_dataset/'
# SAVE_DIR='../rep_work_dirs/exp4-mfn-resnet50ibnls1-288x144-bs16x8-warmup10-flip-pad10-meanstd-erase0502-lbsm-tpl05-nosm_allgem_3-trainVal2/' #(h, w)
# CUDA_VISIBLE_DEVICES=7 python train.py --config_file='configs/naic/mfn.yml' \
#     SOLVER.IMS_PER_BATCH "128" DATALOADER.NUM_INSTANCE "8" \
#     INPUT.SIZE_TRAIN "([288,144])" INPUT.SIZE_TEST "([288,144])" \
#     MODEL.NAME "mfn" MODEL.BACKBONE "('resnet50_ibn_a')" MODEL.MFN.POOL_TYPE "gem_3" MODEL.MFN.AUX_POOL_TYPE "gem_3" MODEL.MFN.AUX_SMOOTH "False"\
#     DATASETS.DATA_PATH "('${DATA_DIR}')" DATASETS.TRAIN_PATH "rep_trainVal2"\
#     MODEL.PRETRAIN_PATH "('${PRETRAIN}')"  \
#     OUTPUT_DIR "('${SAVE_DIR}')" 

# PRETRAIN=../weights/r50_ibn_a.pth
# DATA_DIR='/data/Dataset/PReID/rep_dataset/'
# SAVE_DIR='../rep_work_dirs/exp4-mfn-resnet50ibnls1-384x144-bs16x8-warmup10-flip-pad10-meanstd-erase0502-lbsm-tpl05-nosm_allgem_3-minInst2/' #(h, w)
# CUDA_VISIBLE_DEVICES=7 python train.py --config_file='configs/naic/mfn.yml' \
#     SOLVER.IMS_PER_BATCH "128" DATALOADER.NUM_INSTANCE "8" \
#     INPUT.SIZE_TRAIN "([384,144])" INPUT.SIZE_TEST "([384,144])" \
#     MODEL.NAME "mfn" MODEL.BACKBONE "('resnet50_ibn_a')" MODEL.MFN.POOL_TYPE "gem_3" MODEL.MFN.AUX_POOL_TYPE "gem_3" MODEL.MFN.AUX_SMOOTH "False"\
#     DATASETS.DATA_PATH "('${DATA_DIR}')" DATASETS.TRAIN_PATH "rep_f0_train2"\
#     MODEL.PRETRAIN_PATH "('${PRETRAIN}')"  \
#     OUTPUT_DIR "('${SAVE_DIR}')" 

# PRETRAIN=../weights/r50_ibn_a.pth
# DATA_DIR='/data/Dataset/PReID/rep_dataset/'
# SAVE_DIR='../rep_work_dirs/exp4-mfn-resnet50ibnls1-384x192-bs16x8-warmup10-flip-pad10-meanstd-erase0502-lbsm-tpl05-nosm_allgem_3-minInst2/' #(h, w)
# CUDA_VISIBLE_DEVICES=5 python train.py --config_file='configs/naic/mfn.yml' \
#     SOLVER.IMS_PER_BATCH "128" DATALOADER.NUM_INSTANCE "8" \
#     INPUT.SIZE_TRAIN "([384,192])" INPUT.SIZE_TEST "([384,192])" \
#     MODEL.NAME "mfn" MODEL.BACKBONE "('resnet50_ibn_a')" MODEL.MFN.POOL_TYPE "gem_3" MODEL.MFN.AUX_POOL_TYPE "gem_3" MODEL.MFN.AUX_SMOOTH "False"\
#     DATASETS.DATA_PATH "('${DATA_DIR}')" DATASETS.TRAIN_PATH "rep_f0_train2"\
#     MODEL.PRETRAIN_PATH "('${PRETRAIN}')"  \
#     OUTPUT_DIR "('${SAVE_DIR}')" 

# PRETRAIN=../weights/resnet101_ibn_a.pth.tar
# DATA_DIR='/data/Dataset/PReID/rep_dataset/'
# SAVE_DIR='../rep_work_dirs/exp4-mfn-resnet101ibnls1-288x144-bs16x8-warmup10-flip-pad10-meanstd-erase0502-lbsm-tpl05-nosm_allavg-cj-minInst2/' #(h, w)
# CUDA_VISIBLE_DEVICES=5 python train.py --config_file='configs/naic/mfn.yml' \
#     SOLVER.IMS_PER_BATCH "128" DATALOADER.NUM_INSTANCE "8" SOLVER.START_SAVE_EPOCH "70" \
#     INPUT.USE_CJ "True" INPUT.CJ_PROB "0.5" INPUT.CJ_BRIGHNESS "0.5" INPUT.CJ_CONTRAST "0.5" INPUT.CJ_SATURATION "0.0" INPUT.CJ_HUE "0.5" \
#     INPUT.SIZE_TRAIN "([288,144])" INPUT.SIZE_TEST "([288,144])" \
#     MODEL.NAME "mfn" MODEL.BACKBONE "('resnet101_ibn_a')" MODEL.MFN.POOL_TYPE "avg" MODEL.MFN.AUX_POOL_TYPE "avg" MODEL.MFN.AUX_SMOOTH "False"\
#     DATASETS.DATA_PATH "('${DATA_DIR}')" DATASETS.TRAIN_PATH "rep_f0_train2"\
#     MODEL.PRETRAIN_PATH "('${PRETRAIN}')"  \
#     OUTPUT_DIR "('${SAVE_DIR}')" 

# PRETRAIN=../weights/resnet101_ibn_a.pth.tar
# DATA_DIR='/data/Dataset/PReID/rep_dataset/'
# SAVE_DIR='../rep_work_dirs/exp4-mfn-resnet101ibnls1_sestn-288x144-bs16x8-warmup10-flip-pad10-meanstd-erase0502-lbsm-tpl05-nosm_allavg-cj-minInst2/' #(h, w)
# CUDA_VISIBLE_DEVICES=6 python train.py --config_file='configs/naic/mfn.yml' \
#     SOLVER.IMS_PER_BATCH "128" DATALOADER.NUM_INSTANCE "8" SOLVER.START_SAVE_EPOCH "70" \
#     INPUT.USE_CJ "True" INPUT.CJ_PROB "0.5" INPUT.CJ_BRIGHNESS "0.5" INPUT.CJ_CONTRAST "0.5" INPUT.CJ_SATURATION "0.0" INPUT.CJ_HUE "0.5" \
#     INPUT.SIZE_TRAIN "([288,144])" INPUT.SIZE_TEST "([288,144])" \
#     MODEL.NAME "mfn" MODEL.BACKBONE "('resnet101_ibn_a')" MODEL.MFN.POOL_TYPE "avg" MODEL.MFN.AUX_POOL_TYPE "avg" MODEL.MFN.AUX_SMOOTH "False" MODEL.MFN.USE_SESTN "True"\
#     DATASETS.DATA_PATH "('${DATA_DIR}')" DATASETS.TRAIN_PATH "rep_f0_train2"\
#     MODEL.PRETRAIN_PATH "('${PRETRAIN}')"  \
#     OUTPUT_DIR "('${SAVE_DIR}')" 

# PRETRAIN=../weights/resnet101_ibn_a.pth.tar
# DATA_DIR='/data/Dataset/PReID/rep_dataset/'
# SAVE_DIR='../rep_work_dirs/exp4-mfn-resnet101ibnls1_sestn-384x192-bs16x8-warmup10-flip-pad10-meanstd-erase0502-lbsm-tpl05-nosm_allavg-cj05-minInst2/' #(h, w)
# CUDA_VISIBLE_DEVICES=3 python train.py --config_file='configs/naic/mfn.yml' \
#     SOLVER.IMS_PER_BATCH "128" DATALOADER.NUM_INSTANCE "8" SOLVER.START_SAVE_EPOCH "70" SOLVER.MAX_EPOCHS "90" \
#     INPUT.USE_CJ "True" INPUT.CJ_PROB "0.5" INPUT.CJ_BRIGHNESS "0.5" INPUT.CJ_CONTRAST "0.5" INPUT.CJ_SATURATION "0.0" INPUT.CJ_HUE "0.5" \
#     INPUT.SIZE_TRAIN "([384,192])" INPUT.SIZE_TEST "([384,192])" \
#     MODEL.NAME "mfn" MODEL.BACKBONE "('resnet101_ibn_a')" MODEL.MFN.POOL_TYPE "avg" MODEL.MFN.AUX_POOL_TYPE "avg" MODEL.MFN.AUX_SMOOTH "False" MODEL.MFN.USE_SESTN "True"\
#     DATASETS.DATA_PATH "('${DATA_DIR}')" DATASETS.TRAIN_PATH "rep_f0_train2"\
#     MODEL.PRETRAIN_PATH "('${PRETRAIN}')"  \
#     OUTPUT_DIR "('${SAVE_DIR}')" 