#!/usr/bin/env bash

# PRETRAIN=../weights/r50_ibn_a.pth
# DATA_DIR='/share/Dataset/PReID/all_dataset/'
# SAVE_DIR='../rep_work_dirs/exp4-baseline-resnet50ibnls1-256x128-bs16x4-warmup10-flip-pad10-meanstd-erase0502-lbsm-gem_3/' #(h, w)
# CUDA_VISIBLE_DEVICES=5 python train.py --config_file='configs/naic/baseline.yml' \
#     MODEL.NAME "baseline" MODEL.BACKBONE "('resnet50_ibn_a')" MODEL.BASELINE.POOL_TYPE "gem_3"\
#     DATASETS.DATA_PATH "('${DATA_DIR}')" DATASETS.TRAIN_PATH "rep_f0_train"\
#     MODEL.PRETRAIN_PATH "('${PRETRAIN}')"  \
#     OUTPUT_DIR "('${SAVE_DIR}')" 

# PRETRAIN=../weights/r50_ibn_a.pth
# DATA_DIR='/share/Dataset/PReID/all_dataset/'
# SAVE_DIR='../rep_work_dirs/exp4-baseline-resnet50ibnls1-256x128-bs16x4-warmup10-flip-pad10-meanstd-erase0502-lbsm-gem_3-minInst2/' #(h, w)
# CUDA_VISIBLE_DEVICES=4 python train.py --config_file='configs/naic/baseline.yml' \
#     MODEL.NAME "baseline" MODEL.BACKBONE "('resnet50_ibn_a')" MODEL.BASELINE.POOL_TYPE "gem_3"\
#     DATASETS.DATA_PATH "('${DATA_DIR}')" DATASETS.TRAIN_PATH "rep_f0_train2"\
#     MODEL.PRETRAIN_PATH "('${PRETRAIN}')"  \
#     OUTPUT_DIR "('${SAVE_DIR}')" 

# PRETRAIN=../weights/r50_ibn_a.pth
# DATA_DIR='/data/Dataset/PReID/all_dataset/'
# SAVE_DIR='../rep_work_dirs/exp4-baseline-resnet50ibnls1-256x128-bs16x4-warmup10-flip-pad10-meanstd-erase0502-lbsm-gem-minInst2/' #(h, w)
# CUDA_VISIBLE_DEVICES=6 python train.py --config_file='configs/naic/baseline.yml' \
#     MODEL.NAME "baseline" MODEL.BACKBONE "('resnet50_ibn_a')" MODEL.BASELINE.POOL_TYPE "gem"\
#     DATASETS.DATA_PATH "('${DATA_DIR}')" DATASETS.TRAIN_PATH "rep_f0_train2"\
#     MODEL.PRETRAIN_PATH "('${PRETRAIN}')"  \
#     OUTPUT_DIR "('${SAVE_DIR}')" 

# [20191212]
# PRETRAIN=../weights/resnext101_ibn_a.pth.tar
# DATA_DIR='/data/Dataset/PReID/rep_dataset/'
# SAVE_DIR='../rep_work_dirs/exp4-baseline-resnext101ibnls1-256x128-bs16x4-warmup10-flip-pad10-meanstd-erase0502-lbsm-gem_3-minInst2/' #(h, w)
# CUDA_VISIBLE_DEVICES=5 python train.py --config_file='configs/naic/baseline.yml' \
#     MODEL.NAME "baseline" MODEL.BACKBONE "('resnext101_ibn_a')" MODEL.BASELINE.POOL_TYPE "gem_3"\
#     DATASETS.DATA_PATH "('${DATA_DIR}')" DATASETS.TRAIN_PATH "rep_f0_train2"\
#     MODEL.PRETRAIN_PATH "('${PRETRAIN}')"  \
#     OUTPUT_DIR "('${SAVE_DIR}')" 


# PRETRAIN=../weights/resnext101_ibn_a.pth.tar
# DATA_DIR='/data/Dataset/PReID/rep_dataset/'
# SAVE_DIR='../rep_work_dirs/exp4-baseline-resnext101ibnls1-288x144-bs16x4-warmup10-flip-pad10-meanstd-erase0502-lbsm-gem_3-minInst2/' #(h, w)
# CUDA_VISIBLE_DEVICES=5 python train.py --config_file='configs/naic/baseline.yml' \
#     INPUT.SIZE_TRAIN "([288,144])" INPUT.SIZE_TEST "([288,144])" \
#     MODEL.NAME "baseline" MODEL.BACKBONE "('resnext101_ibn_a')" MODEL.BASELINE.POOL_TYPE "gem_3"\
#     DATASETS.DATA_PATH "('${DATA_DIR}')" DATASETS.TRAIN_PATH "rep_f0_train2"\
#     MODEL.PRETRAIN_PATH "('${PRETRAIN}')"  \
#     OUTPUT_DIR "('${SAVE_DIR}')" 

# [1216]

# PRETRAIN=../weights/r50_ibn_a.pth
# DATA_DIR='/share/Dataset/PReID/all_dataset/'
# SAVE_DIR='../rep_work_dirs/exp4-baseline-resnet50ibnls1-256x128-bs16x4-warmup10-flip-pad10-meanstd-erase0502-lbsm-gem_3-minInst2/' #(h, w)
# CUDA_VISIBLE_DEVICES=4 python train.py --config_file='configs/naic/baseline.yml' \
#     MODEL.NAME "baseline" MODEL.BACKBONE "('resnet50_ibn_a')" MODEL.BASELINE.POOL_TYPE "gem_3"\
#     DATASETS.DATA_PATH "('${DATA_DIR}')" DATASETS.TRAIN_PATH "rep_f0_train2"\
#     MODEL.PRETRAIN_PATH "('${PRETRAIN}')"  \
#     OUTPUT_DIR "('${SAVE_DIR}')" 

# PRETRAIN=../weights/r50_ibn_a.pth
# DATA_DIR='/data/Dataset/PReID/rep_dataset/'
# SAVE_DIR='../rep_work_dirs/exp4-baseline-resnet50ibnls1-384x192-bs16x4-warmup10-flip-pad10-meanstd-erase0502-lbsm-gem-minInst2/' #(h, w)
# CUDA_VISIBLE_DEVICES=6 python train.py --config_file='configs/naic/baseline.yml' \
#     INPUT.SIZE_TRAIN "([384,192])" INPUT.SIZE_TEST "([384,192])" \
#     MODEL.NAME "baseline" MODEL.BACKBONE "('resnet50_ibn_a')" MODEL.BASELINE.POOL_TYPE "gem"\
#     DATASETS.DATA_PATH "('${DATA_DIR}')" DATASETS.TRAIN_PATH "rep_f0_train2"\
#     MODEL.PRETRAIN_PATH "('${PRETRAIN}')"  \
#     OUTPUT_DIR "('${SAVE_DIR}')" 

# PRETRAIN=../weights/r50_ibn_a.pth
# DATA_DIR='/data/Dataset/PReID/rep_dataset/'
# SAVE_DIR='../rep_work_dirs/exp4-cosinebaseline-resnet50ibnls1-384x192-bs16x4-warmup10-flip-pad10-meanstd-erase0502-lbsm-gem-arface30_05_0901-minInst2/' #(h, w)
# CUDA_VISIBLE_DEVICES=4 python train.py --config_file='configs/naic/arcface_baseline.yml' \
#     INPUT.SIZE_TRAIN "([384,192])" INPUT.SIZE_TEST "([384,192])" \
#     MODEL.NAME "cosine_baseline" MODEL.BACKBONE "('resnet50_ibn_a')" MODEL.BASELINE.POOL_TYPE "gem"\
#     DATASETS.DATA_PATH "('${DATA_DIR}')" DATASETS.TRAIN_PATH "rep_f0_train2"\
#     MODEL.PRETRAIN_PATH "('${PRETRAIN}')"  \
#     OUTPUT_DIR "('${SAVE_DIR}')" 

# PRETRAIN=../weights/r50_ibn_a.pth
# DATA_DIR='/data/Dataset/PReID/rep_dataset/'
# SAVE_DIR='../rep_work_dirs/exp4-cosinebaseline-resnet50ibnls1-384x192-bs16x4-warmup10-flip-pad10-meanstd-erase0502-nolbsm-gem-arccos30_05_0901-minInst2/' #(h, w)
# CUDA_VISIBLE_DEVICES=4 python train.py --config_file='configs/naic/arcface_baseline.yml' \
#     MODEL.LABEL_SMOOTH "False" MODEL.BASELINE.COSINE_LOSS_TYPE 'ArcCos' \
#     INPUT.SIZE_TRAIN "([384,192])" INPUT.SIZE_TEST "([384,192])" \
#     MODEL.NAME "cosine_baseline" MODEL.BACKBONE "('resnet50_ibn_a')" MODEL.BASELINE.POOL_TYPE "gem"\
#     DATASETS.DATA_PATH "('${DATA_DIR}')" DATASETS.TRAIN_PATH "rep_f0_train2"\
#     MODEL.PRETRAIN_PATH "('${PRETRAIN}')"  \
#     OUTPUT_DIR "('${SAVE_DIR}')" 

# PRETRAIN=../weights/r50_ibn_a.pth
# DATA_DIR='/data/Dataset/PReID/rep_dataset/'
# SAVE_DIR='../rep_work_dirs/exp4-cosinebaseline-resnet50ibnls1-384x192-bs16x4-warmup10-flip-pad10-meanstd-erase0502-nolbsm-gem-arccos30_05_10033-minInst2/' #(h, w)
# CUDA_VISIBLE_DEVICES=3 python train.py --config_file='configs/naic/arcface_baseline.yml' \
#     MODEL.BASELINE.TPL_WEIGHT "1.0" MODEL.BASELINE.CE_WEIGHT "0.33" \
#     MODEL.LABEL_SMOOTH "False" MODEL.BASELINE.COSINE_LOSS_TYPE 'ArcCos' \
#     INPUT.SIZE_TRAIN "([384,192])" INPUT.SIZE_TEST "([384,192])" \
#     MODEL.NAME "cosine_baseline" MODEL.BACKBONE "('resnet50_ibn_a')" MODEL.BASELINE.POOL_TYPE "gem"\
#     DATASETS.DATA_PATH "('${DATA_DIR}')" DATASETS.TRAIN_PATH "rep_f0_train2"\
#     MODEL.PRETRAIN_PATH "('${PRETRAIN}')"  \
#     OUTPUT_DIR "('${SAVE_DIR}')" 

# PRETRAIN=../weights/r50_ibn_a.pth
# DATA_DIR='/data/Dataset/PReID/rep_dataset/'
# SAVE_DIR='../rep_work_dirs/exp4-cosinebaseline-resnet50ibnls1-384x192-bs16x4-warmup10-flip-pad10-meanstd-erase0502-nolbsm-gem-cosface30_035_10033-minInst2/' #(h, w)
# CUDA_VISIBLE_DEVICES=3 python train.py --config_file='configs/naic/arcface_baseline.yml' \
#     MODEL.BASELINE.TPL_WEIGHT "1.0" MODEL.BASELINE.CE_WEIGHT "0.33" MODEL.BASELINE.S "30.0" MODEL.BASELINE.M "0.35"\
#     MODEL.LABEL_SMOOTH "False" MODEL.BASELINE.COSINE_LOSS_TYPE 'CosFace' \
#     INPUT.SIZE_TRAIN "([384,192])" INPUT.SIZE_TEST "([384,192])" \
#     MODEL.NAME "cosine_baseline" MODEL.BACKBONE "('resnet50_ibn_a')" MODEL.BASELINE.POOL_TYPE "gem"\
#     DATASETS.DATA_PATH "('${DATA_DIR}')" DATASETS.TRAIN_PATH "rep_f0_train2"\
#     MODEL.PRETRAIN_PATH "('${PRETRAIN}')"  \
#     OUTPUT_DIR "('${SAVE_DIR}')" 


# PRETRAIN=../weights/r50_ibn_a.pth
# DATA_DIR='/data/Dataset/PReID/rep_dataset/'
# SAVE_DIR='../rep_work_dirs/exp4-cosinebaseline-resnet50ibnls1-384x192-bs16x4-warmup10-flip-pad10-meanstd-erase0502-nolbsm-gem-cosface60_035_10033-minInst2/' #(h, w)
# CUDA_VISIBLE_DEVICES=4 python train.py --config_file='configs/naic/arcface_baseline.yml' \
#     MODEL.BASELINE.TPL_WEIGHT "1.0" MODEL.BASELINE.CE_WEIGHT "0.33" MODEL.BASELINE.S "60.0" MODEL.BASELINE.M "0.35"\
#     MODEL.LABEL_SMOOTH "False" MODEL.BASELINE.COSINE_LOSS_TYPE 'CosFace' \
#     INPUT.SIZE_TRAIN "([384,192])" INPUT.SIZE_TEST "([384,192])" \
#     MODEL.NAME "cosine_baseline" MODEL.BACKBONE "('resnet50_ibn_a')" MODEL.BASELINE.POOL_TYPE "gem"\
#     DATASETS.DATA_PATH "('${DATA_DIR}')" DATASETS.TRAIN_PATH "rep_f0_train2"\
#     MODEL.PRETRAIN_PATH "('${PRETRAIN}')"  \
#     OUTPUT_DIR "('${SAVE_DIR}')" 


# PRETRAIN=../weights/r50_ibn_a.pth
# DATA_DIR='/data/Dataset/PReID/rep_dataset/'
# SAVE_DIR='../rep_work_dirs/exp4-cosinebaseline-resnet50ibnls1-384x192-bs16x4-warmup10-flip-pad10-meanstd-erase0502-nolbsm-gem-cosface60_035_10033-center000505-minInst2/' #(h, w)
# CUDA_VISIBLE_DEVICES=2 python train.py --config_file='configs/naic/cosface_center.yml' \
#     MODEL.BASELINE.TPL_WEIGHT "1.0" MODEL.BASELINE.CE_WEIGHT "0.333" MODEL.BASELINE.COSINE_LOSS_TYPE 'CosFace' MODEL.BASELINE.S "60.0" MODEL.BASELINE.M "0.35" \
#     INPUT.SIZE_TRAIN "([384,192])" INPUT.SIZE_TEST "([384,192])" \
#     MODEL.NAME "cosine_baseline" MODEL.BACKBONE "('resnet50_ibn_a')" MODEL.BASELINE.POOL_TYPE "gem"\
#     DATASETS.DATA_PATH "('${DATA_DIR}')" DATASETS.TRAIN_PATH "rep_f0_train2"\
#     MODEL.PRETRAIN_PATH "('${PRETRAIN}')"  \
#     OUTPUT_DIR "('${SAVE_DIR}')" 

# PRETRAIN=../weights/r50_ibn_a.pth
# DATA_DIR='/data/Dataset/PReID/rep_dataset/'
# SAVE_DIR='../rep_work_dirs/exp4-cosinebaseline-resnet50ibnls1-384x192-bs16x4-warmup10-flip-pad10-meanstd-erase0502-nolbsm-gem-arface60_05_0901-minInst2/' #(h, w)
# CUDA_VISIBLE_DEVICES=4 python train.py --config_file='configs/naic/arcface_baseline.yml' \
#     MODEL.LABEL_SMOOTH "False" MODEL.BASELINE.S "60.0" \
#     INPUT.SIZE_TRAIN "([384,192])" INPUT.SIZE_TEST "([384,192])" \
#     MODEL.NAME "cosine_baseline" MODEL.BACKBONE "('resnet50_ibn_a')" MODEL.BASELINE.POOL_TYPE "gem"\
#     DATASETS.DATA_PATH "('${DATA_DIR}')" DATASETS.TRAIN_PATH "rep_f0_train2"\
#     MODEL.PRETRAIN_PATH "('${PRETRAIN}')"  \
#     OUTPUT_DIR "('${SAVE_DIR}')" 

# PRETRAIN=../weights/r50_ibn_a.pth
# DATA_DIR='/data/Dataset/PReID/rep_dataset/'
# SAVE_DIR='../rep_work_dirs/exp4-cosinebaseline-resnet50ibnls1-384x192-bs16x4-warmup10-flip-pad10-meanstd-erase0502-lbsm-gem-arface30_05_1010-minInst2/' #(h, w)
# CUDA_VISIBLE_DEVICES=5 python train.py --config_file='configs/naic/arcface_baseline.yml' \
#     MODEL.BASELINE.TPL_WEIGHT "1.0" MODEL.BASELINE.CE_WEIGHT "1.0" \
#     INPUT.SIZE_TRAIN "([384,192])" INPUT.SIZE_TEST "([384,192])" \
#     MODEL.NAME "cosine_baseline" MODEL.BACKBONE "('resnet50_ibn_a')" MODEL.BASELINE.POOL_TYPE "gem"\
#     DATASETS.DATA_PATH "('${DATA_DIR}')" DATASETS.TRAIN_PATH "rep_f0_train2"\
#     MODEL.PRETRAIN_PATH "('${PRETRAIN}')"  \
#     OUTPUT_DIR "('${SAVE_DIR}')" 

# PRETRAIN=../weights/r50_ibn_a.pth
# DATA_DIR='/data/Dataset/PReID/rep_dataset/'
# SAVE_DIR='../rep_work_dirs/exp4-cosinebaseline-resnet50ibnls1-384x192-bs16x4-warmup10-flip-pad10-meanstd-erase0502-nolbsm-gem-cosface30_035_10033-use_bnbias-minInst2/' #(h, w)
# CUDA_VISIBLE_DEVICES=0 python train.py --config_file='configs/naic/arcface_baseline.yml' \
#     MODEL.BASELINE.TPL_WEIGHT "1.0" MODEL.BASELINE.CE_WEIGHT "0.33" MODEL.BASELINE.S "30.0" MODEL.BASELINE.M "0.35" MODEL.BASELINE.USE_BNBAIS "True"\
#     MODEL.LABEL_SMOOTH "False" MODEL.BASELINE.COSINE_LOSS_TYPE 'CosFace' \
#     INPUT.SIZE_TRAIN "([384,192])" INPUT.SIZE_TEST "([384,192])" \
#     MODEL.NAME "cosine_baseline" MODEL.BACKBONE "('resnet50_ibn_a')" MODEL.BASELINE.POOL_TYPE "gem"\
#     DATASETS.DATA_PATH "('${DATA_DIR}')" DATASETS.TRAIN_PATH "rep_f0_train2"\
#     MODEL.PRETRAIN_PATH "('${PRETRAIN}')"  \
#     OUTPUT_DIR "('${SAVE_DIR}')" 

# PRETRAIN=../weights/r50_ibn_a.pth
# DATA_DIR='/data/Dataset/PReID/rep_dataset/'
# SAVE_DIR='../rep_work_dirs/exp4-cosinebaseline-resnet50ibnls1-384x192-bs16x6-warmup10-flip-pad10-meanstd-erase0502-nolbsm-gem-cosface30_035_10033-use_bnbias-minInst2/' #(h, w)
# CUDA_VISIBLE_DEVICES=4 python train.py --config_file='configs/naic/arcface_baseline.yml' \
#     SOLVER.IMS_PER_BATCH "96" DATALOADER.NUM_INSTANCE "6" \
#     MODEL.BASELINE.TPL_WEIGHT "1.0" MODEL.BASELINE.CE_WEIGHT "0.33" MODEL.LABEL_SMOOTH "False" \
#     MODEL.BASELINE.S "30.0" MODEL.BASELINE.M "0.35" MODEL.BASELINE.USE_BNBAIS "True"\
#     MODEL.BASELINE.COSINE_LOSS_TYPE 'CosFace' \
#     INPUT.SIZE_TRAIN "([384,192])" INPUT.SIZE_TEST "([384,192])" \
#     MODEL.NAME "cosine_baseline" MODEL.BACKBONE "('resnet50_ibn_a')" MODEL.BASELINE.POOL_TYPE "gem"\
#     DATASETS.DATA_PATH "('${DATA_DIR}')" DATASETS.TRAIN_PATH "rep_f0_train2"\
#     MODEL.PRETRAIN_PATH "('${PRETRAIN}')"  \
#     OUTPUT_DIR "('${SAVE_DIR}')" 


# PRETRAIN=../weights/r50_ibn_a.pth
# DATA_DIR='/data/Dataset/PReID/rep_dataset/'
# SAVE_DIR='../rep_work_dirs/exp4-cosinebaseline-resnet50ibnls1-384x192-bs16x6-warmup10-flip-pad10-meanstd-erase0502-nolbsm-gem-cosface30_035_10033-use_bnbias-sepnorm-minInst2/' #(h, w)
# # SAVE_DIR='../rep_work_dirs/debug/' #(h, w)
# CUDA_VISIBLE_DEVICES=7 python train.py --config_file='configs/naic/arcface_baseline.yml' \
#     SOLVER.IMS_PER_BATCH "96" DATALOADER.NUM_INSTANCE "6" \
#     MODEL.BASELINE.TPL_WEIGHT "1.0" MODEL.BASELINE.CE_WEIGHT "0.33" \
#     MODEL.BASELINE.COSINE_LOSS_TYPE 'CosFace'  MODEL.BASELINE.S "30.0" MODEL.BASELINE.M "0.35" \
#     INPUT.SEPNORM.USE "True" \
#     INPUT.SEPNORM.PIXEL_MEAN "([[0.1164,0.1567,0.1796],[0.4556,0.5367,0.6493]])" \
#     INPUT.SEPNORM.PIXEL_STD "([[0.1467,0.1497,0.1930],[0.0750,0.0613,0.1040]])" \
#     INPUT.SIZE_TRAIN "([384,192])" INPUT.SIZE_TEST "([384,192])" \
#     MODEL.BASELINE.USE_BNBAIS "True" \
#     MODEL.NAME "cosine_baseline" MODEL.BACKBONE "('resnet50_ibn_a')" MODEL.BASELINE.POOL_TYPE "gem"\
#     DATASETS.DATA_PATH "('${DATA_DIR}')" DATASETS.TRAIN_PATH "rep_f0_train2"\
#     MODEL.PRETRAIN_PATH "('${PRETRAIN}')"  \
#     OUTPUT_DIR "('${SAVE_DIR}')" 



# PRETRAIN=../weights/r50_ibn_a.pth
# DATA_DIR='/data/Dataset/PReID/rep_dataset/'
# SAVE_DIR='../rep_work_dirs/exp4-cosinebaseline-resnet50ibnls1-384x192-bs16x6-warmup10-flip-pad10-meanstd-erase0502-nolbsm-gem-cosface30_035_10033-use_bnbias-minInst2-st2/' #(h, w)
# CUDA_VISIBLE_DEVICES=2 python train.py --config_file='configs/naic/arcface_baseline.yml' \
#     SOLVER.IMS_PER_BATCH "96" DATALOADER.NUM_INSTANCE "6" SOLVER.STEPS "[60, 90]" SOLVER.MAX_EPOCHS "120" \
#     MODEL.BASELINE.TPL_WEIGHT "1.0" MODEL.BASELINE.CE_WEIGHT "0.33" MODEL.LABEL_SMOOTH "False" \
#     MODEL.BASELINE.S "30.0" MODEL.BASELINE.M "0.35" MODEL.BASELINE.USE_BNBAIS "True"\
#     MODEL.BASELINE.COSINE_LOSS_TYPE 'CosFace' \
#     INPUT.SIZE_TRAIN "([384,192])" INPUT.SIZE_TEST "([384,192])" \
#     MODEL.NAME "cosine_baseline" MODEL.BACKBONE "('resnet50_ibn_a')" MODEL.BASELINE.POOL_TYPE "gem"\
#     DATASETS.DATA_PATH "('${DATA_DIR}')" DATASETS.TRAIN_PATH "rep_f0_train2"\
#     MODEL.PRETRAIN_PATH "('${PRETRAIN}')"  \
#     OUTPUT_DIR "('${SAVE_DIR}')" 



# PRETRAIN=../weights/r50_ibn_a.pth
# DATA_DIR='/data/Dataset/PReID/rep_dataset/'
# SAVE_DIR='../rep_work_dirs/exp4-cosinebaseline-resnet50ibnls1-384x192-bs16x6-warmup10-flip-pad10-meanstd-erase0502-nolbsm-gem-cosface30_035_10033-use_bnbias-trainVal-st2/' #(h, w)
# CUDA_VISIBLE_DEVICES=3 python train.py --config_file='configs/naic/arcface_baseline.yml' \
#     SOLVER.IMS_PER_BATCH "96" DATALOADER.NUM_INSTANCE "6" SOLVER.STEPS "[60, 90]" SOLVER.MAX_EPOCHS "120" \
#     MODEL.BASELINE.TPL_WEIGHT "1.0" MODEL.BASELINE.CE_WEIGHT "0.33" MODEL.LABEL_SMOOTH "False" \
#     MODEL.BASELINE.S "30.0" MODEL.BASELINE.M "0.35" MODEL.BASELINE.USE_BNBAIS "True"\
#     MODEL.BASELINE.COSINE_LOSS_TYPE 'CosFace' \
#     INPUT.SIZE_TRAIN "([384,192])" INPUT.SIZE_TEST "([384,192])" \
#     MODEL.NAME "cosine_baseline" MODEL.BACKBONE "('resnet50_ibn_a')" MODEL.BASELINE.POOL_TYPE "gem"\
#     DATASETS.DATA_PATH "('${DATA_DIR}')" DATASETS.TRAIN_PATH "rep_trainVal"\
#     MODEL.PRETRAIN_PATH "('${PRETRAIN}')"  \
#     OUTPUT_DIR "('${SAVE_DIR}')" 


# PRETRAIN=../weights/r50_ibn_a.pth
# DATA_DIR='/share/Dataset/PReID/rep_dataset/'
# SAVE_DIR='../rep_work_dirs/exp4-cosinebaseline-resnet50ibnls1-288x144-bs96-warmup10-flip-pad10-meanstd-erase0502-nolbsm-avg-cosface30_035-use_bnbias-minInst1-st2/' #(h, w)
# CUDA_VISIBLE_DEVICES=4 python train.py --config_file='configs/naic/arcface_baseline.yml' \
#     SOLVER.IMS_PER_BATCH "96" DATALOADER.SAMPLER 'softmax' MODEL.BASELINE.CE_WEIGHT "1.0" SOLVER.STEPS "[60, 90]" SOLVER.MAX_EPOCHS "140" SOLVER.START_SAVE_EPOCH "100"\
#     MODEL.LABEL_SMOOTH "False" \
#     MODEL.BASELINE.S "30.0" MODEL.BASELINE.M "0.35" MODEL.BASELINE.USE_BNBAIS "True"\
#     MODEL.BASELINE.COSINE_LOSS_TYPE 'CosFace' \
#     INPUT.SIZE_TRAIN "([288,144])" INPUT.SIZE_TEST "([288,144])" \
#     MODEL.NAME "cosine_baseline" MODEL.BACKBONE "('resnet50_ibn_a')" MODEL.BASELINE.POOL_TYPE "avg"\
#     DATASETS.DATA_PATH "('${DATA_DIR}')" DATASETS.TRAIN_PATH "rep_f0_train"\
#     MODEL.PRETRAIN_PATH "('${PRETRAIN}')"  \
#     OUTPUT_DIR "('${SAVE_DIR}')" 

# PRETRAIN=../weights/r50_ibn_a.pth
# DATA_DIR='/share/Dataset/PReID/rep_dataset/'
# SAVE_DIR='../rep_work_dirs/exp4-cosinebaseline-resnet50ibnls1-288x144-bs96-warmup10-flip-pad10-meanstd-erase0502-lbsm-avg-use_bnbias-minInst1-st2/' #(h, w)
# CUDA_VISIBLE_DEVICES=5 python train.py --config_file='configs/naic/arcface_baseline.yml' \
#     SOLVER.IMS_PER_BATCH "96" DATALOADER.SAMPLER 'softmax' MODEL.BASELINE.CE_WEIGHT "1.0" SOLVER.STEPS "[60, 90]" SOLVER.MAX_EPOCHS "140" SOLVER.START_SAVE_EPOCH "100"\
#     MODEL.BASELINE.USE_BNBAIS "True"\
#     INPUT.SIZE_TRAIN "([288,144])" INPUT.SIZE_TEST "([288,144])" \
#     MODEL.NAME "cosine_baseline" MODEL.BACKBONE "('resnet50_ibn_a')" MODEL.BASELINE.POOL_TYPE "avg"\
#     DATASETS.DATA_PATH "('${DATA_DIR}')" DATASETS.TRAIN_PATH "rep_f0_train"\
#     MODEL.PRETRAIN_PATH "('${PRETRAIN}')"  \
#     OUTPUT_DIR "('${SAVE_DIR}')" 

# PRETRAIN=../weights/r50_ibn_a.pth
# DATA_DIR='/share/Dataset/PReID/rep_dataset/'
# SAVE_DIR='../rep_work_dirs/exp4-cosinebaseline-resnet50ibnls1-288x144-bs96-warmup10-flip-pad10-meanstd-erase0502-lbsm-avg-cosface30_035-use_bnbias-minInst1-st2/' #(h, w)
# CUDA_VISIBLE_DEVICES=6 python train.py --config_file='configs/naic/arcface_baseline.yml' \
#     SOLVER.IMS_PER_BATCH "96" DATALOADER.SAMPLER 'softmax' MODEL.BASELINE.CE_WEIGHT "1.0" SOLVER.STEPS "[60, 90]" SOLVER.MAX_EPOCHS "140" SOLVER.START_SAVE_EPOCH "100"\
#     MODEL.BASELINE.S "30.0" MODEL.BASELINE.M "0.35" MODEL.BASELINE.USE_BNBAIS "True"\
#     MODEL.BASELINE.COSINE_LOSS_TYPE 'CosFace' \
#     INPUT.SIZE_TRAIN "([288,144])" INPUT.SIZE_TEST "([288,144])" \
#     MODEL.NAME "cosine_baseline" MODEL.BACKBONE "('resnet50_ibn_a')" MODEL.BASELINE.POOL_TYPE "avg"\
#     DATASETS.DATA_PATH "('${DATA_DIR}')" DATASETS.TRAIN_PATH "rep_f0_train"\
#     MODEL.PRETRAIN_PATH "('${PRETRAIN}')"  \
#     OUTPUT_DIR "('${SAVE_DIR}')" 

# PRETRAIN=../weights/r50_ibn_a.pth
# DATA_DIR='/share/Dataset/PReID/rep_dataset/'
# SAVE_DIR='../rep_work_dirs/exp4-cosinebaseline-resnet50ibnls1-288x144-bs96-warmup10-flip-pad10-meanstd-erase0502-lbsm-avg-cosface30_035-use_bnbias-sepnorm-minInst1-st2/' #(h, w)
# CUDA_VISIBLE_DEVICES=7 python train.py --config_file='configs/naic/arcface_baseline.yml' \
#     SOLVER.IMS_PER_BATCH "96" DATALOADER.SAMPLER 'softmax' MODEL.BASELINE.CE_WEIGHT "1.0" SOLVER.STEPS "[60, 90]" SOLVER.MAX_EPOCHS "140" SOLVER.START_SAVE_EPOCH "100" \
#     INPUT.SEPNORM.USE "True" \
#     INPUT.SEPNORM.PIXEL_MEAN "([[0.1164,0.1567,0.1796],[0.4556,0.5367,0.6493]])" \
#     INPUT.SEPNORM.PIXEL_STD "([[0.1467,0.1497,0.1930],[0.0750,0.0613,0.1040]])" \
#     MODEL.BASELINE.S "30.0" MODEL.BASELINE.M "0.35" MODEL.BASELINE.USE_BNBAIS "True"\
#     MODEL.BASELINE.COSINE_LOSS_TYPE 'CosFace' \
#     INPUT.SIZE_TRAIN "([288,144])" INPUT.SIZE_TEST "([288,144])" \
#     MODEL.NAME "cosine_baseline" MODEL.BACKBONE "('resnet50_ibn_a')" MODEL.BASELINE.POOL_TYPE "avg"\
#     DATASETS.DATA_PATH "('${DATA_DIR}')" DATASETS.TRAIN_PATH "rep_f0_train"\
#     MODEL.PRETRAIN_PATH "('${PRETRAIN}')"  \
#     OUTPUT_DIR "('${SAVE_DIR}')" 



# PRETRAIN=../weights/r50_ibn_a.pth
# DATA_DIR='/data/Dataset/PReID/rep_dataset/'
# SAVE_DIR='../rep_work_dirs/exp4-cosinebaseline-resnet50ibnls1-384x192-bs16x6-warmup10-flip-pad10-meanstd-erase0502-nolbsm-gem-cosface30_035_10033-use_bnbias-cj-minInst2/' #(h, w)
# CUDA_VISIBLE_DEVICES=0 python train.py --config_file='configs/naic/arcface_baseline.yml' \
#     SOLVER.IMS_PER_BATCH "96" DATALOADER.NUM_INSTANCE "6" \
#     INPUT.USE_CJ "True" INPUT.CJ_BRIGHNESS "0.5" INPUT.CJ_CONTRAST "0.5" INPUT.CJ_SATURATION "0.0" INPUT.CJ_HUE "0.5" \
#     MODEL.BASELINE.TPL_WEIGHT "1.0" MODEL.BASELINE.CE_WEIGHT "0.33" MODEL.LABEL_SMOOTH "False" \
#     MODEL.BASELINE.S "30.0" MODEL.BASELINE.M "0.35" MODEL.BASELINE.USE_BNBAIS "True"\
#     MODEL.BASELINE.COSINE_LOSS_TYPE 'CosFace' \
#     INPUT.SIZE_TRAIN "([384,192])" INPUT.SIZE_TEST "([384,192])" \
#     MODEL.NAME "cosine_baseline" MODEL.BACKBONE "('resnet50_ibn_a')" MODEL.BASELINE.POOL_TYPE "gem"\
#     DATASETS.DATA_PATH "('${DATA_DIR}')" DATASETS.TRAIN_PATH "rep_f0_train2"\
#     MODEL.PRETRAIN_PATH "('${PRETRAIN}')"  \
#     OUTPUT_DIR "('${SAVE_DIR}')" 

# PRETRAIN=../weights/r50_ibn_a.pth
# DATA_DIR='/data/Dataset/PReID/rep_dataset/'
# SAVE_DIR='../rep_work_dirs/exp4-cosinebaseline-resnet50ibnls1-384x192-bs16x6-warmup10-flip-pad10-meanstd-erase0502-nolbsm-avg-cosface30_035_10033-use_bnbias-cj-minInst2/' #(h, w)
# CUDA_VISIBLE_DEVICES=6 python train.py --config_file='configs/naic/arcface_baseline.yml' \
#     SOLVER.IMS_PER_BATCH "96" DATALOADER.NUM_INSTANCE "6" \
#     INPUT.USE_CJ "True" INPUT.CJ_BRIGHNESS "0.5" INPUT.CJ_CONTRAST "0.5" INPUT.CJ_SATURATION "0.0" INPUT.CJ_HUE "0.5" \
#     MODEL.BASELINE.TPL_WEIGHT "1.0" MODEL.BASELINE.CE_WEIGHT "0.33" MODEL.LABEL_SMOOTH "False" \
#     MODEL.BASELINE.S "30.0" MODEL.BASELINE.M "0.35" MODEL.BASELINE.USE_BNBAIS "True"\
#     MODEL.BASELINE.COSINE_LOSS_TYPE 'CosFace' \
#     INPUT.SIZE_TRAIN "([384,192])" INPUT.SIZE_TEST "([384,192])" \
#     MODEL.NAME "cosine_baseline" MODEL.BACKBONE "('resnet50_ibn_a')" MODEL.BASELINE.POOL_TYPE "avg"\
#     DATASETS.DATA_PATH "('${DATA_DIR}')" DATASETS.TRAIN_PATH "rep_f0_train2"\
#     MODEL.PRETRAIN_PATH "('${PRETRAIN}')"  \
#     OUTPUT_DIR "('${SAVE_DIR}')" 

# PRETRAIN=../weights/r50_ibn_a.pth
# DATA_DIR='/data/Dataset/PReID/rep_dataset/'
# SAVE_DIR='../rep_work_dirs/exp4-cosinebaseline-resnet50ibnls1-384x192-bs16x6-warmup10-flip-pad10-meanstd-erase0502-nolbsm-gem-cosface30_035_10033-use_bnbias-cj-trainVal2/' #(h, w)
# CUDA_VISIBLE_DEVICES=4 python train.py --config_file='configs/naic/arcface_baseline.yml' \
#     SOLVER.IMS_PER_BATCH "96" DATALOADER.NUM_INSTANCE "6" SOLVER.STEPS "[40, 70]" SOLVER.MAX_EPOCHS "90" SOLVER.START_SAVE_EPOCH "75" \
#     INPUT.USE_CJ "True" INPUT.CJ_BRIGHNESS "0.5" INPUT.CJ_CONTRAST "0.5" INPUT.CJ_SATURATION "0.0" INPUT.CJ_HUE "0.5" \
#     MODEL.BASELINE.TPL_WEIGHT "1.0" MODEL.BASELINE.CE_WEIGHT "0.33" MODEL.LABEL_SMOOTH "False" \
#     MODEL.BASELINE.S "30.0" MODEL.BASELINE.M "0.35" MODEL.BASELINE.USE_BNBAIS "True"\
#     MODEL.BASELINE.COSINE_LOSS_TYPE 'CosFace' \
#     INPUT.SIZE_TRAIN "([384,192])" INPUT.SIZE_TEST "([384,192])" \
#     MODEL.NAME "cosine_baseline" MODEL.BACKBONE "('resnet50_ibn_a')" MODEL.BASELINE.POOL_TYPE "gem"\
#     DATASETS.DATA_PATH "('${DATA_DIR}')" DATASETS.TRAIN_PATH "rep_trainVal2"\
#     MODEL.PRETRAIN_PATH "('${PRETRAIN}')"  \
#     OUTPUT_DIR "('${SAVE_DIR}')" 

# PRETRAIN=../weights/r50_ibn_a.pth
# DATA_DIR='/data/Dataset/PReID/rep_dataset/'
# SAVE_DIR='../rep_work_dirs/exp4-cosinebaseline-resnet50ibnls1-384x192-bs16x6-warmup10-flip-pad10-meanstd-erase0502-nolbsm-avg-cosface30_035_10033-use_bnbias-cj-trainVal2/' #(h, w)
# CUDA_VISIBLE_DEVICES=2 python train.py --config_file='configs/naic/arcface_baseline.yml' \
#     SOLVER.IMS_PER_BATCH "96" DATALOADER.NUM_INSTANCE "6" SOLVER.STEPS "[40, 70]" SOLVER.MAX_EPOCHS "90" SOLVER.START_SAVE_EPOCH "75" \
#     INPUT.USE_CJ "True" INPUT.CJ_BRIGHNESS "0.5" INPUT.CJ_CONTRAST "0.5" INPUT.CJ_SATURATION "0.0" INPUT.CJ_HUE "0.5" \
#     MODEL.BASELINE.TPL_WEIGHT "1.0" MODEL.BASELINE.CE_WEIGHT "0.33" MODEL.LABEL_SMOOTH "False" \
#     MODEL.BASELINE.S "30.0" MODEL.BASELINE.M "0.35" MODEL.BASELINE.USE_BNBAIS "True"\
#     MODEL.BASELINE.COSINE_LOSS_TYPE 'CosFace' \
#     INPUT.SIZE_TRAIN "([384,192])" INPUT.SIZE_TEST "([384,192])" \
#     MODEL.NAME "cosine_baseline" MODEL.BACKBONE "('resnet50_ibn_a')" MODEL.BASELINE.POOL_TYPE "avg"\
#     DATASETS.DATA_PATH "('${DATA_DIR}')" DATASETS.TRAIN_PATH "rep_trainVal2"\
#     MODEL.PRETRAIN_PATH "('${PRETRAIN}')"  \
#     OUTPUT_DIR "('${SAVE_DIR}')" 

# PRETRAIN=../weights/r50_ibn_a.pth
# DATA_DIR='/data/Dataset/PReID/rep_dataset/'
# SAVE_DIR='../rep_work_dirs/exp4-cosinebaseline-resnet50ibnls1-384x192-bs16x6-warmup10-flip-pad10-meanstd-erase0502-nolbsm-avg-cosface30_035_10033-cj-minInst2/' #(h, w)
# CUDA_VISIBLE_DEVICES=3 python train.py --config_file='configs/naic/arcface_baseline.yml' \
#     SOLVER.IMS_PER_BATCH "96" DATALOADER.NUM_INSTANCE "6" \
#     INPUT.USE_CJ "True" INPUT.CJ_BRIGHNESS "0.5" INPUT.CJ_CONTRAST "0.5" INPUT.CJ_SATURATION "0.0" INPUT.CJ_HUE "0.5" \
#     MODEL.BASELINE.TPL_WEIGHT "1.0" MODEL.BASELINE.CE_WEIGHT "0.33" MODEL.LABEL_SMOOTH "False" \
#     MODEL.BASELINE.S "30.0" MODEL.BASELINE.M "0.35" \
#     MODEL.BASELINE.COSINE_LOSS_TYPE 'CosFace' \
#     INPUT.SIZE_TRAIN "([384,192])" INPUT.SIZE_TEST "([384,192])" \
#     MODEL.NAME "cosine_baseline" MODEL.BACKBONE "('resnet50_ibn_a')" MODEL.BASELINE.POOL_TYPE "avg"\
#     DATASETS.DATA_PATH "('${DATA_DIR}')" DATASETS.TRAIN_PATH "rep_f0_train2"\
#     MODEL.PRETRAIN_PATH "('${PRETRAIN}')"  \
#     OUTPUT_DIR "('${SAVE_DIR}')" 

# PRETRAIN=../weights/resnet101_ibn_a.pth.tar
# DATA_DIR='/data/Dataset/PReID/rep_dataset/'
# SAVE_DIR='../rep_work_dirs/exp4-cosinebaseline-resnet101ibnls1-384x192-bs16x6-warmup10-flip-pad10-meanstd-erase0502-nolbsm-avg-cosface30_035_10033-cj-minInst2/' #(h, w)
# # SAVE_DIR='../rep_work_dirs/debug/' #(h, w)

# CUDA_VISIBLE_DEVICES=3 python train.py --config_file='configs/naic/arcface_baseline.yml' \
#     SOLVER.IMS_PER_BATCH "96" DATALOADER.NUM_INSTANCE "6" \
#     INPUT.USE_CJ "True" INPUT.CJ_BRIGHNESS "0.5" INPUT.CJ_CONTRAST "0.5" INPUT.CJ_SATURATION "0.0" INPUT.CJ_HUE "0.5" \
#     MODEL.BASELINE.TPL_WEIGHT "1.0" MODEL.BASELINE.CE_WEIGHT "0.33" MODEL.LABEL_SMOOTH "False" \
#     MODEL.BASELINE.S "30.0" MODEL.BASELINE.M "0.35" \
#     MODEL.BASELINE.COSINE_LOSS_TYPE 'CosFace' \
#     INPUT.SIZE_TRAIN "([384,192])" INPUT.SIZE_TEST "([384,192])" \
#     MODEL.NAME "cosine_baseline" MODEL.BACKBONE "('resnet101_ibn_a')" MODEL.BASELINE.POOL_TYPE "avg"\
#     DATASETS.DATA_PATH "('${DATA_DIR}')" DATASETS.TRAIN_PATH "rep_f0_train2"\
#     MODEL.PRETRAIN_PATH "('${PRETRAIN}')"  \
#     OUTPUT_DIR "('${SAVE_DIR}')" 

# PRETRAIN=../weights/resnet101_ibn_a.pth.tar
# DATA_DIR='/data/Dataset/PReID/rep_dataset/'
# SAVE_DIR='../rep_work_dirs/exp4-cosinebaseline-resnet101ibnls1-384x192-bs16x6-warmup10-flip-pad10-meanstd-erase0502-nolbsm-avg-cosface30_035_10033-cj-trainVal2/' #(h, w)
# # SAVE_DIR='../rep_work_dirs/debug/' #(h, w)

# CUDA_VISIBLE_DEVICES=6 python train.py --config_file='configs/naic/arcface_baseline.yml' \
#     SOLVER.IMS_PER_BATCH "96" DATALOADER.NUM_INSTANCE "6" SOLVER.STEPS "[40, 70]" SOLVER.MAX_EPOCHS "90" SOLVER.START_SAVE_EPOCH "75" \
#     INPUT.USE_CJ "True" INPUT.CJ_BRIGHNESS "0.5" INPUT.CJ_CONTRAST "0.5" INPUT.CJ_SATURATION "0.0" INPUT.CJ_HUE "0.5" \
#     MODEL.BASELINE.TPL_WEIGHT "1.0" MODEL.BASELINE.CE_WEIGHT "0.33" MODEL.LABEL_SMOOTH "False" \
#     MODEL.BASELINE.S "30.0" MODEL.BASELINE.M "0.35" \
#     MODEL.BASELINE.COSINE_LOSS_TYPE 'CosFace' \
#     INPUT.SIZE_TRAIN "([384,192])" INPUT.SIZE_TEST "([384,192])" \
#     MODEL.NAME "cosine_baseline" MODEL.BACKBONE "('resnet101_ibn_a')" MODEL.BASELINE.POOL_TYPE "avg"\
#     DATASETS.DATA_PATH "('${DATA_DIR}')" DATASETS.TRAIN_PATH "rep_trainVal2"\
#     MODEL.PRETRAIN_PATH "('${PRETRAIN}')"  \
#     OUTPUT_DIR "('${SAVE_DIR}')" 

# PRETRAIN=../weights/r50_ibn_a.pth
# DATA_DIR='/data/Dataset/PReID/rep_dataset/'
# SAVE_DIR='../rep_work_dirs/exp4-cosinebaseline-resnet50ibnls1_sestn-384x192-bs16x6-warmup10-flip-pad10-meanstd-erase0502-nolbsm-avg-cosface30_035_10033-cj-minInst2/' #(h, w)
# # SAVE_DIR='../rep_work_dirs/debug/' #(h, w)

# CUDA_VISIBLE_DEVICES=5 python train.py --config_file='configs/naic/arcface_baseline.yml' \
#     SOLVER.IMS_PER_BATCH "96" DATALOADER.NUM_INSTANCE "6" \
#     INPUT.USE_CJ "True" INPUT.CJ_BRIGHNESS "0.5" INPUT.CJ_CONTRAST "0.5" INPUT.CJ_SATURATION "0.0" INPUT.CJ_HUE "0.5" \
#     MODEL.BASELINE.TPL_WEIGHT "1.0" MODEL.BASELINE.CE_WEIGHT "0.33" MODEL.LABEL_SMOOTH "False" \
#     MODEL.BASELINE.S "30.0" MODEL.BASELINE.M "0.35" \
#     MODEL.BASELINE.COSINE_LOSS_TYPE 'CosFace' \
#     INPUT.SIZE_TRAIN "([384,192])" INPUT.SIZE_TEST "([384,192])" \
#     MODEL.NAME "cosine_baseline" MODEL.BACKBONE "('resnet50_ibn_a')" MODEL.BASELINE.POOL_TYPE "avg" MODEL.BASELINE.USE_SESTN "True"\
#     DATASETS.DATA_PATH "('${DATA_DIR}')" DATASETS.TRAIN_PATH "rep_f0_train2"\
#     MODEL.PRETRAIN_PATH "('${PRETRAIN}')"  \
#     OUTPUT_DIR "('${SAVE_DIR}')" 

# PRETRAIN=../weights/r50_ibn_a.pth
# DATA_DIR='/data/Dataset/PReID/rep_dataset/'
# SAVE_DIR='../rep_work_dirs/exp4-cosinebaseline-resnet50ibnls1_sestn-384x192-bs16x6-warmup10-flip-pad10-meanstd-erase0502-nolbsm-avg-cosface30_035_10033-cj-trainVal2/' #(h, w)
# # SAVE_DIR='../rep_work_dirs/debug/' #(h, w)

# CUDA_VISIBLE_DEVICES=4 python train.py --config_file='configs/naic/arcface_baseline.yml' \
#     SOLVER.IMS_PER_BATCH "96" DATALOADER.NUM_INSTANCE "6"  SOLVER.STEPS "[40, 70]" SOLVER.MAX_EPOCHS "100" SOLVER.START_SAVE_EPOCH "75" \
#     INPUT.USE_CJ "True" INPUT.CJ_BRIGHNESS "0.5" INPUT.CJ_CONTRAST "0.5" INPUT.CJ_SATURATION "0.0" INPUT.CJ_HUE "0.5" \
#     MODEL.BASELINE.TPL_WEIGHT "1.0" MODEL.BASELINE.CE_WEIGHT "0.33" MODEL.LABEL_SMOOTH "False" \
#     MODEL.BASELINE.S "30.0" MODEL.BASELINE.M "0.35" \
#     MODEL.BASELINE.COSINE_LOSS_TYPE 'CosFace' \
#     INPUT.SIZE_TRAIN "([384,192])" INPUT.SIZE_TEST "([384,192])" \
#     MODEL.NAME "cosine_baseline" MODEL.BACKBONE "('resnet50_ibn_a')" MODEL.BASELINE.POOL_TYPE "avg" MODEL.BASELINE.USE_SESTN "True"\
#     DATASETS.DATA_PATH "('${DATA_DIR}')" DATASETS.TRAIN_PATH "rep_trainVal2"\
#     MODEL.PRETRAIN_PATH "('${PRETRAIN}')"  \
#     OUTPUT_DIR "('${SAVE_DIR}')" 


# PRETRAIN=../weights/resnet101_ibn_a.pth.tar
# DATA_DIR='/data/Dataset/PReID/rep_dataset/'
# SAVE_DIR='../rep_work_dirs/exp4-cosinebaseline-resnet101ibnls1-384x192-bs16x6-warmup10-flip-pad10-meanstd-erase0502-nolbsm-avg-cosface30_035_10033-cj05-minInst2/' #(h, w)
# # SAVE_DIR='../rep_work_dirs/debug/' #(h, w)

# CUDA_VISIBLE_DEVICES=5 python train.py --config_file='configs/naic/arcface_baseline.yml' \
#     SOLVER.IMS_PER_BATCH "96" DATALOADER.NUM_INSTANCE "6" \
#     INPUT.USE_CJ "True"  INPUT.CJ_PROB "0.5" INPUT.CJ_BRIGHNESS "0.5" INPUT.CJ_CONTRAST "0.5" INPUT.CJ_SATURATION "0.0" INPUT.CJ_HUE "0.5" \
#     MODEL.BASELINE.TPL_WEIGHT "1.0" MODEL.BASELINE.CE_WEIGHT "0.33" MODEL.LABEL_SMOOTH "False" \
#     MODEL.BASELINE.S "30.0" MODEL.BASELINE.M "0.35" \
#     MODEL.BASELINE.COSINE_LOSS_TYPE 'CosFace' \
#     INPUT.SIZE_TRAIN "([384,192])" INPUT.SIZE_TEST "([384,192])" \
#     MODEL.NAME "cosine_baseline" MODEL.BACKBONE "('resnet101_ibn_a')" MODEL.BASELINE.POOL_TYPE "avg"\
#     DATASETS.DATA_PATH "('${DATA_DIR}')" DATASETS.TRAIN_PATH "rep_f0_train2"\
#     MODEL.PRETRAIN_PATH "('${PRETRAIN}')"  \
#     OUTPUT_DIR "('${SAVE_DIR}')" 

# PRETRAIN=../weights/resnet101_ibn_a.pth.tar
# DATA_DIR='/data/Dataset/PReID/rep_dataset/'
# SAVE_DIR='../rep_work_dirs/exp4-cosinebaseline-resnet101ibnls1-384x192-bs16x6-warmup10-flip-pad10-meanstd-erase0502-nolbsm-avg-cosface30_035_10033-cj05-trainVal2/' #(h, w)
# # SAVE_DIR='../rep_work_dirs/debug/' #(h, w)

# CUDA_VISIBLE_DEVICES=4 python train.py --config_file='configs/naic/arcface_baseline.yml' \
#     SOLVER.IMS_PER_BATCH "96" DATALOADER.NUM_INSTANCE "6" SOLVER.STEPS "[40, 70]" SOLVER.MAX_EPOCHS "90" SOLVER.START_SAVE_EPOCH "75" \
#     INPUT.USE_CJ "True" INPUT.CJ_PROB "0.5" INPUT.CJ_BRIGHNESS "0.5" INPUT.CJ_CONTRAST "0.5" INPUT.CJ_SATURATION "0.0" INPUT.CJ_HUE "0.5" \
#     MODEL.BASELINE.TPL_WEIGHT "1.0" MODEL.BASELINE.CE_WEIGHT "0.33" MODEL.LABEL_SMOOTH "False" \
#     MODEL.BASELINE.S "30.0" MODEL.BASELINE.M "0.35" \
#     MODEL.BASELINE.COSINE_LOSS_TYPE 'CosFace' \
#     INPUT.SIZE_TRAIN "([384,192])" INPUT.SIZE_TEST "([384,192])" \
#     MODEL.NAME "cosine_baseline" MODEL.BACKBONE "('resnet101_ibn_a')" MODEL.BASELINE.POOL_TYPE "avg"\
#     DATASETS.DATA_PATH "('${DATA_DIR}')" DATASETS.TRAIN_PATH "rep_trainVal2"\
#     MODEL.PRETRAIN_PATH "('${PRETRAIN}')"  \
#     OUTPUT_DIR "('${SAVE_DIR}')" 

# PRETRAIN=../weights/resnet101_ibn_a.pth.tar
# DATA_DIR='/data/Dataset/PReID/rep_dataset/'
# SAVE_DIR='../rep_work_dirs/exp4-cosinebaseline-resnet101ibnls1-384x192-bs16x6-warmup10-flip-pad10-meanstd-erase0502-nolbsm-avg-arcface30_035_10033-cj05-trainVal2/' #(h, w)
# # SAVE_DIR='../rep_work_dirs/debug/' #(h, w)

# CUDA_VISIBLE_DEVICES=4 python train.py --config_file='configs/naic/arcface_baseline.yml' \
#     SOLVER.IMS_PER_BATCH "96" DATALOADER.NUM_INSTANCE "6" SOLVER.STEPS "[40, 70]" SOLVER.MAX_EPOCHS "90" SOLVER.START_SAVE_EPOCH "75" \
#     INPUT.USE_CJ "True" INPUT.CJ_PROB "0.5" INPUT.CJ_BRIGHNESS "0.5" INPUT.CJ_CONTRAST "0.5" INPUT.CJ_SATURATION "0.0" INPUT.CJ_HUE "0.5" \
#     MODEL.BASELINE.TPL_WEIGHT "1.0" MODEL.BASELINE.CE_WEIGHT "0.33" MODEL.LABEL_SMOOTH "False" \
#     MODEL.BASELINE.S "30.0" MODEL.BASELINE.M "0.35" \
#     MODEL.BASELINE.COSINE_LOSS_TYPE 'ArcCos' \
#     INPUT.SIZE_TRAIN "([384,192])" INPUT.SIZE_TEST "([384,192])" \
#     MODEL.NAME "cosine_baseline" MODEL.BACKBONE "('resnet101_ibn_a')" MODEL.BASELINE.POOL_TYPE "avg"\
#     DATASETS.DATA_PATH "('${DATA_DIR}')" DATASETS.TRAIN_PATH "rep_trainVal2"\
#     MODEL.PRETRAIN_PATH "('${PRETRAIN}')"  \
#     OUTPUT_DIR "('${SAVE_DIR}')" 

# PRETRAIN=../weights/resnet101_ibn_a.pth.tar
# DATA_DIR='/data/Dataset/PReID/rep_dataset/'
# SAVE_DIR='../rep_work_dirs/exp5-cosinebaseline-resnet101ibnls1-384x192-bs16x6-warmup10-flip-pad10-meanstd-erase0502-nolbsm-avgmax-arcface30_035_10033-cj05-trainVal2/' #(h, w)
# # SAVE_DIR='../rep_work_dirs/debug/' #(h, w)

# CUDA_VISIBLE_DEVICES=6 python train.py --config_file='configs/naic/arcface_baseline.yml' \
#     SOLVER.IMS_PER_BATCH "96" DATALOADER.NUM_INSTANCE "6" SOLVER.STEPS "[40, 70]" SOLVER.MAX_EPOCHS "90" SOLVER.START_SAVE_EPOCH "75" \
#     INPUT.USE_CJ "True" INPUT.CJ_PROB "0.5" INPUT.CJ_BRIGHNESS "0.5" INPUT.CJ_CONTRAST "0.5" INPUT.CJ_SATURATION "0.0" INPUT.CJ_HUE "0.5" \
#     MODEL.BASELINE.TPL_WEIGHT "1.0" MODEL.BASELINE.CE_WEIGHT "0.33" MODEL.LABEL_SMOOTH "False" \
#     MODEL.BASELINE.S "30.0" MODEL.BASELINE.M "0.35" \
#     MODEL.BASELINE.COSINE_LOSS_TYPE 'ArcCos' \
#     INPUT.SIZE_TRAIN "([384,192])" INPUT.SIZE_TEST "([384,192])" \
#     MODEL.NAME "cosine_baseline" MODEL.BACKBONE "('resnet101_ibn_a')" MODEL.BASELINE.POOL_TYPE "GlobalAvgMaxPool2d"\
#     DATASETS.DATA_PATH "('${DATA_DIR}')" DATASETS.TRAIN_PATH "rep_trainVal2"\
#     MODEL.PRETRAIN_PATH "('${PRETRAIN}')"  \
#     OUTPUT_DIR "('${SAVE_DIR}')" 


# PRETRAIN=../weights/resnet101_ibn_a.pth.tar
# DATA_DIR='/data/Dataset/PReID/rep_dataset/'
# SAVE_DIR='../rep_work_dirs/exp5-cosinebaseline-resnet101ibnls1-384x192-bs16x6-warmup10-flip-pad10-meanstd-erase0502-nolbsm-avgmax-cosface30_035_10033-cj05-trainVal2/' #(h, w)
# # SAVE_DIR='../rep_work_dirs/debug/' #(h, w)

# CUDA_VISIBLE_DEVICES=5 python train.py --config_file='configs/naic/arcface_baseline.yml' \
#     SOLVER.IMS_PER_BATCH "96" DATALOADER.NUM_INSTANCE "6" SOLVER.STEPS "[40, 70]" SOLVER.MAX_EPOCHS "90" SOLVER.START_SAVE_EPOCH "75" \
#     INPUT.USE_CJ "True" INPUT.CJ_PROB "0.5" INPUT.CJ_BRIGHNESS "0.5" INPUT.CJ_CONTRAST "0.5" INPUT.CJ_SATURATION "0.0" INPUT.CJ_HUE "0.5" \
#     MODEL.BASELINE.TPL_WEIGHT "1.0" MODEL.BASELINE.CE_WEIGHT "0.33" MODEL.LABEL_SMOOTH "False" \
#     MODEL.BASELINE.S "30.0" MODEL.BASELINE.M "0.35" \
#     MODEL.BASELINE.COSINE_LOSS_TYPE 'CosFace' \
#     INPUT.SIZE_TRAIN "([384,192])" INPUT.SIZE_TEST "([384,192])" \
#     MODEL.NAME "cosine_baseline" MODEL.BACKBONE "('resnet101_ibn_a')" MODEL.BASELINE.POOL_TYPE "GlobalAvgMaxPool2d"\
#     DATASETS.DATA_PATH "('${DATA_DIR}')" DATASETS.TRAIN_PATH "rep_trainVal2"\
#     MODEL.PRETRAIN_PATH "('${PRETRAIN}')"  \
#     OUTPUT_DIR "('${SAVE_DIR}')" 

# PRETRAIN=../weights/resnet101_ibn_a.pth.tar
# DATA_DIR='/data/Dataset/PReID/rep_dataset/'
# SAVE_DIR='../rep_work_dirs/exp4-cosinebaseline-resnet101ibnls1-384x192-bs16x8-warmup10-flip-pad10-meanstd-erase0502-nolbsm-avg-arcface30_035_10033-cj05-trainVal2/' #(h, w)
# # SAVE_DIR='../rep_work_dirs/debug/' #(h, w)

# CUDA_VISIBLE_DEVICES=7 python train.py --config_file='configs/naic/arcface_baseline.yml' \
#     SOLVER.IMS_PER_BATCH "128" DATALOADER.NUM_INSTANCE "8" SOLVER.STEPS "[40, 70]" SOLVER.MAX_EPOCHS "90" SOLVER.START_SAVE_EPOCH "75" \
#     INPUT.USE_CJ "True" INPUT.CJ_PROB "0.5" INPUT.CJ_BRIGHNESS "0.5" INPUT.CJ_CONTRAST "0.5" INPUT.CJ_SATURATION "0.0" INPUT.CJ_HUE "0.5" \
#     MODEL.BASELINE.TPL_WEIGHT "1.0" MODEL.BASELINE.CE_WEIGHT "0.33" MODEL.LABEL_SMOOTH "False" \
#     MODEL.BASELINE.S "30.0" MODEL.BASELINE.M "0.35" \
#     MODEL.BASELINE.COSINE_LOSS_TYPE 'ArcCos' \
#     INPUT.SIZE_TRAIN "([384,192])" INPUT.SIZE_TEST "([384,192])" \
#     MODEL.NAME "cosine_baseline" MODEL.BACKBONE "('resnet101_ibn_a')" MODEL.BASELINE.POOL_TYPE "avg"\
#     DATASETS.DATA_PATH "('${DATA_DIR}')" DATASETS.TRAIN_PATH "rep_trainVal2"\
#     MODEL.PRETRAIN_PATH "('${PRETRAIN}')"  \
#     OUTPUT_DIR "('${SAVE_DIR}')" 

# PRETRAIN=../weights/resnet101_ibn_a.pth.tar
# DATA_DIR='/data/Dataset/PReID/rep_dataset/'
# SAVE_DIR='../rep_work_dirs/exp4-cosinebaseline-resnet101ibnls1-384x192-bs16x8-warmup10-flip-pad10-meanstd-erase0502-nolbsm-avg-arcface30_035_10033-cj05-trainVal/' #(h, w)
# # SAVE_DIR='../rep_work_dirs/debug/' #(h, w)

# CUDA_VISIBLE_DEVICES=0 python train.py --config_file='configs/naic/arcface_baseline.yml' \
#     SOLVER.IMS_PER_BATCH "128" DATALOADER.NUM_INSTANCE "8" SOLVER.STEPS "[40, 70]" SOLVER.MAX_EPOCHS "90" SOLVER.START_SAVE_EPOCH "70" \
#     INPUT.USE_CJ "True" INPUT.CJ_PROB "0.5" INPUT.CJ_BRIGHNESS "0.5" INPUT.CJ_CONTRAST "0.5" INPUT.CJ_SATURATION "0.0" INPUT.CJ_HUE "0.5" \
#     MODEL.BASELINE.TPL_WEIGHT "1.0" MODEL.BASELINE.CE_WEIGHT "0.33" MODEL.LABEL_SMOOTH "False" \
#     MODEL.BASELINE.S "30.0" MODEL.BASELINE.M "0.35" \
#     MODEL.BASELINE.COSINE_LOSS_TYPE 'ArcCos' \
#     INPUT.SIZE_TRAIN "([384,192])" INPUT.SIZE_TEST "([384,192])" \
#     MODEL.NAME "cosine_baseline" MODEL.BACKBONE "('resnet101_ibn_a')" MODEL.BASELINE.POOL_TYPE "avg"\
#     DATASETS.DATA_PATH "('${DATA_DIR}')" DATASETS.TRAIN_PATH "rep_trainVal"\
#     MODEL.PRETRAIN_PATH "('${PRETRAIN}')"  \
#     OUTPUT_DIR "('${SAVE_DIR}')" 

# PRETRAIN=../weights/resnet101_ibn_a.pth.tar
# DATA_DIR='/data/Dataset/PReID/rep_dataset/'
# SAVE_DIR='../rep_work_dirs/exp4-cosinebaseline-resnet101ibnls1-288x144-bs16x8-warmup10-flip-pad10-meanstd-erase0504-lbsm-avg-arcface30_035_10033-cj05-trainVal/' #(h, w)
# # SAVE_DIR='../rep_work_dirs/debug/' #(h, w)

# CUDA_VISIBLE_DEVICES=2 python train.py --config_file='configs/naic/arcface_baseline.yml' \
#     SOLVER.IMS_PER_BATCH "128" DATALOADER.NUM_INSTANCE "8" SOLVER.STEPS "[40, 70]" SOLVER.MAX_EPOCHS "90" SOLVER.START_SAVE_EPOCH "70" \
#     INPUT.USE_CJ "True" INPUT.CJ_PROB "0.5" INPUT.CJ_BRIGHNESS "0.5" INPUT.CJ_CONTRAST "0.5" INPUT.CJ_SATURATION "0.0" INPUT.CJ_HUE "0.5" \
#     INPUT.RE_MAX_RATIO "0.5" \
#     MODEL.BASELINE.TPL_WEIGHT "1.0" MODEL.BASELINE.CE_WEIGHT "0.33" MODEL.LABEL_SMOOTH "True" \
#     MODEL.BASELINE.S "30.0" MODEL.BASELINE.M "0.35" \
#     MODEL.BASELINE.COSINE_LOSS_TYPE 'ArcCos' \
#     INPUT.SIZE_TRAIN "([288,144])" INPUT.SIZE_TEST "([288,144])" \
#     MODEL.NAME "cosine_baseline" MODEL.BACKBONE "('resnet101_ibn_a')" MODEL.BASELINE.POOL_TYPE "avg"\
#     DATASETS.DATA_PATH "('${DATA_DIR}')" DATASETS.TRAIN_PATH "rep_trainVal"\
#     MODEL.PRETRAIN_PATH "('${PRETRAIN}')"  \
#     OUTPUT_DIR "('${SAVE_DIR}')" 

# PRETRAIN=../weights/resnet101_ibn_a.pth.tar
# DATA_DIR='/data/Dataset/PReID/rep_dataset/'
# SAVE_DIR='../rep_work_dirs/exp4-cosinebaseline-resnet101ibnls1-288x144-bs16x8-warmup10-flip-pad10-nomeanstd-erase0504-lbsm-avg-arcface30_035_10033-cj05-trainVal/' #(h, w)
# # SAVE_DIR='../rep_work_dirs/debug/' #(h, w)

# CUDA_VISIBLE_DEVICES=0 python train.py --config_file='configs/naic/arcface_baseline.yml' \
#     SOLVER.IMS_PER_BATCH "128" DATALOADER.NUM_INSTANCE "8" SOLVER.STEPS "[40, 70]" SOLVER.MAX_EPOCHS "90" SOLVER.START_SAVE_EPOCH "70" \
#     INPUT.USE_CJ "True" INPUT.CJ_PROB "0.5" INPUT.CJ_BRIGHNESS "0.5" INPUT.CJ_CONTRAST "0.5" INPUT.CJ_SATURATION "0.0" INPUT.CJ_HUE "0.5" \
#     INPUT.RE_MAX_RATIO "0.5" \
#     MODEL.BASELINE.TPL_WEIGHT "1.0" MODEL.BASELINE.CE_WEIGHT "0.33" MODEL.LABEL_SMOOTH "True" \
#     INPUT.PIXEL_MEAN "[0.485, 0.456, 0.406]" INPUT.PIXEL_STD "[0.229, 0.224, 0.225]" \
#     MODEL.BASELINE.S "30.0" MODEL.BASELINE.M "0.35" \
#     MODEL.BASELINE.COSINE_LOSS_TYPE 'ArcCos' \
#     INPUT.SIZE_TRAIN "([288,144])" INPUT.SIZE_TEST "([288,144])" \
#     MODEL.NAME "cosine_baseline" MODEL.BACKBONE "('resnet101_ibn_a')" MODEL.BASELINE.POOL_TYPE "avg"\
#     DATASETS.DATA_PATH "('${DATA_DIR}')" DATASETS.TRAIN_PATH "rep_trainVal"\
#     MODEL.PRETRAIN_PATH "('${PRETRAIN}')"  \
#     OUTPUT_DIR "('${SAVE_DIR}')" 


# PRETRAIN=../weights/resnet101_ibn_a.pth.tar
# DATA_DIR='/data/Dataset/PReID/rep_dataset/'
# SAVE_DIR='../rep_work_dirs/exp4-cosinebaseline-resnet101ibnls1-384x192-bs16x6-warmup10-flip-pad10-meanstd-erase0502-nolbsm-avg-cosface30_035_10033-cj05-trainVal2-pseudo/' #(h, w)
# # SAVE_DIR='../rep_work_dirs/debug/' #(h, w)

# CUDA_VISIBLE_DEVICES=5 python train.py --config_file='configs/naic/arcface_baseline.yml' \
#     MODEL.WEIGHT '../rep_work_dirs/exp4-cosinebaseline-resnet101ibnls1-384x192-bs16x6-warmup10-flip-pad10-meanstd-erase0502-nolbsm-avg-cosface30_035_10033-cj05-trainVal2/cosine_baseline_epoch90.pth' \
#     SOLVER.BASE_LR '1e-4' SOLVER.WARMUP_EPOCH "2" SOLVER.STEPS "[6, 8]" SOLVER.MAX_EPOCHS "10" SOLVER.START_SAVE_EPOCH "2" SOLVER.EVAL_PERIOD "1"\
#     SOLVER.IMS_PER_BATCH "96" DATALOADER.NUM_INSTANCE "6" \
#     INPUT.USE_CJ "False" INPUT.CJ_PROB "0.5" INPUT.CJ_BRIGHNESS "0.5" INPUT.CJ_CONTRAST "0.5" INPUT.CJ_SATURATION "0.0" INPUT.CJ_HUE "0.5" \
#     MODEL.BASELINE.TPL_WEIGHT "1.0" MODEL.BASELINE.CE_WEIGHT "0.33" MODEL.LABEL_SMOOTH "False" \
#     MODEL.BASELINE.S "30.0" MODEL.BASELINE.M "0.35" \
#     MODEL.BASELINE.COSINE_LOSS_TYPE 'CosFace' \
#     INPUT.SIZE_TRAIN "([384,192])" INPUT.SIZE_TEST "([384,192])" \
#     MODEL.NAME "cosine_baseline" MODEL.BACKBONE "('resnet101_ibn_a')" MODEL.BASELINE.POOL_TYPE "avg"\
#     DATASETS.DATA_PATH "('${DATA_DIR}')" DATASETS.TRAIN_PATH "/data/Dataset/PReID/pseudo_aqe_eps02_2_dataset/rep_trainVal2"\
#     MODEL.PRETRAIN_PATH "('${PRETRAIN}')"  \
#     OUTPUT_DIR "('${SAVE_DIR}')" 



# PRETRAIN=../weights/resnet101_ibn_a.pth.tar
# DATA_DIR='/data/Dataset/PReID/rep_dataset/'
# SAVE_DIR='../rep_work_dirs/exp4-cosinebaseline-resnet101ibnls1-384x192-bs16x6-warmup10-flip-pad10-meanstd-erase0502-nolbsm-avg-arcface30_035_10033-cj05-trainVal2-pseudo/' #(h, w)

# CUDA_VISIBLE_DEVICES=2 python train.py --config_file='configs/naic/arcface_baseline.yml' \
#     MODEL.WEIGHT '../rep_work_dirs/exp4-cosinebaseline-resnet101ibnls1-384x192-bs16x6-warmup10-flip-pad10-meanstd-erase0502-nolbsm-avg-arcface30_035_10033-cj05-trainVal2/cosine_baseline_epoch90.pth' \
#     SOLVER.BASE_LR '3e-4' SOLVER.WARMUP_EPOCH "5" SOLVER.STEPS "[8, 13]" SOLVER.MAX_EPOCHS "16" SOLVER.START_SAVE_EPOCH "7" SOLVER.EVAL_PERIOD "2"\
#     INPUT.USE_CJ "False" INPUT.CJ_PROB "0.5" INPUT.CJ_BRIGHNESS "0.5" INPUT.CJ_CONTRAST "0.5" INPUT.CJ_SATURATION "0.0" INPUT.CJ_HUE "0.5" \
#     SOLVER.IMS_PER_BATCH "96" DATALOADER.NUM_INSTANCE "6" \
#     MODEL.BASELINE.TPL_WEIGHT "1.0" MODEL.BASELINE.CE_WEIGHT "0.33" MODEL.LABEL_SMOOTH "False" \
#     MODEL.BASELINE.S "30.0" MODEL.BASELINE.M "0.35" \
#     MODEL.BASELINE.COSINE_LOSS_TYPE 'ArcCos' \
#     INPUT.SIZE_TRAIN "([384,192])" INPUT.SIZE_TEST "([384,192])" \
#     MODEL.NAME "cosine_baseline" MODEL.BACKBONE "('resnet101_ibn_a')" MODEL.BASELINE.POOL_TYPE "avg"\
#     DATASETS.DATA_PATH "('${DATA_DIR}')" DATASETS.TRAIN_PATH "/data/Dataset/PReID/pseudo_auto_flip_eps055_2_dataset/rep_trainVal2"\
#     MODEL.PRETRAIN_PATH "('${PRETRAIN}')"  \
#     OUTPUT_DIR "('${SAVE_DIR}')" 

# PRETRAIN=../weights/resnet101_ibn_a.pth.tar
# DATA_DIR='/data/Dataset/PReID/rep_dataset/'
# SAVE_DIR='../rep_work_dirs/exp4-cosinebaseline-resnet101ibnls1-384x192-bs16x6-warmup10-flip-pad10-meanstd-erase0502-nolbsm-avg-arcface30_035_10033-cj05-trainVal2-retrain_pseudo_smlbsm/' #(h, w)

# CUDA_VISIBLE_DEVICES=3 python train.py --config_file='configs/naic/arcface_baseline.yml' \
#     MODEL.WEIGHT '../rep_work_dirs/exp4-cosinebaseline-resnet101ibnls1-384x192-bs16x6-warmup10-flip-pad10-meanstd-erase0502-nolbsm-avg-arcface30_035_10033-cj05-trainVal2/cosine_baseline_epoch90.pth' \
#     SOLVER.BASE_LR '3e-4' SOLVER.WARMUP_EPOCH "5" SOLVER.STEPS "[8, 13]" SOLVER.MAX_EPOCHS "16" SOLVER.START_SAVE_EPOCH "7" SOLVER.EVAL_PERIOD "2"\
#     INPUT.USE_CJ "False" INPUT.CJ_PROB "0.5" INPUT.CJ_BRIGHNESS "0.5" INPUT.CJ_CONTRAST "0.5" INPUT.CJ_SATURATION "0.0" INPUT.CJ_HUE "0.5" \
#     SOLVER.IMS_PER_BATCH "96" DATALOADER.NUM_INSTANCE "6" \
#     MODEL.USE_COS "False" MODEL.BASELINE.COSINE_LOSS_TYPE ''  MODEL.BASELINE.TPL_WEIGHT "1.0" MODEL.BASELINE.CE_WEIGHT "1.0" MODEL.LABEL_SMOOTH "True" \
#     INPUT.SIZE_TRAIN "([384,192])" INPUT.SIZE_TEST "([384,192])" \
#     MODEL.NAME "cosine_baseline" MODEL.BACKBONE "('resnet101_ibn_a')" MODEL.BASELINE.POOL_TYPE "avg"\
#     DATASETS.DATA_PATH "('${DATA_DIR}')" DATASETS.TRAIN_PATH "/data/Dataset/PReID/pseudo_auto_flip_eps055_2_dataset/rep_trainVal2"\
#     MODEL.PRETRAIN_PATH "('${PRETRAIN}')"  \
#     OUTPUT_DIR "('${SAVE_DIR}')" 
# PRETRAIN=../weights/resnet101_ibn_a.pth.tar
# DATA_DIR='/data/Dataset/PReID/rep_dataset/'
# SAVE_DIR='../rep_work_dirs/exp4-cosinebaseline-resnet101ibnls1-384x192-bs16x6-warmup10-flip-pad10-nomeanstd-erase0504-lbsm-avg-arcface30_035_10033-cj05-trainVal2/' #(h, w)
# # SAVE_DIR='../rep_work_dirs/debug/' #(h, w)

# CUDA_VISIBLE_DEVICES=4 python train.py --config_file='configs/naic/arcface_baseline.yml' \
#     SOLVER.IMS_PER_BATCH "96" DATALOADER.NUM_INSTANCE "6" SOLVER.STEPS "[40, 70]" SOLVER.MAX_EPOCHS "90" SOLVER.START_SAVE_EPOCH "75" \
#     INPUT.USE_CJ "True" INPUT.CJ_PROB "0.5" INPUT.CJ_BRIGHNESS "0.5" INPUT.CJ_CONTRAST "0.5" INPUT.CJ_SATURATION "0.0" INPUT.CJ_HUE "0.5" \
#     MODEL.BASELINE.TPL_WEIGHT "1.0" MODEL.BASELINE.CE_WEIGHT "0.33" MODEL.LABEL_SMOOTH "True" \
#     INPUT.RE_MAX_RATIO "0.4" \
#     INPUT.PIXEL_MEAN "[0.485, 0.456, 0.406]" INPUT.PIXEL_STD "[0.229, 0.224, 0.225]" \
#     MODEL.BASELINE.S "30.0" MODEL.BASELINE.M "0.35" \
#     MODEL.BASELINE.COSINE_LOSS_TYPE 'ArcCos' \
#     INPUT.SIZE_TRAIN "([384,192])" INPUT.SIZE_TEST "([384,192])" \
#     MODEL.NAME "cosine_baseline" MODEL.BACKBONE "('resnet101_ibn_a')" MODEL.BASELINE.POOL_TYPE "avg"\
#     DATASETS.DATA_PATH "('${DATA_DIR}')" DATASETS.TRAIN_PATH "rep_trainVal2"\
#     MODEL.PRETRAIN_PATH "('${PRETRAIN}')"  \
#     OUTPUT_DIR "('${SAVE_DIR}')" 


# PRETRAIN=../weights/resnet101_ibn_a.pth.tar
# DATA_DIR='/data/Dataset/PReID/rep_dataset/'
# SAVE_DIR='../rep_work_dirs/exp4-cosinebaseline-resnet101ibnls1-384x192-bs16x6-warmup10-flip-pad10-nomeanstd-erase0504-nolbsm-avg-arcface30_035_10033-cj05-trainVal2/' #(h, w)
# # SAVE_DIR='../rep_work_dirs/debug/' #(h, w)

# CUDA_VISIBLE_DEVICES=5 python train.py --config_file='configs/naic/arcface_baseline.yml' \
#     SOLVER.IMS_PER_BATCH "96" DATALOADER.NUM_INSTANCE "6" SOLVER.STEPS "[40, 70]" SOLVER.MAX_EPOCHS "90" SOLVER.START_SAVE_EPOCH "75" \
#     INPUT.USE_CJ "True" INPUT.CJ_PROB "0.5" INPUT.CJ_BRIGHNESS "0.5" INPUT.CJ_CONTRAST "0.5" INPUT.CJ_SATURATION "0.0" INPUT.CJ_HUE "0.5" \
#     MODEL.BASELINE.TPL_WEIGHT "1.0" MODEL.BASELINE.CE_WEIGHT "0.33" MODEL.LABEL_SMOOTH "False" \
#     INPUT.RE_MAX_RATIO "0.4" \
#     INPUT.PIXEL_MEAN "[0.485, 0.456, 0.406]" INPUT.PIXEL_STD "[0.229, 0.224, 0.225]" \
#     MODEL.BASELINE.S "30.0" MODEL.BASELINE.M "0.35" \
#     MODEL.BASELINE.COSINE_LOSS_TYPE 'ArcCos' \
#     INPUT.SIZE_TRAIN "([384,192])" INPUT.SIZE_TEST "([384,192])" \
#     MODEL.NAME "cosine_baseline" MODEL.BACKBONE "('resnet101_ibn_a')" MODEL.BASELINE.POOL_TYPE "avg"\
#     DATASETS.DATA_PATH "('${DATA_DIR}')" DATASETS.TRAIN_PATH "rep_trainVal2"\
#     MODEL.PRETRAIN_PATH "('${PRETRAIN}')"  \
#     OUTPUT_DIR "('${SAVE_DIR}')" 

# PRETRAIN=../weights/resnet101_ibn_a.pth.tar
# DATA_DIR='/data/Dataset/PReID/rep_dataset/'
# SAVE_DIR='../rep_work_dirs/exp4-cosinebaseline-resnet101ibnls1-384x192-bs16x6-warmup10-flip-pad10-meanstd-erase0502-nolbsm-avg-arcface30_035_10033-cj05-trainVal2-finetune_tpl05/' #(h, w)
# # SAVE_DIR='../rep_work_dirs/debug/' #(h, w)

# CUDA_VISIBLE_DEVICES=2 python train.py --config_file='configs/naic/arcface_baseline.yml' \
#     MODEL.WEIGHT '../rep_work_dirs/exp4-cosinebaseline-resnet101ibnls1-384x192-bs16x6-warmup10-flip-pad10-meanstd-erase0502-nolbsm-avg-arcface30_035_10033-cj05-trainVal2/cosine_baseline_epoch90.pth' \
#     SOLVER.BASE_LR '1e-4' SOLVER.WARMUP_EPOCH "5" SOLVER.STEPS "[8, 13]" SOLVER.MAX_EPOCHS "16" SOLVER.START_SAVE_EPOCH "7" SOLVER.EVAL_PERIOD "2"\
#     SOLVER.IMS_PER_BATCH "96" DATALOADER.NUM_INSTANCE "6" \
#     INPUT.USE_CJ "True" INPUT.CJ_PROB "0.5" INPUT.CJ_BRIGHNESS "0.5" INPUT.CJ_CONTRAST "0.5" INPUT.CJ_SATURATION "0.0" INPUT.CJ_HUE "0.5" \
#     MODEL.BASELINE.TPL_WEIGHT "1.0" MODEL.BASELINE.CE_WEIGHT "0.33" MODEL.LABEL_SMOOTH "False" SOLVER.MARGIN "0.5" \
#     MODEL.BASELINE.S "30.0" MODEL.BASELINE.M "0.35" \
#     MODEL.BASELINE.COSINE_LOSS_TYPE 'ArcCos' \
#     INPUT.SIZE_TRAIN "([384,192])" INPUT.SIZE_TEST "([384,192])" \
#     MODEL.NAME "cosine_baseline" MODEL.BACKBONE "('resnet101_ibn_a')" MODEL.BASELINE.POOL_TYPE "avg"\
#     DATASETS.DATA_PATH "('${DATA_DIR}')" DATASETS.TRAIN_PATH "rep_trainVal2"\
#     MODEL.PRETRAIN_PATH "('${PRETRAIN}')"  \
#     OUTPUT_DIR "('${SAVE_DIR}')" 

# PRETRAIN=../weights/resnet101_ibn_a.pth.tar
# DATA_DIR='/data/Dataset/PReID/rep_dataset/'
# SAVE_DIR='../rep_work_dirs/exp4-cosinebaseline-resnet101ibnls1-384x192-bs16x6-warmup10-flip-pad10-meanstd-erase0502-nolbsm-avg-arcface30_035_10033-cj05-trainVal2-pseudo_resume/' #(h, w)

# CUDA_VISIBLE_DEVICES=3 python train.py --config_file='configs/naic/arcface_baseline.yml' \
#     MODEL.WEIGHT '../rep_work_dirs/exp4-cosinebaseline-resnet101ibnls1-384x192-bs16x6-warmup10-flip-pad10-meanstd-erase0502-nolbsm-avg-arcface30_035_10033-cj05-trainVal2/cosine_baseline_epoch90.pth' \
#     SOLVER.BASE_LR '1e-4' SOLVER.WARMUP_EPOCH "10" SOLVER.STEPS "[25, 45]" SOLVER.MAX_EPOCHS "60" SOLVER.START_SAVE_EPOCH "30" SOLVER.EVAL_PERIOD "2"\
#     INPUT.USE_CJ "True" INPUT.CJ_PROB "0.5" INPUT.CJ_BRIGHNESS "0.5" INPUT.CJ_CONTRAST "0.5" INPUT.CJ_SATURATION "0.0" INPUT.CJ_HUE "0.5" \
#     SOLVER.IMS_PER_BATCH "96" DATALOADER.NUM_INSTANCE "6" \
#     MODEL.BASELINE.TPL_WEIGHT "1.0" MODEL.BASELINE.CE_WEIGHT "0.33" MODEL.LABEL_SMOOTH "False" \
#     MODEL.BASELINE.S "30.0" MODEL.BASELINE.M "0.35" \
#     MODEL.BASELINE.COSINE_LOSS_TYPE 'ArcCos' \
#     INPUT.SIZE_TRAIN "([384,192])" INPUT.SIZE_TEST "([384,192])" \
#     MODEL.NAME "cosine_baseline" MODEL.BACKBONE "('resnet101_ibn_a')" MODEL.BASELINE.POOL_TYPE "avg"\
#     DATASETS.DATA_PATH "('${DATA_DIR}')" DATASETS.TRAIN_PATH "/data/Dataset/PReID/pseudo_auto_flip_eps055_2_dataset/rep_trainVal2"\
#     MODEL.PRETRAIN_PATH "('${PRETRAIN}')"  \
#     OUTPUT_DIR "('${SAVE_DIR}')" 

# PRETRAIN=../weights/resnet101_ibn_a.pth.tar
# DATA_DIR='/data/Dataset/PReID/rep_dataset/'
# SAVE_DIR='../rep_work_dirs/exp4-cosinebaseline-resnet101ibnls1-384x192-bs16x6-warmup10-flip-pad10-meanstd-erase0502-nolbsm-avg-arcface30_035_10033-cj05-trainVal2-pseudo_resume_smlbsm/' #(h, w)

# CUDA_VISIBLE_DEVICES=4 python train.py --config_file='configs/naic/arcface_baseline.yml' \
#     MODEL.WEIGHT '../rep_work_dirs/exp4-cosinebaseline-resnet101ibnls1-384x192-bs16x6-warmup10-flip-pad10-meanstd-erase0502-nolbsm-avg-arcface30_035_10033-cj05-trainVal2/cosine_baseline_epoch90.pth' \
#     SOLVER.BASE_LR '1e-4' SOLVER.WARMUP_EPOCH "10" SOLVER.STEPS "[25, 45]" SOLVER.MAX_EPOCHS "60" SOLVER.START_SAVE_EPOCH "30" SOLVER.EVAL_PERIOD "2"\
#     INPUT.USE_CJ "True" INPUT.CJ_PROB "0.5" INPUT.CJ_BRIGHNESS "0.5" INPUT.CJ_CONTRAST "0.5" INPUT.CJ_SATURATION "0.0" INPUT.CJ_HUE "0.5" \
#     SOLVER.IMS_PER_BATCH "96" DATALOADER.NUM_INSTANCE "6" \
#     MODEL.USE_COS "False" MODEL.BASELINE.COSINE_LOSS_TYPE ''  MODEL.BASELINE.TPL_WEIGHT "1.0" MODEL.BASELINE.CE_WEIGHT "1.0" MODEL.LABEL_SMOOTH "True" \
#     INPUT.SIZE_TRAIN "([384,192])" INPUT.SIZE_TEST "([384,192])" \
#     MODEL.NAME "cosine_baseline" MODEL.BACKBONE "('resnet101_ibn_a')" MODEL.BASELINE.POOL_TYPE "avg"\
#     DATASETS.DATA_PATH "('${DATA_DIR}')" DATASETS.TRAIN_PATH "/data/Dataset/PReID/pseudo_auto_flip_eps055_2_dataset/rep_trainVal2"\
#     MODEL.PRETRAIN_PATH "('${PRETRAIN}')"  \
#     OUTPUT_DIR "('${SAVE_DIR}')" 

# PRETRAIN=../weights/resnet101_ibn_a.pth.tar
# DATA_DIR='/data/Dataset/PReID/rep_dataset/'
# SAVE_DIR='../rep_work_dirs/exp4-cosinebaseline-resnet101ibnls1-384x192-bs16x6-warmup10-flip-pad10-meanstd-erase0502-nolbsm-avg-arcface30_035_10033-cj05-trainVal2-pseudo_retrain_smlbsm/' #(h, w)

# CUDA_VISIBLE_DEVICES=4 python train.py --config_file='configs/naic/arcface_baseline.yml' \
#     SOLVER.BASE_LR '1e-4' SOLVER.WARMUP_EPOCH "10" SOLVER.STEPS "[25, 45]" SOLVER.MAX_EPOCHS "60" SOLVER.START_SAVE_EPOCH "30" SOLVER.EVAL_PERIOD "2"\
#     INPUT.USE_CJ "True" INPUT.CJ_PROB "0.5" INPUT.CJ_BRIGHNESS "0.5" INPUT.CJ_CONTRAST "0.5" INPUT.CJ_SATURATION "0.0" INPUT.CJ_HUE "0.5" \
#     SOLVER.IMS_PER_BATCH "96" DATALOADER.NUM_INSTANCE "6" \
#     MODEL.USE_COS "False" MODEL.BASELINE.COSINE_LOSS_TYPE ''  MODEL.BASELINE.TPL_WEIGHT "1.0" MODEL.BASELINE.CE_WEIGHT "1.0" MODEL.LABEL_SMOOTH "True" \
#     INPUT.SIZE_TRAIN "([384,192])" INPUT.SIZE_TEST "([384,192])" \
#     MODEL.NAME "cosine_baseline" MODEL.BACKBONE "('resnet101_ibn_a')" MODEL.BASELINE.POOL_TYPE "avg"\
#     DATASETS.DATA_PATH "('${DATA_DIR}')" DATASETS.TRAIN_PATH "/data/Dataset/PReID/pseudo_auto_flip_eps055_2_dataset/rep_trainVal2"\
#     MODEL.PRETRAIN_PATH "('${PRETRAIN}')"  \
#     OUTPUT_DIR "('${SAVE_DIR}')" 


# PRETRAIN=../weights/resnet101_ibn_a.pth.tar
# DATA_DIR='/data/Dataset/PReID/rep_dataset/'
# SAVE_DIR='../rep_work_dirs/exp4-cosinebaseline-resnet101ibnls1-384x192-bs16x6-warmup10-flip-pad10-meanstd-erase0502-nolbsm-avg-arcface30_035_10033-cj05-trainVal2-testb_pseudo_retrain_smlbsm/' #(h, w)

# CUDA_VISIBLE_DEVICES=4 python train.py --config_file='configs/naic/arcface_baseline.yml' \
#     SOLVER.BASE_LR '3e-4' SOLVER.WARMUP_EPOCH "6" SOLVER.STEPS "[30, 50]" SOLVER.MAX_EPOCHS "60" SOLVER.START_SAVE_EPOCH "40" SOLVER.EVAL_PERIOD "2"\
#     SOLVER.IMS_PER_BATCH "96" DATALOADER.NUM_INSTANCE "6" \
#     MODEL.USE_COS "False" MODEL.BASELINE.COSINE_LOSS_TYPE ''  MODEL.BASELINE.TPL_WEIGHT "1.0" MODEL.BASELINE.CE_WEIGHT "1.0" MODEL.LABEL_SMOOTH "True" \
#     INPUT.SIZE_TRAIN "([384,192])" INPUT.SIZE_TEST "([384,192])" \
#     MODEL.NAME "cosine_baseline" MODEL.BACKBONE "('resnet101_ibn_a')" MODEL.BASELINE.POOL_TYPE "avg"\
#     DATASETS.DATA_PATH "('${DATA_DIR}')" DATASETS.TRAIN_PATH "/data/Dataset/PReID/testb_pseudo_hist_065_080_dataset/rep_trainVal2"\
#     MODEL.PRETRAIN_PATH "('${PRETRAIN}')"  \
#     OUTPUT_DIR "('${SAVE_DIR}')" 


# PRETRAIN=../weights/resnet101_ibn_a.pth.tar
# DATA_DIR='/data/Dataset/PReID/rep_dataset/'
# SAVE_DIR='../rep_work_dirs/exp4-cosinebaseline-resnet101ibnls1-384x192-bs16x6-warmup10-flip-pad10-meanstd-erase0502-nolbsm-avg-arcface30_035_10033-cj05-trainVal2-testb_pseudo_retrain_smlbsm_cj05/' #(h, w)

# CUDA_VISIBLE_DEVICES=6 python train.py --config_file='configs/naic/arcface_baseline.yml' \
#     SOLVER.BASE_LR '3e-4' SOLVER.WARMUP_EPOCH "6" SOLVER.STEPS "[30, 50]" SOLVER.MAX_EPOCHS "60" SOLVER.START_SAVE_EPOCH "40" SOLVER.EVAL_PERIOD "2"\
#     SOLVER.IMS_PER_BATCH "96" DATALOADER.NUM_INSTANCE "6" \
#     INPUT.USE_CJ "True" INPUT.CJ_PROB "0.5" INPUT.CJ_BRIGHNESS "0.5" INPUT.CJ_CONTRAST "0.5" INPUT.CJ_SATURATION "0.0" INPUT.CJ_HUE "0.5" \
#     MODEL.USE_COS "False" MODEL.BASELINE.COSINE_LOSS_TYPE ''  MODEL.BASELINE.TPL_WEIGHT "1.0" MODEL.BASELINE.CE_WEIGHT "1.0" MODEL.LABEL_SMOOTH "True" \
#     INPUT.SIZE_TRAIN "([384,192])" INPUT.SIZE_TEST "([384,192])" \
#     MODEL.NAME "cosine_baseline" MODEL.BACKBONE "('resnet101_ibn_a')" MODEL.BASELINE.POOL_TYPE "avg"\
#     DATASETS.DATA_PATH "('${DATA_DIR}')" DATASETS.TRAIN_PATH "/data/Dataset/PReID/testb_pseudo_hist_065_080_dataset/rep_trainVal2"\
#     MODEL.PRETRAIN_PATH "('${PRETRAIN}')"  \
#     OUTPUT_DIR "('${SAVE_DIR}')" 



# PRETRAIN=../weights/resnet101_ibn_a.pth.tar
# DATA_DIR='/data/Dataset/PReID/rep_dataset/'
# SAVE_DIR='../rep_work_dirs/exp4-cosinebaseline-resnet101ibnls1-384x192-bs16x6-warmup10-flip-pad10-meanstd-erase0502-nolbsm-avg-arcface30_035_10033-cj05-trainVal2-testb_pseudo_retrain_arface/' #(h, w)

# CUDA_VISIBLE_DEVICES=5 python train.py --config_file='configs/naic/arcface_baseline.yml' \
#     SOLVER.BASE_LR '3e-4' SOLVER.WARMUP_EPOCH "6" SOLVER.STEPS "[30, 50]" SOLVER.MAX_EPOCHS "60" SOLVER.START_SAVE_EPOCH "40" SOLVER.EVAL_PERIOD "2"\
#     SOLVER.IMS_PER_BATCH "96" DATALOADER.NUM_INSTANCE "6" \
#     MODEL.BASELINE.TPL_WEIGHT "1.0" MODEL.BASELINE.CE_WEIGHT "0.33" MODEL.LABEL_SMOOTH "False" MODEL.BASELINE.S "30.0" MODEL.BASELINE.M "0.35" MODEL.BASELINE.COSINE_LOSS_TYPE 'ArcCos' \
#     INPUT.SIZE_TRAIN "([384,192])" INPUT.SIZE_TEST "([384,192])" \
#     MODEL.NAME "cosine_baseline" MODEL.BACKBONE "('resnet101_ibn_a')" MODEL.BASELINE.POOL_TYPE "avg"\
#     DATASETS.DATA_PATH "('${DATA_DIR}')" DATASETS.TRAIN_PATH "/data/Dataset/PReID/testb_pseudo_hist_065_080_dataset/rep_trainVal2"\
#     MODEL.PRETRAIN_PATH "('${PRETRAIN}')"  \
#     OUTPUT_DIR "('${SAVE_DIR}')" 

# PRETRAIN=../weights/resnet101_ibn_a.pth.tar
# DATA_DIR='/data/Dataset/PReID/rep_dataset/'
# SAVE_DIR='../rep_work_dirs/exp4-cosinebaseline-resnet101ibnls1-256x128-bs16x6-warmup10-flip-pad10-meanstd-erase0502-nolbsm-avg-arcface30_035_10033-cj05-trainVal2-testb_pseudo_retrain_smlbsm/' #(h, w)

# CUDA_VISIBLE_DEVICES=2 python train.py --config_file='configs/naic/arcface_baseline.yml' \
#     SOLVER.BASE_LR '3e-4' SOLVER.WARMUP_EPOCH "6" SOLVER.STEPS "[30, 50]" SOLVER.MAX_EPOCHS "60" SOLVER.START_SAVE_EPOCH "40" SOLVER.EVAL_PERIOD "2"\
#     SOLVER.IMS_PER_BATCH "96" DATALOADER.NUM_INSTANCE "6" \
#     MODEL.USE_COS "False" MODEL.BASELINE.COSINE_LOSS_TYPE ''  MODEL.BASELINE.TPL_WEIGHT "1.0" MODEL.BASELINE.CE_WEIGHT "1.0" MODEL.LABEL_SMOOTH "True" \
#     INPUT.SIZE_TRAIN "([256,128])" INPUT.SIZE_TEST "([256,128])" \
#     MODEL.NAME "cosine_baseline" MODEL.BACKBONE "('resnet101_ibn_a')" MODEL.BASELINE.POOL_TYPE "avg"\
#     DATASETS.DATA_PATH "('${DATA_DIR}')" DATASETS.TRAIN_PATH "/data/Dataset/PReID/testb_pseudo_hist_065_080_dataset/rep_trainVal2"\
#     MODEL.PRETRAIN_PATH "('${PRETRAIN}')"  \
#     OUTPUT_DIR "('${SAVE_DIR}')" 


# PRETRAIN=../weights/resnet101_ibn_a.pth.tar
# DATA_DIR='/data/Dataset/PReID/rep_dataset/'
# SAVE_DIR='../rep_work_dirs/exp4-cosinebaseline-resnet101ibnls1-256x128-bs16x6-warmup10-flip-pad10-meanstd-erase0502-nolbsm-avg-arcface30_035_10033-cj05-trainVal2-testb_pseudo_retrain_smlbsm_cj05/' #(h, w)

# CUDA_VISIBLE_DEVICES=0 python train.py --config_file='configs/naic/arcface_baseline.yml' \
#     SOLVER.BASE_LR '3e-4' SOLVER.WARMUP_EPOCH "6" SOLVER.STEPS "[30, 50]" SOLVER.MAX_EPOCHS "60" SOLVER.START_SAVE_EPOCH "40" SOLVER.EVAL_PERIOD "2"\
#     SOLVER.IMS_PER_BATCH "96" DATALOADER.NUM_INSTANCE "6" \
#     INPUT.USE_CJ "True" INPUT.CJ_PROB "0.5" INPUT.CJ_BRIGHNESS "0.5" INPUT.CJ_CONTRAST "0.5" INPUT.CJ_SATURATION "0.0" INPUT.CJ_HUE "0.5" \
#     MODEL.USE_COS "False" MODEL.BASELINE.COSINE_LOSS_TYPE ''  MODEL.BASELINE.TPL_WEIGHT "1.0" MODEL.BASELINE.CE_WEIGHT "1.0" MODEL.LABEL_SMOOTH "True" \
#     INPUT.SIZE_TRAIN "([256,128])" INPUT.SIZE_TEST "([256,128])" \
#     MODEL.NAME "cosine_baseline" MODEL.BACKBONE "('resnet101_ibn_a')" MODEL.BASELINE.POOL_TYPE "avg"\
#     DATASETS.DATA_PATH "('${DATA_DIR}')" DATASETS.TRAIN_PATH "/data/Dataset/PReID/testb_pseudo_hist_065_080_dataset/rep_trainVal2"\
#     MODEL.PRETRAIN_PATH "('${PRETRAIN}')"  \
#     OUTPUT_DIR "('${SAVE_DIR}')" 



# PRETRAIN=../weights/resnet101_ibn_a.pth.tar
# DATA_DIR='/data/Dataset/PReID/rep_dataset/'
# SAVE_DIR='../rep_work_dirs/exp4-cosinebaseline-resnet101ibnls1-384x192-bs16x6-warmup10-flip-pad10-meanstd-erase0502-nolbsm-avg-arcface30_035_10033-cj05-trainVal2-testb_pseudo_retrain_smlbsm-fp16/' #(h, w)

# CUDA_VISIBLE_DEVICES=3 python train.py --config_file='configs/naic/arcface_baseline.yml' \
#     MODEL.OPT_LEVEL "O1" \
#     SOLVER.BASE_LR '3e-4' SOLVER.WARMUP_EPOCH "8" SOLVER.STEPS "[35, 60]" SOLVER.MAX_EPOCHS "70" SOLVER.START_SAVE_EPOCH "50" SOLVER.EVAL_PERIOD "2"\
#     SOLVER.IMS_PER_BATCH "96" DATALOADER.NUM_INSTANCE "6" \
#     MODEL.USE_COS "False" MODEL.BASELINE.COSINE_LOSS_TYPE ''  MODEL.BASELINE.TPL_WEIGHT "1.0" MODEL.BASELINE.CE_WEIGHT "1.0" MODEL.LABEL_SMOOTH "True" \
#     INPUT.SIZE_TRAIN "([384,192])" INPUT.SIZE_TEST "([384,192])" \
#     MODEL.NAME "cosine_baseline" MODEL.BACKBONE "('resnet101_ibn_a')" MODEL.BASELINE.POOL_TYPE "avg"\
#     DATASETS.DATA_PATH "('${DATA_DIR}')" DATASETS.TRAIN_PATH "/data/Dataset/PReID/testb_pseudo_hist_065_080_dataset/rep_trainVal2"\
#     MODEL.PRETRAIN_PATH "('${PRETRAIN}')"  \
#     OUTPUT_DIR "('${SAVE_DIR}')" 


# PRETRAIN=../weights/resnet101_ibn_a.pth.tar
# DATA_DIR='/data/Dataset/PReID/rep_dataset/'
# SAVE_DIR='../rep_work_dirs/exp4-cosinebaseline-resnet101ibnls1-384x192-bs16x6-warmup10-flip-pad10-meanstd-erase0502-nolbsm-avg-arcface30_035_10033-cj05-trainVal2-testb_pseudo_retrain_smlbsm_cj05-fp16/' #(h, w)
# # which python
# CUDA_VISIBLE_DEVICES=7 python train.py --config_file='configs/naic/arcface_baseline.yml' \
#     MODEL.OPT_LEVEL "O1" \
#     SOLVER.BASE_LR '3e-4' SOLVER.WARMUP_EPOCH "8" SOLVER.STEPS "[35, 60]" SOLVER.MAX_EPOCHS "70" SOLVER.START_SAVE_EPOCH "50" SOLVER.EVAL_PERIOD "2"\
#     SOLVER.IMS_PER_BATCH "96" DATALOADER.NUM_INSTANCE "6" \
#     INPUT.USE_CJ "True" INPUT.CJ_PROB "0.5" INPUT.CJ_BRIGHNESS "0.5" INPUT.CJ_CONTRAST "0.5" INPUT.CJ_SATURATION "0.0" INPUT.CJ_HUE "0.5" \
#     MODEL.USE_COS "False" MODEL.BASELINE.COSINE_LOSS_TYPE ''  MODEL.BASELINE.TPL_WEIGHT "1.0" MODEL.BASELINE.CE_WEIGHT "1.0" MODEL.LABEL_SMOOTH "True" \
#     INPUT.SIZE_TRAIN "([384,192])" INPUT.SIZE_TEST "([384,192])" \
#     MODEL.NAME "cosine_baseline" MODEL.BACKBONE "('resnet101_ibn_a')" MODEL.BASELINE.POOL_TYPE "avg"\
#     DATASETS.DATA_PATH "('${DATA_DIR}')" DATASETS.TRAIN_PATH "/data/Dataset/PReID/testb_pseudo_hist_065_080_dataset/rep_trainVal2"\
#     MODEL.PRETRAIN_PATH "('${PRETRAIN}')"  \
#     OUTPUT_DIR "('${SAVE_DIR}')" 


# PRETRAIN=../weights/resnet101_ibn_a.pth.tar
# DATA_DIR='/data/Dataset/PReID/rep_dataset/'
# SAVE_DIR='../rep_work_dirs/exp4-cosinebaseline-resnet101ibnls1-256x128-bs16x6-warmup10-flip-pad10-meanstd-erase0502-nolbsm-avg-arcface30_035_10033-cj05-trainVal2-testb_pseudo_retrain_arcface/' #(h, w)

# CUDA_VISIBLE_DEVICES=2 python train.py --config_file='configs/naic/arcface_baseline.yml' \
#     SOLVER.BASE_LR '3e-4' SOLVER.WARMUP_EPOCH "8" SOLVER.STEPS "[35, 60]" SOLVER.MAX_EPOCHS "70" SOLVER.START_SAVE_EPOCH "50" SOLVER.EVAL_PERIOD "2"\
#     SOLVER.IMS_PER_BATCH "96" DATALOADER.NUM_INSTANCE "6" \
#     MODEL.BASELINE.TPL_WEIGHT "1.0" MODEL.BASELINE.CE_WEIGHT "0.33" MODEL.LABEL_SMOOTH "False" MODEL.BASELINE.S "30.0" MODEL.BASELINE.M "0.35" MODEL.BASELINE.COSINE_LOSS_TYPE 'ArcCos' \
#     INPUT.SIZE_TRAIN "([256,128])" INPUT.SIZE_TEST "([256,128])" \
#     MODEL.NAME "cosine_baseline" MODEL.BACKBONE "('resnet101_ibn_a')" MODEL.BASELINE.POOL_TYPE "avg"\
#     DATASETS.DATA_PATH "('${DATA_DIR}')" DATASETS.TRAIN_PATH "/data/Dataset/PReID/testb_pseudo_hist_065_080_dataset/rep_trainVal2"\
#     MODEL.PRETRAIN_PATH "('${PRETRAIN}')"  \
#     OUTPUT_DIR "('${SAVE_DIR}')" 


# PRETRAIN=../weights/resnet101_ibn_a.pth.tar
# DATA_DIR='/data/Dataset/PReID/rep_dataset/'
# SAVE_DIR='../rep_work_dirs/exp4-cosinebaseline-resnet101ibnls1-256x128-bs16x6-warmup10-flip-pad10-meanstd-erase0502-nolbsm-avg-arcface30_035_10033-cj05-trainVal2-testb_pseudo_retrain_arcface_cj05/' #(h, w)

# CUDA_VISIBLE_DEVICES=0 python train.py --config_file='configs/naic/arcface_baseline.yml' \
#     SOLVER.BASE_LR '3e-4' SOLVER.WARMUP_EPOCH "8" SOLVER.STEPS "[35, 60]" SOLVER.MAX_EPOCHS "70" SOLVER.START_SAVE_EPOCH "50" SOLVER.EVAL_PERIOD "2"\
#     SOLVER.IMS_PER_BATCH "96" DATALOADER.NUM_INSTANCE "6" \
#     INPUT.USE_CJ "True" INPUT.CJ_PROB "0.5" INPUT.CJ_BRIGHNESS "0.5" INPUT.CJ_CONTRAST "0.5" INPUT.CJ_SATURATION "0.0" INPUT.CJ_HUE "0.5" \
#     MODEL.BASELINE.TPL_WEIGHT "1.0" MODEL.BASELINE.CE_WEIGHT "0.33" MODEL.LABEL_SMOOTH "False" MODEL.BASELINE.S "30.0" MODEL.BASELINE.M "0.35" MODEL.BASELINE.COSINE_LOSS_TYPE 'ArcCos' \
#     INPUT.SIZE_TRAIN "([256,128])" INPUT.SIZE_TEST "([256,128])" \
#     MODEL.NAME "cosine_baseline" MODEL.BACKBONE "('resnet101_ibn_a')" MODEL.BASELINE.POOL_TYPE "avg"\
#     DATASETS.DATA_PATH "('${DATA_DIR}')" DATASETS.TRAIN_PATH "/data/Dataset/PReID/testb_pseudo_hist_065_080_dataset/rep_trainVal2"\
#     MODEL.PRETRAIN_PATH "('${PRETRAIN}')"  \
#     OUTPUT_DIR "('${SAVE_DIR}')" 



# [todo] fp32 for arcface
# PRETRAIN=../weights/resnet101_ibn_a.pth.tar
# DATA_DIR='/data/Dataset/PReID/rep_dataset/'
# SAVE_DIR='../rep_work_dirs/exp4-cosinebaseline-resnet101ibnls1-384x192-bs16x6-warmup10-flip-pad10-meanstd-erase0502-nolbsm-avg-arcface30_035_10033-cj05-trainVal2-testb_pseudo_retrain_arcface-fp16/' #(h, w)

# CUDA_VISIBLE_DEVICES=3 python train.py --config_file='configs/naic/arcface_baseline.yml' \
#     MODEL.OPT_LEVEL "O1" \
#     SOLVER.BASE_LR '3e-4' SOLVER.WARMUP_EPOCH "8" SOLVER.STEPS "[35, 58]" SOLVER.MAX_EPOCHS "66" SOLVER.START_SAVE_EPOCH "50" SOLVER.EVAL_PERIOD "2"\
#     SOLVER.IMS_PER_BATCH "96" DATALOADER.NUM_INSTANCE "6" \
#     MODEL.BASELINE.TPL_WEIGHT "1.0" MODEL.BASELINE.CE_WEIGHT "0.33" MODEL.LABEL_SMOOTH "False" MODEL.BASELINE.S "30.0" MODEL.BASELINE.M "0.35" MODEL.BASELINE.COSINE_LOSS_TYPE 'ArcCos' \
#     INPUT.SIZE_TRAIN "([384,192])" INPUT.SIZE_TEST "([384,192])" \
#     MODEL.NAME "cosine_baseline" MODEL.BACKBONE "('resnet101_ibn_a')" MODEL.BASELINE.POOL_TYPE "avg"\
#     DATASETS.DATA_PATH "('${DATA_DIR}')" DATASETS.TRAIN_PATH "/data/Dataset/PReID/testb_pseudo_hist_065_080_dataset/rep_trainVal2"\
#     MODEL.PRETRAIN_PATH "('${PRETRAIN}')"  \
#     OUTPUT_DIR "('${SAVE_DIR}')" 

# PRETRAIN=../weights/resnet101_ibn_a.pth.tar
# DATA_DIR='/data/Dataset/PReID/rep_dataset/'
# SAVE_DIR='../rep_work_dirs/exp4-cosinebaseline-resnet101ibnls1-384x192-bs16x6-warmup10-flip-pad10-meanstd-erase0502-nolbsm-avg-arcface30_035_10033-cj05-trainVal2-testb_pseudo_retrain_arcface_cj05-fp16/' #(h, w)

# CUDA_VISIBLE_DEVICES=6 python train.py --config_file='configs/naic/arcface_baseline.yml' \
#     MODEL.OPT_LEVEL "O1" \
#     SOLVER.BASE_LR '3e-4' SOLVER.WARMUP_EPOCH "8" SOLVER.STEPS "[35, 58]" SOLVER.MAX_EPOCHS "66" SOLVER.START_SAVE_EPOCH "50" SOLVER.EVAL_PERIOD "2"\
#     INPUT.USE_CJ "True" INPUT.CJ_PROB "0.5" INPUT.CJ_BRIGHNESS "0.5" INPUT.CJ_CONTRAST "0.5" INPUT.CJ_SATURATION "0.0" INPUT.CJ_HUE "0.5" \
#     SOLVER.IMS_PER_BATCH "96" DATALOADER.NUM_INSTANCE "6" \
#     MODEL.BASELINE.TPL_WEIGHT "1.0" MODEL.BASELINE.CE_WEIGHT "0.33" MODEL.LABEL_SMOOTH "False" MODEL.BASELINE.S "30.0" MODEL.BASELINE.M "0.35" MODEL.BASELINE.COSINE_LOSS_TYPE 'ArcCos' \
#     INPUT.SIZE_TRAIN "([384,192])" INPUT.SIZE_TEST "([384,192])" \
#     MODEL.NAME "cosine_baseline" MODEL.BACKBONE "('resnet101_ibn_a')" MODEL.BASELINE.POOL_TYPE "avg"\
#     DATASETS.DATA_PATH "('${DATA_DIR}')" DATASETS.TRAIN_PATH "/data/Dataset/PReID/testb_pseudo_hist_065_080_dataset/rep_trainVal2"\
#     MODEL.PRETRAIN_PATH "('${PRETRAIN}')"  \
#     OUTPUT_DIR "('${SAVE_DIR}')" 



# PRETRAIN=../weights/resnet101_ibn_a.pth.tar
# DATA_DIR='/data/Dataset/PReID/rep_dataset/'
# SAVE_DIR='../rep_work_dirs/exp4-cosinebaseline-resnet101ibnls1-384x192-bs16x6-warmup10-flip-pad10-meanstd-erase0502-nolbsm-avg-arcface30_035_10033-cj05-trainVal2-testb_pseudo_retrain_smlbsm_e66/' #(h, w)

# CUDA_VISIBLE_DEVICES=4 python train.py --config_file='configs/naic/arcface_baseline.yml' \
#     SOLVER.BASE_LR '3e-4' SOLVER.WARMUP_EPOCH "8" SOLVER.STEPS "[35, 55]" SOLVER.MAX_EPOCHS "66" SOLVER.START_SAVE_EPOCH "50" SOLVER.EVAL_PERIOD "2" \
#     SOLVER.IMS_PER_BATCH "96" DATALOADER.NUM_INSTANCE "6" \
#     MODEL.USE_COS "False" MODEL.BASELINE.COSINE_LOSS_TYPE ''  MODEL.BASELINE.TPL_WEIGHT "1.0" MODEL.BASELINE.CE_WEIGHT "1.0" MODEL.LABEL_SMOOTH "True" \
#     INPUT.SIZE_TRAIN "([384,192])" INPUT.SIZE_TEST "([384,192])" \
#     MODEL.NAME "cosine_baseline" MODEL.BACKBONE "('resnet101_ibn_a')" MODEL.BASELINE.POOL_TYPE "avg"\
#     DATASETS.DATA_PATH "('${DATA_DIR}')" DATASETS.TRAIN_PATH "/data/Dataset/PReID/testb_pseudo_hist_065_080_dataset/rep_trainVal2"\
#     MODEL.PRETRAIN_PATH "('${PRETRAIN}')"  \
#     OUTPUT_DIR "('${SAVE_DIR}')" 


# PRETRAIN=../weights/resnet101_ibn_a.pth.tar
# DATA_DIR='/data/Dataset/PReID/rep_dataset/'
# SAVE_DIR='../rep_work_dirs/exp4-cosinebaseline-resnet101ibnls1-384x192-bs16x6-warmup10-flip-pad10-meanstd-erase0502-nolbsm-avg-arcface30_035_10033-cj05-trainVal2-testb_pseudo_retrain_arface_e66/' #(h, w)

# CUDA_VISIBLE_DEVICES=5 python train.py --config_file='configs/naic/arcface_baseline.yml' \
#     SOLVER.BASE_LR '3e-4' SOLVER.WARMUP_EPOCH "8" SOLVER.STEPS "[35, 55]" SOLVER.MAX_EPOCHS "66" SOLVER.START_SAVE_EPOCH "50" SOLVER.EVAL_PERIOD "2" \
#     SOLVER.IMS_PER_BATCH "96" DATALOADER.NUM_INSTANCE "6" \
#     MODEL.BASELINE.TPL_WEIGHT "1.0" MODEL.BASELINE.CE_WEIGHT "0.33" MODEL.LABEL_SMOOTH "False" MODEL.BASELINE.S "30.0" MODEL.BASELINE.M "0.35" MODEL.BASELINE.COSINE_LOSS_TYPE 'ArcCos' \
#     INPUT.SIZE_TRAIN "([384,192])" INPUT.SIZE_TEST "([384,192])" \
#     MODEL.NAME "cosine_baseline" MODEL.BACKBONE "('resnet101_ibn_a')" MODEL.BASELINE.POOL_TYPE "avg"\
#     DATASETS.DATA_PATH "('${DATA_DIR}')" DATASETS.TRAIN_PATH "/data/Dataset/PReID/testb_pseudo_hist_065_080_dataset/rep_trainVal2"\
#     MODEL.PRETRAIN_PATH "('${PRETRAIN}')"  \
#     OUTPUT_DIR "('${SAVE_DIR}')" 

# PRETRAIN=../weights/resnet101_ibn_a.pth.tar
# DATA_DIR='/data/Dataset/PReID/rep_dataset/'
# SAVE_DIR='../rep_work_dirs/debug/' #(h, w)

# CUDA_VISIBLE_DEVICES=3 python train.py --config_file='configs/naic/arcface_baseline.yml' \
#     SOLVER.BASE_LR '3e-4' SOLVER.WARMUP_EPOCH "8" SOLVER.STEPS "[35, 55]" SOLVER.MAX_EPOCHS "66" SOLVER.START_SAVE_EPOCH "50" SOLVER.EVAL_PERIOD "2" \
#     SOLVER.IMS_PER_BATCH "96" DATALOADER.NUM_INSTANCE "6" \
#     MODEL.BASELINE.TPL_WEIGHT "1.0" MODEL.BASELINE.CE_WEIGHT "0.33" MODEL.LABEL_SMOOTH "False" MODEL.BASELINE.S "30.0" MODEL.BASELINE.M "0.35" MODEL.BASELINE.COSINE_LOSS_TYPE 'ArcCos' \
#     INPUT.SIZE_TRAIN "([384,192])" INPUT.SIZE_TEST "([384,192])" \
#     MODEL.NAME "cosine_baseline" MODEL.BACKBONE "('resnet101_ibn_a')" MODEL.BASELINE.POOL_TYPE "avg"\
#     DATASETS.DATA_PATH "('${DATA_DIR}')" DATASETS.TRAIN_PATH "/data/Dataset/PReID/testb_pseudo_hist_065_080_dataset/rep_trainVal2"\
#     MODEL.PRETRAIN_PATH "('${PRETRAIN}')"  \
#     OUTPUT_DIR "('${SAVE_DIR}')" 