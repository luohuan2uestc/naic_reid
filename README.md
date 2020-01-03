# naic_reid
This is the code for the compitition of NAIC

### Dependencies
* python==3.6
* pandas==0.25.1
* hickle==3.4.5
* tqdm==4.36.1
* opencv_python==4.1.1.26
* scikit_image==0.15.0
* mlconfig==0.0.4
* visdom==0.1.8.9
* torch==1.0.0
* torchvision==0.2.0
* yacs==0.1.6
* numpy==1.17.4
* scipy==1.3.1
* apex==0.9.10dev
* ipython==7.10.0
* Pillow==6.2.1
* skimage==0.0


## prepare_data
### step 1    
#### modify prepare_rep2.py
```python
    root_dir = '/data/Dataset/PReID/'  # dataset root
    rep_dir = root_dir+'dataset2/'  # rep dataset

    save_dir = root_dir+'rep_dataset/' # save path
```
### step 2
#### run prepare_rep2.py

## Train

### setp1
#### modify train sh file

```
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
```

### step 2
#### run ./shells/rep_train_bl.sh


## Test
### step1
#### modify test sh file

```
DATA_DIR=${ROOT_DIR}rep_dataset/
PRETRAIN=../weights/resnet101_ibn_a.pth.tar
MODEL_DIR=../rep_work_dirs/exp4-cosinebaseline-resnet101ibnls1-384x192-bs16x6-warmup10-flip-pad10-meanstd-erase0502-nolbsm-avg-arcface30_035_10033-cj05-trainVal2-testb_pseudo_retrain_arface_e66/ #(h, w)
WEIGHT=${MODEL_DIR}cosine_baseline_epoch66.pth
SAVE_DIR=${MODEL_DIR}eval/
 
   --flip \
   --aqe --aqe_k2 7 --aqe_alpha 3.0 \
 CUDA_VISIBLE_DEVICES=2 python test2.py --config_file='configs/naic/arcface_baseline.yml' \
     TEST.IMS_PER_BATCH "128" TEST.RANDOMPERM "5" \
     MODEL.BASELINE.TPL_WEIGHT "1.0" MODEL.BASELINE.CE_WEIGHT "0.33" MODEL.LABEL_SMOOTH "False" MODEL.BASELINE.S "30.0" MODEL.BASELINE.M "0.35" MODEL.BASELINE.COSINE_LOSS_TYPE 'ArcCos' \
     INPUT.SIZE_TRAIN "([384,192])" INPUT.SIZE_TEST "([384,192])" \
     MODEL.NAME "cosine_baseline" MODEL.BACKBONE "('resnet101_ibn_a')" MODEL.BASELINE.POOL_TYPE "avg"\
     DATASETS.DATA_PATH "('${DATA_DIR}')" DATASETS.TRAIN_PATH "/data/Dataset/PReID/testb_pseudo_hist_065_080_dataset/rep_trainVal2"\
     MODEL.PRETRAIN_PATH "('${PRETRAIN}')"  \
     OUTPUT_DIR "('${SAVE_DIR}')" \
     TEST.WEIGHT "${WEIGHT}"
```

### step 2
#### run ./shells/rep_test_bl.sh



