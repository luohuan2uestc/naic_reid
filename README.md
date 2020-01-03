# naic_reid
# This is Top19 Code for the Person ReID Compitition of NAIC(首届“全国人工智能大赛”（行人重识别 Person ReID 赛项）)

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

# Models
- [x] Strong Baseline
- [x] MGN
- [x] MFN

# Tricks
- [x] DataAugmention(RandomErase+ColorJittering+RandomHorizontallyFlip)
- [x] ArcFace
- [x] BackBone resnet101_ibn_a
- [x] Size 384*192
- [x] TripleLoss+SoftmaxLoss
- [x] AQE
- [x] Batch GPU ReRanking
- [x] Pseudo Label + Ensemble
- [x] Multi Triple-Margine Ensemble


## prepare_data
### step 1 modify prepare_rep2.py
```python
    root_dir = '/data/Dataset/PReID/'  # dataset root
    rep_dir = root_dir+'dataset2/'  # rep dataset

    save_dir = root_dir+'rep_dataset/' # save path
```
### step 2 run prepare_rep2.py

## Train

### setp1 modify train sh file

```
 PRETRAIN=../weights/resnet101_ibn_a.pth.tar
 DATA_DIR='/data/Dataset/PReID/rep_dataset/'
 SAVE_DIR='../rep_work_dirs/exp4-cosinebaseline-resnet101ibnls1-384x192-bs16x6-warmup10-flip-pad10-meanstd-erase0502-nolbsm-avg-arcface30_035_10033-cj05-trainVal2-testb_pseudo_retrain_arface_e66/' #(h, w)

 CUDA_VISIBLE_DEVICES=5 python train.py --config_file='configs/naic/arcface_baseline.yml' \
     SOLVER.BASE_LR '3e-4' SOLVER.WARMUP_EPOCH "8" SOLVER.STEPS "[35, 55]" SOLVER.MAX_EPOCHS "66" SOLVER.START_SAVE_EPOCH "50" SOLVER.EVAL_PERIOD "2" \
     SOLVER.IMS_PER_BATCH "96" DATALOADER.NUM_INSTANCE "6" \
     MODEL.BASELINE.TPL_WEIGHT "1.0" MODEL.BASELINE.CE_WEIGHT "0.33" MODEL.LABEL_SMOOTH "False" MODEL.BASELINE.S "30.0" MODEL.BASELINE.M "0.35" MODEL.BASELINE.COSINE_LOSS_TYPE 'ArcCos' \
     INPUT.SIZE_TRAIN "([384,192])" INPUT.SIZE_TEST "([384,192])" \
     MODEL.NAME "cosine_baseline" MODEL.BACKBONE "('resnet101_ibn_a')" MODEL.BASELINE.POOL_TYPE "avg"\
     DATASETS.DATA_PATH "('${DATA_DIR}')" DATASETS.TRAIN_PATH "/data/Dataset/PReID/testb_pseudo_hist_065_080_dataset/rep_trainVal2"\
     MODEL.PRETRAIN_PATH "('${PRETRAIN}')"  \
     OUTPUT_DIR "('${SAVE_DIR}')" 

```

### step 2 run ./shells/rep_train_bl.sh


## Test
### step1 modify test sh file

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

### step 2 run ./shells/rep_test_bl.sh

## pseudo label
### step1 modify test sh file for pseudo label
```
# [20191231] testb pseudo

QUERY_DIR=${ROOT_DIR}dataset2/rep_B/query_b/
GALLERY_DIR=${ROOT_DIR}dataset2/rep_B/gallery_b/
DATA_DIR=${ROOT_DIR}rep_dataset/
PRETRAIN=../weights/resnet101_ibn_a.pth.tar
MODEL_DIR=../rep_work_dirs/exp4-cosinebaseline-resnet101ibnls1-384x192-bs16x6-warmup10-flip-pad10-meanstd-erase0502-nolbsm-avg-arcface30_035_10033-cj05-trainVal2/ #(h, w)
WEIGHT=${MODEL_DIR}cosine_baseline_epoch90.pth
SAVE_DIR=${MODEL_DIR}eval/


 SAVE_DIR=${MODEL_DIR}sub/
 CUDA_VISIBLE_DEVICES=2 python test2.py --config_file='configs/naic/arcface_baseline.yml' \
     --sub \
     --pseudo --pseudo_hist --pseudo_visual --pseudo_algorithm "auto" --pseudo_eps 0.55 --pseudo_minpoints 3 --pseudo_maxpoints 100 --pseudo_savepath '../rep_work_dirs/testb_pseudo_hist_065_080/'\
     --query_dir ${QUERY_DIR}\
     --gallery_dir ${GALLERY_DIR}\
     MODEL.BASELINE.S "30.0" MODEL.BASELINE.M "0.35" \
     MODEL.BASELINE.COSINE_LOSS_TYPE 'ArcCos' \
     INPUT.SIZE_TRAIN "([384,192])" INPUT.SIZE_TEST "([384,192])" \
     MODEL.NAME "cosine_baseline" MODEL.BACKBONE "('resnet101_ibn_a')" MODEL.BASELINE.POOL_TYPE "avg"\
     DATASETS.DATA_PATH "('${DATA_DIR}')" DATASETS.TRAIN_PATH "rep_trainVal2"\
     MODEL.PRETRAIN_PATH "('${PRETRAIN}')"  \
     OUTPUT_DIR "('${SAVE_DIR}')" \
     TEST.WEIGHT "${WEIGHT}"
```
then you can get pseudo data in pseudo_savepath

### step2 copy pseudo data to trainVal data
#### modify prepare_pseudo.py
```python
    root_dir = '/data/Dataset/PReID/'  # root of dataset
    
    origin_path = root_dir+'dataset2/'+'pid_dataset/' # original trainval data
    pseudo_path = '../rep_work_dirs/testb_pseudo_hist_065_080/' # pseudo data

    save_dir = root_dir+'testb_pseudo_hist_065_080_dataset/' # save path
```
#### run prepare_pseudo.py

### step3 train the model again using mix dataset(original trainval data and pseudo data)

## Model Ensemble
### step1 modify dist_ensemble.py
```python
 save_dir = '../rep_work_dirs/testb_ensembles/'  # save path
 query_dir='/data/Dataset/PReID/dataset2/query_/' # query path
 gallery_dir='/data/Dataset/PReID/dataset2/gallery_/' # gallery path
 save_fname = 'ensemble1.json' # submit filename
 dist_fnames = [  ## distance matrix
    #     '../rep_work_dirs/exp4-cosinebaseline-resnet101ibnls1-384x192-bs16x6-warmup10-flip-pad10-meanstd-erase0502-nolbsm-avg-arcface30_035_10033-cj05-trainVal2/sub/origin_tpl03_e90_flip_sub_aqe.pkl',
    #     '../rep_work_dirs/exp4-cosinebaseline-resnet101ibnls1-384x192-bs16x6-warmup10-flip-pad10-meanstd-erase0502-nolbsm-avg-arcface30_035_10033-cj05-trainVal2/sub/origin_tpl03_e80_flip_sub_aqe.pkl',

    #     '../rep_work_dirs/exp4-cosinebaseline-resnet101ibnls1-384x192-bs16x6-warmup10-flip-pad10-meanstd-erase0502-nolbsm-avg-arcface30_035_10033-cj05-trainVal2-finetune_tpl05/sub/finetune_tpl05_e16_flip_sub_aqe.pkl',
    #     '../rep_work_dirs/exp4-cosinebaseline-resnet101ibnls1-384x192-bs16x6-warmup10-flip-pad10-meanstd-erase0502-nolbsm-avg-arcface30_035_10033-cj05-trainVal2-finetune_tpl05/sub/finetune_tpl05_e14_flip_sub_aqe.pkl'

    # ]
```

### step2 run dist_ensemble.py

