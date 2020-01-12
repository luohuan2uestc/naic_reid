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

# Support
- [x] Multi-GPU and SyncBN
- [x] fp16

# Models
- [x] Strong Baseline
- [x] MGN
- [x] MFN(Comming Soon)

# Tricks
- [x] DataAugmention(RandomErase + ColorJittering + RandomHorizontallyFlip)
- [x] WarmUp + MultiStepLR 
- [x] ArcFace
- [x] BackBone resnet101_ibn_a
- [x] Size 384*192
- [x] TripleLoss+SoftmaxLoss
- [x] AQE
- [x] adabn
- [x] gem
- [x] all data training(not include single pid image)
- [x] Batch GPU ReRanking
- [x] Pseudo Label + Ensemble
- [x] Multi Triplet-Margine Ensemble

| model | size | backbone | trick |  performance |
|------|------|------|------|
| mgn | 384x128 | resnet50-ibna | adabn + gem | 0.77808439 |
| mgn | 384x144 | resnet50-ibna | adabn + gem | 0.78023715 |
| mgn | 384x144 | resnet50-ibna | adabn + gem + aqe | 0.78967123 |
| mgn | 384x144 | resnet50-ibna | gem + aqe + cj | 0.79911998 |
| baseline | 384x192 | resnet50-ibna | adabn + gem + cosface | 0.76778309 |
| baseline | 384x192 | resnet50-ibna | gem + cosface + cj | 0.79208766 |
| baseline | 384x192 | resnet50-ibna | avg + cosface + cj | 0.79478573 |
| baseline | 384x192 | resnet101-ibna | avg + cosface + cj | 0.80346376 |
| baseline | 384x192 | resnet101-ibna | avg + cosface + cj + all_data | 0.80577292 |
| baseline | 384x192 | resnet101-ibna | avg + cosface + cj05 + all_data | 0.80686313 |
| baseline | 384x192 | resnet101-ibna | avg + arcfaceface + cj05 + all_data | 0.819 |




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
 PRETRAIN=resnet101_ibn_a.pth.tar
 DATA_DIR='your data dir'
 SAVE_DIR='your save dir' #(h, w)

 CUDA_VISIBLE_DEVICES=5 python train.py --config_file='configs/naic/arcface_baseline.yml' \
     SOLVER.BASE_LR '3e-4' SOLVER.WARMUP_EPOCH "8" SOLVER.STEPS "[35, 55]" SOLVER.MAX_EPOCHS "66" SOLVER.START_SAVE_EPOCH "50" SOLVER.EVAL_PERIOD "2" \
     SOLVER.IMS_PER_BATCH "96" DATALOADER.NUM_INSTANCE "6" \
     MODEL.BASELINE.TPL_WEIGHT "1.0" MODEL.BASELINE.CE_WEIGHT "0.33" MODEL.LABEL_SMOOTH "False" MODEL.BASELINE.S "30.0" MODEL.BASELINE.M "0.35" MODEL.BASELINE.COSINE_LOSS_TYPE 'ArcCos' \
     INPUT.SIZE_TRAIN "([384,192])" INPUT.SIZE_TEST "([384,192])" \
     MODEL.NAME "cosine_baseline" MODEL.BACKBONE "('resnet101_ibn_a')" MODEL.BASELINE.POOL_TYPE "avg"\
     DATASETS.DATA_PATH "('${DATA_DIR}')" DATASETS.TRAIN_PATH "your train data folder"\
     MODEL.PRETRAIN_PATH "('${PRETRAIN}')"  \
     OUTPUT_DIR "('${SAVE_DIR}')" 

```

### step 2 run ./shells/rep_train_bl.sh


## Test
### step1 modify test sh file

```
DATA_DIR= 'your data dir'
PRETRAIN=resnet101_ibn_a.pth.tar
MODEL_DIR=your model dir #(h, w)
WEIGHT=${MODEL_DIR}cosine_baseline_epoch66.pth
SAVE_DIR=${MODEL_DIR}eval/
 
   --flip \
   --aqe --aqe_k2 7 --aqe_alpha 3.0 \
 CUDA_VISIBLE_DEVICES=2 python test2.py --config_file='configs/naic/arcface_baseline.yml' \
     TEST.IMS_PER_BATCH "128" TEST.RANDOMPERM "5" \
     MODEL.BASELINE.TPL_WEIGHT "1.0" MODEL.BASELINE.CE_WEIGHT "0.33" MODEL.LABEL_SMOOTH "False" MODEL.BASELINE.S "30.0" MODEL.BASELINE.M "0.35" MODEL.BASELINE.COSINE_LOSS_TYPE 'ArcCos' \
     INPUT.SIZE_TRAIN "([384,192])" INPUT.SIZE_TEST "([384,192])" \
     MODEL.NAME "cosine_baseline" MODEL.BACKBONE "('resnet101_ibn_a')" MODEL.BASELINE.POOL_TYPE "avg"\
     DATASETS.DATA_PATH "('${DATA_DIR}')" DATASETS.TRAIN_PATH "your train data folder"\
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
MODEL_DIR=your model dir #(h, w)
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
    #     'origin_tpl03_e90_flip_sub_aqe.pkl',
    #     'origin_tpl03_e80_flip_sub_aqe.pkl',
    #     'finetune_tpl05_e16_flip_sub_aqe.pkl',
    #     'finetune_tpl05_e14_flip_sub_aqe.pkl'

    # ]
```

### step2 run dist_ensemble.py

## Notes:
1.Due to time constraints, the best solution we eventually adopted was to use a baseline model and a pseudo-labeled baseline model for fusion.  
&nbsp;  
2.We also trained MGN to a better effect, but the final effect of mgn is not as good as the baseline + arcface + cj.  
&nbsp;  
3.The MFN network comes from one of my teammates, and he plans to use the model in a paper. So it cannot be open source now. If you are interested, you can follow his Github：https://github.com/douzi0248/Re-ID  
&nbsp;  
4.The codes are expanded on a ReID-baseline , which is open sourced by Hao Luo.(Thanks for [Hao Luo](https://github.com/michuanhaohao "Hao Luo") and [DTennant](https://github.com/DTennant "DTennant"), our baseline model comes from https://github.com/michuanhaohao/reid-strong-baseline and https://github.com/DTennant/reid_baseline_with_syncbn)
