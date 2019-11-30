# naic_reid
This is the code for the compitition of NAIC

### Dependencies
* python==3.6
* torch==1.0.0
* numpy==1.17.4
* torchvision==0.2.0


## Train
step 1.  
modify config.py  
``` python 
_C.MODEL.NAME = 'mfn' # model to choose 'mfn|mgn|baseline'
_C.MODEL.NAME = 'resnet50_ibn_a' # backbone to choose ''resnet50|resnet50_ibn_a|'
_C.MODEL.PRETRAIN_PATH = '' # path to pretained model

_C.DATASETS.DATA_PATH = '' # path to dataset, which contains the train / query / gallery subfolder
_C.DATALOADER.NUM_INSTANCE = 4 # K
_C.OUTPUT_DIR = '' # path to save log file and model weights

```

setp 2.  
run main.py  

## Test
single model  
run inference.py  
multi models  
run inference_muti_model.py  

最终融合了两种0.6,0.5margin的mfn模型训练后期的模型，一共16个，下载百度云链接：链接: https://pan.baidu.com/s/1zlo-LwxGpPB6SYczp2K4wQ 提取码: bfkg


