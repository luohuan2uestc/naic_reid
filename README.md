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
运行inference_multi_model.py时需要先下载网盘中的16个模型，
先分别利用这16个模型提取特征，然后将特征进行拼接融合
提取特征的步骤为：
* 1.先修改测试数据路径
```python
query_list = [os.path.join(r'E:\data\reid\初赛B榜测试集\初赛B榜测试集\query_b\query_b', x) for x in
                os.listdir(r'E:\data\reid\初赛B榜测试集\初赛B榜测试集\query_b\query_b')]
gallery_list = [os.path.join(r'E:\data\reid\初赛B榜测试集\初赛B榜测试集\gallery_b\gallery_b', x) for x in
                os.listdir(r'E:\data\reid\初赛B榜测试集\初赛B榜测试集\gallery_b\gallery_b')]  # line 54-57
```
* 2.修改模型路径
```python
para_dict = torch.load(r'E:\data\reid\exp\hg/mfn_epoch238.pth') # line 307
```
* 3.取消extract_feature方法的注释，同时注释掉merge_feature_sample方法
```python
extract_feature(model=model, transform=transform, batch_size=64, model_name='mfn_epoch238_0.6') # 注意每次运行不同模型需要同时修改模型名称
# merge_feature_sample(8, 3, 0.8, True) # line 321-322
```
* 4.修改提取特征的保存位置
```python
hickle.dump(all_feature.numpy(), r'E:\data\reid\features/%s.feature.hkl' % (model_name)) # line 107
```
特征提取完毕之后，需要进行特征融合，步骤如下：
* 1.取消merge_feature_sample方法的注释，同时注释掉extract_feature方法
 ```python
# extract_feature(model=model, transform=transform, batch_size=64, model_name='mfn_epoch238_0.6') # 注意每次运行不同模型需要同时修改模型名称
merge_feature_sample(8, 3, 0.8, True)
```
* 2.根据之前提取特征的保存位置修改加载特征的路径
```python
feature_files = [os.path.join(r'E:\data\reid\features', x) for x in os.listdir(r'E:\data\reid\features')] # line 149
```
最终结果文件保存在当前目录的submission_B.json文件中

最终融合了两种0.6,0.5margin的mfn模型训练后期的模型，一共16个，下载百度云链接：链接: https://pan.baidu.com/s/1zlo-LwxGpPB6SYczp2K4wQ 提取码: bfkg


