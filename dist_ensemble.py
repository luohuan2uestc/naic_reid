import os
import numpy as np
import json
import time 
import pprint
import pickle
import mmcv
import functools
import pandas as pd
import argparse
import matplotlib
import shutil

from test2 import get_post_json
if __name__ == "__main__":
    post = False
    post_top_per = 0.7
    max_rank = 200
    save_dir = '../rep_work_dirs/testb_ensembles/'
    # save_fname = 'ensemble1.json'
    # dist_fnames = [
    #     '../rep_work_dirs/exp4-cosinebaseline-resnet101ibnls1-384x192-bs16x6-warmup10-flip-pad10-meanstd-erase0502-nolbsm-avg-arcface30_035_10033-cj05-trainVal2/sub/origin_tpl03_e90_flip_sub_aqe.pkl',
    #     '../rep_work_dirs/exp4-cosinebaseline-resnet101ibnls1-384x192-bs16x6-warmup10-flip-pad10-meanstd-erase0502-nolbsm-avg-arcface30_035_10033-cj05-trainVal2/sub/origin_tpl03_e80_flip_sub_aqe.pkl',

    #     '../rep_work_dirs/exp4-cosinebaseline-resnet101ibnls1-384x192-bs16x6-warmup10-flip-pad10-meanstd-erase0502-nolbsm-avg-arcface30_035_10033-cj05-trainVal2-finetune_tpl05/sub/finetune_tpl05_e16_flip_sub_aqe.pkl',
    #     '../rep_work_dirs/exp4-cosinebaseline-resnet101ibnls1-384x192-bs16x6-warmup10-flip-pad10-meanstd-erase0502-nolbsm-avg-arcface30_035_10033-cj05-trainVal2-finetune_tpl05/sub/finetune_tpl05_e14_flip_sub_aqe.pkl'

    # ]
    # save_fname = 'ensemble2_post.json'
    # dist_fnames = [
    #     '../rep_work_dirs/exp4-cosinebaseline-resnet101ibnls1-384x192-bs16x6-warmup10-flip-pad10-meanstd-erase0502-nolbsm-avg-arcface30_035_10033-cj05-trainVal2/sub/origin_tpl03_e90_flip_sub_aqe.pkl',
    #     '../rep_work_dirs/exp4-cosinebaseline-resnet101ibnls1-384x192-bs16x6-warmup10-flip-pad10-meanstd-erase0502-nolbsm-avg-arcface30_035_10033-cj05-trainVal2/sub/origin_tpl03_e80_flip_sub_aqe.pkl',

    #     '../rep_work_dirs/exp4-cosinebaseline-resnet101ibnls1-384x192-bs16x6-warmup10-flip-pad10-meanstd-erase0502-nolbsm-avg-arcface30_035_10033-cj05-trainVal2-pseudo_resume_smlbsm/sub/pseudo_resume_smlbsm_e60_flip_sub_aqe.pkl',
    #     '../rep_work_dirs/exp4-cosinebaseline-resnet101ibnls1-384x192-bs16x6-warmup10-flip-pad10-meanstd-erase0502-nolbsm-avg-arcface30_035_10033-cj05-trainVal2-pseudo_resume_smlbsm/sub/pseudo_resume_smlbsm_e50_flip_sub_aqe.pkl'
    # ]
    # save_fname = 'ensemble3.json'
    # dist_fnames = [
    #     '../rep_work_dirs/exp4-cosinebaseline-resnet101ibnls1-384x192-bs16x6-warmup10-flip-pad10-meanstd-erase0502-nolbsm-avg-arcface30_035_10033-cj05-trainVal2/sub/origin_tpl03_e90_flip_sub_aqe.pkl',
    #     '../rep_work_dirs/exp4-cosinebaseline-resnet101ibnls1-384x192-bs16x6-warmup10-flip-pad10-meanstd-erase0502-nolbsm-avg-arcface30_035_10033-cj05-trainVal2/sub/origin_tpl03_e80_flip_sub_aqe.pkl',

    #     '../rep_work_dirs/exp5-mgn-resnet50ibnls2-384x144-bs12x8-warmup10-flip-pad10-meanstd-lbsm-layer3_-5-erase0502-allgem_3-minInst2/eval/origin_e110_flip_adabn_mgnsub_aqe.pkl',
    # ]
    query_dir='/data/Dataset/PReID/dataset2/query_/'
    gallery_dir='/data/Dataset/PReID/dataset2/gallery_/'

    # save_fname = 'testb_sm_e56_60.json'
    # dist_fnames = [
    #     '../rep_work_dirs/exp4-cosinebaseline-resnet101ibnls1-384x192-bs16x6-warmup10-flip-pad10-meanstd-erase0502-nolbsm-avg-arcface30_035_10033-cj05-trainVal2-testb_pseudo_retrain_smlbsm/sub/testb_pseudo_retrain_smlbsm_e60_flip_sub_aqe.pkl',
    #     '../rep_work_dirs/exp4-cosinebaseline-resnet101ibnls1-384x192-bs16x6-warmup10-flip-pad10-meanstd-erase0502-nolbsm-avg-arcface30_035_10033-cj05-trainVal2-testb_pseudo_retrain_smlbsm/sub/testb_pseudo_retrain_smlbsm_e56_flip_sub_aqe.pkl',
    # ]
    # save_fname = 'testb_arcface_e56_60.json'
    # dist_fnames = [
    #     '../rep_work_dirs/exp4-cosinebaseline-resnet101ibnls1-384x192-bs16x6-warmup10-flip-pad10-meanstd-erase0502-nolbsm-avg-arcface30_035_10033-cj05-trainVal2-testb_pseudo_retrain_arface/sub/testb_pseudo_retrain_arcface_e60_flip_sub_aqe.pkl',
    #     '../rep_work_dirs/exp4-cosinebaseline-resnet101ibnls1-384x192-bs16x6-warmup10-flip-pad10-meanstd-erase0502-nolbsm-avg-arcface30_035_10033-cj05-trainVal2-testb_pseudo_retrain_arface/sub/testb_pseudo_retrain_arcface_e56_flip_sub_aqe.pkl'
    # ]

    # save_fname = 'testb_en-e84_86-e62_66.json'
    # dist_fnames = [
    #     '../rep_work_dirs/exp4-cosinebaseline-resnet101ibnls1-384x192-bs16x6-warmup10-flip-pad10-meanstd-erase0502-nolbsm-avg-arcface30_035_10033-cj05-trainVal2/sub/testb_origin_tpl03_e90_flip_sub_aqe.pkl',
    #     '../rep_work_dirs/exp4-cosinebaseline-resnet101ibnls1-384x192-bs16x6-warmup10-flip-pad10-meanstd-erase0502-nolbsm-avg-arcface30_035_10033-cj05-trainVal2/sub/testb_origin_tpl03_e84_flip_sub_aqe.pkl',
    #     '../rep_work_dirs/exp4-cosinebaseline-resnet101ibnls1-384x192-bs16x6-warmup10-flip-pad10-meanstd-erase0502-nolbsm-avg-arcface30_035_10033-cj05-trainVal2-testb_pseudo_retrain_arface_e66/sub/testb_pseudo_retrain_arcface_e66_flip_sub_aqe.pkl',
    #     '../rep_work_dirs/exp4-cosinebaseline-resnet101ibnls1-384x192-bs16x6-warmup10-flip-pad10-meanstd-erase0502-nolbsm-avg-arcface30_035_10033-cj05-trainVal2-testb_pseudo_retrain_arface_e66/sub/testb_pseudo_retrain_arcface_e62_flip_sub_aqe.pkl'
    # ]
    save_fname = 'testb_en-e90_86-e66-e66.json'
    dist_fnames = [
        '../rep_work_dirs/exp4-cosinebaseline-resnet101ibnls1-384x192-bs16x6-warmup10-flip-pad10-meanstd-erase0502-nolbsm-avg-arcface30_035_10033-cj05-trainVal2/sub/testb_origin_tpl03_e90_flip_sub_aqe.pkl',
        '../rep_work_dirs/exp4-cosinebaseline-resnet101ibnls1-384x192-bs16x6-warmup10-flip-pad10-meanstd-erase0502-nolbsm-avg-arcface30_035_10033-cj05-trainVal2-testb_pseudo_retrain_arcface-fp16/sub/testb_pseudo_retrain_arcface_e66_flip_sub_aqe.pkl',
        '../rep_work_dirs/exp4-cosinebaseline-resnet101ibnls1-384x192-bs16x6-warmup10-flip-pad10-meanstd-erase0502-nolbsm-avg-arcface30_035_10033-cj05-trainVal2-testb_pseudo_retrain_arcface_cj05-fp16/sub/testb_pseudo_retrain_arcface_e66_flip_sub_aqe.pkl'
    ]
    
    query_dir='/data/Dataset/PReID/dataset2/rep_B/query_b/'
    gallery_dir='/data/Dataset/PReID/dataset2/rep_B/gallery_b/'

    os.makedirs(save_dir,exist_ok=True)
    query_list = [os.path.join(query_dir, x) for x in os.listdir(query_dir)]
    gallery_list = [os.path.join(gallery_dir, x) for x in os.listdir(gallery_dir)]
    num_q = len(query_list)
    query_list = sorted(query_list)
    gallery_list = sorted(gallery_list)
    print(query_list[:10])

    distmat = None
    for i in range(len(dist_fnames)):
        print('loading..')
        print(dist_fnames[i])
        if distmat is None:
            with open(dist_fnames[i],'rb') as fid:
                # distmat = np.power(pickle.load(fid),2)
                distmat = pickle.load(fid)

        else:
            with open(dist_fnames[i],'rb') as fid:
                distmat += pickle.load(fid)
                # distmat += np.power(pickle.load(fid),2)
    
    print("==>saving..")
    if post:
        qfnames = [fname.split('/')[-1] for fname in query_list]
        gfnames = [fname.split('/')[-1] for fname in gallery_list]
        st = time.time()
        print("post json using top_per:",post_top_per)
        res_dict = get_post_json(distmat,qfnames,gfnames,post_top_per)
        print("post cost:",time.time()-st)
    else:
        # [todo] fast test
        print("==>sorting..")
        st = time.time()
        indices = np.argsort(distmat, axis=1)
        print("argsort cost:",time.time()-st)
        # print(indices[:2, :max_rank])
        # st = time.time()
        # indices = np.argpartition( distmat, range(1,max_rank+1))
        # print("argpartition cost:",time.time()-st)
        # print(indices[:2, :max_rank])

        max_200_indices = indices[:, :max_rank]
        res_dict = dict()
        for q_idx in range(num_q):
            filename = query_list[q_idx].split('/')[-1]
            max_200_files = [gallery_list[i].split('/')[-1] for i in max_200_indices[q_idx]]
            res_dict[filename] = max_200_files
    with open(save_dir+save_fname, 'w' ,encoding='utf-8') as f:
        json.dump(res_dict, f)