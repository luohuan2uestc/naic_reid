# -*- encoding: utf-8 -*-
'''
@File    :   inference_muti_model.py    
@Contact :   whut.hexin@foxmail.com
@License :   (C)Copyright 2017-2020, HeXin

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2019/11/14 20:50   xin      1.0         None
'''

from dataset.data import read_image
import os
import torch
import hickle
import numpy as np
import json
from evaluate import eval_func, euclidean_dist, re_rank
import pandas as pd
import shutil

os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# q_img_list = os.listdir(r'E:\data\reid\dataset6\query')
# query_list = list()
# qid_list = list()
# qcid_list = list()
# for q_img in q_img_list:
#     query_list.append(os.path.join(r'E:\data\reid\dataset6\query', q_img))
#     qid_list.append(int(q_img.strip(".png").split("_")[0]))
#     qcid_list.append(int(q_img.strip(".png").split("_")[1].strip("c")))
#
# g_img_list = os.listdir(r'E:\data\reid\dataset6\gallery')
# gallery_list = list()
# gid_list = list()
# gcid_list = list()
# for g_img in g_img_list:
#     gallery_list.append(os.path.join(r'E:\data\reid\dataset6\gallery', g_img))
#     gid_list.append(int(g_img.strip(".png").split("_")[0]))
#     gcid_list.append(int(g_img.strip(".png").split("_")[1].strip("c")))
# query_num = len(query_list)

# query_list = list()
# with open(r'E:\data\reid\初赛A榜测试集\初赛A榜测试集/query_a_list.txt', 'r') as f:
#     lines = f.readlines()
#     for i, line in enumerate(lines):
#         data = line.split(" ")
#         image_name = data[0].split("/")[1]
#         img_file = os.path.join(r'E:\data\reid\初赛A榜测试集\初赛A榜测试集\query_a', image_name)
#         query_list.append(img_file)

query_list = [os.path.join(r'E:\data\reid\初赛B榜测试集\初赛B榜测试集\query_b\query_b', x) for x in
                os.listdir(r'E:\data\reid\初赛B榜测试集\初赛B榜测试集\query_b\query_b')]
gallery_list = [os.path.join(r'E:\data\reid\初赛B榜测试集\初赛B榜测试集\gallery_b\gallery_b', x) for x in
                os.listdir(r'E:\data\reid\初赛B榜测试集\初赛B榜测试集\gallery_b\gallery_b')]
query_num = len(query_list)


def extract_feature(model,  transform, batch_size, model_name):

    img_list = list()
    for q_img in query_list:
        q_img = read_image(q_img)
        q_img = transform(q_img)
        img_list.append(q_img)
    for g_img in gallery_list:
        g_img = read_image(g_img)
        g_img = transform(g_img)
        img_list.append(g_img)

    img_data = torch.Tensor([t.numpy() for t in img_list])

    model = model.to(device)
    model.eval()
    iter_n = len(img_list) // batch_size
    if len(img_list) % batch_size != 0:
        iter_n += 1
    all_feature = list()
    for i in range(iter_n):
        # print("batch ----%d----" % (i))
        batch_data = img_data[i * batch_size:(i + 1) * batch_size]
        with torch.no_grad():
            # batch_feature = model(batch_data).detach().cpu()

            ff = torch.FloatTensor(batch_data.size(0), 2048*2).zero_()
            for i in range(2):
                if i == 1:
                    batch_data = batch_data.index_select(3, torch.arange(batch_data.size(3) - 1, -1, -1).long())
                outputs = model(batch_data.cuda())
                f = outputs.data.cpu()
                # ff = ff + f
                if i == 0:
                    # fnorm = torch.norm(f, p=2, dim=1, keepdim=True)
                    # f = f.div(fnorm.expand_as(f))
                    ff[:, :2048] = f
                if i == 1:
                    # fnorm = torch.norm(f, p=2, dim=1, keepdim=True)
                    # f = f.div(fnorm.expand_as(f))
                    ff[:, 2048:] = f
            # fnorm = torch.norm(ff, p=2, dim=1, keepdim=True)
            # ff = ff.div(fnorm.expand_as(ff))

            all_feature.append(ff)
    all_feature = torch.cat(all_feature)
    hickle.dump(all_feature.numpy(), r'E:\data\reid\features/%s.feature.hkl' % (model_name))


# def merge_feature_val(k1=20, k2=6, p=0.3, use_rerank=False):
#
#     feature_files = [os.path.join(r'E:\data\reid\val_features', x) for x in os.listdir(r'E:\data\reid\val_features')]
#
#     for i, feature_file in enumerate(feature_files):
#         feature = hickle.load(feature_file)
#         if i ==0 :
#             avg_feature = feature
#         else:
#             avg_feature = avg_feature+feature
#
#     avg_feature = avg_feature /(1.0*len(feature_files))
#     gallery_feat = avg_feature[query_num:]
#     query_feat = avg_feature[:query_num]
#     if use_rerank:
#         distmat = re_rank(torch.Tensor(query_feat), torch.Tensor(gallery_feat), k1, k2, p)
#     else:
#         distmat = euclidean_dist(torch.Tensor(query_feat), torch.Tensor(gallery_feat))
#
#     # distmat = euclidean_dist(query_feat, gallery_feat)
#     cmc, mAP, _ = eval_func(distmat, np.array(qid_list), np.array(gid_list),
#                             np.array(qcid_list), np.array(gcid_list))
#     print('Validation Result:')
#     print(str(k1) + "  -  " + str(k2) + "  -  " + str(p))
#     for r in [1, 5, 10]:
#         print('CMC Rank-{}: {:.2%}'.format(r, cmc[r - 1]))
#     print('mAP: {:.2%}'.format(mAP))
#     with open('re_rank.txt', 'a') as f:
#         f.write(str(k1) + "  -  " + str(k2) + "  -  " + str(p) + "\n")
#         for r in [1, 5, 10]:
#             f.write('CMC Rank-{}: {:.2%}'.format(r, cmc[r - 1]) + "\n")
#         f.write('mAP: {:.2%}'.format(mAP) + "\n")
#         f.write('------------------------------------------\n')
#         f.write('------------------------------------------\n')
#         f.write('\n\n')


def merge_feature_sample(k1=20, k2=6, p=0.3, use_rerank=False):

    feature_files = [os.path.join(r'E:\data\reid\features', x) for x in os.listdir(r'E:\data\reid\features')]

    ff = torch.FloatTensor(16246, 4096 * len(feature_files)).zero_()
    for i, feature_file in enumerate(feature_files):
        feature = hickle.load(feature_file)
        feature  =torch.Tensor(feature)
        fnorm = torch.norm(feature, p=2, dim=1, keepdim=True)
        feature = feature.div(fnorm.expand_as(feature))
        ff[:, (i)*4096:(i+1)*4096] = feature
        # if i ==0 :
        #     avg_feature = feature
        # else:
        #     avg_feature = avg_feature+feature
    avg_feature = ff
    # avg_feature = torch.Tensor(avg_feature)
    fnorm = torch.norm(avg_feature, p=2, dim=1, keepdim=True)
    avg_feature = avg_feature.div(fnorm.expand_as(avg_feature))
    avg_feature = avg_feature.numpy()
    # avg_feature = avg_feature /(1.0*len(feature_files))
    gallery_feat = avg_feature[query_num:]
    query_feat = avg_feature[:query_num]
    if use_rerank:
        distmat = re_rank(torch.Tensor(query_feat), torch.Tensor(gallery_feat), k1, k2, p)
    else:
        distmat = euclidean_dist(torch.Tensor(query_feat), torch.Tensor(gallery_feat))

    hickle.dump(distmat, 'final.distmat2.hkl')

    num_q, num_g = distmat.shape
    indices = np.argsort(distmat, axis=1)

    max_200_indices = indices[:, :200]

    res_dict = dict()
    for q_idx in range(num_q):
        print(query_list[q_idx])
        filename = query_list[q_idx][query_list[q_idx].rindex("\\") + 1:]
        max_200_files = [gallery_list[i][gallery_list[i].rindex("\\") + 1:] for i in max_200_indices[q_idx]]
        res_dict[filename] = max_200_files

    with open(r'submission_B.json', 'w', encoding='utf-8') as f:
        json.dump(res_dict, f)


# def merge_feature_sample(k1=20, k2=6, p=0.3, use_rerank=False):
#
#     feature_files = [os.path.join(r'E:\data\reid\features', x) for x in os.listdir(r'E:\data\reid\features')]
#
#     for i, feature_file in enumerate(feature_files):
#         feature = hickle.load(feature_file)
#         if i ==0 :
#             avg_feature = feature
#         else:
#             avg_feature = avg_feature+feature
#     avg_feature = torch.Tensor(avg_feature)
#     fnorm = torch.norm(avg_feature, p=2, dim=1, keepdim=True)
#     avg_feature = avg_feature.div(fnorm.expand_as(avg_feature))
#     avg_feature = avg_feature.numpy()
#     # avg_feature = avg_feature /(1.0*len(feature_files))
#     gallery_feat = avg_feature[query_num:]
#     query_feat = avg_feature[:query_num]
#     if use_rerank:
#         distmat = re_rank(torch.Tensor(query_feat), torch.Tensor(gallery_feat), k1, k2, p)
#     else:
#         distmat = euclidean_dist(torch.Tensor(query_feat), torch.Tensor(gallery_feat))
#
#     hickle.dump(distmat, 'final.distmat.hkl')
#
#     num_q, num_g = distmat.shape
#     indices = np.argsort(distmat, axis=1)
#
#     max_200_indices = indices[:, :300]
#
#     res_dict = dict()
#     for q_idx in range(num_q):
#         print(query_list[q_idx])
#         filename = query_list[q_idx][query_list[q_idx].rindex("\\") + 1:]
#         max_200_files = [gallery_list[i][gallery_list[i].rindex("\\") + 1:] for i in max_200_indices[q_idx]]
#         res_dict[filename] = max_200_files
#
#     with open(r'submission_A.json', 'w', encoding='utf-8') as f:
#         json.dump(res_dict, f)


def pseudo_label(k1, k2, p):
    feature_files = [os.path.join(r'E:\data\reid\features', x) for x in os.listdir(r'E:\data\reid\features')]

    for i, feature_file in enumerate(feature_files):
        feature = hickle.load(feature_file)
        if i == 0:
            avg_feature = feature
        else:
            avg_feature = avg_feature + feature

    avg_feature = avg_feature / (1.0 * len(feature_files))

    gallery_feat = avg_feature[query_num:]
    query_feat = avg_feature[:query_num]

    distmat = re_rank(torch.Tensor(query_feat), torch.Tensor(gallery_feat), k1, k2, p)
    distmat = distmat
    num_q, num_g = distmat.shape
    indices = np.argsort(distmat, axis=1)
    max_200_indices = indices[:, :00]

    res_dict = dict()
    pseudo_res = {"q_imgs": list(), "g_imgs": list(), "probs": list()}
    error_prob = {"q_imgs": list(), "g_imgs": list(), "probs": list()}
    true_prob = {"q_imgs": list(), "g_imgs": list(), "probs": list()}
    for q_idx in range(num_q):
        print(query_list[q_idx])
        filename = query_list[q_idx][query_list[q_idx].rindex("\\") + 1:]
        max_200_files = [gallery_list[i][gallery_list[i].rindex("\\") + 1:] for i in max_200_indices[q_idx]]
        probs = [distmat[q_idx, i] for i in max_200_indices[q_idx]]

        if max_200_files[0].split("_")[0] != filename.split("_")[0]:
            error_prob["q_imgs"].append(filename)
            error_prob["g_imgs"].append(max_200_files[0])
            error_prob["probs"].append(probs[0])
        # for i, prob in enumerate(probs):
        #     if probs[i] < 0.1:
        #         true_prob["q_imgs"].append(filename)
        #         true_prob["g_imgs"].append(max_200_files[i])
        #         true_prob["probs"].append(probs[i])
        if probs[0] < 0.1:
            true_prob["q_imgs"].append(filename)
            true_prob["g_imgs"].append(max_200_files[0])
            pid = max_200_files[0].split("_")[0]
            if not os.path.exists(os.path.join(r'E:\data\reid\gan\clu_data', pid)):
                shutil.copy(os.path.join(r'E:\data\reid\gan\train_all', filename), os.path.join(r'E:\data\reid\gan\clu_data', pid, max_200_files[0]))

        for g_filename, prob in zip(max_200_files, probs):
            pseudo_res["q_imgs"].append(filename)
            pseudo_res["g_imgs"].append(g_filename)
            pseudo_res["probs"].append(prob)

        res_dict[filename] = max_200_files

    columns = [u'q_imgs', u'g_imgs', u'probs']
    save_df = pd.DataFrame(pseudo_res,
                           columns=columns)
    save_df.to_csv('pseudo_res.csv')
    save_df = pd.DataFrame(error_prob,
                           columns=columns)
    save_df.to_csv('error_pseudo_res.csv')
    save_df = pd.DataFrame(true_prob,
                           columns=columns)
    save_df.to_csv('true_pseudo_res.csv')

if __name__ == "__main__":
    import torchvision.transforms as T
    from models.baseline import Baseline

    from config import cfg
    from common.sync_bn import convert_model
    from models import build_model

    model = build_model(cfg, 2772)
    para_dict = torch.load(r'E:\data\reid\exp\hg/mfn_epoch238.pth')
    model = torch.nn.DataParallel(model)
    model = convert_model(model)
    model.load_state_dict(para_dict)

    transform = T.Compose([
        T.Resize((256, 128)),

        T.ToTensor(),
        # T.Normalize(mean=[0.09661545, 0.18356957, 0.21322473], std=[0.13422933, 0.14724616, 0.19259872])
    ])



    # extract_feature(model=model, transform=transform, batch_size=64, model_name='mfn_epoch238_0.6')
    merge_feature_sample(8, 3, 0.8, True)
    # pseudo_label(8, 3, 0.8)




