# -*- encoding: utf-8 -*-
'''
@File    :   inference.py    
@Contact :   whut.hexin@foxmail.com
@License :   (C)Copyright 2017-2020, HeXin

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2019/11/6 20:18   xin      1.0         None
'''

# -*- encoding: utf-8 -*-
'''
@File    :   inference.py    
@Contact :   whut.hexin@foxmail.com
@License :   (C)Copyright 2017-2020, HeXin

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2019/10/28 23:01   xin      1.0         None
'''
import os
import numpy as np
import json

import pandas as pd
import argparse

import torch
import torch.nn.functional as F
from tqdm import tqdm
from dataset.data import read_image
from evaluate import eval_func, euclidean_dist, re_rank
from dataset import make_dataloader

# os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

device = 'cuda'
def inference_val(model,  transform, batch_size, query_dir,gallery_dir,save_dir, k1=20, k2=6, p=0.3, use_rerank=False,use_flip=False,n_randperm=0):
    q_img_list = os.listdir(query_dir)
    query_list = list()
    qid_list = list()
    qcid_list = list()
    for q_img in q_img_list:
        query_list.append(os.path.join(query_dir, q_img))
        qid_list.append(int(q_img.strip(".png").split("_")[0]))
        qcid_list.append(int(q_img.strip(".png").split("_")[1].strip("c")))

    g_img_list = os.listdir(gallery_dir)
    gallery_list = list()
    gid_list = list()
    gcid_list = list()
    for g_img in g_img_list:
        gallery_list.append(os.path.join(gallery_dir, g_img))
        gid_list.append(int(g_img.strip(".png").split("_")[0]))
        gcid_list.append(int(g_img.strip(".png").split("_")[1].strip("c")))
    img_list = list()
    print('==> load query image..')
    with tqdm(total=len(query_list)) as pbar:
        for q_img in query_list:
            q_img = read_image(q_img)
            q_img = transform(q_img)
            img_list.append(q_img)
            pbar.update(1)
    print('==> load gallery image..')
    with tqdm(total=len(gallery_list)) as pbar:
        for g_img in gallery_list:
            g_img = read_image(g_img)
            g_img = transform(g_img)
            img_list.append(g_img)
            pbar.update(1)

    query_num = len(query_list)
    img_data = torch.Tensor([t.numpy() for t in img_list])

    model = model.to(device)
    model.eval()
    iter_n = len(img_list) // batch_size
    if len(img_list) % batch_size != 0:
        iter_n += 1
    all_feature = list()
    with tqdm(total=iter_n) as pbar:
        for i in range(iter_n):
            # print("batch ----%d----" % (i))
            batch_data = img_data[i * batch_size:(i + 1) * batch_size]
            with torch.no_grad():
                if use_flip:
                    ff = torch.FloatTensor(batch_data.size(0), 2048*2).zero_()
                    for i in range(2):
                        if i == 1:
                            batch_data = batch_data.index_select(3, torch.arange(batch_data.size(3) - 1, -1, -1).long())
                        outputs = model(batch_data)
                        f = outputs.data.cpu()

                        if i == 0:
                            fnorm = torch.norm(f, p=2, dim=1, keepdim=True)
                            f = f.div(fnorm.expand_as(f))
                            ff[:, :2048] = f
                        if i == 1:
                            fnorm = torch.norm(f, p=2, dim=1, keepdim=True)
                            f = f.div(fnorm.expand_as(f))
                            ff[:, 2048:] = f
                    fnorm = torch.norm(ff, p=2, dim=1, keepdim=True)
                    ff = ff.div(fnorm.expand_as(ff))
                else:

                    # ff = model(batch_data,output_feature='with_score').data.cpu()
                    ff = model(batch_data).data.cpu()

                    fnorm = torch.norm(ff, p=2, dim=1, keepdim=True)
                    ff = ff.div(fnorm.expand_as(ff))

                all_feature.append(ff)
            pbar.update(1)
    all_feature = torch.cat(all_feature)
    # DBA
    if 0:
        k2 = 10
        alpha = -3.0
        # [todo] heap sort
        distmat = euclidean_dist(all_feature, all_feature)
        initial_rank = distmat.numpy().argsort(axis=1)

        all_feature = all_feature.numpy()

        V_qe = np.zeros_like(all_feature,dtype=np.float32)
        weights = np.logspace(0,alpha,k2).reshape((-1,1))
        for i in range(len(all_feature)):
            V_qe[i,:] = np.mean(all_feature[initial_rank[i,:k2],:]*weights,axis=0)
        # import pdb;pdb.set_trace()
        all_feature = V_qe
        del V_qe
        all_feature = torch.from_numpy(all_feature)

        fnorm = torch.norm(all_feature, p=2, dim=1, keepdim=True)
        all_feature = all_feature.div(fnorm.expand_as(all_feature))
    # aQE: weight query expansion
    if 0:
        k2 = 5
        alpha = 3
        # [todo] remove norma; normalize is used to to make sure the similiar one is itself
        all_feature = F.normalize(all_feature, p=2, dim=1)
        sims = torch.mm(all_feature, all_feature.t()).numpy()

        # [todo] heap sort
        initial_rank = sims.argsort(axis=1)[:,::-1]

        all_feature = all_feature.numpy()

        V_qe = np.zeros_like(all_feature,dtype=np.float32)

        # [todo] update query feature only?
        for i in range(len(all_feature)):
            # get weights from similarity
            weights = sims[i,initial_rank[i,:k2]].reshape((-1,1))
            # weights = (weights-weights.min())/(weights.max()-weights.min())
            weights = np.power(weights,alpha)
            # import pdb;pdb.set_trace()
            
            V_qe[i,:] = np.mean(all_feature[initial_rank[i,:k2],:]*weights,axis=0)
        # import pdb;pdb.set_trace()
        all_feature = V_qe
        del V_qe
        all_feature = torch.from_numpy(all_feature)
        all_feature = F.normalize(all_feature, p=2, dim=1)
     
    print('feature shape:',all_feature.size())

#
        # for k1 in range(8,15,2):
        #     for k2 in range(3,10,2):
        #         for l in range(3,8):
        #             p = l*0.1

    if n_randperm <=0 :
        gallery_feat = all_feature[query_num:]
        query_feat = all_feature[:query_num]
        
        if use_rerank:
            print('==> using rerank')
            distmat = re_rank(query_feat, gallery_feat, k1, k2, p)
        else:
            print('==> using euclidean_dist')
            distmat = euclidean_dist(query_feat, gallery_feat)

        cmc, mAP, _ = eval_func(distmat, np.array(qid_list), np.array(gid_list),
                np.array(qcid_list), np.array(gcid_list))
    else:
        torch.manual_seed(0)
        pids = np.array(qid_list+gid_list)
        camids = np.array(qcid_list+gcid_list)
        cmc = 0
        mAP = 0
        for i in range(n_randperm):
            index = torch.randperm(all_feature.size()[0])
        
            query_feat = all_feature[index][:num_query]
            gallery_feat = all_feature[index][num_query:]

            query_pid = pids[index.numpy()][:num_query]
            query_camid = camids[index.numpy()][:num_query]

            gallery_pid = pids[index.numpy()][num_query:]
            gallery_camid = camids[index.numpy()][num_query:]

            if use_rerank:
                print('==> using rerank')
                distmat = re_rank(query_feat, gallery_feat, k1, k2, p)
            else:
                print('==> using euclidean_dist')
                distmat = euclidean_dist(query_feat, gallery_feat)

            _cmc, _mAP, _ = eval_func(distmat, query_pid, gallery_pid,
                    query_camid, gallery_camid)
            cmc += _cmc/n_randperm
            mAP += _mAP/n_randperm

    print('Validation Result:')
    if use_rerank:
        print(str(k1) + "  -  " + str(k2) + "  -  " + str(p))
    print('mAP: {:.2%}'.format(mAP))
    for r in [1, 5, 10]:
        print('CMC Rank-{}: {:.2%}'.format(r, cmc[r - 1]))
    print('average of mAP and rank1: {:.2%}'.format((mAP+cmc[0])/2.0))

    with open(save_dir+'eval.txt', 'a') as f:
        if use_rerank:
            f.write('==> using rerank\n')
            f.write(str(k1)+"  -  "+str(k2)+"  -  "+str(p) + "\n")
        else:
            f.write('==> using euclidean_dist\n')

        f.write('mAP: {:.2%}'.format(mAP) + "\n")
        for r in [1, 5, 10]:
            f.write('CMC Rank-{}: {:.2%}'.format(r, cmc[r - 1])+"\n")
        f.write('average of mAP and rank1: {:.2%}\n'.format((mAP+cmc[0])/2.0))

        f.write('------------------------------------------\n')
        f.write('------------------------------------------\n')
        f.write('\n\n')



def inference_samples(model,  transform, batch_size, query_txt,query_dir,gallery_dir,save_dir,k1=20, k2=6, p=0.3, use_rerank=False,use_flip=False,max_rank=200):
    query_list = list()
    with open(query_txt, 'r') as f:
        lines = f.readlines()
        for i, line in enumerate(lines):
            data = line.split(" ")
            image_name = data[0].split("/")[1]
            img_file = os.path.join(query_dir, image_name)
            query_list.append(img_file)

    gallery_list = [os.path.join(gallery_dir, x) for x in os.listdir(gallery_dir)]
    query_num = len(query_list)
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
    with tqdm(total=iter_n) as pbar:
        for i in range(iter_n):
            # print("batch ----%d----" % (i))
            batch_data = img_data[i*batch_size:(i+1)*batch_size]
            with torch.no_grad():

                # ff = model(batch_data).data.cpu()

                # fnorm = torch.norm(ff, p=2, dim=1, keepdim=True)
                # ff = ff.div(fnorm.expand_as(ff))
                if use_flip:
                    ff = torch.FloatTensor(batch_data.size(0), 2048*2).zero_()
                    for i in range(2):
                        if i == 1:
                            batch_data = batch_data.index_select(3, torch.arange(batch_data.size(3) - 1, -1, -1).long())
                        outputs = model(batch_data)
                        f = outputs.data.cpu()

                        if i == 0:
                            fnorm = torch.norm(f, p=2, dim=1, keepdim=True)
                            f = f.div(fnorm.expand_as(f))
                            ff[:, :2048] = f
                        if i == 1:
                            fnorm = torch.norm(f, p=2, dim=1, keepdim=True)
                            f = f.div(fnorm.expand_as(f))
                            ff[:, 2048:] = f
                    fnorm = torch.norm(ff, p=2, dim=1, keepdim=True)
                    ff = ff.div(fnorm.expand_as(ff))
                else:

                    ff = model(batch_data).data.cpu()

                    fnorm = torch.norm(ff, p=2, dim=1, keepdim=True)
                    ff = ff.div(fnorm.expand_as(ff))

                all_feature.append(ff)
            pbar.update(1)
    all_feature = torch.cat(all_feature)
    gallery_feat = all_feature[query_num:]
    query_feat = all_feature[:query_num]

    if use_rerank:
        print("use re_rank")
        distmat = re_rank(query_feat, gallery_feat, k1, k2, p)
    else:
        distmat = euclidean_dist(query_feat, gallery_feat)
        distmat = distmat.numpy()
    num_q, num_g = distmat.shape
    indices = np.argsort(distmat, axis=1)

    max_200_indices = indices[:, :max_rank]

    res_dict = dict()
    for q_idx in range(num_q):
        # print(query_list[q_idx])
        # filename = query_list[q_idx][query_list[q_idx].rindex("\\")+1:]
        # max_200_files = [gallery_list[i][gallery_list[i].rindex("\\")+1:] for i in max_200_indices[q_idx]]
        filename = query_list[q_idx].split('/')[-1]
        max_200_files = [gallery_list[i].split('/')[-1] for i in max_200_indices[q_idx]]
        res_dict[filename] = max_200_files
    if use_rerank:
        with open(save_dir+'sub_rerank.json', 'w' ,encoding='utf-8') as f:
            json.dump(res_dict, f)
    else:
        with open(save_dir+'sub.json', 'w' ,encoding='utf-8') as f:
            json.dump(res_dict, f)


def pseudo_label_samples(model, query_list, gallery_list,  transform, batch_size, k1=20, k2=6, p=0.3):


    query_num = len(query_list)
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
        print("batch ----%d----" % (i))
        batch_data = img_data[i*batch_size:(i+1)*batch_size]
        with torch.no_grad():
            batch_feature = model(batch_data).detach().cpu()
            # ff = torch.FloatTensor(batch_data.size(0), 2048).zero_()
            # for i in range(2):
            #     if i == 1:
            #         batch_data = batch_data.index_select(3, torch.arange(batch_data.size(3) - 1, -1, -1).long())
            #
            #     outputs_1, outputs_2, outputs_3, outputs_4 = model(batch_data)
            #     outputs = torch.cat((outputs_1, outputs_2, outputs_3, outputs_4), 1)
            #     f = outputs.data.cpu()
            #     ff = ff + f
            #
            # fnorm = torch.norm(ff, p=2, dim=1, keepdim=True)
            # ff = ff.div(fnorm.expand_as(ff))
            all_feature.append(batch_feature)
    all_feature = torch.cat(all_feature)
    gallery_feat = all_feature[query_num:]
    query_feat = all_feature[:query_num]

    distmat = re_rank(query_feat, gallery_feat, k1, k2, p)
    distmat = distmat
    num_q, num_g = distmat.shape
    indices = np.argsort(distmat, axis=1)
    max_200_indices = indices[:, :200]

    res_dict = dict()
    pseudo_res = {"q_imgs": list(), "g_imgs": list(), "probs": list()}
    error_prob = {"q_imgs": list(), "g_imgs": list(), "probs": list()}
    true_prob = {"q_imgs": list(), "g_imgs": list(), "probs": list()}
    for q_idx in range(num_q):
        print(query_list[q_idx])
        filename = query_list[q_idx][query_list[q_idx].rindex("\\")+1:]
        max_200_files = [gallery_list[i][gallery_list[i].rindex("\\")+1:] for i in max_200_indices[q_idx]]
        probs = [distmat[q_idx, i] for i in max_200_indices[q_idx]]

        if max_200_files[0].split("_")[0] != filename.split("_")[0]:
            error_prob["q_imgs"].append(filename)
            error_prob["g_imgs"].append(max_200_files[0])
            error_prob["probs"].append(probs[0])
        for i, prob in enumerate(probs):
            if probs[0]<0.1:
                true_prob["q_imgs"].append(filename)
                true_prob["g_imgs"].append(max_200_files[i])
                true_prob["probs"].append(probs[i])

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

    parser = argparse.ArgumentParser(description="ReID Baseline Training")
    parser.add_argument("--config_file", default="", help="path to config file", type=str)
    parser.add_argument('--rerank',action='store_true',help='whether to rerank')
    parser.add_argument('--sub',action='store_true',help='whether to sub')

    parser.add_argument('--flip',action='store_true',help='whether to flip test')


    parser.add_argument("--k1", default=8, help="", type=int)
    parser.add_argument("--k2", default=3, help="", type=int)
    parser.add_argument("--lambda_value", default=0.8, help="", type=float)
    parser.add_argument("--max_rank", default=200, help="", type=int)



    parser.add_argument("--query_txt", default="", help="path to query file", type=str)
    parser.add_argument("--query_dir", default="", help="path to query file", type=str)
    parser.add_argument("--gallery_dir", default="", help="path to query file", type=str)



    parser.add_argument("opts", help="Modify config options using the command-line", default=None,nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.config_file != "":
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()


    output_dir = cfg.OUTPUT_DIR
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    num_gpus = torch.cuda.device_count()
    train_dl, val_dl, num_query, num_classes = make_dataloader(cfg, num_gpus)
    
    model = build_model(cfg, num_classes)
    param_dict = torch.load(cfg.TEST.WEIGHT)
    model = torch.nn.DataParallel(model)
    if cfg.SOLVER.SYNCBN:
        print("convert_model to syncbn")
        model = convert_model(model)
    model.load_state_dict(param_dict)
    transform = val_dl.dataset.transform
    print(transform)
    if args.query_txt =="":
        query_dir = cfg.DATASETS.DATA_PATH+cfg.DATASETS.QUERY_PATH
        gallery_dir = cfg.DATASETS.DATA_PATH+cfg.DATASETS.GALLERY_PATH
    else:
        query_dir = args.query_dir
        gallery_dir = args.gallery_dir

    if args.sub == False:
        inference_val(model, transform, 64, query_dir,gallery_dir,cfg.OUTPUT_DIR, args.k1,args.k2, args.lambda_value, use_rerank=args.rerank,use_flip=args.flip,n_randperm=cfg.TEST.RANDOMPERM)
    else:
        inference_samples(model, transform, 64,args.query_txt, query_dir,gallery_dir,cfg.OUTPUT_DIR, args.k1,args.k2, args.lambda_value, use_rerank=args.rerank,use_flip=args.flip,max_rank=args.max_rank)
    # transform = T.Compose([
    #         T.Resize((256, 128)),

    #         T.ToTensor(),
    #         # T.Normalize(mean=[0.09661545, 0.18356957, 0.21322473], std=[0.13422933, 0.14724616, 0.19259872])
    #     ])

    # query_list = list()
    # with open(r'E:\data\reid\初赛A榜测试集\初赛A榜测试集/query_a_list.txt', 'r') as f:
    #     lines = f.readlines()
    #     for i, line in enumerate(lines):
    #         data = line.split(" ")
    #         image_name = data[0].split("/")[1]
    #         img_file = os.path.join(r'E:\data\reid\初赛A榜测试集\初赛A榜测试集\query_a', image_name)
    #         query_list.append(img_file)
    #
    # gallery_list = [os.path.join(r'E:\data\reid\初赛A榜测试集\初赛A榜测试集\gallery_a', x) for x in
    #                 os.listdir(r'E:\data\reid\初赛A榜测试集\初赛A榜测试集\gallery_a')]
    # query_num = len(query_list)

    # inference_val(model, transform, 64, 2048,  6, 3, 0.8, use_rerank=True)
    # q_img_list = [os.path.join(r'E:\data\reid\dataset5\query', x) for x in os.listdir(r'E:\data\reid\dataset5\query')]
    # g_img_list = [os.path.join(r'E:\data\reid\dataset5\gallery', x) for x in os.listdir(r'E:\data\reid\dataset5\gallery')]
    # pseudo_label_samples(model, query_list, gallery_list, transform, 16, 15, 3, 0.7)
    # inference_samples(model, transform, 64, 2048, 8, 3, 0.8, True)
    # tta_inference_samples(model, 64, 15, 3, 0.7)
    # inference_val_with_tta(model, 64)

    # batch_size = 64





    # k1_list = [6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    # k2_list = [2, 3, 4, 5, 6]
    # p_list = [0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    # for k1 in k1_list:
    #     for k2 in k2_list:
    #         for p in p_list:
    #             inference_val(model, transform, 64, 2048,  k1, k2, p, True)






