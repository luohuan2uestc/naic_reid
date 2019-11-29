# -*- encoding: utf-8 -*-
'''
@File    :   prepare_data.py
@Contact :   whut.hexin@foxmail.com
@License :   (C)Copyright 2017-2020, HeXin

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2019/10/28 16:11   xin      1.0         None
'''
import os
import shutil
import pandas as pd
import random

from dataset.data import read_image
from skimage.io import imsave, imread
from PIL import ImageFile, Image


def process_dataset(txt_label, root_path, save_path):
    with open(txt_label, 'r') as f:
        lines = f.readlines()
        for i, line in enumerate(lines):
            data = line.split(" ")
            image_name = data[0].split("/")[1]
            pid = data[1].strip("\n")
            if not os.path.exists(os.path.join(save_path, pid)):
                os.mkdir(os.path.join(save_path, pid))
            new_filename = pid+"_c"+str(i)+".png"
            shutil.copy(os.path.join(root_path, image_name), os.path.join(os.path.join(save_path, pid), new_filename))


def dataset_analyse(root_path):
    pids = os.listdir(root_path)
    counts = list()
    for pid in pids:
        imgs = os.listdir(os.path.join(root_path, pid))
        counts.append(len(imgs))
    columns = [u'pid', u'count']
    save_df = pd.DataFrame({u'pid': pids, u'count': counts},
                           columns=columns)
    save_df.to_csv('dataset_analyse.csv')


def split_dataset(root_path, train_path, query_path, gallery):
    pids = os.listdir(root_path)

    for pid in pids:

        imgs = os.listdir(os.path.join(root_path, pid))

        for img in imgs:
            shutil.copy(os.path.join(os.path.join(root_path, pid), img), os.path.join(train_path, img))
            # k = k+1
            # img1 = str(int(pid) + 6000)+"_"+str(k)+".jpg"
            # shutil.copy(os.path.join(os.path.join(root_path, pid), img), os.path.join(train_path, img1))

        # if len(imgs) > 100:
        #     for img in imgs:
        #         shutil.copy(os.path.join(os.path.join(root_path, pid), img), os.path.join(train_path, img))
        #
        # elif len(imgs)<2:
        #     for img in imgs:
        #         shutil.copy(os.path.join(os.path.join(root_path, pid), img), os.path.join(gallery, img))
        # # elif len(imgs) < 4:
        # #     random.shuffle(imgs)
        # #     for img in imgs[:1]:
        # #         shutil.copy(os.path.join(os.path.join(root_path, pid), img), os.path.join(query_path, img))
        # #     for img in imgs[1:]:
        # #         shutil.copy(os.path.join(os.path.join(root_path, pid), img), os.path.join(gallery, img))
        # else:
        #     for img in imgs:
        #         shutil.copy(os.path.join(os.path.join(root_path, pid), img), os.path.join(train_path, img))
            # p = random.random()
            # if p < 0.8:
            #     for img in imgs:
            #         shutil.copy(os.path.join(os.path.join(root_path, pid), img), os.path.join(train_path, img))
            # else:
            #     random.shuffle(imgs)
            #     for img in imgs[:1]:
            #         shutil.copy(os.path.join(os.path.join(root_path, pid), img), os.path.join(query_path, img))
            #     for img in imgs[1:]:
            #         shutil.copy(os.path.join(os.path.join(root_path, pid), img), os.path.join(gallery, img))

        # if len(imgs) > 100:
        #     for img in imgs:
        #         shutil.copy(os.path.join(os.path.join(root_path, pid), img), os.path.join(train_path, img))
        #     # random.shuffle(imgs)
        #     # for img in imgs[:int(0.7*len(imgs))]:
        #     #     shutil.copy(os.path.join(os.path.join(root_path, pid), img), os.path.join(train_path, img))
        #     # for img in imgs[int(0.7 * len(imgs)):int(0.85 * len(imgs))]:
        #     #     shutil.copy(os.path.join(os.path.join(root_path, pid), img), os.path.join(query_path, img))
        #     # for img in imgs[int(0.85 * len(imgs)):]:
        #     #     shutil.copy(os.path.join(os.path.join(root_path, pid), img), os.path.join(gallery, img))
        # elif len(imgs) < 2:
        #     p = random.random()
        #     if p < 0.8:
        #         for img in imgs:
        #             shutil.copy(os.path.join(os.path.join(root_path, pid), img), os.path.join(train_path, img))
        #     else:
        #         for img in imgs:
        #             shutil.copy(os.path.join(os.path.join(root_path, pid), img), os.path.join(gallery, img))
        #
        # elif len(imgs) < 5:
        #     p = random.random()
        #     if p < 0.8:
        #         for img in imgs:
        #             shutil.copy(os.path.join(os.path.join(root_path, pid), img), os.path.join(train_path, img))
        #     else:
        #         random.shuffle(imgs)
        #         for img in imgs[:1]:
        #             shutil.copy(os.path.join(os.path.join(root_path, pid), img), os.path.join(query_path, img))
        #         for img in imgs[1:]:
        #             shutil.copy(os.path.join(os.path.join(root_path, pid), img), os.path.join(gallery, img))
        # else:
        #     p = random.random()
        #     if p < 0.8:
        #         for img in imgs:
        #             shutil.copy(os.path.join(os.path.join(root_path, pid), img), os.path.join(train_path, img))
        #     else:
        #         random.shuffle(imgs)
        #         for img in imgs[:int(0.20*len(imgs))]:
        #             shutil.copy(os.path.join(os.path.join(root_path, pid), img), os.path.join(query_path, img))
        #         for img in imgs[int(0.20*len(imgs)):]:
        #             shutil.copy(os.path.join(os.path.join(root_path, pid), img), os.path.join(gallery, img))

def mk_pseudo_data(root_path, txt_label, csv_data, save_path):
    query_dic = dict()
    with open(txt_label, 'r') as f:
        lines = f.readlines()
        for i, line in enumerate(lines):
            data = line.split(" ")
            image_name = data[0].split("/")[1]
            pid = data[1].strip("\n")
            query_dic[image_name] = pid
    pseudo_csv_data = pd.read_csv(csv_data)
    for query_file, gallery_file in zip(pseudo_csv_data['q_imgs'], pseudo_csv_data['g_imgs']):
        pid = query_dic[query_file]
        # if not os.path.exists(os.path.join(save_path, pid)):
        #     os.mkdir(os.path.join(save_path, pid))
        new_query_filename = pid + "_c" + query_file
        shutil.copy(os.path.join(root_path, query_file), os.path.join(save_path, new_query_filename))
        new_gallery_filename = pid + "_c" + gallery_file
        shutil.copy(os.path.join(root_path, gallery_file), os.path.join(save_path, new_gallery_filename))


if __name__ == "__main__":
    pass
    # mk_pseudo_data(r'E:\data\reid\初赛A榜测试集\初赛A榜测试集\all', r'E:\data\reid\初赛A榜测试集\初赛A榜测试集/query_a_list.txt',
    #                r'E:\src\python\reid\true_pseudo_res.csv', r'E:\data\reid\初赛A榜测试集\初赛A榜测试集\pseudo_label')
    #

    # from skimage.io import imread
    # import numpy as np
    #
    # means = [0, 0, 0]
    # stdevs = [0, 0, 0]
    #
    # num_imgs = 0
    # img_files = [os.path.join(r'E:\data\reid\初赛训练集\初赛训练集\train_set', x) for x in os.listdir(r'E:\data\reid\初赛训练集\初赛训练集\train_set')]
    # for img_file in img_files:
    #     print(img_file, len(img_files)-num_imgs)
    #     num_imgs += 1
    #     img = imread(img_file)
    #     img = img.astype(np.float32) / 255.
    #     for i in range(3):
    #         means[i] += img[:, :, i].mean()
    #         stdevs[i] += img[:, :, i].std()
    #
    # means = np.asarray(means) / num_imgs
    # stdevs = np.asarray(stdevs) / num_imgs
    #
    # print("normMean = {}".format(means))
    # print("normStd = {}".format(stdevs))
    # print('transforms.Normalize(normMean = {}, normStd = {})'.format(means, stdevs))




    # with open(r'E:\data\reid\初赛A榜测试集\初赛A榜测试集/query_a_list.txt', 'r') as f:
    #     lines = f.readlines()
    #     for i, line in enumerate(lines):
    #         data = line.split(" ")
    #         image_name = data[0].split("/")[1]
    #         pid = data[1].strip("\n")
    #
    #         new_filename = pid + "_c" + str(i) + ".png"
    #         shutil.copy(os.path.join(r'E:\data\reid\初赛A榜测试集\初赛A榜测试集\query_a', image_name),
    #                     os.path.join(r'E:\data\reid\初赛A榜测试集\初赛A榜测试集\gg', new_filename))

    # pid_list = []
    # for img in os.listdir(r'E:\data\reid\初赛A榜测试集\初赛A榜测试集\gg'):
    #     pid = img.split("_")[0]
    #     if pid not in pid_list:
    #         pid_list.append(pid)
    # print(len(pid_list))



    split_dataset(r'E:\data\reid\初赛训练集\初赛训练集\all_dataset', r'E:\data\reid\dataset10\train', r'E:\data\reid\dataset10\query',
                  r'E:\data\reid\dataset10\gallery')
    # dataset_analyse(r'E:\data\reid\初赛训练集\初赛训练集\all_dataset')
    # process_dataset(r'E:\data\reid\初赛训练集\初赛训练集/train_list.txt', r'E:\data\reid\初赛训练集\初赛训练集\train_set', r'E:\data\reid\初赛训练集\初赛训练集\all_dataset')