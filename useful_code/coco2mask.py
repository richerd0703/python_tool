# -*- coding:utf-8 -*-

from __future__ import print_function

import os
from pycocotools.coco import COCO
import os, sys, zipfile
import urllib.request
import shutil
import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt
import pylab

pylab.rcParams['figure.figsize'] = (8.0, 10.0)

import os
from tqdm import tqdm
import json

# json文件的地址 需要手动设置

def single_obg(json_file, save_path):
    # person_keypoints_val2017.json  # Object Keypoint 类型的标注格式
    # captions_val2017.json  # Image Caption的标注格式
    data = json.load(open(json_file, 'r'))
    # 设置需要提取的图片数量 我设置提取82000张
    for i in tqdm(range(8000)):
        data_2 = {}
        data_2['info'] = data['info']
        # data_2['licenses'] = data['licenses']
        data_2['images'] = [data['images'][i]]  # 只提取第一张图片
        data_2['categories'] = data['categories']
        annotation = []
        # 通过imgID 找到其所有对象
        imgID = data_2['images'][0]['id']
        for ann in data['annotations']:
            if ann['image_id'] == imgID:
                annotation.append(ann)
        data_2['annotations'] = annotation
        # 保存到新的JSON文件，便于查看数据特点
        # img_file 获取图片名称
        img_file = data_2['images'][0]['file_name']
        img_first = img_file.split(".")[0]
        # 将提取出的图片写入data.txt文件中并换行 (optional)
        # with open('./coco_single_object/data.txt',mode='a') as f:
        #         f.write(img_file)
        #         f.write("\n")
        # 设置存储目录 我的是存在当前目录下coco_single_object文件夹下 需要手动创建空文件夹
        # json.dump(data_2, open(r'G:\dataset\mapping_challenge_dataset\raw\train\singal_obj' + img_first + '.json', 'w'), indent=4)  # indent=4 更加美观显示
        json.dump(data_2, open(os.path.join(save_path, img_first + '.json'), 'w'), indent=4)  # indent=4 更加美观显示


def json2mask(json_path, img_path, color_img_save, binary_img_save):
    # json_path json文件路径  从coco数据集的annotations标注json文件中提取出的单个json文件
    #  img_path 原图目录   下载coco数据集时的原图目录
    # color_img_save 原图存放目录
    # binary_img_save 二值图存放目录
    dir = os.listdir(json_path)
    for jfile in dir:
        annFile = os.path.join(json_path, jfile)
        coco = COCO(annFile)
        imgIds = coco.getImgIds()
        img = coco.loadImgs(imgIds[0])[0]
        dataDir = img_path
        shutil.copy(os.path.join(dataDir, img['file_name']), color_img_save)
        # load and display instance annotations
        # 加载实例掩膜
        catIds = []
        for ann in coco.dataset['annotations']:
            if ann['image_id'] == imgIds[0]:
                catIds.append(ann['category_id'])
        annIds = coco.getAnnIds(imgIds=img['id'], catIds=catIds, iscrowd=None)
        width = img['width']
        height = img['height']
        anns = coco.loadAnns(annIds)
        mask_pic = np.zeros((height, width))
        for single in anns:
            mask_single = coco.annToMask(single)
            mask_pic += mask_single
        for row in range(height):
            for col in range(width):
                if (mask_pic[row][col] > 0):
                    mask_pic[row][col] = 255
        imgs = np.zeros(shape=(height, width, 3), dtype=np.float32)
        imgs[:, :, 0] = mask_pic[:, :]
        imgs[:, :, 1] = mask_pic[:, :]
        imgs[:, :, 2] = mask_pic[:, :]
        imgs = imgs.astype(int)
        img_name = img['file_name'].split(".")[0]
        # plt.imsave(binary_img_save + "/" + img_name + ".png", imgs)
        plt.imsave(os.path.join(binary_img_save, img_name + ".png"), imgs.astype(np.uint8))


if __name__ == '__main__':
    json_file = r'G:\dataset\mapping_challenge_dataset\raw\train\annotation.json'  # # Object Instance 类型的标注
    single_savepath = r"G:\dataset\mapping_challenge_dataset\raw\train\singal_obj"
    # single_obg(json_file, single_savepath)

    json_path = r"G:\dataset\mapping_challenge_dataset\raw\val\singal_obj"
    img_path = r"G:\dataset\mapping_challenge_dataset\raw\val\images"
    color_img_save = r"G:\dataset\mapping_challenge_dataset\raw\val\color"
    binary_img_save = r"G:\dataset\mapping_challenge_dataset\raw\val\mask"
    json2mask(json_path, img_path, color_img_save, binary_img_save)


# # -*- coding: utf-8 -*-
# """
# Created on Wed Jul  1 14:45:07 2020
#
# @author: mhshao
# """
# from pycocotools.coco import COCO
# import os
# import shutil
# from tqdm import tqdm
# import matplotlib.pyplot as plt
# import cv2
# from PIL import Image, ImageDraw
# import skimage.io as io
# import json
# import numpy as np
#
# '''
# 路径参数
# '''
# # 原coco数据集的路径
# dataDir = 'newdata/'
# # 用于保存新生成的mask数据的路径
# savepath = "newdata/"
#
# '''
# 数据集参数
# '''
# # coco有80类，这里写要进行二值化的类的名字
# # 其他没写的会被当做背景变成黑色
# # 如我只需要car、bus、truck这三类数据
# classes_names = ['car', 'bus', 'truck']
# # 要处理的数据集，比如val2017、train2017等
# # 不建议多个数据集在一个list中
# # 一次提取一个数据集安全点_(:3」∠❀)_
# datasets_list = ['val2017']
#
#
# # 生成保存路径，函数抄的(›´ω`‹ )
# # if the dir is not exists,make it,else delete it
# def mkr(path):
#     if os.path.exists(path):
#         shutil.rmtree(path)
#         os.mkdir(path)
#     else:
#         os.mkdir(path)
#
#
# # 生成mask图
# def mask_generator(coco, width, height, anns_list):
#     mask_pic = np.zeros((height, width))
#     # 生成mask
#     for single in anns_list:
#         mask_single = coco.annToMask(single)
#         mask_pic += mask_single
#     # 转化为255
#     for row in range(height):
#         for col in range(width):
#             if (mask_pic[row][col] > 0):
#                 mask_pic[row][col] = 255
#     mask_pic = mask_pic.astype(int)
#     '''
#     #转为三通道
#     imgs = np.zeros(shape=(height, width, 3), dtype=np.float32)
#     imgs[:, :, 0] = mask_pic[:, :]
#     imgs[:, :, 1] = mask_pic[:, :]
#     imgs[:, :, 2] = mask_pic[:, :]
#     imgs = imgs.astype(int)
#     '''
#     return mask_pic
#
#
# # 处理json数据并保存二值mask
# def get_mask_data(annFile, mask_to_save):
#     # 获取COCO_json的数据
#     coco = COCO(annFile)
#     # 拿到所有需要的图片数据的id
#     classes_ids = coco.getCatIds(catNms=classes_names)
#     # 取所有类别的并集的所有图片id
#     # 如果想要交集，不需要循环，直接把所有类别作为参数输入，即可得到所有类别都包含的图片
#     imgIds_list = []
#     for idx in classes_ids:
#         imgidx = coco.getImgIds(catIds=idx)
#         imgIds_list += imgidx
#     # 去除重复的图片
#     imgIds_list = list(set(imgIds_list))
#
#     # 一次性获取所有图像的信息
#     image_info_list = coco.loadImgs(imgIds_list)
#
#     # 对每张图片生成一个mask
#     for imageinfo in image_info_list:
#         # 获取对应类别的分割信息
#         annIds = coco.getAnnIds(imgIds=imageinfo['id'], catIds=classes_ids, iscrowd=None)
#         anns_list = coco.loadAnns(annIds)
#         # 生成二值mask图
#         mask_image = mask_generator(coco, imageinfo['width'], imageinfo['height'], anns_list)
#         # 保存图片
#         file_name = mask_to_save + '/' + imageinfo['file_name'][:-4] + '.png'
#         plt.imsave(file_name, mask_image)
#         # 按单个数据集进行处理
#         for dataset in datasets_list:
#         # 用来保存最后生成的mask图像目录
#             mask_to_save = savepath + 'masks/' + dataset
#         mkr(savepath + 'masks/')
#         # 生成路径
#         mkr(mask_to_save)
#
#         # 获取要处理的json文件路径
#         # 我这里用了之前自己生成的部分类别json
#         # 具体方法见我前一篇博客
#         annFile = '{}/annotations/instances_{}_sub.json'.format(dataDir, dataset)
#         # 处理数据
#         get_mask_data(annFile, mask_to_save)
#         print('Got all the masks of {} from {}'.format(classes_names, dataset))
