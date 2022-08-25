import os
import numpy as np
from PIL import Image
os.environ["OPENCV_IO_MAX_IMAGE_PIXELS"] = pow(2, 40).__str__()
import cv2
from osgeo import gdal_array
from tqdm import tqdm
import random
import shutil


Image.MAX_IMAGE_PIXELS = 1000000000000000
TARGET_W, TARGET_H = 512, 512
stride = TARGET_W
count = 0

def cut_images(image_name, image_path, save_dir, is_label, count):
    # 初始化路径
    # image_save_dir = os.path.join(save_dir, "crops_" + str(count) + "//")
    image_save_dir = os.path.join(save_dir, "crops/")
    if not os.path.exists(image_save_dir): os.makedirs(image_save_dir)

    n = 0
    target_w, target_h = TARGET_W, TARGET_H

    # print(image_path)
    image = np.asarray(cv2.imread(image_path))
    # print(image.shape)
    # image = cv2.imread(image_path, 1)
    # print("=========", image.shape[0], image.shape[1])
    h, w = image.shape[0], image.shape[1]
    # print("origal size: ", w, h)
    # new_h, new_w = h, w
    if (w - target_w) % stride:
        new_w = ((w - target_w) // stride + 1) * stride + target_w
    if (h - target_h) % stride:
        new_h = ((h - target_h) // stride + 1) * stride + target_h
    image = cv2.copyMakeBorder(image, 0, new_h - h, 0, new_w - w, cv2.BORDER_CONSTANT, 0)
    # label = cv.copyMakeBorder(label, 0, new_h - h, 0, new_w - w, cv.BORDER_CONSTANT, 1)
    h, w = image.shape[0], image.shape[1]
    # print("padding to : ", w, h)

    def crop(cnt, crop_image, n):
        _name = image_name.split(".")[0]
        image_save_path = os.path.join(image_save_dir, _name + "_" + str(cnt[1]) + "_" + str(cnt[0]) + ".png")
        #  按照 1 2 3命名
        # image_save_path = os.path.join(image_save_dir, str(n) + ".jpg")

        cv2.imwrite(image_save_path, crop_image)

    h, w = image.shape[0], image.shape[1]
    for i in tqdm(range((w - target_w) // stride + 1)):
        for j in range((h - target_h) // stride + 1):
            topleft_x = i * stride
            topleft_y = j * stride

            # 处理label 转8位
            if is_label:
                crop_image = image[topleft_y:topleft_y + target_h, topleft_x:topleft_x + target_w]
                gray_image = cv2.cvtColor(crop_image, cv2.COLOR_BGR2GRAY)
                crop((i, j), gray_image, n)
                n += 1
            else:
                # 处理image
                crop_image = image[topleft_y:topleft_y + target_h, topleft_x:topleft_x + target_w]
                # if len(np.flatnonzero(crop_image)) / (TARGET_H * TARGET_W) >= 0.75:
                crop((i, j), crop_image, n)
                n += 1


if __name__ == "__main__":
    # 单个文件

    # data_dir = r"I:\temp\build_512\orinal"
    # img_name1 = "mosaic_2013panyu13.TIF"
    # cut_images(img_name1, os.path.join(data_dir, img_name1), data_dir)

    # 文件夹
    path = r"G:\dataset\tianchi\jingwei_round1_train_20190619"
    is_label = False

    name_list = [x for x in os.listdir(path) if x.endswith(".png")]
    for file in name_list:
        # print(file)
        cut_images(file, os.path.join(path, file), path, is_label, count)
        count += 1



# def crop_img(img, cropsize, overlap):
#     """
#     裁剪图像为指定格式并保存成tiff
#     输入为array形式的数组
#     """
#     num = 0
#     height = img.shape[1]
#     width = img.shape[2]
#     print(height)
#     print(width)
#
#     # 从左上开始裁剪
#     for i in range(int(height / (cropsize * (1 - overlap)))):  # 行裁剪次数
#         for j in range(int(width / (cropsize * (1 - overlap)))):  # 列裁剪次数
#             cropped = img[:,  # 通道不裁剪
#                       int(i * cropsize * (1 - overlap)): int(i * cropsize * (1 - overlap) + cropsize),
#                       int(j * cropsize * (1 - overlap)): int(j * cropsize * (1 - overlap) + cropsize),
#                       ]  # max函数是为了防止i，j为0时索引为负数
#
#             num = num + 1
#             target = 'tiff_crop' + '/cropped{n}.tif'.format(n=num)
#             gdal_array.SaveArray(cropped, target, format="GTiff")
#
#     #  向前裁剪最后的列
#     for i in range(int(height / (cropsize * (1 - overlap)))):
#         cropped = img[:,  # 通道不裁剪
#                   int(i * cropsize * (1 - overlap)): int(i * cropsize * (1 - overlap) + cropsize),  # 所有行
#                   width - cropsize: width,  # 最后256列
#                   ]
#
#         num = num + 1
#         target = 'tiff_crop' + '/cropped{n}.tif'.format(n=num)
#         gdal_array.SaveArray(cropped, target, format="GTiff")
#
#     # 向前裁剪最后的行
#     for j in range(int(width / (cropsize * (1 - overlap)))):
#         cropped = img[:,  # 通道不裁剪
#                   height - cropsize: height,  # 最后256行
#                   int(j * cropsize * (1 - overlap)): int(j * cropsize * (1 - overlap) + cropsize),  # 所有列
#                   ]
#
#         num = num + 1
#         target = 'tiff_crop' + '/cropped{n}.tif'.format(n=num)
#         gdal_array.SaveArray(cropped, target, format="GTiff")
#
#
#     # 裁剪右下角
#     cropped = img[:,  # 通道不裁剪
#               height - cropsize: height,
#               width - cropsize: width,
#               ]
#
#     num = num + 1
#     target = 'tiff_crop' + '/cropped{n}.tif'.format(n=num)
#     gdal_array.SaveArray(cropped, target, format="GTiff")
#
# def read_landsat8_bands(base_path):
#     """保存landsat8不同波段的路径(共11个波段)
#     base_path: 存储了多张影像的文件夹
#     mid_path: 存储了不同波段的文件夹，对应着单张影像
#     final_path: 最后一层可以直接打开的单张波段文件
#
#     bands：包含了波段路径的列表
#     """
#
#     # 用于存储不同波段路径,维度是影像数量*波段数（11）
#     bands = []
#     num_bands = 11
#
#     # 用于定位波段用的关键字列表
#     keys = []
#     for k in range(num_bands):
#         key = 'B{num}.TIF'.format(num = k + 1)
#         keys.append(key)
#
#     # 读取最外层文件
#     base_files = os.listdir(base_path)
#     for i in range(len(base_files)):
#         bands.append([])
#
#         # 读取中层文件
#         mid_path = base_path + '\\' + base_files[i]
#         mid_file = os.listdir(mid_path)
#
#         # 得到最内层的波段文件
#         for final_file in mid_file:
#             final_path = mid_path + '\\' + final_file
#
#             for j in range(num_bands):
#                 if keys[j] in final_file:
#                     bands[i].append(final_path)
#
#         # 原始列表排序是1,10,11,2,3，...
#         # 按照倒数第5个字符进行排序（XXXB1.TIF）
#         bands[i].sort(key=lambda arr: (arr[:-5], int(arr[-5])))
#
#     # 返回波段列表和影像数量
#     return bands
#
#
# def get_img(base_path):
#     """
#     叠加波段的简单demo
#     与mask矩阵一起构建成3维数组结构"""
#     bands = read_landsat8_bands(base_path)
#
#     # 读取数据
#     B1_gdal = gdal_array.LoadFile(bands[0][0])
#     B2_gdal = gdal_array.LoadFile(bands[0][1])
#     B3_gdal = gdal_array.LoadFile(bands[0][2])
#
#     # 转化成ndarray形式
#     B1_np = np.array(B1_gdal)
#     B2_np = np.array(B2_gdal)
#     B3_np = np.array(B3_gdal)
#     print(B1_np.shape)
#
#     B123 = np.stack([B1_np, B2_np, B3_np], axis=0)
#     print(B123.shape)  # 3,7301,7341
#
#     # 构建0-1 mask矩阵
#     height = B123.shape[1]
#     width = B123.shape[2]
#     mask = np.random.randint(0, 2, (1, height, width))
#
#     # 按照通道堆叠
#     img = np.concatenate([B123, mask], axis=0)
#
#     return img
#
# if __name__ == '__main__':
#     base_path = 'I:/test/12'
#     img = get_img(base_path)
#     cropsize = 256  # 裁剪尺寸
#     overlap = 0 # 重叠率
#     crop_img(img, cropsize, overlap)
#     print('finish')