# -*- coding: utf-8 -*-
import os
import gdal
import numpy as np
import cv2


#读取Tif数据集
def readTif(fileName):
    dataset = gdal.Open(fileName)
    if dataset == None:
        print(fileName + "文件无法打开")
    print(dataset)
    return dataset


#滑动裁剪主题函数
def TifCrop(TifPath, SavePath, Size, RepetitionRate):
    """
    :param TifPath: 需要裁剪的图像
    :param SavePath: 保存路径
    :param Size: 裁剪大小
    :param RepetitionRate:重复率
    :return:
    """
    dataset_img = readTif(TifPath)
    width = dataset_img.RasterXSize
    height = dataset_img.RasterYSize
    proj = dataset_img.GetProjection()
    geotrans = dataset_img.GetGeoTransform()
    img = dataset_img.ReadAsArray(0, 0, width, height)  # 获取数据

    # num = len(os.listdir(SavePath)) + 1  # 获取当前文件夹的文件个数len,并以len+1命名即将裁剪得到的图像
    # num = 0
    datatype = dataset_img.GetRasterBand(1).DataType
    outbandsize = dataset_img.RasterCount
    im_band1 = dataset_img.GetRasterBand(1)
    im_band2 = dataset_img.GetRasterBand(2)
    im_band3 = dataset_img.GetRasterBand(3)
    # 四通道图
    # im_band4 = dataset_img.GetRasterBand(4)


    col_num1 = int((height - (Size * RepetitionRate)) / (Size * (1 - RepetitionRate)))
    row_num1 = int((width - (Size * RepetitionRate)) / (Size * (1 - RepetitionRate)))
    l = 0
    if height % Size == 0:
        l = height // Size
    else:
        l = height // Size + 1

    # 三通道图：
    if len(img.shape) == 3:
        img_with_suffix = TifPath.split('\\')[-1]
        img_without_suffix = img_with_suffix.split('.')[0]
        num = 0
        for i in range(l):
            for j in range(l):
                num += 1
                offset_x = i * Size * (1 - RepetitionRate)
                offset_y = j * Size * (1 - RepetitionRate)

                if (height - offset_y < 1000):
                    offset_y = height - Size

                if (width - offset_x < 1000):
                    offset_x = width - Size
                out_band1 = im_band1.ReadAsArray(offset_x, offset_y, Size, Size)
                out_band2 = im_band2.ReadAsArray(offset_x, offset_y, Size, Size)
                out_band3 = im_band3.ReadAsArray(offset_x, offset_y, Size, Size)
                # out_band4 = im_band4.ReadAsArray(offset_x, offset_y, Size, Size)
                gtif_driver = gdal.GetDriverByName("GTiff")
                # file = os.path.join(SavePath, 'hagfclip_%04d.tif' % num)
                file = SavePath + '//' + img_without_suffix + '_' + str(num) + '.jpg'
                out_ds = gtif_driver.Create(file, Size, Size, 3, datatype)
                print("create new tif file succeed")
                ori_transform = dataset_img.GetGeoTransform()
                if ori_transform:
                    print(ori_transform)
                    print("Origin = ({}, {})".format(ori_transform[0], ori_transform[3]))
                    print("Pixel Size = ({}, {})".format(ori_transform[1], ori_transform[5]))
                top_left_x = ori_transform[0]  # 左上角x坐标
                w_e_pixel_resolution = ori_transform[1]  # 东西方向像素分辨率
                top_left_y = ori_transform[3]  # 左上角y坐标
                n_s_pixel_resolution = ori_transform[5]  # 南北方向像素分辨率
                top_left_x = top_left_x + offset_x * w_e_pixel_resolution
                top_left_y = top_left_y + offset_y * n_s_pixel_resolution
                dst_transform = (
                top_left_x, ori_transform[1], ori_transform[2], top_left_y, ori_transform[4], ori_transform[5])
                out_ds.SetGeoTransform(dst_transform)
                out_ds.SetProjection(dataset_img.GetProjection())
                out_ds.GetRasterBand(1).WriteArray(out_band1)
                out_ds.GetRasterBand(2).WriteArray(out_band2)
                out_ds.GetRasterBand(3).WriteArray(out_band3)
                # out_ds.GetRasterBand(4).WriteArray(out_band4)
                out_ds.FlushCache()
                print("FlushCache succeed")
                # del out_ds, out_band1, out_band2, out_band3, out_band4
                del out_ds, out_band1, out_band2, out_band3
    # del dataset_img

        # # 裁剪最后一列  (包含右下角)
        # for i in range(0, row_num1 + 1):
        #     offset_x = width - Size
        #     offset_y = i * Size
        #     if (height - offset_y < 1000):
        #         offset_y = height - Size
        #     out_band1 = im_band1.ReadAsArray(offset_x, offset_y, Size, Size)
        #     out_band2 = im_band2.ReadAsArray(offset_x, offset_y, Size, Size)
        #     out_band3 = im_band3.ReadAsArray(offset_x, offset_y, Size, Size)
        #     out_band4 = im_band4.ReadAsArray(offset_x, offset_y, Size, Size)
        #     gtif_driver = gdal.GetDriverByName("GTiff")
        #     file = os.path.join(SavePath, 'hagfclip_%04d.tif' % num)
        #     num += 1
        #     out_ds = gtif_driver.Create(file, Size, Size, outbandsize, datatype)
        #     print("create new tif file succeed")
        #     ori_transform = dataset_img.GetGeoTransform()
        #     if ori_transform:
        #         print(ori_transform)
        #         print("Origin = ({}, {})".format(ori_transform[0], ori_transform[3]))
        #         print("Pixel Size = ({}, {})".format(ori_transform[1], ori_transform[5]))
        #     top_left_x = ori_transform[0]
        #     w_e_pixel_resolution = ori_transform[1]
        #     top_left_y = ori_transform[3]
        #     n_s_pixel_resolution = ori_transform[5]
        #     top_left_x = top_left_x + offset_x * w_e_pixel_resolution
        #     top_left_y = top_left_y + offset_y * n_s_pixel_resolution
        #     dst_transform = (
        #     top_left_x, ori_transform[1], ori_transform[2], top_left_y, ori_transform[4], ori_transform[5])
        #     out_ds.SetGeoTransform(dst_transform)
        #     out_ds.SetProjection(dataset_img.GetProjection())
        #     out_ds.GetRasterBand(1).WriteArray(out_band1)
        #     out_ds.GetRasterBand(2).WriteArray(out_band2)
        #     out_ds.GetRasterBand(3).WriteArray(out_band3)
        #
        #     out_ds.GetRasterBand(4).WriteArray(out_band4)
        #
        #     out_ds.FlushCache()
        #     print("FlushCache succeed")
        #     del out_ds, out_band1, out_band2, out_band3, out_band4

        # # 裁剪最后一行( 同样包含右下角，手动剔除相同的那张)
        # for i in range(0, col_num1 + 1):
        #     offset_x = i * Size
        #     if (width - offset_x < 1000):
        #         offset_x = width - Size
        #     offset_y = height - Size
        #     out_band1 = im_band1.ReadAsArray(offset_x, offset_y, Size, Size)
        #     out_band2 = im_band2.ReadAsArray(offset_x, offset_y, Size, Size)
        #     out_band3 = im_band3.ReadAsArray(offset_x, offset_y, Size, Size)
        #
        #     out_band4 = im_band4.ReadAsArray(offset_x, offset_y, Size, Size)
        #
        #
        #     gtif_driver = gdal.GetDriverByName("GTiff")
        #     file = os.path.join(SavePath, 'hagfclip_%04d.tif' % num)
        #     num += 1
        #     out_ds = gtif_driver.Create(file, Size, Size, outbandsize, datatype)
        #     print("create new tif file succeed")
        #     ori_transform = dataset_img.GetGeoTransform()
        #     if ori_transform:
        #         print(ori_transform)
        #         print("Origin = ({}, {})".format(ori_transform[0], ori_transform[3]))
        #         print("Pixel Size = ({}, {})".format(ori_transform[1], ori_transform[5]))
        #     top_left_x = ori_transform[0]
        #     w_e_pixel_resolution = ori_transform[1]
        #     top_left_y = ori_transform[3]
        #     n_s_pixel_resolution = ori_transform[5]
        #     top_left_x = top_left_x + offset_x * w_e_pixel_resolution
        #     top_left_y = top_left_y + offset_y * n_s_pixel_resolution
        #     dst_transform = (
        #     top_left_x, ori_transform[1], ori_transform[2], top_left_y, ori_transform[4], ori_transform[5])
        #     out_ds.SetGeoTransform(dst_transform)
        #     out_ds.SetProjection(dataset_img.GetProjection())
        #     out_ds.GetRasterBand(1).WriteArray(out_band1)
        #     out_ds.GetRasterBand(2).WriteArray(out_band2)
        #     out_ds.GetRasterBand(3).WriteArray(out_band3)
        #
        #     out_ds.GetRasterBand(4).WriteArray(out_band4)
        #
        #     out_ds.FlushCache()
        #     print("FlushCache succeed")
        #     del out_ds, out_band1, out_band2, out_band3, out_band4

    if len(img.shape) == 2:
        num = 0
        img_with_suffix = TifPath.split('\\')[-1]
        img_without_suffix = img_with_suffix.split('.')[0]
        for i in range(l):
            for j in range(l):
                num += 1
                offset_x = i * Size * (1 - RepetitionRate)
                offset_y = j * Size * (1 - RepetitionRate)

                if (height - offset_y < 1000):
                    offset_y = height - Size
                if (width - offset_x < 1000):
                    offset_x = width - Size

                out_band1 = im_band1.ReadAsArray(offset_x, offset_y, Size, Size)
                gtif_driver = gdal.GetDriverByName("GTiff")
                file = SavePath + '//' + img_without_suffix + '_' + str(num) + '.png'
                out_ds = gtif_driver.Create(file, Size, Size, outbandsize, datatype)
                print("create new tif file succeed")
                ori_transform = dataset_img.GetGeoTransform()
                if ori_transform:
                    print(ori_transform)
                    print("Origin = ({}, {})".format(ori_transform[0], ori_transform[3]))
                    print("Pixel Size = ({}, {})".format(ori_transform[1], ori_transform[5]))
                top_left_x = ori_transform[0]  # 左上角x坐标
                w_e_pixel_resolution = ori_transform[1]  # 东西方向像素分辨率
                top_left_y = ori_transform[3]  # 左上角y坐标
                n_s_pixel_resolution = ori_transform[5]  # 南北方向像素分辨率
                top_left_x = top_left_x + offset_x * w_e_pixel_resolution
                top_left_y = top_left_y + offset_y * n_s_pixel_resolution
                dst_transform = (
                top_left_x, ori_transform[1], ori_transform[2], top_left_y, ori_transform[4], ori_transform[5])
                out_ds.SetGeoTransform(dst_transform)
                out_ds.SetProjection(dataset_img.GetProjection())
                out_ds.GetRasterBand(1).WriteArray(out_band1)
                out_ds.FlushCache()
                print("FlushCache succeed")
                del out_ds, out_band1
        del dataset_img

        # # 裁剪最后一列  (包含右下角)
        # for i in range(0, row_num1 + 1):
        #     print('########################################################################')
        #     print(i)
        #     print('########################################################################')
        #     offset_x = width - Size
        #     offset_y = i * Size
        #
        #     out_band1 = im_band1.ReadAsArray(offset_x, offset_y, Size, Size)
        #     gtif_driver = gdal.GetDriverByName("GTiff")
        #     file = os.path.join(SavePath, 'hagfclip_%04d.tif' % num)
        #     num += 1
        #     out_ds = gtif_driver.Create(file, Size, Size, outbandsize, datatype)
        #     print("create new tif file succeed")
        #     ori_transform = dataset_img.GetGeoTransform()
        #     if ori_transform:
        #         print(ori_transform)
        #         print("Origin = ({}, {})".format(ori_transform[0], ori_transform[3]))
        #         print("Pixel Size = ({}, {})".format(ori_transform[1], ori_transform[5]))
        #     top_left_x = ori_transform[0]
        #     w_e_pixel_resolution = ori_transform[1]
        #     top_left_y = ori_transform[3]
        #     n_s_pixel_resolution = ori_transform[5]
        #     top_left_x = top_left_x + offset_x * w_e_pixel_resolution
        #     top_left_y = top_left_y + offset_y * n_s_pixel_resolution
        #     dst_transform = (
        #     top_left_x, ori_transform[1], ori_transform[2], top_left_y, ori_transform[4], ori_transform[5])
        #     out_ds.SetGeoTransform(dst_transform)
        #     out_ds.SetProjection(dataset_img.GetProjection())
        #     out_ds.GetRasterBand(1).WriteArray(out_band1)
        #     out_ds.FlushCache()
        #     print("FlushCache succeed")
        #     del out_ds, out_band1
        #
        # # 裁剪最后一行( 同样包含右下角，手动剔除相同的那张)
        # for i in range(0, col_num1 + 1):
        #     offset_x = i * Size
        #     if (width - offset_x < 1000):
        #         offset_x = width - Size
        #     offset_y = height - Size
        #     out_band1 = im_band1.ReadAsArray(offset_x, offset_y, Size, Size)
        #     gtif_driver = gdal.GetDriverByName("GTiff")
        #     file = os.path.join(SavePath, 'hagfclip_%04d.tif' % num)
        #     num += 1
        #     out_ds = gtif_driver.Create(file, Size, Size, outbandsize, datatype)
        #     print("create new tif file succeed")
        #     ori_transform = dataset_img.GetGeoTransform()
        #     if ori_transform:
        #         print(ori_transform)
        #         print("Origin = ({}, {})".format(ori_transform[0], ori_transform[3]))
        #         print("Pixel Size = ({}, {})".format(ori_transform[1], ori_transform[5]))
        #     top_left_x = ori_transform[0]
        #     w_e_pixel_resolution = ori_transform[1]
        #     top_left_y = ori_transform[3]
        #     n_s_pixel_resolution = ori_transform[5]
        #     top_left_x = top_left_x + offset_x * w_e_pixel_resolution
        #     top_left_y = top_left_y + offset_y * n_s_pixel_resolution
        #     dst_transform = (
        #     top_left_x, ori_transform[1], ori_transform[2], top_left_y, ori_transform[4], ori_transform[5])
        #     out_ds.SetGeoTransform(dst_transform)
        #     out_ds.SetProjection(dataset_img.GetProjection())
        #     out_ds.GetRasterBand(1).WriteArray(out_band1)
        #     out_ds.FlushCache()
        #     print("FlushCache succeed")
        #     del out_ds, out_band1


def deleteNodataimg(img):
    h, w, c = img.shape
    temp = 0
    for hc in range(h):
        for wc in range(w):
            for cc in range(c):
                if cc == 0:
                    temp += img[hc, wc, cc]
    return temp


# imgPath = r"F:\data\graduationProject\newxihugf2_clip_20210728"
# imglist = os.listdir(imgPath)
# for imgfile in imglist:
#     img = cv2.imread(os.path.join(imgPath, imgfile), -1)
#     count = deleteNodataimg(img)
#     if count == 0:
#         os.remove(os.path.join(imgPath, imgfile))











# TifCrop(r"H:\zjut\graduationproject\高分建筑标注检查（0816）\train\img\hagfclip_0228.tif",
#         r"H:\zjut\graduationproject\高分建筑标注检查（0816）\train\test", 512, 0)

img_path = r"F:\wd_data\Tianchi\Train\image"
label_path = r"F:\wd_data\Tianchi\Train\label"
save_img_path = r"F:\wd_data\Tianchi\Train\image\crop"
save_label_path = r"F:\wd_data\Tianchi\Train\label\crop"

if not os.path.exists(save_img_path):
    os.mkdir(save_img_path)
if not os.path.exists(save_label_path):
    os.mkdir(save_label_path)
img_list = [x for x in os.listdir(img_path) if x.endswith(".png")]
for i in range(len(img_list)):
    img_path_name = os.path.join(img_path, img_list[i])
    label_path_name = os.path.join(label_path, img_list[i])
    TifCrop(img_path_name, save_img_path, 512, 0)
    TifCrop(label_path_name, save_label_path, 512, 0)

# TifCrop(r"E:\xmxdata\hz_buildingExtraction\image_pre\4_poly\xihu4.tif",
#          r"E:\xmxdata\hz_buildingExtraction\dataset1\new_bdcn_dataset\train_temp\4_poly_clip", 512, 0)


