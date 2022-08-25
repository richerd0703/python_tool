# 将4波段的遥感影像提取出前3波段

import os
import gdal
import numpy as np

#  读取tif数据集
def readTif(fileName):
    dataset = gdal.Open(fileName)
    if dataset == None:
        print(fileName + "文件无法打开")
    return dataset

#  保存tif文件函数
def writeTiff(im_data, im_geotrans, im_proj, path):
    if 'int8' in im_data.dtype.name:
        datatype = gdal.GDT_Byte
    elif 'int16' in im_data.dtype.name:
        datatype = gdal.GDT_UInt16
    else:
        datatype = gdal.GDT_Float32
    if len(im_data.shape) == 3:
        im_bands, im_height, im_width = im_data.shape
    elif len(im_data.shape) == 2:
        im_data = np.array([im_data])
        im_bands, im_height, im_width = im_data.shape
    # 创建文件
    driver = gdal.GetDriverByName("GTiff")
    # 这个写入函数比较特殊，跟数据的shape不一样，不一样才对，数据对应行列，这里必须是列行，先写列，再写行
    # https://vimsky.com/examples/detail/python-method-gdal.GetDriverByName.html
    dataset = driver.Create(path, int(im_width), int(im_height), int(3), datatype)
    print("============", dataset[0])
    if (dataset != None):
        dataset.SetGeoTransform(im_geotrans)  # 写入仿射变换参数
        dataset.SetProjection(im_proj)  # 写入投影
    # 这里写入前3个波段，全部输出im_bands
    for i in range(3):
        dataset.GetRasterBand(i + 1).WriteArray(im_data[i])
    del dataset


def bands4to3(TifPath, SavePath):
    i = 0
    filenames = os.listdir(TifPath)
    print(len(filenames))
    for filename in filenames:
        i +=1
        if i%1000==0:
            print("number：%d"%i)
        # 输出文件名
        file_path = os.path.join(TifPath, filename)
        # 读取栅格文件
        dataset_img = readTif(file_path)
        proj = dataset_img.GetProjection()
        geotrans = dataset_img.GetGeoTransform()
        img = dataset_img.ReadAsArray()  # 获取数据
        writeTiff(img, geotrans, proj, os.path.join(SavePath, filename))

if __name__ == '__main__':
    bands4 = r"F:\he\data\samples-高二\gf2\temp"
    bands3 = r"F:\he\data\samples-高二\gf2\out"

    bands4to3(bands4, bands3)
