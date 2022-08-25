import ctypes
import os
import shutil
import configparser

# _*_ coding: utf-8 _*_

import sys
import os
import glob
import io
import cv2
import numpy as np
import argparse

# import tensorboard.plugins.projector.metadata

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf8')  # 打印出中文字符
try:
    from osgeo import gdal
    from osgeo import ogr
    from osgeo import osr
except ImportError:
    import gdal
    import ogr
    import osr

from PIL import Image
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from tqdm import tqdm


def world2Pixel(geoMatrix, x, y):
    """
    Uses a gdal geomatrix (gdal.GetGeoTransform()) to calculate
    the pixel location of a geospatial coordinate
    """
    ulX = geoMatrix[0]
    ulY = geoMatrix[3]
    xDist = geoMatrix[1]
    pixel = int((x - ulX) / xDist)
    line = int((ulY - y) / xDist)
    return (pixel, line)


def agrifield(pInput, pResult, pTemp, DLLPath, DLLName):
    # 将当前目录设置为dll的路径
    os.chdir(DLLPath)
    print(os.getcwd())
    SimDeepDll = ctypes.cdll.LoadLibrary(DLLName)

    SimDeepDll.CreateAgriField(pInput.encode(encoding='UTF-8'),
                               pResult.encode(encoding='UTF-8'),
                               pTemp.encode(encoding='UTF-8'), )


def building(pInput, pResult, pTemp, DLLPath, DLLName):
    # 将当前目录设置为dll的路径
    os.chdir(DLLPath)
    print(os.getcwd())
    SimDeepDll = ctypes.cdll.LoadLibrary(DLLName)

    SimDeepDll.CreateBuilding(pInput.encode(encoding='UTF-8'),
                              pResult.encode(encoding='UTF-8'),
                              pTemp.encode(encoding='UTF-8'), )


def all_element(pInput, pResult, pTemp, DLLPath, DLLName):
    # 将当前目录设置为dll的路径
    os.chdir(DLLPath)
    print(os.getcwd())
    SimDeepDll = ctypes.cdll.LoadLibrary(DLLName)

    SimDeepDll.CreateAllElement(pInput.encode(encoding='UTF-8'),
                                pResult.encode(encoding='UTF-8'),
                                pTemp.encode(encoding='UTF-8'), )


def transfer_16_to_8(pInput, pOutput, DLLPath, DLLName):
    os.chdir(DLLPath)
    print(os.getcwd())
    SimDeepDll = ctypes.cdll.LoadLibrary(os.path.join(DLLPath, DLLName))
    SimDeepDll.gDosStrech16To8(pInput.encode(encoding='UTF-8'),
                               pOutput.encode(encoding='UTF-8'))


def image2shp(pInput, pTempPath, pAThinPath, pRetPath, pRetGEOPath, DLLPath, DLLName):
    os.chdir(DLLPath)
    print(os.getcwd())
    SimDeepDll = ctypes.cdll.LoadLibrary(os.path.join(DLLPath, DLLName))
    SimDeepDll.ExtractFieldEdgeLargeFile(pInput.encode(encoding='UTF-8'),
                                         pTempPath.encode(encoding='UTF-8'),
                                         pAThinPath.encode(encoding='UTF-8'),
                                         pRetPath.encode(encoding='UTF-8'),
                                         pRetGEOPath.encode(encoding='UTF-8'))


def image2shp_small(pInput, pTempPath, pAThinPath, pRetPath, DLLPath, DLLName):
    os.chdir(DLLPath)
    # print(os.getcwd())
    SimDeepDll = ctypes.cdll.LoadLibrary(os.path.join(DLLPath, DLLName))
    SimDeepDll.ExtractFieldEdgePatch(pInput.encode(encoding='UTF-8'),
                                     pAThinPath.encode(encoding='UTF-8'),
                                     pTempPath.encode(encoding='UTF-8'),
                                     pRetPath.encode(encoding='UTF-8'), True)


def poly2shp(pInput, pShpPath, DLLPath, DLLName):
    os.chdir(DLLPath)
    # print(os.getcwd())
    SimDeepDll = ctypes.cdll.LoadLibrary(os.path.join(DLLPath, DLLName))
    SimDeepDll.DetectKeyPoint(pInput.encode(encoding='UTF-8'),
                                     pShpPath.encode(encoding='UTF-8'),)
if __name__ == '__main__':

    img_root = r'G:\temp\t1\image'
    save_root = r'G:\temp\t1\shp'
    print(img_root, save_root)
    if not os.path.exists(save_root):
        os.makedirs(save_root)
    name_list = [i for i in os.listdir(img_root) if i.endswith('.png')]

    DLLPath = r"F:\x64\819\x64\Debug"  # 封装好的dll所在文件夹路径
    DLLName = "SimDeep.dll"
    # pOutput = os.path.join(save_root)
    # pTempPath = os.path.join(pOutput, "Temp")
    # pAThinPath = os.path.join(pOutput, "AT")
    # pRetPath = os.path.join(pOutput, "Result")
    # pRetGEOPath = os.path.join(pOutput, "ResultGEO")
    # if not os.path.exists(pTempPath):
    #     os.makedirs(pTempPath)
    # if not os.path.exists(pAThinPath):
    #     os.makedirs(pAThinPath)
    # if not os.path.exists(pRetPath):
    #     os.makedirs(pRetPath)
    # if not os.path.exists(pRetGEOPath):
    #     os.makedirs(pRetGEOPath)
    # 预测多张图片
    for name in tqdm(name_list):
        poly2shp(os.path.join(img_root, name), os.path.join(save_root, name.split(".")[0] + ".shp"), DLLPath, DLLName)
        # image2shp(os.path.join(img_root, name), pTempPath, pAThinPath, pRetPath, pRetGEOPath, DLLPath, DLLName)

    # 预测单张图片
    # img_path = r''
    # save_path = r''
    # transfer_16_to_8(img_path, save_path, DLLPath, DLLName)
