import os
import ctypes
from ctypes import *
import argparse


def DoCreateValidTSImages(pTSImages, pDays, nImageNum, pMask, pResult):
    # 将当前目录设置为dll的路径
    os.chdir(DLLPath)
    print(os.getcwd())
    pTSImages = pTSImages.replace("\\", "\\\\")
    pResult = pResult.replace("\\", "\\\\")
    print(pTSImages)
    print(pResult)
    # SimDeepDll = ctypes.cdll.LoadLibrary(DLLName)
    lib = ctypes.cdll.LoadLibrary(Dllname)

    lib.CreateValidTSImage(pTSImages.encode("UTF-8"), pDays, nImageNum, pMask.encode("UTF-8"), pResult.encode("UTF-8"))


def DoFeatureExtract(pInputTSImage, pDays, nImageSize, pInPlotVec, pResultCSV):
    os.chdir(DLLPath)
    print(os.getcwd())
    pInputTSImage = pInputTSImage.replace("\\", "\\\\")
    pResultCSV = pResultCSV.replace("\\", "\\\\")
    pInPlotVec = pInPlotVec.replace("\\", "\\\\")
    print(pInputTSImage)
    print(pInPlotVec)
    print(pResultCSV)
    # SimDeepDll = ctypes.cdll.LoadLibrary(DLLName)
    lib = ctypes.cdll.LoadLibrary(Dllname)

    lib.ExtractTimeFeatures(pInputTSImage.encode("UTF-8"), pDays, nImageSize, pInPlotVec.encode("UTF-8"),
                            pResultCSV.encode("UTF-8"))


def DoWaterIndexExtration(pInputTSImage, pDays, nImageSize, pInPlotVec, pResultCSV):
    os.chdir(DLLPath)
    print(os.getcwd())
    pInputTSImage = pInputTSImage.replace("\\", "\\\\")
    pInPlotVec = pInPlotVec.replace("\\", "\\\\")
    pResultCSV = pResultCSV.replace("\\", "\\\\")
    print(pResultCSV)
    print(pInputTSImage)
    print(pInPlotVec)
    lib = ctypes.cdll.LoadLibrary(Dllname)
    lib.WaterIndexExtration(pInputTSImage.encode("UTF-8"), pDays, nImageSize, pInPlotVec.encode("UTF-8"),
                            pResultCSV.encode("UTF-8"))


def DoBuidingIndexExtraction(pInputTSImage, pDays, nImageSize, pInPlotVec, pResultCSV):
    os.chdir(DLLPath)
    print(os.getcwd())
    pInputTSImage = pInputTSImage.replace("\\", "\\\\")
    pInPlotVec = pInPlotVec.replace("\\", "\\\\")
    pResultCSV = pResultCSV.replace("\\", "\\\\")
    print(pInputTSImage)
    print(pInPlotVec)
    print(pResultCSV)
    lib = ctypes.cdll.LoadLibrary(Dllname)
    lib.BuidingIndexExtraction(pInputTSImage.encode("UTF-8"), pDays, nImageSize, pInPlotVec.encode("UTF-8"),
                               pResultCSV.encode("UTF-8"))


def DoGreenLandIndexExtraction(pInputTSImage, pDays, nImageSize, pInPlotVec, pResultCSV):
    os.chdir(DLLPath)
    print(os.getcwd())
    pInputTSImage = pInputTSImage.replace("\\", "\\\\")
    pInPlotVec = pInPlotVec.replace("\\", "\\\\")
    pResultCSV = pResultCSV.replace("\\", "\\\\")
    print(pInputTSImage)
    print(pInPlotVec)
    print(pResultCSV)
    lib = ctypes.cdll.LoadLibrary(Dllname)
    lib.GreenLandIndexExtraction(pInputTSImage.encode("UTF-8"), pDays, nImageSize, pInPlotVec.encode("UTF-8"),
                                 pResultCSV.encode("UTF-8"))


def DoDecisionClassify(strCSVFeature, strSample,strPlotVec,strResult):
    os.chdir(DLLPath)
    print(os.getcwd())
    strCSVFeature = strCSVFeature.replace("\\", "\\\\")
    strSample = strSample.replace("\\", "\\\\")
    strPlotVec = strPlotVec.replace("\\", "\\\\")
    strResult = strResult.replace("\\", "\\\\")
    print(strCSVFeature)
    print(strSample)
    print(strPlotVec)
    print(strResult)
    lib = ctypes.cdll.LoadLibrary(Dllname)
    lib.DecisionByClassify(strCSVFeature.encode("UTF-8"), strSample.encode("UTF-8"), strPlotVec.encode("UTF-8"),
                               strResult.encode("UTF-8"))

def cat_path(srcpath, count, savepath):
    input_path = ""
    output_path = ""
    img_list = [x for x in os.listdir(srcpath) if x.endswith(".img")]  # 获取目录中所有tif格式图像列表
    for img in img_list:
        print(img)
    a = len(img_list)
    print(a)
    for i in range(len(img_list)):
        input_img_path = os.path.join(srcpath, img_list[i])
        output_img_path = os.path.join(savepath, img_list[i])
        input_path += input_img_path + "|"
        output_path += output_img_path + "|"
    return input_path.strip("|"), output_path.strip("|")


if __name__ == '__main__':
    DLLPath = r"I:\tools\Code\TSProc\x64\Debug"
    Dllname = "TSProc.dll"
    pInput = "F:\wd_data\dll_tool\Data\IGSNRR\TimeSeries\GF1_WFV4_E120.7_N31.8_20160126_L1A0001370935_DOM_subset.img|" \
             "F:\wd_data\dll_tool\Data\IGSNRR\TimeSeries\GF1_WFV2_E121.1_N30.9_20160227_L1A0001437707_DOM_subset.img|" \
             "F:\wd_data\dll_tool\Data\IGSNRR\TimeSeries\GF1_WFV1_E119.5_N31.3_20160302_L1A0001446480_DOM_subset.img|" \
             "F:\wd_data\dll_tool\Data\IGSNRR\TimeSeries\GF1_WFV2_E121D1_N31D0_20160408_GF1_WFV2_DOM_subset.img|" \
             "F:\wd_data\dll_tool\Data\IGSNRR\TimeSeries\GF1_WFV4_E119D0_N31D7_20160504_GF1_WFV4_DOM_subset.img|" \
             "F:\wd_data\dll_tool\Data\IGSNRR\TimeSeries\GF1_WFV4_E119D4_N31D9_20160630_GF1_WFV4_DOM_subset.img|" \
             "F:\wd_data\dll_tool\Data\IGSNRR\TimeSeries\GF1_WFV1_E120D4_N31D3_20160715_GF1_WFV1_DOM_subset.img|" \
             "F:\wd_data\dll_tool\Data\IGSNRR\TimeSeries\GF1_WFV4_E120D1_N31D8_20160814_GF1_WFV4_DOM_subset.img|" \
             "F:\wd_data\dll_tool\Data\IGSNRR\TimeSeries\GF1_WFV2_E119D8_N31D0_20160903_GF1_WFV2_DOM_subset.img|" \
             "F:\wd_data\dll_tool\Data\IGSNRR\TimeSeries\GF1_WFV2_E120D5_N31D0_20161128_GF1_WFV2_DOM0_subset.img|" \
             "F:\wd_data\dll_tool\Data\IGSNRR\TimeSeries\GF1_WFV3_E120D1_N30D6_20161231_GF1_WFV3_DOM0_subset.img"

    temp = [2016026, 2016058, 2016062, 2016099, 2016125, 2016182, 2016197, 2016227, 2016247, 2016333, 2016366]
    nImageSize = 11
    input = c_int * nImageSize
    pDays = input()
    for i in range(len(temp)):
        pDays[i] = temp[i]

    pMask = r"F:\\wd_data\\dll_tool\\Data\\IGSNRR\\TimeSeries\\Gaoxin-Mask-16m.tif"
    pResult = "F:\wd_data\dll_tool\Data\out\GF1_WFV4_E120.7_N31.8_20160126_L1A0001370935_DOM_subset.img|" \
              "F:\wd_data\dll_tool\Data\out\GF1_WFV2_E121.1_N30.9_20160227_L1A0001437707_DOM_subset.img|" \
              "F:\wd_data\dll_tool\Data\out\GF1_WFV1_E119.5_N31.3_20160302_L1A0001446480_DOM_subset.img|" \
              "F:\wd_data\dll_tool\Data\out\GF1_WFV2_E121D1_N31D0_20160408_GF1_WFV2_DOM_subset.img|" \
              "F:\wd_data\dll_tool\Data\out\GF1_WFV4_E119D0_N31D7_20160504_GF1_WFV4_DOM_subset.img|" \
              "F:\wd_data\dll_tool\Data\out\GF1_WFV4_E119D4_N31D9_20160630_GF1_WFV4_DOM_subset.img|" \
              "F:\wd_data\dll_tool\Data\out\GF1_WFV1_E120D4_N31D3_20160715_GF1_WFV1_DOM_subset.img|" \
              "F:\wd_data\dll_tool\Data\out\GF1_WFV4_E120D1_N31D8_20160814_GF1_WFV4_DOM_subset.img|" \
              "F:\wd_data\dll_tool\Data\out\GF1_WFV2_E119D8_N31D0_20160903_GF1_WFV2_DOM_subset.img|" \
              "F:\wd_data\dll_tool\Data\out\GF1_WFV2_E120D5_N31D0_20161128_GF1_WFV2_DOM0_subset.img|" \
              "F:\wd_data\dll_tool\Data\out\GF1_WFV3_E120D1_N30D6_20161231_GF1_WFV3_DOM0_subset.img"

    # srcpath = r"F:\wd_data\dll_tool\Data\IGSNRR\TimeSeries"
    # savepath = r"F:\wd_data\dll_tool\Data\out"
    # pInput, pResult = cat_path(srcpath, nImageSize, savepath)
    # print(pInput)
    # print(pResult)
    shp_path = r"F:\wd_data\dll_tool\Data\IGSNRR\Plot\SanDiao-Albers-classify.shp"
    # shp_path = r"F:\wd_data\dll_tool\Data\IGSNRR\Plot\SanDiao-Albers.shp"
    csv_path = r"F:\wd_data\dll_tool\Data\out\19\out.csv"
    # DoCreateValidTSImages(pInput, pDays, nImageSize, pMask, pResult)  # 15 ok
    # DoFeatureExtract(pInput, pDays, nImageSize, shp_path, csv_path)  # 16 ok SanDiao-Albers.shp
    # DoWaterIndexExtration(pInput, pDays, nImageSize, shp_path, csv_path)  # 19 ok
    # DoBuidingIndexExtraction(pInput, pDays, nImageSize, shp_path, csv_path)  # 18 ok
    # DoGreenLandIndexExtraction(pInput, pDays, nImageSize, shp_path, csv_path)  # 20 ok

    strCSVFeature = "F:\wd_data\dll_tool\Data\IGSNRR\Plot\SanDiao-Albers_TSFeateure.csv"
    strSample = "F:\wd_data\dll_tool\Data\IGSNRR\Plot\Sample-DT.csv"
    strPlotVec = "F:\wd_data\dll_tool\Data\IGSNRR\Plot\SanDiao-Albers.shp"
    strResult = "F:\wd_data\dll_tool\Data\out\SanDiao-Albers-classify.shp"
    DoDecisionClassify(strCSVFeature, strSample, strPlotVec, strResult)

# parser = argparse.ArgumentParser('argument')
# parser.add_argument('--arg1', type=str, default=None)
# parser.add_argument('--arg2', type=str, default=None)
# parser.add_argument('--arg3', type=str, default=None)
# parser.add_argument('--arg4', type=str, default=None)
# parser.add_argument('--arg5', type=int)

# from osgeo import ogr
# from osgeo import gdal
# import numpy as np
# from sklearn import datasets
# from sklearn.tree import DecisionTreeClassifier
# import os
# import shutil
# import ctypes
#
#
#
# def run(strCSVFeature, strSample, strPlotVec, strResult):
#     # 读取Sample文件
#     Y = np.loadtxt(strSample, dtype=np.int, delimiter=',')
#     # print(Y)
#     C = np.loadtxt(strCSVFeature, dtype=np.str, delimiter=',', skiprows=1, usecols=(-1))
#     nFeatureNum = len(C[0].split("|"))
#     X = np.zeros((C.shape[0], nFeatureNum))
#     for i in range(len(X)):
#         X[i][0] = 100
#         X[i][1:] = np.array(C[i].split("|"))[:-1].astype(float)
#         # X = np.array(X)
#     # X=np.loadtxt(strCSVFeature,dtype=np.str,delimiter=',',usecols=(-1))
#     nFeatureNum = len(C[0].split("|")) - 1
#     nSampleNum = len(Y)
#     nInputNum = len(X)
#
#     print(nFeatureNum)
#
#     samples = np.zeros((nSampleNum, nFeatureNum), dtype=np.float)
#     labels = np.zeros((nSampleNum), dtype=np.int)
#     Z = np.zeros((nInputNum, nFeatureNum), dtype=np.float)
#
#     Flag = 0
#     for j in range(0, nSampleNum):
#         for i in range(0, nInputNum):
#             if (X[i][0] == Y[j][0]):
#                 for k in range(1, nFeatureNum):
#                     samples[Flag][k - 1] = X[i][k]
#                     labels[Flag] = Y[j][1]
#                 Flag += 1
#                 break
#     print(samples)
#     print(labels)
#     for i in range(0, nInputNum):
#         for j in range(0, nFeatureNum):
#             Z[i][j] = X[i][j + 1]
#
#     # 训练模型，限制树的最大深度4
#     clf = DecisionTreeClassifier(max_depth=nFeatureNum)
#     # 拟合模型
#     clf.fit(samples, labels)
#
#     MyPredict = clf.predict(Z)
#
#     # print(MyPredict)
#
#     ###############################################
#     # print("----------------strPlotVec---------------------")
#     foldname = strPlotVec.replace(strPlotVec.split("/")[-1], "")
#     filenames = os.listdir(foldname)
#     need_changename = strPlotVec.split("/")[-1].split(".")[0] + '.'
#     # print(foldname,filenames,need_changename)
#     for filename in filenames:
#         if need_changename in filename:
#             if os.path.exists(
#                     os.path.join(foldname, filename.replace(need_changename, "SanDiao-Albers-classify."))):
#                 os.remove(os.path.join(foldname, filename.replace(need_changename, "SanDiao-Albers-classify.")))
#             shutil.copy(os.path.join(foldname, filename),
#                         os.path.join(foldname, filename.replace(need_changename, "SanDiao-Albers-classify.")))
#
#     ###############################################
#
#     resultDS = ogr.Open(strResult, 1)
#     if resultDS is None:
#         print("Can't Open File", strResult)
#         return False
#
#     ptLayer = resultDS.GetLayer(0)
#     NewFieldDefn = ogr.FieldDefn("ClassID", ogr.OFTInteger)
#     ptLayer.CreateField(NewFieldDefn)
#     featurenDefn = ptLayer.GetLayerDefn()
#     # print(featurenDefn)
#     pFeature = ptLayer.GetNextFeature()
#     # pFeature=ptLayer.GetFeature(1)
#     # print(pFeature)
#     i = 0
#     while (pFeature != None):
#         pFeature.SetField("ClassID", int(MyPredict[i]))
#         ptLayer.SetFeature(pFeature)
#         pFeature = ptLayer.GetNextFeature()
#         i += 1
#
#     print("Result is Good")
#
#     return 0
#
#
# if __name__ == '__main__':
#
#     strCSVFeature = "F:\wd_data\地理大数据集成\Data\IGSNRR\Plot\SanDiao-Albers_TSFeateure.csv"
#     strSample = "F:\wd_data\地理大数据集成\Data\IGSNRR\Plot\Sample-DT.csv"
#     strPlotVec = "F:\wd_data\地理大数据集成\Data\IGSNRR\Plot\SanDiao-Albers.shp"
#     strResultr = "F:\wd_data\地理大数据集成\Data\out\SanDiao-Albers-classify.shp"
#
#     run(strCSVFeature, strSample, strPlotVec, strResultr)
