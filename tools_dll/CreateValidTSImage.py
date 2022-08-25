import os
import ctypes
from ctypes import *
import argparse
# 15

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


def cat_path(srcpath, count, savepath):
    input_path = ""
    output_path = ""
    img_list = [x for x in os.listdir(srcpath) if x.endswith(".tif")]  # 获取目录中所有tif格式图像列表
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
    DLLPath = r"F:\wu_dll\TSProc\x64\Debug"
    Dllname = "TSProc.dll"
    tif_path = r"F:\wd_data\change_geo\tif_data"
    output = r"F:\wd_data\change_geo\output"

    # pInput = "F:\wd_data\dll_tool\Data\IGSNRR\TimeSeries\GF1_WFV4_E120.7_N31.8_20160126_L1A0001370935_DOM_subset.img|" \
    #          "F:\wd_data\dll_tool\Data\IGSNRR\TimeSeries\GF1_WFV2_E121.1_N30.9_20160227_L1A0001437707_DOM_subset.img|" \
    #          "F:\wd_data\dll_tool\Data\IGSNRR\TimeSeries\GF1_WFV1_E119.5_N31.3_20160302_L1A0001446480_DOM_subset.img|" \
    #          "F:\wd_data\dll_tool\Data\IGSNRR\TimeSeries\GF1_WFV2_E121D1_N31D0_20160408_GF1_WFV2_DOM_subset.img|" \
    #          "F:\wd_data\dll_tool\Data\IGSNRR\TimeSeries\GF1_WFV4_E119D0_N31D7_20160504_GF1_WFV4_DOM_subset.img|" \
    #          "F:\wd_data\dll_tool\Data\IGSNRR\TimeSeries\GF1_WFV4_E119D4_N31D9_20160630_GF1_WFV4_DOM_subset.img|" \
    #          "F:\wd_data\dll_tool\Data\IGSNRR\TimeSeries\GF1_WFV1_E120D4_N31D3_20160715_GF1_WFV1_DOM_subset.img|" \
    #          "F:\wd_data\dll_tool\Data\IGSNRR\TimeSeries\GF1_WFV4_E120D1_N31D8_20160814_GF1_WFV4_DOM_subset.img|" \
    #          "F:\wd_data\dll_tool\Data\IGSNRR\TimeSeries\GF1_WFV2_E119D8_N31D0_20160903_GF1_WFV2_DOM_subset.img|" \
    #          "F:\wd_data\dll_tool\Data\IGSNRR\TimeSeries\GF1_WFV2_E120D5_N31D0_20161128_GF1_WFV2_DOM0_subset.img|" \
    #          "F:\wd_data\dll_tool\Data\IGSNRR\TimeSeries\GF1_WFV3_E120D1_N30D6_20161231_GF1_WFV3_DOM0_subset.img"

    temp = [2016026, 2016058, 2016062, 2016099, 2016125, 2016182, 2016197, 2016227, 2016247, 2016333, 2016366]
    nImageSize = 11
    input = c_int * nImageSize
    pDays = input()
    for i in range(len(temp)):
        pDays[i] = temp[i]

    pMask = r"F:\\wd_data\\dll_tool\\Data\\IGSNRR\\TimeSeries\\Gaoxin-Mask-16m.tif"
    # pResult = "F:\wd_data\dll_tool\Data\out\GF1_WFV4_E120.7_N31.8_20160126_L1A0001370935_DOM_subset.img|" \
    #           "F:\wd_data\dll_tool\Data\out\GF1_WFV2_E121.1_N30.9_20160227_L1A0001437707_DOM_subset.img|" \
    #           "F:\wd_data\dll_tool\Data\out\GF1_WFV1_E119.5_N31.3_20160302_L1A0001446480_DOM_subset.img|" \
    #           "F:\wd_data\dll_tool\Data\out\GF1_WFV2_E121D1_N31D0_20160408_GF1_WFV2_DOM_subset.img|" \
    #           "F:\wd_data\dll_tool\Data\out\GF1_WFV4_E119D0_N31D7_20160504_GF1_WFV4_DOM_subset.img|" \
    #           "F:\wd_data\dll_tool\Data\out\GF1_WFV4_E119D4_N31D9_20160630_GF1_WFV4_DOM_subset.img|" \
    #           "F:\wd_data\dll_tool\Data\out\GF1_WFV1_E120D4_N31D3_20160715_GF1_WFV1_DOM_subset.img|" \
    #           "F:\wd_data\dll_tool\Data\out\GF1_WFV4_E120D1_N31D8_20160814_GF1_WFV4_DOM_subset.img|" \
    #           "F:\wd_data\dll_tool\Data\out\GF1_WFV2_E119D8_N31D0_20160903_GF1_WFV2_DOM_subset.img|" \
    #           "F:\wd_data\dll_tool\Data\out\GF1_WFV2_E120D5_N31D0_20161128_GF1_WFV2_DOM0_subset.img|" \
    #           "F:\wd_data\dll_tool\Data\out\GF1_WFV3_E120D1_N30D6_20161231_GF1_WFV3_DOM0_subset.img"

    # pResult = pResult.replace(".img", ".tif")
    pInput, pResult = cat_path(tif_path, nImageSize, output)
    # print(pInput)
    # print(pResult)
    DoCreateValidTSImages(pInput, pDays, nImageSize, pMask, pResult)