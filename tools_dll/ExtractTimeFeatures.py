import os
import ctypes
from ctypes import *
import argparse
# 16



def DoFeatureExtract(pInputTSImage, pDays, nImageSize, pInPlotVec, pResultCSV, pResultSHP):
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

    # lib.ExtractTimeFeatures(pInputTSImage.encode("UTF-8"), pDays, nImageSize, pInPlotVec.encode("UTF-8"),
    #                         pResultCSV.encode("UTF-8"))

    lib.DoFeatureExtractSHP(pInputTSImage.encode("UTF-8"), pDays, nImageSize, pInPlotVec.encode("UTF-8"),
                            pResultCSV.encode("UTF-8"), pResultSHP.encode("UTF-8"))


if __name__ == '__main__':
    DLLPath = r"F:\wu_dll\TSProc\x64\Debug"
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

    shp_path = r"F:\wd_data\dll_tool\Data\IGSNRR\Plot\SanDiao-Albers.shp"
    csv_path = r"F:\wd_data\dll_tool\Data\out\16\16.csv"
    out_shp = r"F:\wd_data\dll_tool\Data\out\16\16.shp"
    DoFeatureExtract(pInput, pDays, nImageSize, shp_path, csv_path, out_shp)