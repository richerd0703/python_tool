# -*- coding: utf-8 -*-
#
# import numpy as np
# import os
# import SimpleITK as sitk
#
#
# def file_name(file_dir):
#     L = []
#     path_list = os.listdir(file_dir)
#     path_list.sort()  # 对读取的路径进行排序
#     for filename in path_list:
#         # if 'nii' in filename:
#         L.append(os.path.join(filename))
#     return L
#
#
# def computeQualityMeasures(lP, lT):
#     quality = dict()
#     labelPred = sitk.GetImageFromArray(lP, isVector=False)
#     labelTrue = sitk.GetImageFromArray(lT, isVector=False)
#     hausdorffcomputer = sitk.HausdorffDistanceImageFilter()
#     hausdorffcomputer.Execute(labelTrue > 0.5, labelPred > 0.5)
#     quality["avgHausdorff"] = hausdorffcomputer.GetAverageHausdorffDistance()
#     quality["Hausdorff"] = hausdorffcomputer.GetHausdorffDistance()
#
#     dicecomputer = sitk.LabelOverlapMeasuresImageFilter()
#     dicecomputer.Execute(labelTrue > 0.5, labelPred > 0.5)
#     quality["dice"] = dicecomputer.GetDiceCoefficient()
#
#     return quality
#
#
# gtpath = r'G:\dataset\mapping_challenge_dataset\1\crop'
# predpath = r'G:\dataset\mapping_challenge_dataset\1\crop'
#
# gtnames = file_name(gtpath)
# prednames = file_name(predpath)
#
# labels_num = np.zeros(len(prednames))
# NUM = []
# P = []
#
# for i in range(len(gtnames)):
#     temp = gtpath + "\\" + gtnames[i]
#     gt = sitk.ReadImage(gtpath + "\\" + gtnames[i])
#     pred = sitk.ReadImage(predpath + "\\" + gtnames[i])
#     quality = computeQualityMeasures(pred, gt)

import cv2
def get_contours(img):

    # 灰度化, 二值化, 连通域分析
    # img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # _, img_bin = cv2.threshold(img_gray, 200, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    return contours[0]


def main():
    # 1.导入图片
    img_cs1 = cv2.imread(r"G:\dataset\mapping_challenge_dataset\1\point\gt\000000000010.png",0)
    img_cs2 = cv2.imread(r"G:\dataset\mapping_challenge_dataset\1\point\gt\000000000010.png",0)
    img_hand = cv2.imread(r"G:\dataset\mapping_challenge_dataset\1\point\gt\000000000010.png",0)

    # 2.获取图片连通域
    cnt_cs1 = get_contours(img_cs1)
    cnt_cs2 = get_contours(img_cs2)
    cnt_hand = get_contours(img_hand)

    # 3.创建计算距离对象
    hausdorff_sd = cv2.createHausdorffDistanceExtractor()

    # 4.计算轮廓之间的距离
    d1 = hausdorff_sd.computeDistance(cnt_hand, cnt_hand)
    print("与自身的距离hausdorff\t d1=", d1)

    d2 = hausdorff_sd.computeDistance(cnt_hand, cnt_cs2)
    print("与相似图片的距离hausdorff\t d2=", d2)

    d3 = hausdorff_sd.computeDistance(cnt_hand, cnt_cs1)
    print("与不同图片的距离hausdorff\t d3=", d3)


if __name__ == '__main__':
    main()
