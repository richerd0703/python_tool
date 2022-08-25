import numpy as np
from hausdorff import hausdorff_distance
import numba
from math import sqrt
import imutils
import cv2
from tqdm import tqdm
import os

# 得到中心点坐标
def get_center(path):
    point = []
    image = cv2.imread(path, 0)
    cnts = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    # loop over the contours
    for c in cnts:
        # compute the center of the contour
        M = cv2.moments(c)
        try:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
        except:
            continue
        point.append([cX, cY])
    return point


# 计算两个数组距离 返回B到A的最短点 数组
def dis(XA, XB):
    nA = XA.shape[0]
    nB = XB.shape[0]
    l = []
    for j in range(nB):
        cmin = 100000
        for i in range(nA):
            d = custom_dist(XA[i, :], XB[j, :])
            if d == 0:
                cmin = 0
                break
            if d < cmin:
                cmin = d
        l.append(cmin)
    return l


def custom_dist(array_x, array_y):
    n = array_x.shape[0]
    ret = 0.
    for i in range(n):
        ret += (array_x[i] - array_y[i]) ** 2
    return sqrt(ret)


def count_distence(label, pre):
    all_pic_dis = []
    nums = []
    for file in tqdm(os.listdir(pre)):
        labelfile = os.path.join(label, file)
        prefile = os.path.join(pre, file)

        label_center = np.array(get_center(labelfile))
        pre_center = np.array(get_center(prefile))
        nums.append(len(pre_center) / len(label_center))

        dis_list = dis(label_center, pre_center)
        if len(dis_list) < 2:
            continue
        dis_list.sort()
        dis_list = dis_list[0: int(len(dis_list) * 0.9)]

        mean_distence = np.array(dis_list).mean()

        # print(mean_distence)
        all_pic_dis.append(mean_distence)
        # print("=========", max_distence)
    print("mean dis = ", np.array(all_pic_dis).mean())
    print(np.array(nums).mean())
if __name__ == '__main__':
    label_path = r"G:\dataset\publicdataset\ISPRS\512\test\point_5"
    # pre_path = r"G:\dataset\publicdataset\ISPRS\my\pre\point"
    pre_path = r"G:\dataset\publicdataset\ISPRS\my\pre\point0-255"
    count_distence(label_path, pre_path)