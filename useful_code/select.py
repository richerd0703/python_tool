import cv2
import os
import shutil
import numpy as np
import tqdm
from PIL import Image


def movefile(srcpath, path1, path2):
    for root, dirs, filenames in os.walk(srcpath):
        n = 1
        for filename in filenames:
            file = filename[:-4]+'.jpg'
            print(n)
            n += 1
            # img = cv2.imread("H:\\2c_zw\\test\\label\\11.png")
            # cv2.bitwise_not(img, img)
            # cv2.imwrite("H:\\2c_zw\\test\\label\\111.png",img)
            # # cv2.imshow('a',img)
            # # cv2.waitKey(0)
            # exit(0)
            # if len(img[img == 255]) / (1024 * 1024 * 3)==0.0:
            shutil.move(os.path.join(path1, file),path2)


def delfile(path):
    name_list = [x for x in os.listdir(path) if x.endswith(".png")]
    for file in name_list:
        if "_10" in file or "_7_" in file:
            print(os.path.join(path, file))
            # shutil.rmtree(os.path.join(path, file))
            os.remove(os.path.join(path, file))


def select_bypilxo(path):
    name_list = [x for x in os.listdir(path) if x.endswith(".png")]
    count = 0
    for file in name_list:
        img = cv2.imread(os.path.join(path, file))
        # print(np.sum(np.where(img[:, :, 0] > 0)))
        # print(512 * 512 * 4 / 5)
        temp = np.zeros((img.shape[0], img.shape[1]))
        temp[np.where(img[:, :, 0] > 0)] = 1  #  非黑色区域   img[:, :, 0] = 0 黑色
        # print(np.sum(temp))
        if np.sum(temp) < (512 * 512) * 0.7:  # 非黑色区域大于 4/5 移除
            count += 1
            os.remove(os.path.join(path, file))
            print("remove ---> ", file)
    print("finished remove file :", count)

# 2347 1/7  2452 1/5 47161 50141

def gray2rgb(path):

    im_gray = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    im_color = cv2.applyColorMap(im_gray, cv2.COLORMAP_JET)
    cv2.imwrite('ndvi_color.jpg', im_color)


if __name__ == '__main__':
    # sourcefile_dir = r"D:\dataset\cskin\flw\train_up2400_down1000_flw_1024\select_label"
    # destfile_dir = r"D:\dataset\cskin\flw\train_up2400_down1000_flw_1024\image"
    # destfile_dir1 = r"D:\dataset\cskin\flw\train_up2400_down1000_flw_1024\select_image"
    # movefile()
    path = r"G:\dataset\tianchi\label\crops"
    # delfile(path)
    select_bypilxo(path)
    # gray2rgb(path)