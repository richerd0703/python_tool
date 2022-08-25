import cv2
import numpy as np
import os
from tqdm import tqdm


def get_outline(src_path, save_path):
    for file in tqdm(os.listdir(src_path)):
        image = cv2.imread(os.path.join(src_path, file), 0)
        ret, thresh = cv2.threshold(image, 0, 255, 0)
        countours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        img_gray = np.zeros(image.shape)

        # 参数    image, contours, contourIdx, color[, thickness[
        cv2.drawContours(img_gray, countours, -1, 255, 1)
        cv2.imwrite(os.path.join(save_path, file), img_gray)


def get_diff_outline(src_path, save_path):
    l = [1, 2, 3, 5, 7, 9]
    for i in tqdm(l):
        save_path_i = os.path.join(save_path, str(i))
        if not os.path.exists(save_path_i):
            os.makedirs(save_path_i)

        for file in os.listdir(src_path):
            image = cv2.imread(os.path.join(src_path, file), 0)
            ret, thresh = cv2.threshold(image, 0, 255, 0)
            countours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
            img_gray = np.zeros(image.shape)

            # 参数    image, contours, contourIdx, color[, thickness[
            cv2.drawContours(img_gray, countours, -1, 255, i)
            cv2.imwrite(os.path.join(save_path_i, (str(i) + "_" + file)), img_gray)


def get_pixel_outline(src_path, save_path):
    l = [1, 2, 3, 5, 7, 9]
    for p in tqdm(l):
        save_path_i = os.path.join(save_path, str(p))
        if not os.path.exists(save_path_i):
            os.makedirs(save_path_i)

        for file in tqdm(os.listdir(src_path)):
            image = cv2.imread(os.path.join(src_path, file), 0)
            out = np.zeros(image.shape)
            for i in range(1, 25):
                img1 = np.zeros(image.shape)
                img1[image == i] = 1
                cv2.imwrite("img1.png", img1)
                img2 = cv2.imread("img1.png", 0)

                ret, thresh = cv2.threshold(img2, 0, 255, 0)
                countours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
                img_gray = np.zeros(image.shape)

                # 参数    image, contours, contourIdx, color[, thickness[
                # 1， 2， 4， 6， 8
                cv2.drawContours(img_gray, countours, -1, 255, p)
                # cv2.imwrite("{}_gray.png".format(i), img_gray)
                out[img_gray == 255] = i
            cv2.imwrite(os.path.join(save_path_i, file), out)


if __name__ == '__main__':
    src_path = r"G:\exp\crowdAI\my\train\labels"
    save_path = r"G:\temp\pointExtract\edge"
    if not os.path.exists(save_path): os.makedirs(save_path)
    get_outline(src_path, save_path)  # 所有的边界 1
    # get_diff_outline(src_path, save_path) # 不同像素的边界 同一颜色
    # get_pixel_outline(src_path, save_path) # 不同像素的边界 不同颜色

