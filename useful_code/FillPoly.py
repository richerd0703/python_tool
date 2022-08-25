import cv2
import os
import numpy as np


def fill(imaPath, output):
    imaList = os.listdir(imaPath)
    for files in imaList:
        path_ima = os.path.join(imaPath, files)
        path_processed = os.path.join(output, files)
        img = cv2.imread(path_ima, 0)

        mask = np.zeros_like(img)
        # print(np.shape(img))

        # 先利用二值化去除图片噪声
        ret, img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

        contours, _ = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        n = len(contours)  # 轮廓的个数
        cv_contours = []
        for contour in contours:
            rect = cv2.minAreaRect(contour)  # 最小外接矩形
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            area = cv2.contourArea(box)
            print(area)
            if area <= 200:
                cv_contours.append(contour)
                # x, y, w, h = cv2.boundingRect(contour)
                # img[y:y + h, x:x + w] = 255
            else:
                continue

        # def contourArea(cnt):  # 传入一个轮廓
        #     rect = cv2.minAreaRect(cnt)  # 最小外接矩形
        #     box = cv2.boxPoints(rect)
        #     box = np.int0(box)
        #     return cv2.contourArea(box)
        cv2.fillPoly(img, cv_contours, (255, 255, 255))
        cv2.imwrite(path_processed, img)


if __name__ == '__main__':
    src_path = r"G:\dataset\mapping_challenge_dataset\pre\pre_poly"
    save_path = r"G:\dataset\mapping_challenge_dataset\pre\fill"
    fill(src_path, save_path)