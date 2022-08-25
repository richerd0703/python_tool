import cv2
import numpy as np
import os
from tqdm import tqdm


def detect(src_path, save_path):
    name_list = [x for x in os.listdir(src_path) if x.endswith(".png")]
    for file in tqdm(name_list):
        img = cv2.imread(os.path.join(src_path, file))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # gray = np.float32(gray)
        dst = cv2.cornerHarris(gray, 4, 5, 0.04)  # 执行焦点检测
        # print('dst.shape:', dst.shape)
        # print('dst', dst)
        # img[dst > 0.01 * dst.max()] = [0, 0, 255]
        out_img = np.zeros(img.shape)
        out_img[dst > 0.005 * dst.max()] = [255, 255, 255]
        # cv2.imshow('dst', img) cv2.rectangle(out_img, (x, y), (x+3, y+3), [255, 255, 255], -1)
        # cv2.imwrite("dst2.png", img)
        cv2.imwrite(os.path.join(save_path, file), out_img)


if __name__ == '__main__':
    src_path = r"G:\temp\img"
    save_path = r"G:\temp\2"
    if not os.path.exists(save_path): os.makedirs(save_path)
    detect(src_path, save_path)