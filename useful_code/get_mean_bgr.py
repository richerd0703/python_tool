import os
import cv2
import numpy as np
from tqdm import tqdm


def computer(path):
    files = os.listdir(path)
    per_img_Bmean = []
    per_img_Gmean = []
    per_img_Rmean = []
    for file in tqdm(files):
        img = cv2.imread(os.path.join(path, file), 1)
        per_img_Bmean.append(img[:, :, 0])
        per_img_Gmean.append(img[:, :, 1])
        per_img_Rmean.append(img[:, :, 2])

    B_mean = np.mean(per_img_Bmean)
    G_mean = np.mean(per_img_Gmean)
    R_mean = np.mean(per_img_Rmean)

    return B_mean, G_mean, R_mean


if __name__ == '__main__':
    path = r"H:\data\tianchi\select\train\images"
    B, G, R = computer(path)
    print("mean_bgr--> b: {:.4f}, g: {:.4f}, r: {:.4f}".format(B, G, R))

    # mean_bgr--> b: 144.5993, g: 140.8897, r: 124.0062