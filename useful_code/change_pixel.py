import numpy as np
import cv2
import os
from tqdm import tqdm
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

def change_pixel(path, save_path):
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    for file in tqdm(os.listdir(path)):
    # for file in os.listdir(path):
        # print(file)
        image = cv2.imdecode(np.fromfile(os.path.join(path, file), dtype=np.uint8), 0)

        # image = Image.open(os.path.join(path, file)).save("w.png")
        # image = cv2.imdecode(np.fromfile("w.png", dtype=np.uint8), 0)

        # print(image.shape)
        img = np.zeros(image.shape)

        # print(type(img))
        # print(img)
        # temp_img = image.copy()

        # img[image == 127] = 0
        # img[image == 11] = 255

        img[image == 1] = 255
        # img[image == 75] = 2
        # img[image == 113] = 3

        # for i in range(512):
        #     for j in range(512):
        #         if image[i][j] == 38:
        #             # print("pixel == 38")
        #             img[i][j] = 1
        #             # print(img[i][j])
        #         elif image[i][j] == 75:
        #             # print("pixel == 75")
        #             img[i][j] = 2
        #         elif image[i][j] == 113:
        #             # print("pixel == 113")
        #             img[i][j] = 3
        #         else:
        #             img[i][j] = 0


        cv2.imwrite(os.path.join(save_path, file), img)


def cal_percent(path):
    label = []

    P = 512 * 512
    for file in tqdm(os.listdir(path)):
        # print(file)
        image = cv2.imdecode(np.fromfile(os.path.join(path, file), dtype=np.uint8), 0)
        # print(image)
        img1 = np.zeros(image.shape)

        img1[image == 0] = 1
        label.append(round(len(np.flatnonzero(img1)) / P, 4))

    print(np.mean(label))

            # 1: 烤烟  2: 玉米    3:薏仁米
# 0: 0.1 -- 1: 0.3 -- 2: 0.17 -- 3 : 0.43
'''            红色      绿色      黄色
    0: 背景  1: 烤烟  2: 玉米   3:薏仁米
    0: 0.1, 1: 0.3,  2: 0.17, 3 : 0.43
    1 棕色 分布于具有明显垂直间距的条带
    2 分布于具有明显灰白相间垂直间距的条带
    3 绿色 个体,密集分布
'''


def reset_pixel(path, save_path):
    name_list = [x for x in os.listdir(path) if x.endswith(".png")]
    for file in tqdm(name_list):
    # for file in os.listdir(path):
        # print(file)
        image = cv2.imdecode(np.fromfile(os.path.join(path, file), dtype=np.uint8), 0)
        # img = np.zeros(image.shape)
        img = image.copy()

        img[image == 255] = 11
        # img[image == 8] = 2
        # img[image == 7] = 3
        # img[image == 6] = 4
        # img[image == 18] = 20
        # for i in range(512):
        #     for j in range(512):
        #         if img[i][j] == 127:
        #             img[i][j] = 0
                # else:
                #     img[i][j] = 255
        cv2.imwrite(os.path.join(save_path, file), img)


# 再最外层添加一圈轮廓
def outlineadd(path, save_path):
    name_list = [x for x in os.listdir(path) if x.endswith(".png")]
    for file in tqdm(name_list):
        image = cv2.imdecode(np.fromfile(os.path.join(path, file), dtype=np.uint8), 0)
        img = image.copy()

        l = [0, 319]
        for i in l:
            for j in range(319):
                img[i][j] = 0
                img[j][i] = 0

        img2 = np.zeros(image.shape)
        # img2[img < 100] = 0
        img2[img == 255] = 255

        cv2.imwrite(os.path.join(save_path, file), img2)

if __name__ == '__main__':
    path = r"G:\exp\crowdAI\deeplab\pre\0-1"
    save_path = r"G:\exp\crowdAI\deeplab\pre\0-255"
    change_pixel(path, save_path)
    # outlineadd(path, save_path)
    # cal_percent_path = r"G:\dataset\gm_data_voc\gm_voc\SegmentationClass"
    # cal_percent(cal_percent_path)
    # reset_pixel(path, save_path)