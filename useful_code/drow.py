import os
os.environ["OPENCV_IO_MAX_IMAGE_PIXELS"] = pow(2,40).__str__()
import cv2
from tqdm import tqdm
import numpy as np
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None

# edge = cv2.conv_tri(image, 1)
# ox, oy = cv2.grad2(conv_tri(edge, 4))
# oxx, _ = cv2.grad2(ox)
# oxy, oyy = cv2.grad2(oy)
# ori = np.mod(np.arctan(oyy * np.sign(-oxy) / (oxx + 1e-5)), np.pi)

color_list = [
    [255, 255, 255], [255, 0, 0], [0, 0, 255], [0, 255, 0]
]

color_list1 = [
    [255, 255, 255],  # 0
    [49, 130, 189],  # 1
    [253, 208, 163],  # 2
    [253, 174, 107],  # 3
    [253, 141, 60],  # 4
    [255, 255, 255],  # 0
    [198, 219, 239],  # 6
    [255, 255, 255],  # 0
    [255, 255, 255],  # 0
    [255, 255, 255],  # 0
    [255, 255, 255],  # 0
    [255, 255, 255],  # 0
    [255, 255, 255],  # 0
    [255, 255, 255],  # 0
    [255, 255, 255],  # 0
    [188, 189, 220],  # 15
    [218, 218, 235],  # 16
    [99, 99, 99],  # 17
    [255, 255, 255],  # 0
    [255, 255, 255],  # 0
    [150, 150, 150],  # 20
    [255, 255, 255],  # 0
    [82, 84, 163],  # 22
    [107, 110, 207],  # 23
    [255, 255, 255]  # 0
]

color_list2 = [
    [255, 255, 255],  # 0
    [49, 130, 189],  # 1
    [253, 208, 163],  # 2
    [82, 84, 163]  # 3

]

'''
4 #c6dbef
5 #e6550d
6 #fd8d3c
7 #fdae6b
8 #fdd0a2
9 #31a354
10 #74c476
11 #a1d99b
12 #c7e9c0
13 #756bb1
14 #9e9ac8
15 #bcbddc
16 #dadaeb
17 #636363
18 #969696
19 #bdbdbd
20 #d9d9d9
21 #393b79
22 #5254a3
23 #6b6ecf
24 #9c9ede

'''

def process(label_path, pre_path, save_path):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    name_list = [x for x in os.listdir(label_path) if x.endswith(".png")]
    for file in tqdm(name_list):
        label_image = cv2.imdecode(np.fromfile(os.path.join(label_path, file), dtype=np.uint8), 0)
        pre_image = cv2.imdecode(np.fromfile(os.path.join(pre_path, file), dtype=np.uint8), 0)
        # print(image.shape)
        img = np.zeros(label_image.shape)
        ret, thresh = cv2.threshold(label_image, 0, 255, 0)
        countours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

        # 参数    image, contours, contourIdx, color[, thickness[
        cv2.drawContours(img, countours, -1, 3, 2)
        n = label_image.shape[0]
        for i in range(n):
            for j in range(n):
                if i < 2 or j < 2 or i > n - 3 or j > n - 3:
                    img[i][j] = 0
                elif label_image[i][j] == pre_image[i][j]:
                    # 真确
                    pass
                # elif label_image[i][j] != pre_image[i][j] and label_image[i][j] == 0 and pre_image[i][j] != 0:
                #     # 漏检
                #     img[i][j] = 1
                elif label_image[i][j] != pre_image[i][j] and pre_image[i][j] == 0 and label_image[i][j] != 0:
                    img[i][j] = 1
                else:
                    # 错检
                    img[i][j] = 2

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

        output_data = label2rgb(img, color_list)
        # Image.fromarray(output_data).save(os.path.join(save_path, image))
        cv2.imwrite(os.path.join(save_path, file), output_data)


def label2rgb(gt, colors):
    # 数字标签转彩色标签
    h, w = gt.shape
    label_rgb = np.zeros((h, w, 3)).astype(np.uint8)
    for i, rgb in enumerate(colors):
        label_rgb[gt == i] = rgb
    return label_rgb


def label2rgb2(label_path, save_path):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    # 数字标签转彩色标签
    for file in tqdm(os.listdir(label_path)):
        # gt = cv2.imdecode(np.fromfile(os.path.join(label_path, file), dtype=np.uint8), 0)
        gt = np.array(Image.open(os.path.join(label_path, file)))
        output_data = label2rgb(gt, color_list2)
        # Image.fromarray(output_data).save(os.path.join(save_path, image))
        Image.fromarray(output_data).save(os.path.join(save_path, file))


if __name__ == '__main__':
    # label  G:\dataset\1\temp\image\labels
    # label_path = r"G:\dataset\tianchi\trainData\total\val\labels"
    label_path = r"I:\data\exp\fcn_3\select\pre"
    pre_path = r"I:\data\exp\ENCNET_8_7\pre"
    save_path = r"I:\data\exp\fcn_3\select\vis"
    # process(label_path, pre_path, save_path)
    label2rgb2(label_path, save_path)
