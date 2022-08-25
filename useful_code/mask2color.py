import os
import glob
import cv2
import numpy as np
from tqdm import tqdm
from PIL import Image

paths = r"G:\little\predict\predict_result\hangzhouAB\20220416\B\mask"
save_path = r"G:\little\predict\predict_result\hangzhouAB\20220416\B\color"
if not os.path.exists(save_path):
    os.makedirs(save_path)
images = os.listdir(paths)

print(images)

# color_list = [
#     [255, 255, 255], [0, 0, 255], [0, 255, 255], [0, 255, 0], [255, 255, 0], [255, 0, 0]
# ]

color_list = [
    [190, 255, 232], [70, 247, 9], [128, 128, 128], [222, 213, 110], [240, 240, 0], [70, 136, 2], [172, 140, 86],
    [1, 127, 240], [190, 230, 232]
]


def label2rgb(gt, colors):
    # 数字标签转彩色标签
    h, w = gt.shape
    label_rgb = np.zeros((h, w, 3)).astype(np.uint8)
    for i, rgb in enumerate(colors):
        label_rgb[gt == i] = rgb
    return label_rgb


for image in tqdm(images):
    path = os.path.join(paths, image)

    data = np.array(Image.open(path))
    output_data = label2rgb(data, color_list)
    Image.fromarray(output_data).save(os.path.join(save_path, image))
