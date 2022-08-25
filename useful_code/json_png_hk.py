import os
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm
from PIL import Image
import re

img_path = r'H:\hk\temp\img'
mask_path = r'H:\hk\temp\1000x1000'
# color_label_path = r'E:\hk\res\color_label'
json_label_path = r'H:\hk\20210416_浙工大标定结果\B0001_Drone_ZJUT_china\1_0.05m_drone_png.json'

classes = {'草地': 1, '道路': 2, '耕地': 3, '建筑物': 4, '林地': 5, '裸地': 6, '水体': 7, '其他人工用地': 8}

data = pd.read_json(json_label_path)
info = data.iloc[3]
info = info.iloc[0][0]['VideoInfo']
data = info['mapFrameInfos']

for i in tqdm(range(len(data))):
    info = data[i]
    img_name = info['key']['FrameNum']
    # num = img_name.split('-')[0]
    # s = int(re.findall("\d+", num)[0])
    # if s < 12:
    mask = np.zeros([1000, 1000], dtype=np.uint8)
    # else:
    #     mask = np.zeros([2000, 2000], dtype=np.uint8)
    labels = info['value']['mapTargets']

    for j in range(len(labels)):
        label = labels[j]['value']
        land_type = label['PropertyPages'][0]['PropertyPageDescript']
        ploy = []
        points = label['Vertex']
        for k in range(len(points)):
            # if s < 12:
            temp_x = int(points[k]['fX'] * 1000)
            temp_y = int(points[k]['fY'] * 1000)
            # else:
            #     temp_x = int(points[k]['fX'] * 2000)
            #     temp_y = int(points[k]['fY'] * 2000)
            ploy.append([temp_x, temp_y])
        ploy = [np.array(ploy)]
        cv2.fillPoly(mask, ploy, classes[land_type])
    mask = Image.fromarray(mask)
    mask.save(os.path.join(mask_path, img_name))