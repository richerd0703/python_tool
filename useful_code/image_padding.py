import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import numpy as np

def img_padding_1():
    path = r'G:\dataset\mapping_challenge_dataset\raw\val\color'
    label = False
    fileset = []
    fileLists = os.listdir(path)
    for file in fileLists:
        if file.split(".")[-1] == ("png") or file.split(".")[-1] == ("jpg"):
            # print(file)
            whole_path = path + "\\" + file
            fileset.append(whole_path)
    num = len(fileset)
    for i in tqdm(range(len(fileset))):
        # print(fileset[i], '-----', i)
        # img = cv2.imread(fileset[i], 0)  # 灰度图

        img = cv2.imread(fileset[i], 1)  # 彩图
        # img = cv2.imread(r"I:\map_cut\test\1\20_2.png")
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        top_size, bottom_size, left_size, right_size = (10, 10, 10, 10)

        # resize = cv2.copyMakeBorder(img, top_size, bottom_size, left_size, right_size, borderType=cv2.BORDER_REPLICATE) # 也就是复制最边缘像素。
        resize = cv2.copyMakeBorder(img, top_size, bottom_size, left_size, right_size, cv2.BORDER_REFLECT) # 对感兴趣的图像中的像素在两边进行复制
        # resize = cv2.copyMakeBorder(img, top_size, bottom_size, left_size, right_size, cv2.BORDER_REFLECT_101) # 也就是以最边缘像素为轴，对称
        # resize = cv2.copyMakeBorder(img, top_size, bottom_size, left_size, right_size, cv2.BORDER_WRAP) # 外包装法
        # resize = cv2.copyMakeBorder(img, top_size, bottom_size, left_size, right_size, cv2.BORDER_CONSTANT, value=0)  # 常量法，常数值填充
        name = fileset[i].split('\\')[-1]
        if label:
            resize = cv2.cvtColor(resize, cv2.COLOR_BGR2GRAY)
        cv2.imwrite(r"G:\dataset\mapping_challenge_dataset\temp\test\enlarge\{}".format(name), resize)


def img_padding_2(path, save_path):
    new_width, new_height = 320, 320
    for root, dir, files in os.walk(path):
        for file in tqdm(files):
            # 读入原图片
            img = cv2.imdecode(np.fromfile(os.path.join(root, file), dtype=np.uint8), -1)
            # print(img)
            # 将图片高和宽分别x赋值给x，y
            # height, width = img.shape[0:2]
            # 显示原图
            # cv2.imshow('OriginalPicture', img)
            # (width, int(height / 3)) 元组形式，高度缩放到原来的三分之一
            img_change1 = cv2.resize(img, (new_width, new_height))
            # img_change1 = cv2.resize(img, (new_width, new_height), cv2.INTER_LINEAR)
            ext = file[-4:]
            cv2.imencode(ext, img_change1)[1].tofile(os.path.join(save_path, file))
            # cv2.imencode('.jpg', img_change1)[1].tofile(save_path + file.split('.')[0]
            #                                             + '_' + 'overlength' + '.jpg')


if __name__ == '__main__':
    src_path = r"G:\dataset\mapping_challenge_dataset\temp\test\label"
    save_path = r"G:\dataset\mapping_challenge_dataset\temp\test\label"
    img_padding_2(src_path, save_path)