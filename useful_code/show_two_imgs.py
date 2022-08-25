import cv2
import numpy as np
from matplotlib import pyplot as plt
import math
import os
import re
from tqdm import tqdm

path1 = r"G:\dataset\tianchi\trainData\out\eptnet_new\draw"                      # 图片1
path2 = r"G:\dataset\tianchi\trainData\out\eptnet_new\outdir_encnet_gm_tianchi_4_16"      # 图片2
path3 = r"G:\dataset\tianchi\trainData\out\eptnet_new\show2image"   # 保存路径
fileList = os.listdir(path3)
def get_path_name(path):
    fileset = []
    name = []
    fileLists = os.listdir(path)
    # num = len(fileLists)
    for file in fileLists:
        if file.split(".")[-1] == "png" or file.split(".")[-1] == "jpg":
            # print(file)
            name.append(file)
            whole_path = path + "\\" + file
            fileset.append(whole_path)
    fileset.sort(key=lambda x: (re.findall(r"\d+", x)))
    name.sort(key=lambda x: (re.findall(r"\d+", x)))
    return fileset, name


(fileset1, name1) = get_path_name(path1)
(fileset2, name2) = get_path_name(path2)
# print(name1)
# print(name2)
# exit(0)
# (fileset3, name3) = get_path_name(path3)
num = len((fileset1))

for i in tqdm(range(len(fileset1))):
    arr = []
    n=name1[i].split('.')[0]+'.png'
    if n in fileList:
        pass
    else:
        arr.append({
            'name': name1[i],
            'path': fileset1[i]
        })
        arr.append({
            'name': name2[i],
            'path': fileset2[i]

        })
        # arr.append({
        #     'name': name3[i],
        #     'path': fileset3[i]
        #
        # })

        # print(type(arr), arr)

        rows = math.ceil(len(arr) / 2)
        # print('rows', rows)
        #
        # print(arr)
        # exit(0)
        plt.figure(figsize=(16,8))
        plt.tight_layout()  # 调整整体空白
        for index, item in enumerate(arr):
            # print(type(item), item, index)
            # print(item['name'], item['path'])

            name = item['name']
            path = item['path']
            # print(item)
            # exit(0)
            img = cv2.imread(path,1)
            # print(path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # cv2默认为bgr顺序
            # print(path)
            # img=np.array(img)
            # print(img)
            # cv2.imshow('a',img)
            # cv2.waitKey

            plt.subplot(1, 2, index + 1)
            # if index<2:
            #     plt.subplot(1, 3, index + 1)
            # else:
            #     plt.subplot(2, 2, 3)
            plt.imshow(img)

            # plt.figure()
            # plt.subplot(rows, 2, index + 1)
            # img_2 = img[:, :, [2, 1, 0]]
            # plt.imshow(img[:, :, ::-1])

            plt.title(name)
            plt.xticks([])
            plt.yticks([])
            # plt.subplots_adjust(left=0.03, top=0.9, right=0.97, bottom=0.1, wspace=0.06, hspace=0.2)
            # print("正在保存--{},{}".format(name1[i], name2[i]))
            # plt.show()
        # exit(0)
        # plt.show()
        # plt.savefig(r'F:\wd_data\he\show\out\{}.png'.format(name1[i].split(".")[-2]), dpi=200)
        plt.savefig(os.path.join(path3, "{}".format(name1[i].split(".")[-2])), dpi=200)
    # exit(0)# plt.savefig('plt.png', dpi=500) #可调节图片的清晰度


    # exit(0)






