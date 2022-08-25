# -*- coding: utf-8 -*-
import os
from tqdm import tqdm
import shutil
# 设定文件路径

def rename_by_str(path, save_path):
    # path = r'G:\data\tianchi\select\val\8bit_out'
# "戴村镇", "_dcz"  "党湾镇", "_dwz" "益农镇", "_ynz" "进化镇", "_jhz" "新湾街道", "_xwjd"
# 对目录下的文件进行遍历 "所前镇", "_sqz" "新塘街道(0)", "_xtjd" "河上镇", "_hsz"
    for file in tqdm(os.listdir(path)):
        # 判断是否是文件
        if os.path.isfile(os.path.join(path, file)) == True:
            # 设置新文件名 image_1_1_86_out.png
            name = file
            new_name = file.replace("_edge", "")
            # 重命名
            # 原地修改
            os.rename(os.path.join(path, file), os.path.join(save_path, new_name))
            # 到另一个文件夹
            # shutil.copy(os.path.join(path, file), os.path.join(save_path, new_name))
            # print('running ----->', name)
    # 结束
    print("End")


def rename_by_txt(txt_path, src_path, save_path):
    file_list = txt_path
    file_list = tuple(open(file_list, 'r'))
    file_list = [id_.rstrip() for id_ in file_list]

    # print(len(file_list))
    # image_list = [x for x in os.listdir(src_path) if x.endswith(".png")]
    # print(len(image_list))

    for i in tqdm(range(len(file_list))):
        a = os.path.join(src_path, "{}.png".format(i))
        b = os.path.join(save_path, (file_list[i] + ".png"))
        os.rename(a, b)
        # print(a, "-----", b)


def copy_file_by_txt(txt_path, src_path, save_path):
    '''
    :param txt_path: txt文件名路径
    :param src_path: 文件路径
    :param save_path: 保存路径
    '''
    file_list = txt_path
    file_list = tuple(open(file_list, 'r'))
    file_list = [id_.rstrip() for id_ in file_list]

    for i in tqdm(file_list):
        file_whole_path = os.path.join(src_path, "{}.png".format(i))
        # b = os.path.join(save_path, (file_list[i] + ".png"))
        shutil.copy(file_whole_path, save_path)


def rename_by_add_str(src_path, save_path):
    for file in tqdm(os.listdir(src_path)):
        # 判断是否是文件
        # 设置新文件名 image_1_1_86_out.png
        # name = file.split("_")
        name = file.split(".")
        # new_name = name[1].split(".")[0] + "_" + name[0] + ".png"
        # new_name = name[0] + "_image.png"
        # new_name = name[0] + "_label.png"
        new_name = name[0] + "_bound.png"
        # new_name = name[0] + "_sgd" + ".png"
        # 重命名
        shutil.copy(os.path.join(src_path, file), os.path.join(save_path, new_name))



if __name__ == '__main__':
    txt_path = r"G:\dataset\gm_data_voc\VOC2012\ImageSets\Segmentation\val.txt"
    src_path = r"G:\exp\crowdAI\my\test\edge_3"
    save_path = r"G:\exp\crowdAI\my\pre\test_all"
    # rename_by_txt(txt_path, src_path, save_path)
    # rename_by_str(src_path, save_path)
    # copy_file_by_txt(txt_path, src_path, save_path)
    rename_by_add_str(src_path, save_path)