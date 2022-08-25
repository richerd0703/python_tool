import os
import shutil
from PIL import Image
from tqdm import tqdm
# 遍历文件夹
def iter_files(old_path, new_path):
    # 遍历根目录 group
    for root, dirs, files in os.walk(old_path):
        for file in files:
            group = root.split("\\")[-1]
            # file_name = os.path.join(root, file)
            path1 = os.path.join(root, file)
            print("path1", path1)
            # print(path1.size())
            path2 = os.path.join(new_path, group + "_" + file)
            print("path2",path2)
            shutil.copy(path1, path2)
    print("finished")
    exit(0)
def print_imgsize(path1):
    # tif_list = [x for x in os.listdir(path) if x.endswith(".tiff")]
    # l = sorted(tif_list)
    # print(l)
    for file in os.listdir(path1):
        im = Image.open(os.path.join(save_path, file))  # 返回一个Image对象
        print(file, "=====", im.size)
        # print('宽：%d,高：%d' % (im.size[0], im.size[1]))
def print_2imgsize(path1, path2):
    info1 = []
    for file in os.listdir(path1):
        im = Image.open(os.path.join(save_path, file))  # 返回一个Image对象
        size = im.size
        info1.append(file + "===" + str(size))
    info2 = []
    for file in os.listdir(path1):
        im = Image.open(os.path.join(save_path, file))  # 返回一个Image对象
        size = im.size
        info2.append(file + "===" + str(size))

    for i in range(len(info2)):
        print(info2[i], "====", info1[i])
if __name__ == '__main__':
    src_path = r"H:\海康_全要素\hk_drone"
    save_path = r"H:\海康_全要素\drone\poly"
    path2 = r"H:\海康_全要素\drone\poly"
    # iter_files(src_path, save_path)
    # print_imgsize(save_path)
    print_2imgsize(save_path, path2)