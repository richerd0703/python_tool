import os
import shutil
from tqdm import tqdm


def copyFiles(path, path2, obj_path):
    if not os.path.exists(obj_path):
        os.makedirs(obj_path)
    count = 0
    name_list = [x for x in os.listdir(path2) if x.endswith(".png")]  # path2 后缀
    for i in range(len(name_list)):
        name_list[i] = name_list[i].split('.')[0]
    l = len(name_list)
    # 遍历path路径下，所有文件的根目录，文件名，文件名加扩展名
    for root, dirpath, filename in os.walk(path):
        # 获取每个filename列表的长度，即每个filaname文件夹所含的文件个数
        for index in tqdm(range(len(filename))):
            a = filename[index].split('.')[0]
            if filename[index].split('.')[0] in name_list:
                # print(filename[index].split('.')[0])
                count += 1

                # 获取你想要copy的文件，带扩展名的完整路径
                old_path = os.path.join(root, filename[index])
                # 获取你想要copy到的路径，此处路径依旧是文件的完整路径，即绝对路径
                new_path = os.path.join(obj_path + '/', filename[index])
                # shutil.copy将文件复制到目标文件夹，如果目标文件夹已有该文件会覆盖
                # shutil.copyfile跟copy用法相同，但如果目标文件夹已有该文件会报错

                shutil.copy(old_path, new_path)

                # # 删除文件
                # os.remove(old_path)
                # print(old_path)
    print('完成 ', count, ' 个文件')


if __name__ == '__main__':

    path = r'G:\dataset\tianchi\trainData\out\ENCNet\vis'  # 需要找的文件夹
    # path = r'I:\data\exp\fcn_3\drow'  # 需要找的文件夹
    path2 = r'G:\dataset\tianchi\trainData\out\eptnet_new\select\vis'  # 已经确定的文件夹
    obj_path = r'G:\dataset\tianchi\trainData\out\ENCNet\select\vis'  # 保存路径

    copyFiles(path, path2, obj_path)