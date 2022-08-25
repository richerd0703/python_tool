import os
import shutil
from tqdm import tqdm
# 遍历文件夹
def iter_files(path):
    # 遍历根目录
    count = 0
    for root, dirs, files in os.walk(path):
        for file in files:
            if file[7:12] == "image" and file[-4:] == ".xml":
                count += 1
            # field = file[7:12]
            #     print(file[7:12], file[-4:])
                # file_name = os.path.join(root, file)
                path1 = os.path.join(root, file)
                # path2 = os.path.join(new_path, file)
                print(path1)
                # 删除
                os.remove(path1)
                # 复制
            # shutil.copy(path1, path2)
    print(count, "finished")
    exit(0)

def deldir(dir):
    if not os.path.exists(dir):
        return False
    if os.path.isfile(dir):
        os.remove(dir)
        return
    for i in os.listdir(dir):
        t = os.path.join(dir, i)
        if os.path.isdir(t):
            deldir(t)#重新调用次方法
        else:
            os.unlink(t)
    os.rmdir(dir)#递归删除目录下面的空文件夹

def copy_file(srcpath, savepath):
    # 遍历根目录
    count = 0
    for root, dirs, files in os.walk(srcpath):
        for file in files:
            if file[-9:-4] == "3band":
                # print(file)
                count += 1
                # 源文件
                path1 = os.path.join(root, file)
                #  保存路径
                path2 = os.path.join(savepath, file)
                print(path1)
                # 复制
                shutil.copy(path1, path2)
    print(count, "finished")
    exit(0)

if __name__ == '__main__':
    src_path = r"G:\dataset\mapping_challenge_dataset\temp\val\dfhh"
    save_path = r"F:\he\data\samples-高二\gf2\t"
    # iter_files(src_path)
    # copy_file(src_path, save_path)
    deldir(src_path)