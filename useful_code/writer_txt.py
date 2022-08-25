import os

'''
将文件名写入txt、csv、xlsx
'''
file_path = r"G:\exp\crowdAI\frame\val\images"
# 判断文件保存路径是否存在，不存在就创建保存路径
count = 0
if os.path.exists(file_path):
    t = open(r"G:\exp\crowdAI\deeplab\voc\ImageSets\Segmentation\val.txt", 'w')
    for root, dirs, files in os.walk(file_path):
        for file in files:
            # 获取文件路径
            # print(os.path.join(root, file))
            # upload_file = os.path.join(root, file)  # 完整路径
            print(file)  # 文件名
            t.write(file + "\n")
            count += 1
    t.close()
    print("finished", count)
else:
    print("此路径不存在....")

# import os
#
# # train.txt 写入   /home/yanghp/WorkSpace/richerd/data/gm/total
# path1 = r'G:\dataset\gm_data_voc\val\images'
# labelpath  = r"G:\dataset\gm_data_voc\val\labels"
# f = open(r'G:\dataset\gm_data_voc\val\val_list.txt', 'w')
# for root, dir, filename in os.walk(path1):
#     for i in filename:
#         # name = i[:-4]
#         # print(i)
#         image = os.path.join(root, i)  # 完整路径
#         label = os.path.join(labelpath, i.split(".")[0] + ".png")
#         f.write(image + " " + label)
#         f.write("\n")
# f.close()

# test.txt 写入
# path1 = r'F:\he\data\edge\512x512\val\images'
# f = open(r'F:\he\data\edge\512x512\test.txt', 'w')
# for root, dir, filename in os.walk(path1):
#     for i in filename:
#         png = i[:-4]
#
#         f.write(os.path.join(root, i))
#         f.write("\n")
# f.close()

# def run(img1_path, img2_path, out_path):
#     # cmd = os.path.dirname(os.path.realpath(__file__))
#     # cmd = os.path.join(cmd, 'build')
#     # os.chdir(cmd)
#     img1_path = img1_path.replace("/", "\\")
#     data = [["[path]"], ['img1_path =', img1_path, ], ['img2_path =', img2_path], ['out_path =', out_path]]
#     with open("../file/test1.ini", "w") as f:  # 设置文件对象
#
#         for i in data:  # 对于双层列表中的数据
#
#             i = str(i).strip('[').strip(']').replace(',', '').replace('\'', '') + '\n'
#             f.write(i)
#
#
# print('finished')
#
# if __name__ == '__main__':
#     p1 = r"path1:/sdf\d"
#     p2 = "path2"
#     p3 = "path3"
#     run(p1, p2, p3)

# [path]
# img1_path = D:\code\build_tool\input\lable_a\suzhou1_clip1.tif
# img2_path = D:\code\build_tool\input\lable_b\suzhou2_clip1.tif
# out_path = D:\code\build_tool\input\change
