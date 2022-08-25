import os
from PIL import Image

path1 = r"F:\wd_data\cut\2"
path2 = r"F:\wd_data\cut\1"

files1 = os.listdir(path1)
files2 = os.listdir(path1)
l1 = files1


# def listdir(path, list_name):  # 传入存储的list
#     for file in os.listdir(path):
#         file_path = os.path.join(path, file)
#         if os.path.isdir(file_path):
#             listdir(file_path, list_name)
#         else:
#             list_name.append(file_path)
#
#
# list_name = []
# listdir(path, list_name)
# print(list_name)
#
# with open('./list.txt', 'w') as f:  # 要存入的txt
#     write = ''
#     for i in list_name:
#         write = write + str(i) + '\n'
#     f.write(write)

# dirs = os.listdir(path)
# for file in dirs:
#     wholepath = path +"\\" + file
#     print(wholepath)
#     image = Image.open(file)
#     image.show()
#     r_image = unet.detect_image(image)
#     r_image.save("./img/predict/{}".format(file))
