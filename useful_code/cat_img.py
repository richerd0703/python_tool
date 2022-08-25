import PIL.Image as Image
import os
import re
from tqdm import tqdm

IMAGES_PATH = r'G:\dataset\tianchi\jingwei_round1_test_a_20190619\image_2'  # 图片集地址
IMAGE_SIZE = 512  # 每张小图片的大小
IMAGE_SIZE_x = 512
IMAGE_ROW = 90  # 图片间隔，也就是合并成一张图后，一共有几行
IMAGE_COLUMN = 152  # 图片间隔，也就是合并成一张图后，一共有几列
IMAGE_SAVE_PATH = r'G:\dataset\tianchi\jingwei_round1_test_a_20190619\compose\image_2_pre.png'  # 图片转换后的地址

image_names = []
for file in os.listdir(IMAGES_PATH):
    if file.split(".")[-1] == ("png"):
        image_names.append(file)

# 简单的对于参数的设定和实际图片集的大小进行数量判断
if len(image_names) != IMAGE_ROW * IMAGE_COLUMN:
    raise ValueError("合成图片的参数和要求的数量不能匹配！")
# 这里是对图片排序 要根据不同情况修改
image_names1 = sorted(image_names, key=lambda x: int(x.split("_")[-1][:-4]))
# print(image_names1)
image_names = sorted(image_names1, key=lambda x: int(x.split("_")[-2]))
# print(image_names)


# 定义图像拼接函数
def image_compose():
    to_image = Image.new('RGB', (IMAGE_COLUMN * IMAGE_SIZE_x, IMAGE_ROW * IMAGE_SIZE))  # 创建一个新图
    # 循环遍历，把每张图片按顺序粘贴到对应位置上
    for y in tqdm(range(1, IMAGE_ROW + 1)):
        for x in range(1, IMAGE_COLUMN + 1):
            from_image = Image.open(IMAGES_PATH + "\\" + image_names[IMAGE_COLUMN * (y - 1) + x - 1]).resize(
                (IMAGE_SIZE_x, IMAGE_SIZE), Image.ANTIALIAS)
            name = image_names[IMAGE_COLUMN * (y - 1) + x - 1]
            to_image.paste(from_image, ((x - 1) * IMAGE_SIZE_x, (y - 1) * IMAGE_SIZE))
    return to_image.save(IMAGE_SAVE_PATH)  # 保存新图


image_compose()  # 调用函数
exit()
print("finished")