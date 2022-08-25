import PIL.Image as Image
import os
from tqdm import tqdm

# IMAGES_PATH = r'G:\dataset\temp\pre\all'  # 图片集地址
# IMAGES_FORMAT = ['.jpg', '.png']  # 图片格式
# IMAGE_SIZE = 512  # 每张小图片的大小
# IMAGE_SIZE_x = 512
# IMAGE_ROW = 2  # 图片间隔，也就是合并成一张图后，一共有几行
# IMAGE_COLUMN = 3  # 图片间隔，也就是合并成一张图后，一共有几列
# IMAGE_SAVE_PATH = r'G:\dataset\temp\pre\pre2_complax4.jpg'  # 图片转换后的地址
#
# # 获取图片集地址下的所有图片名称
# image_names = [name for name in os.listdir(IMAGES_PATH) for item in IMAGES_FORMAT if
#                os.path.splitext(name)[1] == item]
#
# # 简单的对于参数的设定和实际图片集的大小进行数量判断
# # if len(image_names) != IMAGE_ROW * IMAGE_COLUMN:
# #     raise ValueError("合成图片的参数和要求的数量不能匹配！")
#
# # 对图片排序
# # image_names.sort(key=lambda x:int(x.split("_")[0][:-4]))
# image_names.sort()
# # print(image_names)
#
# # # 定义图像拼接函数
# def image_compose():
#     to_image = Image.new('RGB', (IMAGE_COLUMN * IMAGE_SIZE_x, IMAGE_ROW * IMAGE_SIZE)) #创建一个新图
#     # 循环遍历，把每张图片按顺序粘贴到对应位置上
#     for y in range(1, IMAGE_ROW + 1):
#         for x in range(1, IMAGE_COLUMN + 1):
#             from_image = Image.open(IMAGES_PATH + "//" + image_names[IMAGE_COLUMN * (y - 1) + x - 1]).resize(
#                 (IMAGE_SIZE_x, IMAGE_SIZE),Image.ANTIALIAS)
#             if x != 1 and y != 1:
#                 # print((x - 1) * IMAGE_SIZE_x - 10, (y - 1) * IMAGE_SIZE - 10)
#                 to_image.paste(from_image, ((x - 1) * IMAGE_SIZE_x - 10, (y - 1) * IMAGE_SIZE - 10))
#                 print("do if")
#             else:
#                 to_image.paste(from_image, ((x - 1) * IMAGE_SIZE_x, (y - 1) * IMAGE_SIZE))
#                 print("do else")
#     return to_image.save(IMAGE_SAVE_PATH) # 保存新图
# image_compose() #调用函数


# IMAGES_FORMAT = ['.jpg', '.png']  # 图片格式
IMAGE_SIZE = 320 # 每张小图片的大小
IMAGE_SIZE_x = 320
IMAGE_ROW = 2  # 图片间隔，也就是合并成一张图后，一共有几行
IMAGE_COLUMN = 3  # 图片间隔，也就是合并成一张图后，一共有几列
STEP = IMAGE_ROW * IMAGE_COLUMN

def image_compose_by_index(file_path, save_path):
    name_list = [x for x in os.listdir(file_path) if x.endswith(".png")]
    name_list.sort()
    index = STEP
    while index <= len(name_list):
        image_names = name_list[index - STEP: index]
        index += STEP
        to_image = Image.new('RGB', (IMAGE_COLUMN * IMAGE_SIZE_x, IMAGE_ROW * IMAGE_SIZE))  # 创建一个新图
        # 循环遍历，把每张图片按顺序粘贴到对应位置上
        name = image_names[0]
        for y in range(1, IMAGE_ROW + 1):
            for x in range(1, IMAGE_COLUMN + 1):
                from_image = Image.open(file_path + "\\" + image_names[IMAGE_COLUMN * (y - 1) + x - 1]).resize(
                    (IMAGE_SIZE_x, IMAGE_SIZE), Image.ANTIALIAS)
                to_image.paste(from_image, ((x - 1) * IMAGE_SIZE_x, (y - 1) * IMAGE_SIZE))
        to_image.save(os.path.join(save_path, name))  # 保存新图


if __name__ == '__main__':
    src_path = r"G:\exp\crowdAI\my\pre\test_all"
    save_path = r"G:\exp\crowdAI\my\pre\com"
    image_compose_by_index(src_path, save_path)

# 定义图像拼接函数
# def image_compose():
#     to_image = Image.new('RGB', (IMAGE_COLUMN * IMAGE_SIZE_x, IMAGE_ROW * IMAGE_SIZE)) #创建一个新图
#     # 循环遍历，把每张图片按顺序粘贴到对应位置上
#     for y in range(1, IMAGE_ROW + 1):
#         for x in range(1, IMAGE_COLUMN + 1):
#             from_image = Image.open(IMAGES_PATH + image_names[IMAGE_COLUMN * (y - 1) + x - 1]).resize(
#                 (IMAGE_SIZE_x, IMAGE_SIZE),Image.ANTIALIAS)
#             to_image.paste(from_image, ((x - 1) * IMAGE_SIZE_x, (y - 1) * IMAGE_SIZE))
#     return to_image.save(IMAGE_SAVE_PATH) # 保存新图
# image_compose() #调用函数