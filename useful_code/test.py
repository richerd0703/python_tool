import PIL.Image as Image
import os

IMAGES_FORMAT = ['.jpg', '.png']  # 图片格式
IMAGE_SIZE = 512  # 每张小图片的大小
IMAGE_SIZE_x = 512
IMAGE_ROW = 2  # 图片间隔，也就是合并成一张图后，一共有几行
IMAGE_COLUMN = 3  # 图片间隔，也就是合并成一张图后，一共有几列


def image_compose_by_index(file_path, save_path):
    name_list = [x for x in os.listdir(file_path) if x.endswith(".png")]
    name_list.sort()
    index = 6
    while index <= len(name_list):
        image_names = name_list[index - 6: index]
        index += 6
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
    src_path = r"G:\dataset\temp\pre\pre_3\all"
    save_path = r"G:\dataset\temp\pre\pre_3\com"
    image_compose_by_index(src_path, save_path)