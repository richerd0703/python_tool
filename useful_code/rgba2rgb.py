from PIL import Image
import os
from tqdm import tqdm
Image.MAX_IMAGE_PIXELS = 1000000000000000
file_type = ['tif', 'tiff', 'png', 'jpg']

def get_file_names(data_dir, file_type):
    result_dir = []
    result_name = []
    for maindir, subdir, file_name_list in os.walk(data_dir):
        for filename in file_name_list:
            apath = maindir + '/' + filename
            ext = apath.split('.')[-1]
            if ext in file_type:
                result_dir.append(apath)
                result_name.append(filename)
            else:
                pass
    return result_dir, result_name


def fourTo3(data_dir, out_dir, file_type):
    img_dir, img_name = get_file_names(data_dir, file_type)
    count = 0
    for each_dir, each_name in tqdm(zip(img_dir, img_name)):
        image = Image.open(each_dir)
        r, g, b, a = image.split()
        image = Image.merge("RGB", (r, g, b))
        # out_dir_images = out_dir + '/' + each_name
        out_dir_images = os.path.join(out_dir, each_name.split()[0] + ".png")
        image.save(out_dir_images)


if __name__ == '__main__':
    data_dir = r'I:\data\preliminary\test\image'  # 需转换得图片路径
    out_dir = r'I:\data\preliminary\test\image_png'  # 转换后存储的路径
    fourTo3(data_dir, out_dir, file_type)
