import cv2
import os
from tqdm import tqdm

#   按照坐标点裁剪
def image_cut(path):
    listdir = os.listdir(path)
    # 新建split文件夹用于保存
    newdir = os.path.join(path, 'crop_image')
    if (os.path.exists(newdir) == False):
        os.mkdir(newdir)
    count = 0
    for i in listdir:
        # if i.split('.')[1] == "png" or i.split('.')[1] == "JPG" or i.split('.')[1] == "jpg" :
        if i.split('.')[1] == 'png':
            filepath = os.path.join(path, i)

            filename = i.split('.')[0]
            leftpath1 = os.path.join(newdir, filename) + "_crop1.png"
            leftpath2 = os.path.join(newdir, filename) + "_crop2.png"
            rightpath1 = os.path.join(newdir, filename) + "_crop3.png"
            rightpath2 = os.path.join(newdir, filename) + "_crop4.png"
            img = cv2.imread(filepath)
            [h, w] = img.shape[:2]
            print(filepath, (h, w))

            # 处理image
            if (image):
                l1img = img[0: 512, 0: 512, :]
                l2img = img[400: 912, 0: 512, :]
                r1img = img[0: 512, 400: 912, :]
                r2img = img[400: 912, 400: 912, :]

                cv2.imwrite(leftpath1, l1img)
                cv2.imwrite(leftpath2, l2img)
                cv2.imwrite(rightpath1, r1img)
                cv2.imwrite(rightpath2, r2img)

            # 处理label，将24位转为8位
            if (label):
                l1img = img[0: 512, 0: 512, :]
                l2img = img[400: 912, 0: 512, :]
                r1img = img[0: 512, 400: 912, :]
                r2img = img[400: 912, 400: 912, :]

                croppedl1 = cv2.cvtColor(l1img, cv2.COLOR_BGR2GRAY)
                croppedl2 = cv2.cvtColor(l2img, cv2.COLOR_BGR2GRAY)
                croppedr1 = cv2.cvtColor(r1img, cv2.COLOR_BGR2GRAY)
                croppedr2 = cv2.cvtColor(r2img, cv2.COLOR_BGR2GRAY)

                cv2.imwrite(leftpath1, croppedl1)
                cv2.imwrite(leftpath2, croppedl2)
                cv2.imwrite(rightpath1, croppedr1)
                cv2.imwrite(rightpath2, croppedr2)

            count += 1
    print("num : ", count)


def image_cut2(src_path, save_path):
    for file in tqdm(os.listdir(src_path)):
        imdir = os.path.join(src_path, file)
        # name = os.path.basename(file)
        # filename = name.split('.')[0]
        leftpath1 = os.path.join(save_path, file)
        img = cv2.imread(imdir)
        # [h, w] = img.shape[:2]
        # print(name, (h, w))

        # 处理image
        #              [上下，        左右]
        l1img = img[6: 294, 6: 294, :]  # 12
        # l1img = img[350: 1185, 0: 835, :]  # 27

        # 处理label 转8为
        if label:
            l1img = cv2.cvtColor(l1img, cv2.COLOR_BGR2GRAY)
        cv2.imwrite(leftpath1, l1img)

    # 处理label，将24位转为8位
    # if (label):
    #     l1img = img[0: 512, 0: 512, :]
    #     l2img = img[400: 912, 0: 512, :]
    #     r1img = img[0: 512, 400: 912, :]
    #     r2img = img[400: 912, 400: 912, :]
    #
    #     croppedl1 = cv2.cvtColor(l1img, cv2.COLOR_BGR2GRAY)
    #     croppedl2 = cv2.cvtColor(l2img, cv2.COLOR_BGR2GRAY)
    #     croppedr1 = cv2.cvtColor(r1img, cv2.COLOR_BGR2GRAY)
    #     croppedr2 = cv2.cvtColor(r2img, cv2.COLOR_BGR2GRAY)
    #
    #     cv2.imwrite(leftpath1, croppedl1)
    #     cv2.imwrite(leftpath2, croppedl2)
    #     cv2.imwrite(rightpath1, croppedr1)
    #     cv2.imwrite(rightpath2, croppedr2)


if __name__ == "__main__":
    # 图像文件原始路径
    # path = r"F:\he\data\edge\train\label"
    image = True
    label = True
    # image_cut(path)
    src_path = r"G:\dataset\mapping_challenge_dataset\all_data\test\labels"
    save_path = r"G:\dataset\mapping_challenge_dataset\all_data\test\labels"
    image_cut2(src_path, save_path)

# -*- coding: utf-8 -*-
# import os
# from PIL import  Image
# from tqdm import tqdm
#
# path_img = r'G:\dataset\Massachusetts buildings dataset\resize\label'
# save_path = r"G:\dataset\Massachusetts buildings dataset\resize\1000x1000\label"
#
# img_dir = os.listdir(path_img)
# print(img_dir)
# print(len(img_dir))
# for i in tqdm(range(len(img_dir))):
#     # 根据图片名称提取id,方便重命名
#
#     # id = int((img_dir[i].split('.')[0]).split('_')[0]
#     img = Image.open(path_img + '/' + img_dir[i])
#     size_img = img.size
#     name = img_dir[i].split(".")[0]
#     # print(size_img)
#     print("--------->", img_dir[i])
#     # 准备将图片切割成4张小图片,这里后面的2是开根号以后的数，比如你想分割为9张，将2改为3即可
#     weight = int(size_img[0] // 3)
#     height = int(size_img[1] // 3)
#     for j in range(3):
#         for k in range(3):
#             box = (weight * k, height * j, weight * (k + 1), height * (j + 1))
#             region = img.crop(box)
#             # 输出路径
#             region.save(save_path + '\{}-{}{}.png'.format(name, j, k))
