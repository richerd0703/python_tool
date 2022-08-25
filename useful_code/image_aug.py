import os
import random
import glob
import numpy as np
import imgaug as ia
import imgaug.augmenters as iaa
from imgaug.augmentables.segmaps import SegmentationMapsOnImage
from PIL import Image

class ImageAugmentation(object):
    def __init__(self, image_aug_dir, segmentationClass_aug_dir, image_start_num=1):
        self.image_aug_dir = image_aug_dir
        self.segmentationClass_aug_dir = segmentationClass_aug_dir
        self.image_start_num = image_start_num  # 增强后图片的起始编号
        self.seed_set()
        if not os.path.exists(self.image_aug_dir):
            os.mkdir(self.image_aug_dir)
        if not os.path.exists(self.segmentationClass_aug_dir):
            os.mkdir(self.segmentationClass_aug_dir)

    def seed_set(self, seed=1):
        np.random.seed(seed)
        random.seed(seed)
        ia.seed(seed)

    def array2p_mode(self, alpha_channel):
        """alpha_channel is a binary image."""
        # assert set(alpha_channel.flatten().tolist()) == {0, 1}, "alpha_channel is a binary image."
        alpha_channel[alpha_channel == 1] = 128
        h, w = alpha_channel.shape
        image_arr = np.zeros((h, w, 3))
        image_arr[:, :, 0] = alpha_channel
        img = Image.fromarray(np.uint8(image_arr))
        img_p = img.convert("P")
        return img_p

    def augmentor(self, image):
        height, width, _ = image.shape
        # resize = iaa.Sequential([
        #     iaa.Resize({"height": int(height/2), "width": int(width/2)}),
        # ])  # 缩放

        fliplr_flipud = iaa.Sequential([
            iaa.Fliplr(),
            iaa.Flipud(),
        ])  # 左右+上下翻转

        # rotate = iaa.Sequential([
        #     iaa.Affine(rotate=(-70, 70))
        # ])  # 旋转
        #
        # translate = iaa.Sequential([
        #     iaa.Affine(translate_percent=(0.2, 0.35))
        # ])  # 平移

        # crop_and_pad = iaa.Sequential([
        #     iaa.CropAndPad(percent=(-0.25, 0), keep_size=False),
        # ])  # 裁剪

        # rotate_and_crop = iaa.Sequential([
        #     iaa.Affine(rotate=45),
        #     iaa.CropAndPad(percent=(-0.25, 0), keep_size=False)
        # ])  # 旋转 + 裁剪

        guassian_blur = iaa.Sequential([
            iaa.GaussianBlur(sigma=(2, 3)),
        ])  # 增加高斯噪声

        sharpen = iaa.Sequential([
            iaa.Sharpen(alpha=(0, 1.0), lightness=(0.75, 1.5)),
        ])  # 锐化

        piecewise_affine = iaa.Sequential([
            iaa.PiecewiseAffine(scale=(0.01, 0.05)),
        ])  # 扭曲图像的局部区域

        ops = [fliplr_flipud, guassian_blur, sharpen, piecewise_affine]
        # ops = piecewise_affine
        #      镜像+上下翻转、   旋转、    xy平移、     高斯平滑         锐化         扭曲
        return ops


    def augment_img(self, image_name, segmap_name):
        # 1.Load an image.
        img_name = image_name.split('\\')[-1]
        image = Image.open(image_name)  # RGB
        segmap = Image.open(segmap_name)  # P

        name = f"{self.image_start_num:04d}"
        # image.save(self.image_aug_dir + str(int(name)) + ".png")
        # segmap.save(self.segmentationClass_aug_dir + str(int(name)) + ".png")
        # for i in range(5):
        # image.save(self.image_aug_dir + img_name.split('.')[0] + '_' + str(i) + '.png')
        # segmap.save(self.segmentationClass_aug_dir + img_name.split('.')[0] + '_' + str(i) + '.png')
        self.image_start_num += 1

        image = np.array(image)
        segmap = SegmentationMapsOnImage(np.array(segmap), shape=image.shape)

        # 2. define the ops
        ops = self.augmentor(image)

        # 3.execute ths ops
        for _, op in enumerate(ops):
            name = f"{self.image_start_num:04d}"
            print(f"当前增强了{self.image_start_num:04d}张数据...")
            images_aug_i, segmaps_aug_i = op(image=image, segmentation_maps=segmap)
            images_aug_i = Image.fromarray(images_aug_i)
            images_aug_i.save(self.image_aug_dir + str(int(name)) + ".png")
            segmaps_aug_i_ = segmaps_aug_i.get_arr()
            segmaps_aug_i_[segmaps_aug_i_ > 0] = 1
            segmaps_aug_i_ = self.array2p_mode(segmaps_aug_i_)
            segmaps_aug_i_.save(self.segmentationClass_aug_dir + str(int(name)) + ".png")
            self.image_start_num += 1

    def augment_images(self, image_dir, segmap_dir):
        image_names = sorted(glob.glob(image_dir + "*"))
        segmap_names = sorted(glob.glob(segmap_dir + "*"))
        image_num = len(image_names)

        count = 1
        for image_name, segmap_name in zip(image_names, segmap_names):
            print("*"*20, f"正在增强第【{count:04d}/{image_num:04d}】张图片...", "*"*20)
            self.augment_img(image_name, segmap_name)
            count += 1


if __name__ == "__main__":
    image_dir = r"G:\dataset\temp\image/"                                 # image目录，必须是RGB
    segmap_dir = r"G:\dataset\temp\label\0-255/"                         # segmap目录，必须是p模式的图片
    image_aug_dir = r"G:\dataset\temp\1/"                           # image增强后存放目录
    segmentationClass_aug_dir = r"G:\dataset\temp\2/"  # segmap增强后存放目录

    image_start_num = 1
    image_augmentation = ImageAugmentation(image_aug_dir, segmentationClass_aug_dir, image_start_num)
    image_augmentation.augment_images(image_dir, segmap_dir)



