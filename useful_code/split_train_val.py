import shutil
import os
import random


def get_train_val(data_dir):
    all_images_dir = os.path.join(data_dir, "images/")  # image
    all_labels_dir = os.path.join(data_dir, "labels/")      # label
    train_imgs_dir = os.path.join(data_dir, "train/images/")
    if not os.path.exists(train_imgs_dir): os.makedirs(train_imgs_dir)
    val_imgs_dir = os.path.join(data_dir, "val/images/")
    if not os.path.exists(val_imgs_dir): os.makedirs(val_imgs_dir)
    train_labels_dir = os.path.join(data_dir, "train/labels/")
    if not os.path.exists(train_labels_dir): os.makedirs(train_labels_dir)
    val_labels_dir = os.path.join(data_dir, "val/labels/")
    if not os.path.exists(val_labels_dir): os.makedirs(val_labels_dir)
    for name in os.listdir(all_images_dir):
        image_path = os.path.join(all_images_dir, name)
        label_path = os.path.join(all_labels_dir, name)
        #调整 训练集和验证集的比例
        if random.randint(0, 10) < 2:
            image_save = os.path.join(val_imgs_dir, name)
            label_save = os.path.join(val_labels_dir, name)
        else:
            image_save = os.path.join(train_imgs_dir, name)
            label_save = os.path.join(train_labels_dir, name)
        shutil.move(image_path, image_save)
        shutil.move(label_path, label_save)
    total_nums = len(os.listdir(all_images_dir))
    train_nums = len(os.listdir(train_imgs_dir))
    val_nums = len(os.listdir(val_imgs_dir))
    print("all: " + str(total_nums))
    print("train: " + str(train_nums))
    print("val: " + str(val_nums))


if __name__ == '__main__':
    # 路径下包含两个文件夹 images labels
    path = r"G:\dataset\publicdataset\mapcup\train"
    get_train_val(path)