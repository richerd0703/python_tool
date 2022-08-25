import os
from PIL import Image, ImageChops, ImageEnhance
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np

path = r'E:\hk\res\drone\mask'
fileset = []
fileLists = os.listdir(path)
# num = len(fileLists)
for file in fileLists:
    if file.split(".")[-1] == ("png"):
        # print(file)
        whole_path = path + "\\" + file
        fileset.append(whole_path)

for i in range(len(fileset)):
    img_data = Image.open(fileset[i], 'r')
    name = fileset[i].split('\\')[-1]
    # print(img_data.size)  # (1000,1000)
    print('----------->', name)

    # plt.subplot(1, 2, 1)
    # plt.title("origin")
    plt.imsave(r'E:\hk\res\drone\mask_vision\{}'.format(name), img_data)
# plt.imshow(img_data)

# img = np.array(img_data)
# img = img.reshape(1000, 1000, 1)
# img_tensor = tf.convert_to_tensor(img)
# img_tensor = tf.image.grayscale_to_rgb(img_tensor)
#
# sess = tf.Session()
# img = sess.run(img_tensor)
# print(img_tensor.shape)  # (625,1000,3)
#
# plt.subplot(1, 2, 2)
# plt.title("rgb")
# plt.imsave('rgb.png', img)
# plt.imshow(img)
# plt.show()


