# -*- coding: utf-8 -*-
import os
import cv2


path = r"G:\dataset\Massachusetts buildings dataset\label"
save_path = r"G:\dataset\Massachusetts buildings dataset\resize\label"

tif_list = [x for x in os.listdir(path) if x.endswith(".tif")]
for num, i in enumerate(tif_list):      # 遍历列表
    img_path = os.path.join(path, i)
    img = cv2.imread(img_path)
    print('正在处理  ------>', i)
    img = cv2.resize(img, dsize=(3000, 3000), interpolation=cv2.INTER_CUBIC)  # img
    # img = cv2.resize(img, dsize=(3000, 3000), interpolation=cv2.INTER_NEAREST) #    label

    cv2.imwrite(os.path.join(save_path, '{}'.format(i.split('.')[0] + ".png")), img)

print("  finished")

# img = cv2.imread(r'G:\dataset\Massachusetts buildings dataset\test\label\22678915_15.tif')
# print(img.shape)    # (1280, 1920, 3)
# # img = cv2.pyrDown(img)
# # img = cv2.pyrDown(img)
# # img = cv2.resize(img, dsize=(3000, 3000), interpolation=cv2.INTER_CUBIC)  # img
# img = cv2.resize(img, dsize=(3000, 3000), interpolation=cv2.INTER_NEAREST) #    label
# # img = cv2.pyrUp(img, dstsize=(3000, 3000))
# # print(img.shape)    # (320, 480, 3)
# # cv2.imshow('win', img)
# # cv2.waitKey(0)
# cv2.imwrite("label.png",img)


# img = cv2.pyrDown(img, dstsize=(960, 640))