from PIL import Image
import numpy as np
import cv2
import os


path = r'H:\海康_全要素\drone\image'
fileset = []
fileLists = os.listdir(path)
# num = len(fileLists)
for file in fileLists:
    if file.split(".")[-1] == ("png"):
        # print(file)
        whole_path = path + "\\" + file
        fileset.append(whole_path)
# color = [(255, 255, 255), (255, 0, 0), (255, 255, 0), (0, 255, 0), (0, 255, 255),
#          (255, 0, 255), (0, 0, 255), (120, 123, 234), (255, 34, 100)]
l = len(fileset)
for j in range(len(fileset)):

    image = cv2.imdecode(np.fromfile(fileset[j], dtype=np.uint8), 0)

    # image = cv2.imread(fileset[j], 0)
    # print('.......', image)

    img = np.zeros(image.shape)
    # img = np.zeros(1000)
    # for i in range(1, 9):
    temp_img = image.copy()
    # temp_img = image / 255
    temp_img[temp_img != 0] = 255
    # temp_img[temp_img == 1] = 255

    contours, hierarchy = cv2.findContours(temp_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # contours = cv2.findContours(temp_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(img, contours, -1, (255, 0, 0), 1)
    # cv2.drawContours(img, contours, -1, (1, 1, 1), -1)
    # 255,255,255 全白
    name = fileset[j].split('\\')[-1]
    print("running -------> ", name, ':', j)

    cv2.imwrite(r"H:\海康_全要素\drone\edge\{}".format(name), img)
    # cv2.imencode('.png', img)[1].tofile(r"G:\hk\res\2_1m_hz_gf2\poly\{}".format(name))


def find_outline():
    import os
    import glob
    import cv2
    import numpy as np
    from tqdm import tqdm
    paths = "./mask"
    save_path = "./label"
    images = os.listdir(paths)

    print(images)

    for image in tqdm(images):
        path = os.path.join(paths, image)
        # if path.split('.')[2] == 'tif':
        #     print(path[:-4]+'.png')
        #     os.rename(path, path[:-4]+'.png')
        data = cv2.imread(path, 0)
        # path = path[:-5]+".png"
        # cv2.imwrite(path, data)
        # print(data.shape)
        output_data = np.zeros(data.shape, dtype=np.uint8)
        for i in range(1, 8, 1):
            mask = np.zeros(data.shape, dtype=np.uint8)
            mask[data == i] = 1
            contours, hir = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(output_data, contours, -1, 255, 1)

        cv2.imwrite(os.path.join(save_path, image), output_data)

# import cv2
# import numpy as np
#
#
# def FillHole(mask):
#     img, contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
#     len_contour = len(contours)
#     contour_list = []
#     for i in range(len_contour):
#         drawing = np.zeros_like(mask, np.uint8)  # create a black image
#         img_contour = cv2.drawContours(drawing, contours, i, (255, 255, 255), -1)
#         contour_list.append(img_contour)
#
#     out = sum(contour_list)
#     return out
#
#
# if __name__ == '__main__':
#     mask_in = cv2.imread(r'E:\bianhuajiance\test\val\new_add\bold\E120D6_N31D1_20161215_ZY3_DO1_clip2_class_1_val_val_bold_bold.tif', 0)
#     mask_out = FillHole(mask_in)
#     cv2.imwrite(r'E:\bianhuajiance\test\val\new_add\bold\E120D6_N31D1_20161215_ZY3_DO1_clip2_class_1_val_val_bold_bold_out.tif', mask_out)