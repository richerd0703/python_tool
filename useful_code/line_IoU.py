import cv2
import numpy as np
from tqdm import tqdm
from skimage import morphology, draw
import os


# 这是计算线IOU，所用方法：将结果图和label图中的线膨胀N个像素值，再做IOU计算

# #dinknet
# txt_path = r"D:\binary\line_txt_path"


def line_iou(result_path, label_edge_path, iou_txt_path):
    t = [1, 3, 5, 7, 9]
    for k in t:
        sum_IoU = 0
        iou_txt = open(iou_txt_path, "w")
        iou_txt.write(result_path + "\n")
        files = os.listdir(result_path)
        p = 512 * 512
        s = 0
        for file in tqdm(files):
            if file[-4:] == '.png':
                result_name = os.path.join(result_path, file)
                label_name = os.path.join(label_edge_path, file)
                # label_name = label_name.replace(".tif", ".png")
                result_img = cv2.imread(result_name, 0)
                label_img = cv2.imread(label_name, 0)

                kernel0 = np.ones((k, k), np.uint8)
                # kernel1 = np.ones((1, 1), np.uint8)
                # kernel2 = np.ones((2, 2), np.uint8)
                # kernel3 = np.ones((3, 3), np.uint8)
                # kernel5 = np.ones((5, 5), np.uint8)
                # kernel7 = np.ones((7, 7), np.uint8)
                # kernel9 = np.ones((9, 9), np.uint8)

                # result_dilation = cv2.dilate(result_img, kernel3)
                # label_dilation = cv2.dilate(label_img, kernel3)

                result_dilation = cv2.dilate(result_img, kernel0)
                label_dilation = cv2.dilate(label_img, kernel0)

                overLap = np.where((result_dilation[:, :] == 255) & (label_dilation[:, :] == 255))[0].size
                u1 = np.where((result_dilation[:, :] == 0) & (label_dilation[:, :] == 255))[0].size
                u2 = np.where((result_dilation[:, :] == 255) & (label_dilation[:, :] == 0))[0].size

                # overLap = np.where((mask[min(y1, y2):max(y1, y2) + 1, min(x1, x2):max(x1, x2) + 1] == True) & \
                #                    (my_label[min(y1, y2):max(y1, y2) + 1, min(x1, x2):max(x1, x2) + 1, 0] > 0))[0].size
                # u1 = np.where((mask[min(y1, y2):max(y1, y2) + 1, min(x1, x2):max(x1, x2) + 1] == 0) & \
                #               (my_label[min(y1, y2):max(y1, y2) + 1, min(x1, x2):max(x1, x2) + 1, 0] > 0))[0].size
                # u2 = np.where((mask[min(y1, y2):max(y1, y2) + 1, min(x1, x2):max(x1, x2) + 1] == True) & \
                #               (my_label[min(y1, y2):max(y1, y2) + 1, min(x1, x2):max(x1, x2) + 1, 0] == 0))[0].size
                sum = overLap + u1 + u2
                eiou = 0
                # if sum != 0:
                #     IoU = overLap / sum
                # else:
                #     IoU = 0
                # # print("IoU:" +str(IoU))
                if sum != 0:
                    IoU = overLap / sum

                    # eiou = overLap / (u1 + u2)
                    # temp3 = overLap / p
                    # temp1 = u1 / p
                    # temp2 = u2 / p
                elif sum == 0 and overLap == 0:
                    IoU = 1.0
                else:
                    IoU = 0

                sum_IoU += IoU
                # s += eiou
                iou_txt.write(file + " :" + str(round(IoU, 6)) + "\n")
                # iou_txt.write(file + " :" + str(round(IoU, 6)) + " " + str(round(temp3, 6)) + " " +
                #               str(round(temp1, 6)) + " " + str(round(temp2, 6)) + "\n")
        iou_txt.write("Ave_IoU :" + str(sum_IoU / len(files)) + "\n")
        # iou_txt.write("Ave_IoU :" + str(s / len(files)) + "\n")
        print("Ave_IoU :" + str(sum_IoU / len(files)) + "\n")


def line_iou_diff(result_path, label_edge_path, iou_txt_path):
    t = [1, 3, 5, 7, 9]
    for n in t:
        sum_IoU = 0
        iou_txt = open(iou_txt_path, "w")
        iou_txt.write(result_path + "\n")
        files = os.listdir(result_path)
        for file in tqdm(files):
            if file[-4:] == '.png':
                result_name = os.path.join(result_path, file)
                label_name = os.path.join(label_edge_path, file)
                # label_name = label_name.replace(".tif", ".png")
                result_img = cv2.imread(result_name, 0)
                label_img = cv2.imread(label_name, 0)

                kernel = np.ones((n, n), np.uint8)

                # kernel5 = np.ones((7, 7), np.uint8)

                # result_dilation = cv2.dilate(result_img, kernel3)
                # label_dilation = cv2.dilate(label_img, kernel3)

                result_dilation = cv2.dilate(result_img, kernel)
                label_dilation = cv2.dilate(label_img, kernel)

                path1 = result_path + "-" + str(n)
                path2 = label_edge_path + "-" + str(n)
                if not os.path.exists(path1) or not os.path.exists(path2):
                    os.makedirs(path1)
                    os.makedirs(path2)

                cv2.imwrite(os.path.join(path1, file), result_dilation)
                cv2.imwrite(os.path.join(path2, file), label_dilation)

                overLap = np.where((result_dilation[:, :] == 255) & (label_dilation[:, :] == 255))[0].size
                u1 = np.where((result_dilation[:, :] == 0) & (label_dilation[:, :] == 255))[0].size
                u2 = np.where((result_dilation[:, :] == 255) & (label_dilation[:, :] == 0))[0].size

                # overLap = np.where((mask[min(y1, y2):max(y1, y2) + 1, min(x1, x2):max(x1, x2) + 1] == True) & \
                #                    (my_label[min(y1, y2):max(y1, y2) + 1, min(x1, x2):max(x1, x2) + 1, 0] > 0))[0].size
                # u1 = np.where((mask[min(y1, y2):max(y1, y2) + 1, min(x1, x2):max(x1, x2) + 1] == 0) & \
                #               (my_label[min(y1, y2):max(y1, y2) + 1, min(x1, x2):max(x1, x2) + 1, 0] > 0))[0].size
                # u2 = np.where((mask[min(y1, y2):max(y1, y2) + 1, min(x1, x2):max(x1, x2) + 1] == True) & \
                #               (my_label[min(y1, y2):max(y1, y2) + 1, min(x1, x2):max(x1, x2) + 1, 0] == 0))[0].size
                sum = overLap + u1 + u2
                eiou = 0
                # if sum != 0:
                #     IoU = overLap / sum
                # else:
                #     IoU = 0
                # # print("IoU:" +str(IoU))
                if sum != 0:
                    IoU = overLap / sum

                    # eiou = overLap / (u1 + u2)
                    # temp3 = overLap / p
                    # temp1 = u1 / p
                    # temp2 = u2 / p
                elif sum == 0 and overLap == 0:
                    IoU = 1.0
                else:
                    IoU = 0

                sum_IoU += IoU
                # s += eiou
                iou_txt.write(file + " :" + str(round(IoU, 6)) + "\n")
                # iou_txt.write(file + " :" + str(round(IoU, 6)) + " " + str(round(temp3, 6)) + " " +
                #               str(round(temp1, 6)) + " " + str(round(temp2, 6)) + "\n")
        iou_txt.write("Ave_IoU :" + str(sum_IoU / len(files)) + "\n")
        # iou_txt.write("Ave_IoU :" + str(s / len(files)) + "\n")
        print("Ave_IoU :" + str(sum_IoU / len(files) / 4) + "\n")


if __name__ == '__main__':
    label_edge_path = r"G:\dataset\gm_data_voc\val\0"
    pre_edge_path = r"G:\dataset\gm_data_voc\deeplabV3+\0"
    iou_txt_path = r"G:\dataset\gm_data_voc\deeplabV3+\line_iou.txt"

    line_iou(pre_edge_path, label_edge_path, iou_txt_path)
    # line_iou_diff(pre_edge_path, label_edge_path, iou_txt_path)

# img=cv2.imread(r"F:\Building_sample\test\0000000001_0.tif",0)
# kernel=np.ones((3, 3), np.uint8)
# dilation=cv2.dilate(img,kernel)
# cv2.imwrite(r"F:\Building_sample\test\diltaion1.png",dilation)
