import numpy as np
import cv2
import os
from tqdm import tqdm


def point2rectangle(src_path, save_path):
    name_list = [x for x in os.listdir(src_path) if x.endswith(".png")]
    for file in tqdm(name_list):
        img = cv2.imread(os.path.join(src_path, file))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        corners = cv2.goodFeaturesToTrack(gray, 0, 0.04, 17)  # , blockSize=5
        # 返回的结果是[[ 311., 250.]] 两层括号的数组。
        try:
            corners = np.int0(corners)
        except:
            cv2.imwrite(os.path.join(save_path, file), img)
            continue
        # point = corners.tolist()
        # lists = [x[0] for x in point]  # 一行代码搞定！
        # print(len(lists))
        # l = []
        # for x in lists:
        #     l.append(x)
        #     l.append([x[0] + 1, x[1]])
        #     l.append([x[0] - 1, x[1]])
        #     l.append([x[0], x[1] + 1])
        #     l.append([x[0], x[1] - 1])
        # for i, p in enumerate(l):
        #     print(p, end="")
        #     if (i+1) % 5 == 0:
        #         print()
        out_img = np.zeros(img.shape)

        for i in corners:
            x, y = i.ravel()
            # cv2.circle(out_img, (x, y), 2, [255, 255, 255], -1)
            cv2.rectangle(out_img, (x - 1, y - 1), (x + 3, y + 3), [255, 255, 255], -1)

        # out_img = cv2.cvtColor(out_img, cv2.COLOR_BGR2GRAY)
        cv2.imwrite(os.path.join(save_path, file), out_img)



def detect_point(src_path, save_path):
    name_list = [x for x in os.listdir(src_path) if x.endswith(".png")]
    for file in tqdm(name_list):
        img = cv2.imread(os.path.join(src_path, file))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # gray = np.float32(gray)
        dst = cv2.cornerHarris(gray, 4, 5, 0.04)  # 执行焦点检测
        # print('dst.shape:', dst.shape)
        # print('dst', dst)
        # img[dst > 0.01 * dst.max()] = [0, 0, 255]
        out_img = np.zeros(img.shape)
        out_img[dst > 0.005 * dst.max()] = [255, 255, 255]
        cv2.imwrite(os.path.join(save_path, file), out_img)


def detect2(filepath, save_path):
    img = cv2.imread(filepath)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = np.float32(gray)
    # 角点检测
    dst = cv2.cornerHarris(gray, 2, 3, 0.04)
    dst = cv2.dilate(dst, None)
    img[dst > 0.05 * dst.max()] = [0, 0, 255]
    # cv2.imshow('dst', img)
    cv2.imwrite(r"G:\dataset\temp\label\outline\out.png", img)
    # if cv2.waitKey(0) & 0xff == 27:
    #     cv2.destroyAllWindows()


if __name__ == '__main__':
    src_path = r"G:\temp\img"
    save_path = r"G:\temp\2"

    if not os.path.exists(save_path): os.makedirs(save_path)
    detect_point(src_path, save_path)
    point2rectangle(save_path, save_path)
