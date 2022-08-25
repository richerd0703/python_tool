# import cv2.cv2 as cv
# import os
# os.environ["OPENCV_IO_MAX_IMAGE_PIXELS"] = pow(2, 40).__str__()
import fitz
import os
os.environ["OPENCV_IO_MAX_IMAGE_PIXELS"] = pow(2,40).__str__()
import cv2
from tqdm import tqdm

def tif2png(path, save_path):
    tif_list = [x for x in os.listdir(path) if x.endswith(".png")]  # 获取目录中所有tif格式图像列表
    for num, i in tqdm(enumerate(tif_list)):  # 遍历列表
        img_path = os.path.join(path, i)
        img = cv2.imread(img_path,0)  # 读取列表中的tif图像
        print('------>', i)
        # cv.imwrite(r'I:\hezhiyu\cut\png\{}'.format(i.split('.')[0]+".png"), img)    # tif 格式转 jpg 并按原名称命名
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        cv2.imwrite(os.path.join(save_path, '{}'.format(i.split('.')[0] + ".png")), img)


# 安装库 pip install pymupdf


def get_file(filepath):
    file_list = []
    files = os.listdir(filepath)
    for file in files:
        if os.path.splitext(file)[1] == ".pdf":
            file_list.append(file)
    return file_list


def conver_img(filepath, savepath):
    pdf_dir = get_file(filepath)
    for pdf in pdf_dir:
        doc = fitz.open(os.path.join(filepath, pdf))
        # pdf_name = os.path.splitext(pdf)[0]
        pdf_name = os.path.splitext(pdf)[0]
        print("====================================")
        print("开始转换%s.PDF文档" % pdf_name)
        print("====================================")
        print("共", doc.pageCount, "页")
        for pg in range(0, doc.pageCount):
            print("\r转换为图片", pg + 1, "/", doc.pageCount, end=";")
            page = doc[pg]
            rotate = int(0)  # 旋转角度
            # 每个尺寸的缩放系数为2，这将为我们生成分辨率提高四倍的图像
            zoom_x = 2.0
            zoom_y = 2.0
            print("")
            trans = fitz.Matrix(zoom_x, zoom_y).preRotate(rotate)
            pm = page.getPixmap(matrix=trans, alpha=False)
            pm.writePNG(os.path.join(savepath, str(pdf_name) + '-' + '{:02}.png'.format(pg)))


if __name__ == '__main__':
    path = r'I:\data\tianchi\out'  # 获取图片所在目录
    save_path = r'I:\data\tianchi\png'
    tif2png(path, save_path)

    # pdf_file = r"G:\dataset\temp\val\image\poly_viz.acm.tol_0.125"
    # conver_img(pdf_file, save_path)

# path = r'I:\data\drone\hk_drone'
#
# jpg_path = r'I:\data\drone\jpg'
#
# for j in range(1, 41):
#     tif_path = path + '\\' + 'group{}'.format(j)
#     imgs = os.listdir(tif_path)
#
#     for i, img in enumerate(imgs):
#
#         img_name = os.path.join(tif_path, img)
#         file = cv2.imread(img_name)
#         save_file = os.path.join(jpg_path, 'group{}_'.format(j) + img.strip('.tiff')+'.jpg')
#         cv2.imwrite(save_file, file)
#         print(j, ':', i)
