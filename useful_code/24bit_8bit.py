#读取整个bacepath文件夹下的文件并且转换为8位保存到savepath
import os
os.environ["OPENCV_IO_MAX_IMAGE_PIXELS"] = pow(2, 40).__str__()
import cv2
from tqdm import tqdm
from PIL import Image
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = 1000000000000000

def trans(bacepath, savepath):
    f_n = [x for x in os.listdir(bacepath) if x.endswith(".png")]
    for n in tqdm(f_n):
        imdir = os.path.join(bacepath, n)
        # print(n)
        size = os.path.getsize(imdir)  # 根据尺寸判断
        # print(size)
        # if size > 2048:

        # image = Image.open(os.path.join(bacepath, n)).save("w.png")
        # img = cv2.imread("w.png")
        #

        img = cv2.imread(imdir)

        cropped = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # cropped[cropped == 15] = 0
        cv2.imwrite(savepath + '\\' + n.split('.')[0] + '.png', cropped)  # NOT CAHNGE THE TYPE

if __name__ == '__main__':
    bacepath = r"G:\exp\crowdAI\my\test\point_shrink"
    savepath = r'G:\exp\crowdAI\my\test\point_shrink'
    trans(bacepath, savepath)