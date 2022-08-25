# import cv2
# import numpy as np
#
# def transfer_16bit_to_8bit(image_path):
#     image_16bit = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
#     min_16bit = np.min(image_16bit)
#     max_16bit = np.max(image_16bit)
#     # image_8bit = np.array(np.rint((255.0 * (image_16bit - min_16bit)) / float(max_16bit - min_16bit)), dtype=np.uint8)
#     # 或者下面一种写法
#     image_8bit = np.array(np.rint(255 * ((image_16bit - min_16bit) / (max_16bit - min_16bit))), dtype=np.uint8)
#     print(image_16bit.dtype)
#     print('16bit dynamic range: %d - %d' % (min_16bit, max_16bit))
#     print(image_8bit.dtype)
#     print('8bit dynamic range: %d - %d' % (np.min(image_8bit), np.max(image_8bit)))
#
# image_path = r'F:\he\data\test\1.tif'
# transfer_16bit_to_8bit(image_path)

#
# import os
# import gdal
# from cv2 import cv2
# import numpy as np
# import sys
#
#
# # 拉伸图像  #图片的16位转8位
# def uint16to8(bands, lower_percent=0.001, higher_percent=99.999):
#     out = np.zeros_like(bands, dtype=np.uint8)
#     n = bands.shape[0]
#     for i in range(n):
#         a = 0  # np.min(band)
#         b = 255  # np.max(band)
#         c = np.percentile(bands[i, :, :], lower_percent)
#         d = np.percentile(bands[i, :, :], higher_percent)
#         t = a + (bands[i, :, :] - c) * (b - a) / (d - c)
#         t[t < a] = a
#         t[t > b] = b
#         out[i, :, :] = t
#     return out
#
#
# path = r"F:\he\data\samples-高二\gf2\16-8"  # 获取当前代码路径
# tif_list = [x for x in os.listdir(path) if x.endswith(".tif")]
# for num, i in enumerate(tif_list):
#     print(path + '\\' + i)
#     dataset = gdal.Open(path + '\\' + i)
#     width = dataset.RasterXSize  # 获取数据宽度
#     height = dataset.RasterYSize  # 获取数据高度
#     outbandsize = dataset.RasterCount  # 获取数据波段数
#     im_geotrans = dataset.GetGeoTransform()  # 获取仿射矩阵信息
#     im_proj = dataset.GetProjection()  # 获取投影信息
#     datatype = dataset.GetRasterBand(1).DataType
#     im_data = dataset.ReadAsArray()  # 获取数据
#     # print(im_data.shape)
#     img3 = uint16to8(im_data)
#     img2 = img3[0:3, :, :]
#     img2 = np.transpose(img2, (1, 2, 0))
#     out = img2[:, :, ::-1]  # rgb->bgr
#     cv2.imwrite(path + '\\' + i, out)
#     print(num)

import sys,os
import argparse
import glob
import numpy as np
import cv2
from matplotlib import cm

# python 16bit_8bit.py  G:\dataset\pansharpen\rgb\dataset\MS
parser = argparse.ArgumentParser(description='16bit2rgb.py: script for 16 bit TIFF to 24bit RGB PNG conversion')
parser.add_argument('--input_files', default= r"G:\dataset\pansharpen\rgb\dataset\MS", help='Input image Files')
parser.add_argument('--out', '-o', default=r'G:\dataset\pansharpen\rgb\dataset\out', help='Output directory')
parser.add_argument('--min_val', type=int, default=-1, help='Min value')
parser.add_argument('--max_val', type=int, default=-1, help='Max value')
parser.add_argument('--map', '-m', default='CMRmap', help='Colour Map')

args = parser.parse_args()

if hasattr(cm, args.map):
    cmap = getattr(cm, args.map)
else:
    raise NotImplementedError(args.map)

max_val = 0 if args.max_val < 0 else args.max_val
min_val = 255**2 if args.min_val < 0 else args.min_val

if args.max_val < 0 and args.min_val < 0:
    print('finding minmax values')
    for filepath in glob.glob(args.input_files):
        print(' target file: %s' % filepath)
        img = cv2.imread(filepath, cv2.IMREAD_ANYDEPTH)
        ma = img.max()
        mi = img.min()
        if ma>max_val:
            max_val = ma
        if mi<min_val:
            min_val = mi

print('converting files')
for filepath in glob.glob(args.input_files):
    print(' target file: %s' % filepath)
    img = cv2.imread(filepath, cv2.IMREAD_ANYDEPTH)
    img = ((img-min_val).astype('float64')) / (max_val-min_val)
    img2 = (cmap(img) * 255).astype('uint8')
    img2 = cv2.cvtColor(img2, cv2.COLOR_RGB2BGR)
    # filename = os.path.basename(filepath).replace('.tiff','.png')
    filename = os.path.basename(filepath)
    output_filename = os.path.join(args.out, filename)
    print('  saving: %s' % output_filename)
    cv2.imwrite(output_filename, img2)

print('done (max: %d, min: %d)' % (max_val, min_val))