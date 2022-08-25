import numpy as np
import cv2
import os
from tqdm import tqdm
from osgeo import gdal, ogr
from osgeo.ogr import Geometry

def draw(label_path, point_path, save_path):
    name_list = [x for x in os.listdir(label_path) if x.endswith(".png")]
    for file in tqdm(name_list):
        label_file = cv2.imread(os.path.join(label_path, file))
        point_file = cv2.imread(os.path.join(point_path, file))
        gray = cv2.cvtColor(point_file, cv2.COLOR_BGR2GRAY)
        corners = cv2.goodFeaturesToTrack(gray, 0, 0.04, 17)  # , blockSize=5
        try:
            corners = np.int0(corners)
        except:
            cv2.imwrite(os.path.join(save_path, file), label_file)
            continue
        print(len(corners), file)
        for i in corners:
            x, y = i.ravel()
            cv2.circle(label_file, (x, y), 3, (255, 0, 0), -1)

        # 保存图像
        cv2.imwrite(os.path.join(save_path, file), label_file)


def drawbyshp(label_path, shp_path, save_path):
    name_list = [x for x in os.listdir(shp_path) if x.endswith(".shp")]
    for file in tqdm(name_list):
        name = file.split(".")[0]
        label_file = cv2.imread(os.path.join(label_path, name + ".png"))
        shp = ogr.Open(os.path.join(shp_path, file))
        layer = shp.GetLayer()
        # feature = layer.GetFeature(0)
        feat = layer.GetNextFeature()

        # out_img = np.zeros((320,320))
        # print(name)
        while feat:
            geometry = feat.GetGeometryRef()
            if not isinstance(geometry, Geometry):
                # print("geometry none")
                feat = layer.GetNextFeature()
                continue
            s = str(geometry).split("((")[1][:-2]
            stringPoint = s.split(",")
            for i in stringPoint:
                # print(i)
                p = i.split(" ")
                try:
                    x = int(p[0])
                    y = -int(p[1])
                except:
                    print(name)
                    print("x-->", p[0])
                    print("y-->", p[1])
                cv2.circle(label_file, (x, y), 3, (255, 0, 0), -1)

            cv2.imwrite(os.path.join(save_path, name + ".png"), label_file)
            feat = layer.GetNextFeature()
        layer.ResetReading()


if __name__ == '__main__':
    label_path = r"G:\exp\ISPRS\my\val\labels"
    point_path = r"G:\exp\ISPRS\my\val\shp"
    save_path = r"G:\exp\ISPRS\my\val\show"

    # draw(label_path, point_path, save_path)
    drawbyshp(label_path, point_path, save_path)
