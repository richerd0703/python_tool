import types

from osgeo import gdal, ogr
from osgeo.ogr import Geometry
from tqdm import tqdm
import os
import cv2
import numpy as np


def extract_shp_point(src_path, save_path):
    name_list = [x for x in os.listdir(src_path) if x.endswith(".shp")]
    for file in tqdm(name_list):
        name = file.split(".")[0]
        shp = ogr.Open(os.path.join(src_path, file))
        layer = shp.GetLayer()
        # feature = layer.GetFeature(0)
        feat = layer.GetNextFeature()
        out_img = np.zeros((512, 512))
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
                # temp = []
                # temp.append(p[0])
                # temp.append(p[1])
                try:
                    x = int(p[0])
                    y = -int(p[1])
                except:
                    print(name)
                    print("x-->", p[0])
                    print("y-->", p[1])
                # print("point==", x, y)
                # cv2.rectangle(out_img, (x - 1, y - 1), (x + 2, y + 2), [255, 255, 255], -1)
                cv2.rectangle(out_img, (x - 1, y - 1), (x + 1, y + 1), [255, 255, 255], -1)
                # point.append(temp)
            # print("point  ==", point)
            cv2.imwrite(os.path.join(save_path, name + ".png"), out_img)
            feat = layer.GetNextFeature()
        layer.ResetReading()


def print_poly(src_path):
    name_list = [x for x in os.listdir(src_path) if x.endswith(".shp")]
    for file in tqdm(name_list):
        name = file.split(".")[0]
        shp = ogr.Open(os.path.join(src_path, file))
        layer = shp.GetLayer()
        feat = layer.GetNextFeature()
        print(name)
        while feat:
            geometry = feat.GetGeometryRef()
            # print(type(geometry))
            # print(geometry)
            if not isinstance(geometry, Geometry):
                print("geometry none")
                feat = layer.GetNextFeature()
                continue
            print("-----")
            feat = layer.GetNextFeature()
        layer.ResetReading()


if __name__ == '__main__':
    src_path = r"G:\temp\t1\shp"
    # src_path = r"G:\temp\pointExtract\temp_shp"
    save_path = r"G:\temp\t1\point"
    extract_shp_point(src_path, save_path)
    # print_poly(src_path)