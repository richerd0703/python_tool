# from osgeo import ogr
# from osgeo import gdal
# from osgeo import osr
# import pandas as pd
#
# shapefile = r"E:\PythonProject\shp2wkt\work\test.shp"
#
# driver = ogr.GetDriverByName("ESRI Shapefile")
# dataSource = driver.Open(shapefile, 0)
# layer = dataSource.GetLayer()
#
# layerDefinition = layer.GetLayerDefn()
# # 获取字段名称列表
# fields = []
# for i in range(layerDefinition.GetFieldCount()):
#     fields.append(layerDefinition.GetFieldDefn(i).GetName())
# print(fields)
# # 获取字段值和相应的几何
# file = list()
# GeoWKT = list()
# i = 0
# spatialRef = layer.GetSpatialRef()
# df = pd.DataFrame()
# for feature in layer:
#     list1 = list()
#     j = 0
#     for field in fields:
#         name = feature.GetField(field)
#         if name == None:
#             name = 0
#         list1.insert(j, name)
#         j += 1
#     file.append(list1)
#     geom = feature.GetGeometryRef()
#     target = osr.SpatialReference()
#     target.ImportFromEPSG(4326)
#     transform = osr.CoordinateTransformation(spatialRef, target)
#     geomwkt = geom.ExportToWkt()
#     Pology = ogr.CreateGeometryFromWkt(geomwkt)
#     Pology.Transform(transform)
#     GeoWKT.insert(i, Pology.ExportToWkt())
#     i += 1
# df = pd.DataFrame(file)
# print(df)
# df['WKT'] = GeoWKT
#
# df.to_csv('./test.csv', header=None, sep='\t')

from osgeo import ogr
import pandas as pd
import os
from tqdm import tqdm



def shp2wkt1(shapefile):
    savepath = os.path.dirname(shapefile)
    filename = os.path.basename(shapefile).split(".")[0] + ".csv"

    # table = {
    #     101: "水田",
    #     102: "旱地",
    #     301: "林地",
    #     702: "住宅",
    #     701: "住宅",
    #     601: "工厂",
    #     501: "住宅"
    # }
    # tablelist = [101, 102, 301, 501, 601, 701, 702]
    driver = ogr.GetDriverByName("ESRI Shapefile")
    dataSource = driver.Open(shapefile, 0)
    layer = dataSource.GetLayer()

    layerDefinition = layer.GetLayerDefn()
    # 获取字段名称列表
    fields = []
    for i in range(layerDefinition.GetFieldCount()):
        fields.append(layerDefinition.GetFieldDefn(i).GetName())
    print(fields)
    # 获取字段值和相应的几何
    file = list()
    GeoWKT = list()
    i = 0

    df = pd.DataFrame()
    for feature in tqdm(layer):
        list1 = list()
        j = 0
        for field in fields:
            # print(field)
            name = feature.GetField(field)
            # print(field, ": ---->", type(name))
            if name == None:
                print(field, j, i)
                name = 0
            # if name == "1104A":
            #     name = 1104
            # if name == "05H1":
            #     name = 501
            # if name == "08H1":
            #     name = 801
            # if name == "08H2":
            #     name = 802
            # list1.insert(j, int(name))
            list1.insert(j, name)
            j += 1
        file.append(list1)
        geom = feature.GetGeometryRef()
        geomwkt = geom.ExportToWkt()
        GeoWKT.insert(i, geomwkt)
        i += 1
    df = pd.DataFrame(file)
    # print(df)
    df['WKT'] = GeoWKT

    # df.to_csv('./test.csv', header=None, sep='\t')
    df.to_csv(os.path.join(savepath, filename), header=None, sep='\t')


def shp2wkt2(shapefile):
    savepath = os.path.dirname(shapefile)
    filename = os.path.basename(shapefile).split(".")[0] + ".csv"

    # table = {
    #     11: "水田",
    #     13: "旱地",
    #     21: "果园",
    #     22: "茶园",
    #     23: "其他园地",
    #     31: "有林地",
    #     32: "疏林地",
    #     33: "其他林地",
    #     43: "其他草地",
    #     101: "铁路用地",
    #     102: "公路用地",
    #     104: "农村道路",
    #     106: "港口码头用地",
    #     111: "河流水面",
    #     113: "水库水面",
    #     114: "坑塘水面",
    #     116: "内陆滩涂",
    #     117: "沟渠",
    #     118: "水工建筑用地",
    #     122: "设施农用地",
    #     127: "裸地",
    #     201: "城市",
    #     202: "建制镇",
    #     203: "村庄",
    #     204: "采矿用地",
    #     205: "风景名胜及特殊用地",
    # }

    table = {
        "0101": "水田",
        "0102": "水浇地",
        "0103": "旱地",
        "0201": "果园",
        "0202": "茶园",
        "0204": "其他园地",
        "0301": "其他园地",
        "0305": "灌木林地",
        "0307": "其他林地",
        "0402": "沼泽草地",
        "0404": "其他草地",
        "05H1": "商服用地",
        "0601": "工业仓储用地",
        "0602": "采矿用地",
        "0603": "盐田",
        "06H1": "工业仓储用地",
        "0701": "城镇住宅用地",
        "0702": "农村宅基地",
        "0809": "公共设施用地",
        "0810": "公园与绿地",
        "08H1": "机关用地",
        "08H2": "科教文卫用地",
        "09": "特殊用地",
        "1001": "铁路用地",
        "1002": "轨道交通用地",
        "1003": "公路用地",
        "1004": "城镇村道路用地",
        "1005": "交通服务场站用地",
        "1006": "农村道路",
        "1008": "港口码头用地",
        "1101": "河流水面",
        "1102": "湖泊水面",
        "1104": "坑塘水面",
        "1104A": "养殖坑塘",
        "1108": "沼泽地",
        "1201": "空闲地",
        "1202": "设施农用地",
        "1205": "沙地",
        "1207": "裸岩石砾地",
        "201": "城市",
        "203": "村庄",
    }

    # table = {
    #     1: "建筑物",
    #     2: "林地",
    #     3: "汽车",
    #     4: "低矮植被",
    #     5: "背景",
    #     6: "不透水面",
    # }

    # table = {
    #     0: "水体",
    #     2: "建成区",
    #     1: "绿地",
    #     3: "建成区"
    # }

    # table = {
    #     "bus_dis": "商业区",
    #     "cul_dis": "文教区",
    #     "oth_dis": "其他区域",
    #     "ind_dis": "工业区",
    #     "rel_dis": "休闲区",
    #     "res_dis": "住宅区"
    # }

    # table = {
    #     1: "耕地",
    #     2: "林地",
    #     3: "草地",
    #     4: "水体",
    #     5: "建筑用地",
    #     6: "其他",
    #
    # }
    tablelist = [101, 201, 301, 601, 701, 702, 1101]
    field_table = ["LC_1", "LC_2", "LC_ID", "LU_1_1", "LU_1", "LU_2", "LU_3"]
    driver = ogr.GetDriverByName("ESRI Shapefile")
    dataSource = driver.Open(shapefile, 0)
    layer = dataSource.GetLayer()

    layerDefinition = layer.GetLayerDefn()
    # 获取字段名称列表
    fields = []
    for i in range(layerDefinition.GetFieldCount()):
        fields.append(layerDefinition.GetFieldDefn(i).GetName())
    print(fields)
    # 获取字段值和相应的几何
    file = list()
    GeoWKT = list()
    i = 0

    df = pd.DataFrame()
    for feature in tqdm(layer):
        list1 = list()
        j = 0
        for field in fields:
            if field == "LU_1":

            # if field == "city_type":
                name = feature.GetField(field)
                # print(type(name))
                if name == None:
                    print(field, j, i)
                    name = 11
                # if name == "05H1":
                #     name = 501
                # if name == "08H2":
                #     name = 802
                # if name == "1104A":
                #     name = 1104
                # name = int(name)
                # if name in tablelist:
                #     name = table[name]
                # else:
                #     name = "其他"
                # print(name)
                # print(name)
                name = table[name]
                list1.insert(j, name)
                j += 1
            else:
                # continue
                name = feature.GetField(field)
                if name == None:
                    print(field, j, i)
                    name = 0
                list1.insert(j, name)
                j += 1
        file.append(list1)
        geom = feature.GetGeometryRef()
        geomwkt = geom.ExportToWkt()
        GeoWKT.insert(i, geomwkt)
        i += 1
    df = pd.DataFrame(file)
    # print(df)
    df['WKT'] = GeoWKT

    # df.to_csv('./test.csv', header=None, sep='\t')
    df.to_csv(os.path.join(savepath, filename), header=None, sep='\t')


if __name__ == '__main__':
    path = r"G:\temp\data\out\test2_sample_classify.shp"
    shp2wkt1(path)
    # shp2wkt2(path)