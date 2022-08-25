# -*- coding: utf-8 -*-
import sys
from osgeo import ogr
import csv

# fn = r"G:\tool\2\data\chunan_clip2.shp"
# ds = ogr.Open(fn, 1)
# lyr = shapef.GetLayer()
#     ####
# oFieldID = ogr.FieldDefn("LUType", ogr.OFTInteger)  # 创建一个叫LuType的整型属性
# lyr.CreateField(oFieldID, 1)
# i = 0
# lyr = ds.GetLayer()
# oFieldID = ogr.FieldDefn("new", ogr.OFTInteger)  # 创建一个叫LuType的整型属性
# lyr.CreateField(oFieldID, 1)

# with open(r'H:\tool\2\out2\2.csv','r') as csvfile:
#     reader = csv.reader(csvfile)
#     rows = [row for row in reader]
#     print len(rows[0])
#     print rows[0]
#     column1 = [row[1]for row in reader]
#
# for feat in lyr:
#     # fid_ = feat.GetField("FID_")
#     # fid = feat.GetField(1)
#     # print (fid)
#     i += 1
#     feat.SetField('new', int(i) + 1)
#     lyr.SetFeature(feat)
#
#     print(column1)
# for i in f:
#     print(i)


fn = r"G:\tools\2\data\chunan_clip2.shp"
ds = ogr.Open(fn, 1)

# lyr = shapef.GetLayer()
#     ####
# oFieldID = ogr.FieldDefn("LUType", ogr.OFTInteger)  # 创建一个叫LuType的整型属性
# lyr.CreateField(oFieldID, 1)



with open(r'G:\tools\2\out2\3-1.csv', 'r') as csvfile:
    reader = csv.reader(csvfile)
    rows = [row for row in reader]

leng = len(rows)
temp = []
for j in range(9):
    temp1 = []
    for i in range(1, len(rows)):
        temp1.append(float(rows[i][j]))
    temp.append(temp1)

# with open(r'H:\tool\2\out2\2.csv', 'r') as csvfile1:
#     reader = csv.reader(csvfile1)
#     # column1 = [row[1] for row in reader]
field_list = ['FID_', 'COUNT', 'AREA', 'MIN', 'MAX', 'RANGE', 'MEAN', 'STD', 'SUM']

# field_list = rows[0]
print field_list
print len(temp)
print len(temp[1])
for k in range(len(field_list)):
    print (field_list[k], temp[k])
    ds = ogr.Open(fn, 1)
    lyr = ds.GetLayer()
    oFieldID = ogr.FieldDefn(field_list[k], ogr.OFTReal)  # 创建一个叫LuType的整型属性
    lyr.CreateField(oFieldID, 1)
    n = 0
    print temp[k][n]
    print '=========================='
    print (len(lyr))
    for feat in lyr:

        # fid_ = feat.GetField("FID_")
        # fid = feat.GetField(1)
        # print (fid)

        feat.SetField(field_list[k], temp[k][n])
        lyr.SetFeature(feat)
        n += 1

    print ("sdggsdg", n)
    print k