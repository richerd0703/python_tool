# -*- coding: utf-8 -*-
from osgeo import ogr
import arcpy
from arcpy.sa import *
import struct
import datetime
import decimal
import itertools
import os
import shutil
import argparse
import time
import math

# 计算坡度
def run(shp_path, dem_path, save_path, road):
    # Set environment settings
    # env.workspace = "H:/tool/1"

    # Set local variables
    inRaster = dem_path
    outMeasurement = "DEGREE"
    zFactor = 1

    # Check out the ArcGIS Spatial Analyst extension license
    arcpy.CheckOutExtension("Spatial")

    # Execute Slope
    outSlope = Slope(inRaster, outMeasurement, zFactor)

    # 得到坡度图
    pd_path = os.path.join(save_path, "pd.tif")
    outSlope.save(pd_path)

    # 得到dbf
    inZoneData = shp_path
    zoneField = "FID"
    inValueRaster = pd_path
    # 输出dbf
    outTable = os.path.join(save_path, "table.dbf")

    # Check out the ArcGIS Spatial Analyst extension license
    arcpy.CheckOutExtension("Spatial")

    # Execute ZonalStatisticsAsTable
    outZSaT = ZonalStatisticsAsTable(inZoneData, zoneField, inValueRaster,
                                     outTable, "NODATA", "ALL")

    # outZSaT = ZonalStatisticsAsTable("H:/tool/test_data/chunan_clip2.shp", "FID", "H:/tool/out/1.tif",
    #                                  "H:/tool/out/table", "NODATA", "ALL")

    filename = outTable
    f = open(filename, 'rb')
    db = list(dbfreader(f))
    f.close()
    # for record in db:
    #     print record

    temp = []
    for j in range(9):
        temp1 = []
        for i in range(2, len(db)):
            temp1.append(float(db[i][j]))
        temp.append(temp1)

    leng = len(db)
    temp = []
    for j in range(9):
        temp1 = []
        for i in range(2, len(db)):
            temp1.append(float(db[i][j]))
        temp.append(temp1)

    field_list = ['FID_', 'COUNT', 'AREA', 'MIN', 'MAX', 'RANGE', 'MEAN', 'STD', 'SUM']

    # field_list = rows[0]
    # print field_list
    # print len(temp)
    # print len(temp[1])
    fn = shp_path
    for k in range(len(field_list)):
        # print (field_list[k], temp[k])
        ds = ogr.Open(fn, 1)
        lyr = ds.GetLayer()
        oFieldID = ogr.FieldDefn(field_list[k], ogr.OFTReal)  # 创建一个叫LuType的整型属性
        lyr.CreateField(oFieldID, 1)
        n = 0
        # print temp[k][n]
        # print '=========================='
        # print (len(lyr))
        for feat in lyr:
            # fid_ = feat.GetField("FID_")
            # fid = feat.GetField(1)
            # print (fid)

            feat.SetField(field_list[k], temp[k][n])
            lyr.SetFeature(feat)
            n += 1

        # 计算距离
    poly_point = GetGeoObjectsLoc(shp_path)
    road_point = Getline(road)
    dis = distance(poly_point, road_point)
    write2shp(shp_path, dis)
        #
        # print ("sdggsdg", n)
        # print k
    print "code has finished"
def dbfreader(f):
    """Returns an iterator over records in a Xbase DBF file.
    The first row returned contains the field names.
    The second row contains field specs: (type, size, decimal places).
    Subsequent rows contain the data records.
    If a record is marked as deleted, it is skipped.
    File should be opened for binary reads.
    """
    # See DBF format spec at:
    #     http://www.pgts.com.au/download/public/xbase.htm#DBF_STRUCT

    numrec, lenheader = struct.unpack('<xxxxLH22x', f.read(32))
    numfields = (lenheader - 33) // 32

    fields = []
    for fieldno in xrange(numfields):
        name, typ, size, deci = struct.unpack('<11sc4xBB14x', f.read(32))
        name = name.replace('\0', '')  # eliminate NULs from string
        fields.append((name, typ, size, deci))
    yield [field[0] for field in fields]
    yield [tuple(field[1:]) for field in fields]

    terminator = f.read(1)
    assert terminator == '\r'

    fields.insert(0, ('DeletionFlag', 'C', 1, 0))
    fmt = ''.join(['%ds' % fieldinfo[2] for fieldinfo in fields])
    fmtsiz = struct.calcsize(fmt)
    for i in xrange(numrec):
        record = struct.unpack(fmt, f.read(fmtsiz))
        if record[0] != ' ':
            continue  # deleted record
        result = []
        for (name, typ, size, deci), value in itertools.izip(fields, record):
            if name == 'DeletionFlag':
                continue
            if typ == "N":
                value = value.replace('\0', '').lstrip()
                if value == '':
                    value = 0
                elif deci:
                    value = decimal.Decimal(value)
                else:
                    value = int(value)
            elif typ == 'D':
                y, m, d = int(value[:4]), int(value[4:6]), int(value[6:8])
                value = datetime.date(y, m, d)
            elif typ == 'L':
                value = (value in 'YyTt' and 'T') or (value in 'NnFf' and 'F') or '?'
            elif typ == 'F':
                value = float(value)
            result.append(value)
        yield result


def GetGeoObjectsLoc(path):
    objectLocs = []
    shapef = ogr.Open(path, 0) # 0 is read-only 1 is read-write
    lyr = shapef.GetLayer()
    for feature in lyr:
        geom = feature.GetGeometryRef()
        objectLocs.append(geom.Centroid().GetPoint()[0:2])
    return objectLocs

# 获取ploygen类型
def GetGeoObjectsType(path, field):
    objectLocs = []
    shapef = ogr.Open(path, 0) # 0 is read-only 1 is read-write
    lyr = shapef.GetLayer()
    for feature in lyr:
        fieldtype = feature.GetField(field)
        objectLocs.append(fieldtype)

    return objectLocs


# 根据经纬度计算距离（公里）
def haversine_dis(lon1, lat1, lon2, lat2):
    #将十进制转为弧度
    lon1, lat1, lon2, lat2 = map(math.radians, [lon1, lat1, lon2, lat2])
    #haversine公式
    d_lon = lon2 - lon1
    d_lat = lat2 - lat1
    aa = math.sin(d_lat/2)**2 + math.cos(lat1)*math.cos(lat2)*math.sin(d_lon/2)**2
    c = 2 * math.asin(math.sqrt(aa))
    r = 6371 # 地球半径，千米
    return c*r * 1000

def get_point_line_distance(self, point, line):
    point_x = point[0]
    point_y = point[1]
    line_s_x = line[0][0]
    line_s_y = line[0][1]
    line_e_x = line[1][0]
    line_e_y = line[1][1]
    # 若直线与y轴平行，则距离为点的x坐标与直线上任意一点的x坐标差值的绝对值
    if line_e_x - line_s_x == 0:
        return math.fabs(point_x - line_s_x)
    # 若直线与x轴平行，则距离为点的y坐标与直线上任意一点的y坐标差值的绝对值
    if line_e_y - line_s_y == 0:
        return math.fabs(point_y - line_s_y)
    # 斜率
    k = (line_e_y - line_s_y) / (line_e_x - line_s_x)
    # 截距
    b = line_s_y - k * line_s_x
    # 带入公式得到距离dis
    dis = math.fabs(k * point_x - point_y + b) / math.pow(k * k + 1, 0.5)
    return dis

def Getline(path):
    objectLocs = []
    shapef = ogr.Open(path, 0) # 0 is read-only 1 is read-write
    lyr = shapef.GetLayer()
    for feature in lyr:
        geom = feature.GetGeometryRef()
        print (geom.GetPoints())
        objectLocs.append(geom.GetPoints())

    return objectLocs


def distance(poly_point, road_point):
    min_dis = []
    # 面的点
    for i in range(len(poly_point)):
        # 线的点
        one_point_dis = []
        for j in range(len(road_point)):
            one_line_dis = []
            for k in range(len(road_point[j])):
                temp_dis = haversine_dis(poly_point[i][0], poly_point[i][1], road_point[j][k][0], road_point[j][k][1])
                # '%.3f' % x
                one_line_dis.append('%.3f' % temp_dis)
            one_min_dis = min(one_line_dis)
            if one_min_dis > 20:
                one_point_dis.append(one_min_dis)
            else:
                one_point_dis.append(0)
        min_dis.append(min(one_point_dis))
    return min_dis

def write2shp(shp_path , dis):
    fn = shp_path
    ds = ogr.Open(fn, 1)
    lyr = ds.GetLayer()
    oFieldID = ogr.FieldDefn("Dis", ogr.OFTReal)  # 创建一个叫LuType的整型属性
    lyr.CreateField(oFieldID, 1)
    n = 0
    for feat in lyr:
        feat.SetField("Dis", dis[n])
        lyr.SetFeature(feat)
        n += 1

def dbfwriter(f, fieldnames, fieldspecs, records):
    """ Return a string suitable for writing directly to a binary dbf file.
    File f should be open for writing in a binary mode.
    Fieldnames should be no longer than ten characters and not include \x00.
    Fieldspecs are in the form (type, size, deci) where
        type is one of:
            C for ascii character data
            M for ascii character memo data (real memo fields not supported)
            D for datetime objects
            N for ints or decimal objects
            L for logical values 'T', 'F', or '?'
        size is the field width
        deci is the number of decimal places in the provided decimal object
    Records can be an iterable over the records (sequences of field values).
    """
    # header info
    ver = 3
    now = datetime.datetime.now()
    yr, mon, day = now.year - 1900, now.month, now.day
    numrec = len(records)
    numfields = len(fieldspecs)
    lenheader = numfields * 32 + 33
    lenrecord = sum(field[1] for field in fieldspecs) + 1
    hdr = struct.pack('<BBBBLHH20x', ver, yr, mon, day, numrec, lenheader, lenrecord)
    f.write(hdr)

    # field specs
    for name, (typ, size, deci) in itertools.izip(fieldnames, fieldspecs):
        name = name.ljust(11, '\x00')
        fld = struct.pack('<11sc4xBB14x', name, typ, size, deci)
        f.write(fld)

    # terminator
    f.write('\r')

    # records
    for record in records:
        f.write(' ')  # deletion flag
        for (typ, size, deci), value in itertools.izip(fieldspecs, record):
            if typ == "N":
                value = str(value).rjust(size, ' ')
            elif typ == 'D':
                value = value.strftime('%Y%m%d')
            elif typ == 'L':
                value = str(value)[0].upper()
            else:
                value = str(value)[:size].ljust(size, ' ')
            assert len(value) == size
            f.write(value)

    # End of file
    f.write('\x1A')

def copyShp(fileFullPath, targetDic):
    assFile = ['dbf', 'shp', 'shx', 'sbn', 'sbx', 'cpg', 'xml', 'prj', 'mxs', 'ixs', 'atx', 'ain', 'fbn', 'sbn']
    Path, name = os.path.split(fileFullPath)
    # print(Path)
    # print(name)
    nameSplit = name.split(".")
    for ex in assFile:
        src = os.path.join(Path, "{0}.{1}".format(nameSplit[0], ex))
        if os.path.exists(src):
            # print(src)
            targFile = os.path.join(targetDic, "{0}.{1}".format(nameSplit[0], ex))
            print(targFile)
            if os.path.exists(targFile):
                os.remove(targFile)
            shutil.copy(src, targFile)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='this is a description')
    parser.add_argument('--shp', type=str)
    parser.add_argument('--dem', type=str)
    parser.add_argument('--road', type=str)
    parser.add_argument('--out', type=str)
    args = parser.parse_args()

    copyShp(args.shp, args.out)
    file_name = os.path.basename(args.shp)
    new_shp_path = os.path.join(args.out, file_name)
    # pwd = os.getcwd()
    pwd = os.path.dirname(__file__)
    now_time = time.strftime("%H%M%S_%m%d_%Y", time.localtime())
    temp_save_path = os.path.join(args.out, now_time)
    os.mkdir(temp_save_path)
    try:
        run(new_shp_path, args.dem, temp_save_path, args.road)
    finally:
        shutil.rmtree(temp_save_path)
# --shp H:\tool\test\data\chunan_clip2.shp --dem H:\tool\test\data\3-1.tif --road H:\tool\out\road_hz.shp --out H:\tool\test\out