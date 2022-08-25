# encoding: utf-8
import ogr
import math
# 获取polygen位置
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
            if one_min_dis > 50.0:
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

if __name__ == '__main__':
    path = r"H:\tool\test\data\chunan_clip2.shp"
    res = GetGeoObjectsLoc(path)
    # print len(res)
    # print res

    path2 = r"H:\tool\out\road_hz.shp"
    res2 = Getline(path2)
    # print len(res2)
    # print res2
    dist = distance(res, res2)
    # print dist
    # write2shp(path, dist)