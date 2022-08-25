from osgeo import gdal, ogr, osr
import os
import datetime
import numpy as np

path = "10_err.tif"


if __name__ == '__main__':
    start_time = datetime.datetime.now()

    inraster = gdal.Open(path)  # 读取路径中的栅格数据
    inband = inraster.GetRasterBand(1)  # 这个波段就是最后想要转为矢量的波段，如果是单波段数据的话那就都是1
    prj = osr.SpatialReference()
    prj.ImportFromWkt(inraster.GetProjection())  # 读取栅格数据的投影信息，用来为后面生成的矢量做准备

    outshp = path[:-4] + ".shp"  # 给后面生成的矢量准备一个输出文件名，这里就是把原栅格的文件名后缀名改成shp了
    drv = ogr.GetDriverByName("ESRI Shapefile")
    if os.path.exists(outshp):  # 若文件已经存在，则删除它继续重新做一遍
        drv.DeleteDataSource(outshp)
    Polygon = drv.CreateDataSource(outshp)  # 创建一个目标文件
    Poly_layer = Polygon.CreateLayer(path[:-4], srs=prj, geom_type=ogr.wkbMultiPolygon)  # 对shp文件创建一个图层，定义为多个面类
    newField = ogr.FieldDefn('value', ogr.OFTReal)  # 给目标shp文件添加一个字段，用来存储原始栅格的pixel value,浮点型，
    Poly_layer.CreateField(newField)

    gdal.Polygonize(inband, None, Poly_layer, 0)  # 核心函数，执行的就是栅格转矢量操作
    # gdal.FPolygonize(inband, None, Poly_layer, 0)  # 只转矩形，不合并
    Polygon.SyncToDisk()
    Polygon = None
    end_time = datetime.datetime.now()
    print("Succeeded at", end_time)
    print("Elapsed Time:", end_time - start_time)  # 输出程序运行所需时间

