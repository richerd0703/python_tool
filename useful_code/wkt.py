from osgeo import ogr
import pandas as pd
import os

shapefile = r"F:\wd_data\shp2wkt\chunan_clip2.shp"
savepath = os.path.dirname(shapefile)
filename = os.path.basename(shapefile).split(".")[0] + ".csv"
print(filename, savepath)
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
FID = list()
GeoWKT = list()
i = 0
for feature in layer:
    name = feature.GetField("FID")
    FID.insert(i, name)
    geom = feature.GetGeometryRef()
    geomwkt = geom.ExportToWkt()
    GeoWKT.insert(i, geomwkt)
    i += 1

df = pd.DataFrame()
df['FID'] = FID
df['WKT'] = GeoWKT

df.to_csv(os.path.join(savepath, filename), sep='\t')
