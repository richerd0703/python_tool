from _datetime import datetime
import shapefile
from pandas import read_csv

data = read_csv(r"H:\tool\out\ta1.csv")

#打开shp
w=shapefile.Writer(shapefile.POINT)
#shapefile文件要求”几何数据”与”属性数据”要有一一对应的关系，如果有”几何数据”而没有相应的属性值存在，那么在使用ArcGIS软件打开所创建的shapefile文件时会出错
#为了避免这种情况的发生，可以设置 sf.autoBalance = 1，以确保每创建一个”几何数据”，该库会自动创建一个属性值(空的属性值)来进行对应。
#autoBalance默认为0

w.autoBalance = 1

#增加属性字段 设置类型与长度
w.field('id', 'N', 12)
w.field('date', 'D')
w.field('city', 'C', 100)
w.field('location', 'C', 100)
w.field('lng', 'F', 10, 5)
w.field('lat', 'F', 10, 5)

for r in data[1:]:  #从第二行开始
    record = [
        int(r[0]),
        datetime.strftime(datetime.strptime(r[1], '%d/%m/%Y'),'%Y%m%d'),#把日/月/年转为年\月\日格式
        r[2],
        r[3],
        float(r[4]),
        float(r[5])]
    w.record(*record)
    w.point(float(r[-2]), float(r[-1]))
w.save("sites.shp")