# -*- coding: utf-8 -*-
import sys
import csv
# import struct
# import datetime
# import decimal
# import itertools
# # from cStringIO import StringIO
# from operator import itemgetter
# import csv
# # import shapefile
# def dbfreader(f):
#     """Returns an iterator over records in a Xbase DBF file.
#
#     The first row returned contains the field names.
#     The second row contains field specs: (type, size, decimal places).
#     Subsequent rows contain the data records.
#     If a record is marked as deleted, it is skipped.
#
#     File should be opened for binary reads.
#
#     """
#     # See DBF format spec at:
#     #     http://www.pgts.com.au/download/public/xbase.htm#DBF_STRUCT
#
#     numrec, lenheader = struct.unpack('<xxxxLH22x', f.read(32))
#     numfields = (lenheader - 33) // 32
#
#     fields = []
#     for fieldno in xrange(numfields):
#         name, typ, size, deci = struct.unpack('<11sc4xBB14x', f.read(32))
#         name = name.replace('\0', '')  # eliminate NULs from string
#         fields.append((name, typ, size, deci))
#     yield [field[0] for field in fields]
#     yield [tuple(field[1:]) for field in fields]
#
#     terminator = f.read(1)
#     assert terminator == '\r'
#
#     fields.insert(0, ('DeletionFlag', 'C', 1, 0))
#     fmt = ''.join(['%ds' % fieldinfo[2] for fieldinfo in fields])
#     fmtsiz = struct.calcsize(fmt)
#     for i in xrange(numrec):
#         record = struct.unpack(fmt, f.read(fmtsiz))
#         if record[0] != ' ':
#             continue  # deleted record
#         result = []
#         for (name, typ, size, deci), value in itertools.izip(fields, record):
#             if name == 'DeletionFlag':
#                 continue
#             if typ == "N":
#                 value = value.replace('\0', '').lstrip()
#                 if value == '':
#                     value = 0
#                 elif deci:
#                     value = decimal.Decimal(value)
#                 else:
#                     value = int(value)
#             elif typ == 'D':
#                 y, m, d = int(value[:4]), int(value[4:6]), int(value[6:8])
#                 value = datetime.date(y, m, d)
#             elif typ == 'L':
#                 value = (value in 'YyTt' and 'T') or (value in 'NnFf' and 'F') or '?'
#             elif typ == 'F':
#                 value = float(value)
#             result.append(value)
#         yield result
#
#
# def dbfwriter(f, fieldnames, fieldspecs, records):
#     """ Return a string suitable for writing directly to a binary dbf file.
#
#     File f should be open for writing in a binary mode.
#
#     Fieldnames should be no longer than ten characters and not include \x00.
#     Fieldspecs are in the form (type, size, deci) where
#         type is one of:
#             C for ascii character data
#             M for ascii character memo data (real memo fields not supported)
#             D for datetime objects
#             N for ints or decimal objects
#             L for logical values 'T', 'F', or '?'
#         size is the field width
#         deci is the number of decimal places in the provided decimal object
#     Records can be an iterable over the records (sequences of field values).
#
#     """
#     # header info
#     ver = 3
#     now = datetime.datetime.now()
#     yr, mon, day = now.year - 1900, now.month, now.day
#     numrec = len(records)
#     numfields = len(fieldspecs)
#     lenheader = numfields * 32 + 33
#     lenrecord = sum(field[1] for field in fieldspecs) + 1
#     hdr = struct.pack('<BBBBLHH20x', ver, yr, mon, day, numrec, lenheader, lenrecord)
#     f.write(hdr)
#
#     # field specs
#     for name, (typ, size, deci) in itertools.izip(fieldnames, fieldspecs):
#         name = name.ljust(11, '\x00')
#         fld = struct.pack('<11sc4xBB14x', name, typ, size, deci)
#         f.write(fld)
#
#     # terminator
#     f.write('\r')
#
#     # records
#     for record in records:
#         f.write(' ')  # deletion flag
#         for (typ, size, deci), value in itertools.izip(fieldspecs, record):
#             if typ == "N":
#                 value = str(value).rjust(size, ' ')
#             elif typ == 'D':
#                 value = value.strftime('%Y%m%d')
#             elif typ == 'L':
#                 value = str(value)[0].upper()
#             else:
#                 value = str(value)[:size].ljust(size, ' ')
#             assert len(value) == size
#             f.write(value)
#
#     # End of file
#     f.write('\x1A')
#
#     ###################################################################################3
# def dbf2csv():
#     filename = r'H:\tool\2\out2\1.dbf'
#     f = open(filename, 'rb')
#     db = list(dbfreader(f))
#     f.close()
#     for record in db:
#         print record
#     ##### fieldnames is first row means fieldname,fieldspecs is second row means fieldType,records is afterRows means records
#     fieldnames, fieldspecs, records = db[0], db[1], db[2:]
#
#
#     # Remove a field
#     # del fieldnames[0]
#     # del fieldspecs[0]
#     # records = [rec[1:] for rec in records]
#
#     # Create a new DBF
#     filename1 =r'H:\tool\2\out2\copy1.dbf'
#     f1 = open(filename1, 'wb+')
#     dbfwriter(f1, fieldnames, fieldspecs, records)
#
#     # Read the data back from the new DBF
#     print '-' * 50
#     f1.seek(0)
#     for line in dbfreader(f1):
#         print line
#     f1.close()
#
#     # Convert to CSV
#     print '.' * 50
#     filename1 =r'H:\tool\2\out2\2.csv'
#     f1 = open(filename1, 'wb+')
#     csv.writer(f1).writerow(fieldnames)
#     csv.writer(f1).writerows(records)
#     # print f1.getvalue()
#     f1.close()

# def read_csv(fp):
#     ret = []
#     with open(fp, 'rb') as f:
#         for line in f:
#             ret.append(line.decode('utf-8').strip().split(","))
#     return ret
#
# def csv2shp():
#     # csvfile = open(r'H:\tool\out\ta1.csv', 'rb')  # 读文件内容
#     # data = csv.reader(csvfile)
#     data = read_csv(r'H:\tool\out\ta1.csv')
#
#     # 打开shp
#     w = shapefile.Writer(shapefile.POINT)
#     # shapefile文件要求”几何数据”与”属性数据”要有一一对应的关系，如果有”几何数据”而没有相应的属性值存在，那么在使用ArcGIS软件打开所创建的shapefile文件时会出错
#     # 为了避免这种情况的发生，可以设置 sf.autoBalance = 1，以确保每创建一个”几何数据”，该库会自动创建一个属性值(空的属性值)来进行对应。
#     # autoBalance默认为0
#
#     w.autoBalance = 1
#
#     # 增加属性字段 设置类型与长度
#     w.field('id', 'N', 12)
#     # w.field('date', 'D')
#     # w.field('city', 'C', 100)
#     # w.field('location', 'C', 100)
#     w.field('COUNT', 'N', 12)
#     w.field('AREA', 'F', 10, 5)
#     w.field('MAX', 'F', 10, 5)
#     w.field('RANGE', 'F', 10, 5)
#     w.field('MEAN', 'F', 10, 5)
#     w.field('STD', 'F', 10, 5)
#     w.field('SUM', 'F', 10, 5)
#
#     for r in data[1:]:  # 从第二行开始
#         record = [
#             int(r[0]),
#             int(r[1]),
#             float(r[2]),
#             float(r[3]),
#             float(r[4]),
#             float(r[5]),
#             float(r[6]),
#             float(r[7])]
#         w.record(*record)
#         w.point(float(r[-2]), float(r[-1]))
#     w.save(r"H:\tool\out\ta1.shp")

# # coding:utf-8
# #功能：批量导出栅格文件的属性表。
# #使用步骤 1：在相应文件夹下新建“文件地理数据库”，并将需要导出属性表的栅格文件“导入”到该数据库中。
# #使用步骤 2：更改第二行代码[ws = r'D:\test\test1.gdb']为自己的文件存放地址和数据库名称，第三行同样的处理。
# #使用步骤 3：复制代码在ArcGIS中运行即可。
# import arcpy, os
# def run():
#      ws = r'H:\tool\2\out2\1.gdb'
#      outPath = r'H:\tool\2'
#      outExt = ".csv"
#      arcpy.env.workspace = ws
#      rasters = arcpy.ListRasters("*")
#      for raster in rasters:
#           rasloc = ws + os.sep + raster
#           fields = "*"
#           try:
#                lstFlds = arcpy.ListFields(rasloc)
#                header = ''
#                for fld in lstFlds:
#                     header += ",{0}".format(fld.name)
#                     if len(lstFlds) != 0:
#                          outCSV = outPath + os.sep + raster + outExt
#                          f = open(outCSV,'w')
#                          header = header[1:] + ',RasterName\n'
#                          f.write(header)
#                          with arcpy.da.SearchCursor(rasloc, fields) as cursor:
#                               for row in cursor:
#                                    f.write(str(row).replace("(","").replace(")","") + "," + raster + '\n')
#                          f.close()
#           except Exception as e:
#                print (e)
# import arcpy
# def readshpfile():
#     inFC = r'H:\tool\2\data\chunan_clip2.shp'
#     theFields = arcpy.ListFields(inFC)
#     FieldsArray = []
#     for Field in theFields:
#         FieldsArray.append(Field.aliasName)
#     for row in arcpy.da.SearchCursor(inFC, FieldsArray):
#         print row
# if __name__ == '__main__':
#     readshpfile()
#     # dbf2csv()
    # csv2shp()
    # run()

import sys
# from osgeo import ogr
# import csv
#
# fn = r"H:\tool\2\data\chunan_clip2.shp"
# ds = ogr.Open(fn, 1)
#
# # lyr = shapef.GetLayer()
# #     ####
# # oFieldID = ogr.FieldDefn("LUType", ogr.OFTInteger)  # 创建一个叫LuType的整型属性
# # lyr.CreateField(oFieldID, 1)
# i = 0
# lyr = ds.GetLayer()
# oFieldID = ogr.FieldDefn("new", ogr.OFTInteger)  # 创建一个叫LuType的整型属性
# lyr.CreateField(oFieldID, 1)
#
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

    # print(column1)
# for i in f:
#     print(i)
# import os
# import shutil
#
# def copyShp(inputFullPath, targetFullPath):
#     assFile = ['dbf', 'shp', 'shx', 'sbn', 'sbx', 'cpg', 'xml', 'prj', 'mxs', 'ixs', 'atx', 'ain', 'fbn', 'sbn']
#     inputPath = os.path.dirname(inputFullPath)
#     targetPath = os.path.dirname(targetFullPath)
#     inputName = os.path.basename(inputFullPath).split(".")[0]
#     targetName = os.path.basename(targetFullPath).split(".")[0]
#     for ex in assFile:
#         src = os.path.join(inputPath, "{0}.{1}".format(inputName, ex))
#         if os.path.exists(src):
#             print(src)
#             targFile = os.path.join(targetPath, "{0}.{1}".format(targetName, ex))
#             print(targFile)
#             if os.path.exists(targFile):
#                 os.remove(targFile)
#             shutil.copy(src, targFile)

from osgeo import ogr
import csv

fn = r"H:\tool\2\data\chunan_clip2.shp"
ds = ogr.Open(fn, 1)

# lyr = shapef.GetLayer()
#     ####
# oFieldID = ogr.FieldDefn("LUType", ogr.OFTInteger)  # 创建一个叫LuType的整型属性
# lyr.CreateField(oFieldID, 1)



with open(r'H:\tool\2\out2\2.csv', 'r') as csvfile:
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
    lyr = ds.GetLayer()
    oFieldID = ogr.FieldDefn(field_list[k], ogr.OFTReal)  # 创建一个叫LuType的整型属性
    lyr.CreateField(oFieldID, 1)
    l = 0
    print temp[k][l]
    print '=========================='
    print (len(lyr))
    for feat in lyr:

        # fid_ = feat.GetField("FID_")
        # fid = feat.GetField(1)
        # print (fid)

        feat.SetField(field_list[k], temp[k][l])
        lyr.SetFeature(feat)
        l += 1
        print l