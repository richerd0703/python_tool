# -*- coding: utf-8 -*-
from osgeo import ogr
import arcpy
from arcpy.sa import *
import struct
import datetime
import decimal
import itertools
import csv
import os
import shutil
import argparse
import time


# 计算坡度
def pd():
    # Set environment settings
    # env.workspace = "H:/tool/1"

    # Set local variables
    inRaster = r"H:\tool\2\out1\3-1.tif"
    outMeasurement = "DEGREE"
    zFactor = 1

    # Check out the ArcGIS Spatial Analyst extension license
    arcpy.CheckOutExtension("Spatial")

    # Execute Slope
    outSlope = Slope(inRaster, outMeasurement, zFactor)

    # Save the output
    outSlope.save(r"H:\tool\2\out3\pd.tif")

def table():
    # Set local variables
    inZoneData = r"H:\tool\2\data\chunan_clip2.shp"
    zoneField = "FID"
    inValueRaster = r"H:\tool\2\out3\pd.tif"
    outTable = r"H:\tool\2\out1\ta.dbf"

    # Check out the ArcGIS Spatial Analyst extension license
    arcpy.CheckOutExtension("Spatial")

    # Execute ZonalStatisticsAsTable
    outZSaT = ZonalStatisticsAsTable(inZoneData, zoneField, inValueRaster,
                                     outTable, "NODATA", "ALL")

    # outZSaT = ZonalStatisticsAsTable("H:/tool/test_data/chunan_clip2.shp", "FID", "H:/tool/out/1.tif",
    #                                  "H:/tool/out/table", "NODATA", "ALL")


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

###################################################################################3
def dbf2csv():
    filename = r'H:\tool\2\out3\ta.dbf'
    f = open(filename, 'rb')
    db = list(dbfreader(f))
    f.close()
    for record in db:
        print record
    ##### fieldnames is first row means fieldname,fieldspecs is second row means fieldType,records is afterRows means records
    fieldnames, fieldspecs, records = db[0], db[1], db[2:]


    # Remove a field
    # del fieldnames[0]
    # del fieldspecs[0]
    # records = [rec[1:] for rec in records]

    # Create a new DBF
    filename1 =r'H:\tool\2\out3\copyta.dbf'
    f1 = open(filename1, 'wb+')
    dbfwriter(f1, fieldnames, fieldspecs, records)

    # Read the data back from the new DBF
    print '-' * 50
    f1.seek(0)
    for line in dbfreader(f1):
        print line
    f1.close()

    # Convert to CSV
    print '.' * 50
    filename1 =r'H:\tool\2\out3\copyta.csv'
    f1 = open(filename1, 'wb+')
    csv.writer(f1).writerow(fieldnames)
    csv.writer(f1).writerows(records)
    # print f1.getvalue()
    f1.close()


def csv2shp():
    fn = r"H:\tool\2\data\chunan_clip2.shp"
    with open(r'H:\tool\2\out3\copyta.csv', 'r') as csvfile:
        reader = csv.reader(csvfile)
        rows = [row for row in reader]

    leng = len(rows)
    temp = []
    for j in range(9):
        temp1 = []
        for i in range(1, len(rows)):
            temp1.append(float(rows[i][j]))
        temp.append(temp1)

    field_list = ['FID_', 'COUNT', 'AREA', 'MIN', 'MAX', 'RANGE', 'MEAN', 'STD', 'SUM']

    # field_list = rows[0]
    # print field_list
    # print len(temp)
    # print len(temp[1])
    for k in range(len(field_list)):
        # print (field_list[k], temp[k])
        ds = ogr.Open(fn, 1)
        lyr = ds.GetLayer()
        oFieldID = ogr.FieldDefn(field_list[k], ogr.OFTReal)  # 创建一个叫LuType的整型属性
        lyr.CreateField(oFieldID, 1)
        n = 0
        # print temp[k][n]
        # print '=========================='
        print (len(lyr))
        for feat in lyr:
            # fid_ = feat.GetField("FID_")
            # fid = feat.GetField(1)
            # print (fid)

            feat.SetField(field_list[k], temp[k][n])
            lyr.SetFeature(feat)
            n += 1
        #
        # print ("sdggsdg", n)
        # print k

def copyShp(fileFullPath, targetDic):
    assFile = ['dbf', 'shp', 'shx', 'sbn', 'sbx', 'cpg', 'xml', 'prj', 'mxs', 'ixs', 'atx', 'ain', 'fbn', 'sbn']
    Path, name = os.path.split(fileFullPath)
    # print(Path)
    # print(name)
    nameSplit = name.split(".")
    for ex in assFile:
        src = os.path.join(Path, "{0}.{1}".format(nameSplit[0], ex))
        if os.path.exists(src):
            print(src)
            targFile = os.path.join(targetDic, "{0}.{1}".format(nameSplit[0], ex))
            print(targFile)
            if os.path.exists(targFile):
                os.remove(targFile)
            shutil.copy(src, targFile)


if __name__ == '__main__':
    # parser = argparse.ArgumentParser(description='this is a description')
    # parser.add_argument('--shp', type=str)
    # parser.add_argument('--dem', type=str)
    # parser.add_argument('--out', type=str)
    # args = parser.parse_args()
    #
    # # pwd = os.getcwd()
    # pwd = os.path.dirname(__file__)
    # now_time = time.strftime("%H%M%S_%m%d_%Y", time.localtime())
    # temp_save_path = os.path.join(args.outShp, now_time)
    # os.mkdir(temp_save_path)

    pd()
    # table()
    # dbf2csv()
    # csv2shp()
    # copyShp(r"H:\tool\test_data\chunan_clip2.shp", r"H:\tool\2\out3")