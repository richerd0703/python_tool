import shutil
from osgeo import osr, gdal


def add_info(src_path, dst_path):
    src_ds = gdal.Open(src_path, gdal.GA_ReadOnly)
    sr = osr.SpatialReference()
    sr.ImportFromWkt(src_ds.GetProjectionRef())
    geoTransform = src_ds.GetGeoTransform()
    dst_ds = gdal.Open(dst_path, gdal.GA_Update)
    dst_ds.SetProjection(sr.ExportToWkt())
    dst_ds.SetGeoTransform(geoTransform)
    dst_ds = None
    src_ds = None


# def assign_spatial_reference(filepath):
#     sr = osr.SpatialReference()
#     sr.ImportFromEPSG(4326)  # 地理坐标投影
#
#     ds = gdal.Open(filepath, gdal.GA_Update)
#     ds.SetProjection(sr.ExportToWkt())
#     ds.SetGeoTransform([0, 1, 0, 0, 0, 1])  # 地理6参数
#
#     ds = None


if __name__ == '__main__':
    src_file = r"G:\临时文件\data2\xialiegang\2\data\hangzhou31_1_clip1.tif"
    dst_file = r"G:\临时文件\data2\xialiegang\2\data\label.tif"
    # assign_spatial_reference(src_file)
    add_info(src_file, dst_file)

    # file1 = r"G:\test\addgeo\image\pre_clip1.tif"
    # file2 = r"G:\test\addgeo\move_pre_clip1.tif"
    # shutil.copy(file1, file2)