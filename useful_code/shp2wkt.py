import os
import pandas as pd
from tqdm import tqdm
path = r"I:\wutianjun\Raster_Parcel_Func_Predict\Tool\data\e\RdWtAtt.shp"
savepath = r"I:\wutianjun\Raster_Parcel_Func_Predict\Tool\data\e\res.csv"


infile = open(path, "r", encoding='utf-8')  # 打开文件
shpfile = pd.read_csv(path, header=None, sep="\t")
p = list()
i = 0
for line in tqdm(infile):  # 按行读文件，可避免文件过大，内存消耗
    grammer = line.split('\t')[-1].split("(")
    nums = grammer[2].split(",")
    point1 = nums[0]

    point2 = nums[-1].split(")")[0]

    if point2 != point1:
        print(i, "point1 {}  point2 {}".format(point1, point2))
    result_part = ""
    for point in nums[1:-2]:
        result_part = result_part + point + ","
    result = grammer[0] + "((" + point1 + "," + result_part + point1 + "))"
    p.insert(i, result)
    i += 1

shpfile = shpfile.iloc[:, :-1]
shpfile['polygon'] = p
# shpfile.to_csv(savepath, header=None, sep='\t')
shpfile.to_csv(savepath, sep='\t')
# print(shpfile)
infile.close()  # 文件关闭
# outfile.close()
