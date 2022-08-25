from osgeo import ogr
import numpy as np
from sklearn.tree import DecisionTreeClassifier
import os
import argparse
import time


def decisionclassify(strCSVFeature, strSample, strPlotVec, strResult):
    Y = np.loadtxt(strSample, dtype=np.int, delimiter=',')
    C = np.loadtxt(strCSVFeature, dtype=np.float, delimiter=',', skiprows=2)
    nFeatureNum = C.shape[1]
    nInputNum = C.shape[0]
    nSampleNum = Y.shape[0]
    samples = np.zeros((nSampleNum, nFeatureNum), dtype=np.float)
    labels = np.zeros((nSampleNum), dtype=np.int)
    Z = np.zeros((nInputNum, nFeatureNum), dtype=np.float)

    print("Sample Number:", nSampleNum)
    print("Input Number:", nInputNum)
    print("Feaute Dim:", nFeatureNum)

    Flag = 0
    for j in range(0, nSampleNum):
        for i in range(0, nInputNum):
            if (C[i][0] == Y[j][0]):
                for k in range(1, nFeatureNum):
                    samples[Flag][k - 1] = C[i][k]
                    labels[Flag] = Y[j][1]
                Flag += 1
                break

    print("Sample shape", samples.shape)
    print("Sample shape", labels.shape)

    for i in range(0, nInputNum):
        for j in range(0, nFeatureNum - 1):
            Z[i][j] = C[i][j + 1]

    # 训练模型，限制树的最大深度4
    clf = DecisionTreeClassifier(max_depth=nFeatureNum)
    # 拟合模型
    clf.fit(samples, labels)
    MyPredict = clf.predict(Z)

    driverSHP = ogr.GetDriverByName('ESRI Shapefile')
    resultIn = ogr.Open(strPlotVec, 0)
    resultDS = driverSHP.CopyDataSource(resultIn, strResult)
    if resultDS is None:
        print("Can't Open File", strResult)
        return False

    ptLayer = resultDS.GetLayer(0)
    NewFieldDefn = ogr.FieldDefn("ClassID", ogr.OFTInteger)
    ptLayer.CreateField(NewFieldDefn)
    featurenDefn = ptLayer.GetLayerDefn()
    pFeature = ptLayer.GetNextFeature()
    i = 0
    while (pFeature != None):
        pFeature.SetField("ClassID", int(MyPredict[i]))
        ptLayer.SetFeature(pFeature)
        pFeature = ptLayer.GetNextFeature()
        i += 1

    print("Result is Good")

    return 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser('argument')
    parser.add_argument('--feature_path', type=str)
    parser.add_argument('--sample_path', type=str)
    parser.add_argument('--shp_path', type=str)
    parser.add_argument('--output', type=str)
    args = parser.parse_args()

    pwd = os.path.dirname(__file__)
    now_time = time.strftime("%H%M%S_%m%d_%Y", time.localtime())
    # strCSVFeature = r"I:\xialiegang\DecisionClassify\data\input\SanDiao-Albers_TSFeateure.csv"
    # strSample = r"I:\xialiegang\DecisionClassify\data\input\Sample-DT.csv"
    # strPlotVec = r"I:\xialiegang\DecisionClassify\data\input\SanDiao-Albers.shp"
    # strResultr = r"I:\xialiegang\DecisionClassify\data\output\SanDiao-Albers.shp"

    file_name = os.path.basename(args.shp_path)
    strResultr = os.path.join(args.output, now_time + "_" + file_name)

    decisionclassify(args.feature_path, args.sample_path, args.shp_path, strResultr)

# --feature_path I:\xialiegang\DecisionClassify\data\input\SanDiao-Albers_TSFeateure.csv
# --sample_path I:\xialiegang\DecisionClassify\data\input\Sample-DT.csv
# --shp_path I:\xialiegang\DecisionClassify\data\input\SanDiao-Albers.shp
# --output I:\xialiegang\DecisionClassify\data\output

