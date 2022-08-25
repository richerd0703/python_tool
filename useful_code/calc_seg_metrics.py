
"""
refer to https://github.com/jfzhang95/pytorch-deeplab-xception/blob/master/utils/metrics.py
"""
import numpy as np
import cv2
from PIL import Image
import os, glob
from tqdm import tqdm

__all__ = ['SegmentationMetric']

"""
confusionMetric  # 注意：此处横着代表预测值，竖着代表真实值，与之前介绍的相反
P\L     P    N
P      TP    FP
N      FN    TN
"""


class SegmentationMetric(object):
    def __init__(self, numClass):
        self.numClass = numClass
        # self.confusionMatrix = np.zeros((self.numClass,) * 2)  # 混淆矩阵（空）
        self.confusionMatrix = np.zeros((self.numClass,) * 2)  # 混淆矩阵（空）

    def pixelAccuracy(self):
        # return all class overall pixel accuracy 正确的像素占总像素的比例
        #  PA = acc = (TP + TN) / (TP + TN + FP + TN)
        acc = np.diag(self.confusionMatrix).sum() / self.confusionMatrix.sum()
        return acc

    def classPixelAccuracy(self):  # recall
        # return each category pixel accuracy(A more accurate way to call it precision)
        # acc = (TP) / TP + FP
        classAcc = np.diag(self.confusionMatrix) / self.confusionMatrix.sum(axis=1)
        return classAcc  # 返回的是一个列表值，如：[0.90, 0.80, 0.96]，表示类别1 2 3各类别的预测准确率

    def Precision(self):  
    #  返回所有类别的精确率precision  
        precision = np.diag(self.confusionMatrix) / self.confusionMatrix.sum(axis = 0)
        return precision 

    def F1Score(self):
        precision = np.diag(self.confusionMatrix) / self.confusionMatrix.sum(axis = 0)
        recall = np.diag(self.confusionMatrix) / self.confusionMatrix.sum(axis = 1)
        f1score = 2 * precision * recall / (precision + recall)
        return f1score

    def meanPixelAccuracy(self):
        """
        Mean Pixel Accuracy(MPA，均像素精度)：是PA的一种简单提升，计算每个类内被正确分类像素数的比例，之后求所有类的平均。
        :return:
        """
        classAcc = self.classPixelAccuracy()
        meanAcc = np.nanmean(classAcc)  # np.nanmean 求平均值，nan表示遇到Nan类型，其值取为0
        return meanAcc  # 返回单个值，如：np.nanmean([0.90, 0.80, 0.96, nan, nan]) = (0.90 + 0.80 + 0.96） / 3 =  0.89

    def IntersectionOverUnion(self):
        # Intersection = TP Union = TP + FP + FN
        # IoU = TP / (TP + FP + FN)
        intersection = np.diag(self.confusionMatrix)  # 取对角元素的值，返回列表
        union = np.sum(self.confusionMatrix, axis=1) + np.sum(self.confusionMatrix, axis=0) - np.diag(
            self.confusionMatrix)  # axis = 1表示混淆矩阵行的值，返回列表； axis = 0表示取混淆矩阵列的值，返回列表
        IoU = intersection / union  # 返回列表，其值为各个类别的IoU
        return IoU

    def meanIntersectionOverUnion(self):
        mIoU = np.nanmean(self.IntersectionOverUnion())  # 求各类别IoU的平均
        return mIoU

    def genConfusionMatrix(self, imgPredict, imgLabel):  #
        """
        同FCN中score.py的fast_hist()函数,计算混淆矩阵
        :param imgPredict:
        :param imgLabel:
        :return: 混淆矩阵
        """
        # remove classes from unlabeled pixels in gt image and predict
        mask = (imgLabel >= 0) & (imgLabel < self.numClass)
        label = self.numClass * imgLabel[mask] + imgPredict[mask]
        count = np.bincount(label, minlength=self.numClass ** 2)
        confusionMatrix = count.reshape(self.numClass, self.numClass)
        # print(confusionMatrix)
        return confusionMatrix

    def Frequency_Weighted_Intersection_over_Union(self):
        """
        FWIoU，频权交并比:为MIoU的一种提升，这种方法根据每个类出现的频率为其设置权重。
        FWIOU =     [(TP+FN)/(TP+FP+TN+FN)] *[TP / (TP + FP + FN)]
        """
        freq = np.sum(self.confusion_matrix, axis=1) / np.sum(self.confusion_matrix)
        iu = np.diag(self.confusion_matrix) / (
                np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) -
                np.diag(self.confusion_matrix))
        FWIoU = (freq[freq > 0] * iu[freq > 0]).sum()
        return FWIoU

    def addBatch(self, imgPredict, imgLabel):
        assert imgPredict.shape == imgLabel.shape
        self.confusionMatrix += self.genConfusionMatrix(imgPredict, imgLabel)  # 得到混淆矩阵
        # print(self.confusionMatrix, "===\n")
        return self.confusionMatrix

    def reset(self):
        self.confusionMatrix = np.zeros((self.numClass, self.numClass))


def fast_hist(a, b, n):
    # --------------------------------------------------------------------------------#
    #   a是转化成一维数组的标签，形状(H×W,)；b是转化成一维数组的预测结果，形状(H×W,)
    # --------------------------------------------------------------------------------#
    k = (a >= 0) & (a < n)
    # --------------------------------------------------------------------------------#
    #   np.bincount计算了从0到n**2-1这n**2个数中每个数出现的次数，返回值形状(n, n)
    #   返回中，写对角线上的为分类正确的像素点
    # --------------------------------------------------------------------------------#
    return np.bincount(n * a[k].astype(int) + b[k], minlength=n ** 2).reshape(n, n)
# 测试内容


def edge(Label, Predict, num):

# 标签图像文件夹
    l = [1, 3, 5, 7, 9]
    L = Label
    #  预测图像文件夹
    P = Predict

    for i in l:
        print("-------------", i)
        LabelPath = os.path.join(L, str(i))
        #  预测图像文件夹
        PredictPath = os.path.join(P, str(i))
        class_num = num
        # print(LabelPath, PredictPath, class_num)
        hist = np.zeros((class_num, class_num, class_num))
        LabelPaths = glob.glob(os.path.join(LabelPath, "*.png"))
        PredictPaths = glob.glob(os.path.join(PredictPath, "*.png"))
        metric = SegmentationMetric(class_num)  # 2表示有2个分类，有几个分类就填几
        for i, path in tqdm(enumerate(LabelPaths)):
            # print(path, PredictPaths[i])
            imgPredict = Image.open(PredictPaths[i])  # cv2.imread('1.png')
            imgLabel = Image.open(path)  # cv2.imread('2.png')
            imgLabel = np.array(imgLabel)
            imgPredict = np.array(imgPredict)
            if class_num == 2:
                imgLabel = np.array(imgLabel) / 255.0
                imgPredict = np.array(imgPredict) / 255.0
                imgLabel = imgLabel.astype(np.uint8)
                imgPredict = imgPredict.astype(np.uint8)
            #
            #
            # imgPredict = np.array(imgPredict)
            # imgPredict[imgPredict>128] = 1
            # imgPredict[imgPredict<=128] = 0.0
            #
            # imgPredict = np.array(cv2.cvtColor(imgPredict, cv2.COLOR_BGR2GRAY) / 255., dtype=np.uint8)
            # imgLabel = np.array(cv2.cvtColor(imgLabel, cv2.COLOR_BGR2GRAY) / 255., dtype=np.uint8)
            # imgPredict = np.array([0, 0, 1, 1, 2, 2])  # 可直接换成预测图片
            # imgLabel = np.array([0, 0, 1, 1, 2, 2])  # 可直接换成标注图片

            hist = metric.addBatch(imgPredict, imgLabel)
            hist += fast_hist(imgLabel.flatten(), imgPredict.flatten(),class_num)
            # print(hist)
        metric.confusionMatrix=hist

        pa = metric.pixelAccuracy()
        cpa = metric.classPixelAccuracy()
        mpa = metric.meanPixelAccuracy()
        IoU = metric.IntersectionOverUnion()
        mIoU = metric.meanIntersectionOverUnion()
        f1 = metric.F1Score()
        p = metric.Precision()
        print("\t")
        print('PA is : %f' % pa)
        print('recall is :', cpa)  # 列表
        print('mPA is : %f' % mpa)
        print('IoU is : ', IoU)
        print('mIoU is : ', mIoU)
        print('f1:{}', f1)
        print("prec:", p)


def poly(LabelPath, PredictPath, class_num):
    LabelPath = LabelPath
    #  预测图像文件夹
    PredictPath = PredictPath

    class_num = class_num
    hist = np.zeros((class_num, class_num))
    LabelPaths = glob.glob(os.path.join(LabelPath, "*.png"))
    PredictPaths = glob.glob(os.path.join(PredictPath, "*.png"))
    metric = SegmentationMetric(class_num)  # 2表示有2个分类，有几个分类就填几
    for i, path in tqdm(enumerate(LabelPaths)):
        # print(path, PredictPaths[i])
        imgPredict = Image.open(PredictPaths[i])  # cv2.imread('1.png')
        imgLabel = Image.open(path)  # cv2.imread('2.png')
        imgLabel = np.array(imgLabel)
        imgPredict = np.array(imgPredict)
        if class_num == 2:
            imgLabel = np.array(imgLabel) / 255.0
            imgPredict = np.array(imgPredict) / 255.0
            imgLabel = imgLabel.astype(np.uint8)
            imgPredict = imgPredict.astype(np.uint8)
        #
        #
        # imgPredict = np.array(imgPredict)
        # imgPredict[imgPredict>128] = 1
        # imgPredict[imgPredict<=128] = 0.0
        #
        # imgPredict = np.array(cv2.cvtColor(imgPredict, cv2.COLOR_BGR2GRAY) / 255., dtype=np.uint8)
        # imgLabel = np.array(cv2.cvtColor(imgLabel, cv2.COLOR_BGR2GRAY) / 255., dtype=np.uint8)
        # imgPredict = np.array([0, 0, 1, 1, 2, 2])  # 可直接换成预测图片
        # imgLabel = np.array([0, 0, 1, 1, 2, 2])  # 可直接换成标注图片

        # hist = metric.addBatch(imgPredict, imgLabel)
        hist += fast_hist(imgLabel.flatten(), imgPredict.flatten(), class_num)
    # print(hist)
    metric.confusionMatrix = hist
    pa = metric.pixelAccuracy()
    cpa = metric.classPixelAccuracy()
    mpa = metric.meanPixelAccuracy()
    IoU = metric.IntersectionOverUnion()
    mIoU = metric.meanIntersectionOverUnion()
    f1 = metric.F1Score()
    p = metric.Precision()
    print('PA is : %f' % pa)
    print('recall is :', cpa)  # 列表
    print('mPA is : %f' % mpa)
    print('IoU is : ', IoU)
    print('mIoU is : ', mIoU)
    print('f1:{}', f1)
    print("prec:", p)


if __name__ == '__main__':
    # G:\dataset\tianchi\trainData\total\val\labels
    # LabelPath = r"G:\dataset\tianchi\trainData\total\val\edge_diff"  # edge
    LabelPath = r"G:\exp\crowdAI\my\test\labels"  # poly
    # LabelPath = r"G:\dataset\tianchi\trainData\total\val\0"  # poly
    #  预测图像文件夹
    PredictPath = r"G:\exp\crowdAI\deeplab\pre\0-255"
    class_num = 2
    # edge(LabelPath, PredictPath, class_num)
    poly(LabelPath, PredictPath, class_num)

'''
f1: 0.6685601408689454
PA is : 0.941612
recall is : [0.96854753 0.8593631 ]
mPA is : 0.913955
IoU is :  [0.92590348 0.78406034]
mIoU is :  0.8549819118163373
f1:{} [0.96152636 0.87896169]
prec: [0.95460625 0.89947506]

PA is : 0.952796
recall is : [0.97374014 0.88367448]
mPA is : 0.928707
IoU is :  [0.94058702 0.81319887]
mIoU is :  0.8768929431660334
f1:{} [0.96938402 0.89697703]
prec: [0.9650667  0.91068621]


'''