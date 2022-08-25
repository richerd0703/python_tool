import os
import numpy as np
from PIL import Image
import cv2
from tqdm import tqdm

__all__ = ['SegmentationMetric']

"""
confusionMetric
P\L     P    N
P      TP    FP
N      FN    TN
"""


class SegmentationMetric(object):
    def __init__(self, numClass):
        self.numClass = numClass
        self.confusionMatrix = np.zeros((self.numClass,) * 2)

    def pixelAccuracy(self):
        # return all class overall pixel accuracy
        # acc = (TP + TN) / (TP + TN + FP + TN)
        acc = np.diag(self.confusionMatrix).sum() / self.confusionMatrix.sum()
        return acc

    def classPixelAccuracy(self):
        # return each category pixel accuracy(A more accurate way to call it precision)
        # acc = (TP) / TP + FP
        classAcc = np.diag(self.confusionMatrix) / self.confusionMatrix.sum(axis=1)
        return classAcc

    def meanPixelAccuracy(self):
        classAcc = self.classPixelAccuracy()
        meanAcc = np.nanmean(classAcc)
        return meanAcc

    def meanIntersectionOverUnion(self):
        # Intersection = TP Union = TP + FP + FN
        # IoU = TP / (TP + FP + FN)
        intersection = np.diag(self.confusionMatrix)
        union = np.sum(self.confusionMatrix, axis=1) + np.sum(self.confusionMatrix, axis=0) - np.diag(
            self.confusionMatrix)
        IoU = intersection / union
        mIoU = np.nanmean(IoU)
        return mIoU

    def genConfusionMatrix(self, imgPredict, imgLabel):
        # remove classes from unlabeled pixels in gt image and predict
        mask = (imgLabel >= 0) & (imgLabel < self.numClass)
        label = self.numClass * imgLabel[mask] + imgPredict[mask]
        count = np.bincount(label, minlength=self.numClass ** 2)
        confusionMatrix = count.reshape(self.numClass, self.numClass)
        return confusionMatrix

    def Frequency_Weighted_Intersection_over_Union(self):
        # FWIOU =     [(TP+FN)/(TP+FP+TN+FN)] *[TP / (TP + FP + FN)]
        freq = np.sum(self.confusion_matrix, axis=1) / np.sum(self.confusion_matrix)
        iu = np.diag(self.confusion_matrix) / (
                np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) -
                np.diag(self.confusion_matrix))
        FWIoU = (freq[freq > 0] * iu[freq > 0]).sum()
        return FWIoU

    def addBatch(self, imgPredict, imgLabel):
        assert imgPredict.shape == imgLabel.shape
        self.confusionMatrix += self.genConfusionMatrix(imgPredict, imgLabel)

    def reset(self):
        self.confusionMatrix = np.zeros((self.numClass, self.numClass))


if __name__ == '__main__':
    predict_path = r"G:\data\tianchi\select\val\outdir_2_23\pre"
    label_path = r"G:\data\tianchi\select\val\labels"
    total_pa = total_mpa = total_mIoU = count = 0
    for file in tqdm(os.listdir(label_path)):
        ppath = os.path.join(predict_path, file)  # 用PIL中的Image.open打开图像  预测
        mpath = os.path.join(label_path, file)  # label

        imgPredict =cv2.imread(ppath) # 转化成numpy数组
        imgLabel = cv2.imread(mpath)

        # print("imgPredict:", imgPredict.shape)
        # print("imgPredict:", type(imgPredict))
        #
        # print("imgLabel:", imgLabel.shape)
        # print("imgLabel:", type(imgLabel))

        metric = SegmentationMetric(4)  #3表示有3个分类，有几个分类就填几
        metric.addBatch(imgPredict, imgLabel)
        pa = metric.pixelAccuracy()  #pa 像素准确率
        cpa = metric.classPixelAccuracy()
        mpa = metric.meanPixelAccuracy()
        mIoU = metric.meanIntersectionOverUnion()  #平均像素准确率
        count += 1
        total_mpa += mpa
        total_pa += pa
        total_mIoU += mIoU
        # print('pa is : %f' % pa)
        # print('cpa is :')  # 列表
        # print(cpa)
        # print('mpa is : %f' % mpa)
        # print('mIoU is : %f' % mIoU)

    print('pa is : %f' % (total_pa / count))
    # print('cpa is :')  # 列表
    # print(cpa)
    print('mpa is : %f' % (total_mpa / count))
    print('mIoU is : %f' % (total_mIoU / count))

# if __name__ == '__main__':
#     ppath = r"G:\data\tianchi\select\val\labels\image_1_1_86.png"  # 用PIL中的Image.open打开图像  预测
#     mpath = r"G:\data\tianchi\select\val\labels\image_1_1_86.png"  # label
#
#     imgPredict = cv2.imread(ppath)  # 转化成numpy数组
#     imgLabel = cv2.imread(mpath)
#
#     print("imgPredict:", imgPredict.shape)
#     print("imgPredict:", type(imgPredict))
#
#     print("imgLabel:", imgLabel.shape)
#     print("imgLabel:", type(imgLabel))
#
#     metric = SegmentationMetric(3)
#     metric.addBatch(imgPredict, imgLabel)
#     acc = metric.pixelAccuracy()  # pa 像素准确率
#     mIoU = metric.meanIntersectionOverUnion()  # 平均像素准确率
#     print(acc, mIoU)
