#coding:utf-8
import os
import random

trainval_percent = 1  # 训练验证数据集的百分比
train_percent = 0.75 		# 训练集的百分比
filepath = r'I:\data\oil_tank\JPEGImages' # 位置
total_img = os.listdir(filepath)
num = len(total_img)  				# 列表的长度
list = range(num)
tv = int(num*trainval_percent)  # 训练验证集的图片个数
tr = int(tv*train_percent)  	  # 训练集的图片个数	# sample(seq, n) 从序列seq中选择n个随机且独立的元素；
trainval = random.sample(list, tv)
train = random.sample(trainval, tr)
# 创建文件trainval.txt,test.txt,train.txt,val.txt
ftrain = open(r'I:\data\oil_tank\ImageSets\Segmentation\train.txt', 'w')
fval = open(r'I:\data\oil_tank\ImageSets\Segmentation\val.txt', 'w')
for i in list:
    name = total_img[i][:-4]+'\n'
    if i in train:
        ftrain.write(name)
    else:
        fval.write(name)
ftrain.close()
fval.close()
print('finished')
