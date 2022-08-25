#coding:utf-8
import yaml

with open('../file/config_test.yaml', 'r')as f:
    doc =yaml.load(f, Loader=yaml.FullLoader)

    doc["input"]["img1_path"] = "1"
    doc["input"]["img2_path"] = "4"
    doc["output"]["out_path"] = "7"
#通过doc取yaml的内容,然后赋值
with open('../file/config_test.yaml', 'w')as f:
    yaml.dump(doc,f)
