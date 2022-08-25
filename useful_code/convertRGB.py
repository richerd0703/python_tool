from PIL import Image
import os


path = r"G:\dataset\pansharpen\1"  # 最后要加双斜杠，不然会报错
filelist = os.listdir(path)

for file in filelist:
    whole_path = os.path.join(path, file)
    img = Image.open(whole_path)  # 打开图片img = Image.open(dir)#打开图片
    img = img.convert("RGB")  # 将一个4通道转化为rgb三通道
    save_path = r'G:\dataset\pansharpen'
    # img.save(save_path + img1)
    img.save(save_path + file)