from PIL import Image
import os
'''
result_path_target  图片存放目标路径
cut_pictures 待拼接图片存放路径
num 图片命名以数字按序增加

'''

cut_pictures = r'G:\data\cut\complex\1'
result_path_target = r'G:\data\cut\complex'
num = 1

# ims = [Image.open(cut_pictures+'\\'+fn)for fn in listdir(cut_pictures) if fn.endswith(".png")]       #  打开路径下的所有图片
name_list = [x for x in os.listdir(cut_pictures) if x.endswith(".png")]

for file in name_list:
    ims = Image.open(os.path.join(cut_pictures, file))
    if ims.mode == "P":
        img = ims.convert('RGB')
    width,height = 512, 512  #获取拼接图片的宽和高
    print(ims)
    result = Image.new(ims[0].mode,(width,height*len(ims)))
    for j , im in enumerate(ims):
        result.paste(im,box=(0,j*height))
        print(j)
    result.save(result_path_target+'\\'+'%s.jpg'%num)