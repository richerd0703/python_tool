import os


def cat_path(srcpath, count, savepath):
    input_path = ""
    output_path = ""
    img_list = [x for x in os.listdir(srcpath) if x.endswith(".img")]  # 获取目录中所有tif格式图像列表
    for img in img_list:
        print(img)
    a = len(img_list)
    print(a)
    for i in range(len(img_list)):
        input_img_path = os.path.join(srcpath, img_list[i])
        output_img_path = os.path.join(savepath, img_list[i])
        input_path += input_img_path + "|"
        output_path += output_img_path + "|"
    return input_path.strip("|"), output_path.strip("|")




if __name__ == '__main__':
    srcpath = r"F:\wd_data\dll_tool\Data\IGSNRR\TimeSeries"
    count = 5
    savepath = r"F:\wd_data\dll_tool\Data\IGSNRR"
    res_in, res_out = cat_path(srcpath, count, savepath)
    print(res_in)
    print(res_out)
    # for i in res_in.split("|"):
    #     print(i.split("\\")[-1])
    #
    # for i in res_out.split("|"):
    #     print(i.split("\\")[-1])
