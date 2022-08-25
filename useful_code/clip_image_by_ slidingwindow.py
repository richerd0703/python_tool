import glob, os
from PIL import Image
import numpy as np
import cv2
from tqdm import tqdm
import scipy.io as sio

image_dir = r"F:\wd_data\Tianchi\Train\image"
label_dir = r"F:\wd_data\Tianchi\Train\label"

val_mode = False#True # 多标注验证集模式 
image_ext = "png"
label_ext = "png"
output_ext = "png"

slide_size = 512 #640#128
clip_height = 512 #640#480
clip_width = 512
# seg resize 321*321

output_dir = r"F:\wd_data\Tianchi"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
image_output_dir = os.path.join(output_dir, "image")
if not os.path.exists(image_output_dir):
    os.makedirs(image_output_dir)
label_output_dir = os.path.join(output_dir, "mask")
if not os.path.exists(label_output_dir):
    os.makedirs(label_output_dir)

if val_mode:
    mat_output_dir = os.path.join(output_dir, "mat")
    if not os.path.exists(mat_output_dir):
        os.makedirs(mat_output_dir)
    edges = ["HED_thin_edge","RCF_thin_edge", "BDCN_thin_edge", "DexiNed_thin_edge", "edge"]
    percent = 1.0 / len(edges)
    root_dir = r"H:\EvLab-SSBenchmark\EvLab-SSBenchmark\val"
image_paths = glob.glob(os.path.join(image_dir, "*.{}".format(image_ext)))
label_paths = glob.glob(os.path.join(label_dir, "*.{}".format(label_ext)))

print(len(image_paths), image_paths)
print(len(label_paths), label_paths)


for i, p in tqdm(enumerate(image_paths)):
    idx = 1
    cur_image_path = p
    cur_label_path = label_paths[i]
    if val_mode:
        cur_image = Image.open(cur_image_path)
        cur_image = np.array(cur_image)
        edge_name = os.path.basename(cur_label_path)
        cur_label = []
        for dir_name in edges:
            cur_label.append(np.array(Image.open(os.path.join(root_dir, dir_name, edge_name))))
        cur_label = np.array(cur_label)
        print(cur_image.shape, cur_label.shape)
        h, w, c = cur_image.shape
    else:   
        cur_image = Image.open(cur_image_path)
        cur_label = Image.open(cur_label_path)
        cur_image = np.array(cur_image)
        cur_label = np.array(cur_label)

        print(cur_image_path, cur_label_path)
        print(cur_image.shape, cur_label.shape)
        h, w, c = cur_image.shape
    
    name = os.path.basename(p).split(".")[0]
    for hh in tqdm(range(0, h, slide_size)):
        h_flag = False
        if hh + clip_height >= h:
            hh = h - clip_height
            h_flag = True
        for ww in range(0, w, slide_size):
            w_flag = False
            if ww + clip_width >= w:
                ww = w - clip_width
                w_flag = True
            clip_image = cur_image[hh:hh+clip_height, ww:ww+clip_width, :]
            if val_mode:
                mat_data = {'groundTruth':[]}
                for v in range(len(edges)):
                    if v == 0:
                        clip_label = cur_label[v,hh:hh+clip_height, ww:ww+clip_width] * percent
                    else:
                        clip_label += cur_label[v,hh:hh+clip_height, ww:ww+clip_width] * percent
                mat_label = clip_label / 255.0
                mat_label[mat_label > 0.2] += 0.6# 0.5 for BIPED/BSDS-RIND
                mat_label = np.clip(mat_label, 0., 1.) # BIPED/BSDS-RIND
                mat_data['groundTruth'].append({'Boundaries':np.uint8(mat_label)})
            else:
                clip_label = cur_label[hh:hh+clip_height, ww:ww+clip_width]
            
            clip_image_path = os.path.join(image_output_dir, "{}_{}.{}".format(name,idx, output_ext))
            clip_label_path = os.path.join(label_output_dir, "{}_{}.{}".format(name,idx, output_ext))
            # clip_image = clip_image[:,:,::-1] # for opencv rgb -> bgr
            # cv2.imwrite(clip_image_path, clip_image)
            # cv2.imwrite(clip_label_path, np.uint8(clip_label))
            Image.fromarray(clip_image).save(clip_image_path)
            Image.fromarray(np.uint8(clip_label)).save(clip_label_path)
            if val_mode:
                clip_mat_path = os.path.join(mat_output_dir, "{}_{}.{}".format(name,idx, "mat"))
                sio.savemat(clip_mat_path,mat_data) 
            idx = idx + 1
            if w_flag:
                break
        if h_flag:
            break
    
    print(idx)
