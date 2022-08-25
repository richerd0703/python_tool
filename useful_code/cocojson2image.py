import cv2
import numpy as np
import os
import json
import numpy as np
from collections import defaultdict
from tqdm import tqdm


ROOT_DIR = r"G:\exp\preliminary\frame\val"
ANNOTATION_FILE = os.path.join(ROOT_DIR, "annotation.json")

with open(ANNOTATION_FILE, 'r', encoding='utf-8') as train_new:
    val = json.load(train_new)
    images = [i['id'] for i in val['images']]

img_anno = defaultdict(list)
for anno in val['annotations']:
    for img_id in images:
        if anno['image_id'] == img_id:
            img_anno[img_id].append(anno)
imgid_file = {}
for im in val['images']:
    imgid_file[im['id']] = im['file_name']

for img_idx in tqdm(img_anno):
    instance_png = np.zeros((512, 512), dtype=np.uint8)
    for idx, ann in enumerate(img_anno[img_idx]):
        im_mask = np.zeros((512, 512), dtype=np.uint8)
        mask = []
        for an in ann['segmentation']:
            ct = np.expand_dims(np.array(an), 0).astype(int)
            contour = np.stack((ct[:, ::2], ct[:, 1::2])).T
            mask.append(contour)
        imm = cv2.drawContours(im_mask, mask, -1, 1, -1)
        imm = imm * (1000 * anno['category_id'] + idx)
        instance_png = instance_png + imm
        cv2.imwrite(os.path.join(ROOT_DIR, imgid_file[img_idx].split('.')[0]+".png"), instance_png)
        # plt.imshow(instance_png)