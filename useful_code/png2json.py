import fnmatch
import os
import argparse

import skimage.io
import skimage.morphology
import pycocotools.mask
import numpy as np
from tqdm import tqdm
import json
import skimage.measure


# def get_args():
#     argparser = argparse.ArgumentParser(description=__doc__)
#     argparser.add_argument(
#         '--mask_dirpath',
#         required=True,
#         type=str,
#         help='Path to the directory where the mask .png files are.')
#     argparser.add_argument(
#         '--output_filepath',
#         required=True,
#         type=str,
#         help='Filepath of the final .json.')
#     args = argparser.parse_args()
#     return args


def masks_to_json(mask_dirpath, output_filepath):
    filenames = fnmatch.filter(os.listdir(mask_dirpath), "*.png")
    categories = [
        {
            "id": 100,
            "name": "building",
            "supercategory": "building"
        }
    ]
    images = None
    ann = None
    all = {
        "categories": categories,
        "annotations": ann,
        "images": images,

    }
    annotations = []

    for filename in tqdm(filenames, desc="Process masks:"):
        image_id = int(os.path.splitext(filename)[0])
        seg = skimage.io.imread(os.path.join(mask_dirpath, filename))
        labels = skimage.morphology.label(seg)
        properties = skimage.measure.regionprops(labels, cache=True)
        for i, contour_props in enumerate(properties):
            skimage_bbox = contour_props["bbox"]
            coco_bbox = [skimage_bbox[1], skimage_bbox[0],
                         skimage_bbox[3] - skimage_bbox[1], skimage_bbox[2] - skimage_bbox[0]]

            image_mask = labels == (i + 1)  # The mask has to span the whole image
            rle = pycocotools.mask.encode(np.asfortranarray(image_mask))
            # rle["counts"] = rle["counts"].decode("utf-8")
            rle = [skimage_bbox[1], skimage_bbox[0], skimage_bbox[3], skimage_bbox[2]]
            annotation = {
                "id": i,
                "bbox": coco_bbox,
                "image_id": image_id,
                "category_id": 100,
                "segmentation": [rle],
                # "area": 42.0,
                "iscrowd": 0

            }
            annotations.append(annotation)
            all["annotations"].append(annotations)
        images = {
            "id": image_id,
            "file_name": filename,
            "weight": 512,
            "height": 512,
        }
        all["images"].append(images)

    with open(output_filepath, 'w') as outfile:
        json.dump(annotations, outfile)


def write_json(file_path):
    with open(file_path, 'r') as load_f:
        load_dict = json.load(load_f)
        out = str(load_dict["annotations"])
        out.split(",")[2]
    # print(file["annotations"]["segmentation"])
    load_f.close()


def read_npy(filepath):
    # 导入npy文件路径位置
    test = np.load(filepath)

    print(test)


if __name__ == "__main__":
    mask_dirpath = r"G:\dataset\temp\label"
    output_filepath = r"G:\dataset\temp\json\test.json"
    # masks_to_json(mask_dirpath, output_filepath)

    json_file = r"G:\dataset\temp\json\convert.json"
    # write_json(json_file)
    npy_file = r"G:\dataset\temp\val\image\crossfield\51.npy"
    read_npy(npy_file)
