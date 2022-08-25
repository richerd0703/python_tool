#!/usr/bin/env python

# Copyright (c) 2017-present, Facebook, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
##############################################################################

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
# import h5py
import json
import os
import scipy.misc
import cv2
import instances2dict_with_polygons as cs
# import cityscapesscripts.evaluation.instances2dict_with_polygons as cs
import numpy as np
import pycocotools.mask as mask_util
from tqdm import tqdm
# Type used for storing masks in polygon format
_POLY_TYPE = list
# Type used for storing masks in RLE format
_RLE_TYPE = dict


def is_poly(segm):
    """Determine if segm is a polygon. Valid segm expected (polygon or RLE)."""
    assert isinstance(segm, (_POLY_TYPE, _RLE_TYPE)), \
        'Invalid segm type: {}'.format(type(segm))
    return isinstance(segm, _POLY_TYPE)


def flip_segms(segms, height, width):
    """Left/right flip each mask in a list of masks."""

    def _flip_poly(poly, width):
        flipped_poly = np.array(poly)
        flipped_poly[0::2] = width - np.array(poly[0::2]) - 1
        return flipped_poly.tolist()

    def _flip_rle(rle, height, width):
        if 'counts' in rle and type(rle['counts']) == list:
            # Magic RLE format handling painfully discovered by looking at the
            # COCO API showAnns function.
            rle = mask_util.frPyObjects([rle], height, width)
        mask = mask_util.decode(rle)
        mask = mask[:, ::-1, :]
        rle = mask_util.encode(np.array(mask, order='F', dtype=np.uint8))
        return rle

    flipped_segms = []
    for segm in segms:
        if is_poly(segm):
            # Polygon format
            flipped_segms.append([_flip_poly(poly, width) for poly in segm])
        else:
            # RLE format
            flipped_segms.append(_flip_rle(segm, height, width))
    return flipped_segms


def polys_to_mask(polygons, height, width):
    """Convert from the COCO polygon segmentation format to a binary mask
    encoded as a 2D array of data type numpy.float32. The polygon segmentation
    is understood to be enclosed inside a height x width image. The resulting
    mask is therefore of shape (height, width).
    """
    rle = mask_util.frPyObjects(polygons, height, width)
    mask = np.array(mask_util.decode(rle), dtype=np.float32)
    # Flatten in case polygons was a list
    mask = np.sum(mask, axis=2)
    mask = np.array(mask > 0, dtype=np.float32)
    return mask


def mask_to_bbox(mask):
    """Compute the tight bounding box of a binary mask."""
    xs = np.where(np.sum(mask, axis=0) > 0)[0]
    ys = np.where(np.sum(mask, axis=1) > 0)[0]

    if len(xs) == 0 or len(ys) == 0:
        return None

    x0 = xs[0]
    x1 = xs[-1]
    y0 = ys[0]
    y1 = ys[-1]
    return np.array((x0, y0, x1, y1), dtype=np.float32)


def polys_to_mask_wrt_box(polygons, box, M):
    """Convert from the COCO polygon segmentation format to a binary mask
    encoded as a 2D array of data type numpy.float32. The polygon segmentation
    is understood to be enclosed in the given box and rasterized to an M x M
    mask. The resulting mask is therefore of shape (M, M).
    """
    w = box[2] - box[0]
    h = box[3] - box[1]

    w = np.maximum(w, 1)
    h = np.maximum(h, 1)

    polygons_norm = []
    for poly in polygons:
        p = np.array(poly, dtype=np.float32)
        p[0::2] = (p[0::2] - box[0]) * M / w
        p[1::2] = (p[1::2] - box[1]) * M / h
        polygons_norm.append(p)

    rle = mask_util.frPyObjects(polygons_norm, M, M)
    mask = np.array(mask_util.decode(rle), dtype=np.float32)
    # Flatten in case polygons was a list
    mask = np.sum(mask, axis=2)
    mask = np.array(mask > 0, dtype=np.float32)
    return mask


def polys_to_boxes(polys):
    """Convert a list of polygons into an array of tight bounding boxes."""
    boxes_from_polys = np.zeros((len(polys), 4), dtype=np.float32)
    for i in range(len(polys)):
        poly = polys[i]
        x0 = min(min(p[::2]) for p in poly)
        x1 = max(max(p[::2]) for p in poly)
        y0 = min(min(p[1::2]) for p in poly)
        y1 = max(max(p[1::2]) for p in poly)
        boxes_from_polys[i, :] = [x0, y0, x1, y1]

    return boxes_from_polys


def rle_mask_voting(
        top_masks, all_masks, all_dets, iou_thresh, binarize_thresh, method='AVG'):
    """Returns new masks (in correspondence with `top_masks`) by combining
    multiple overlapping masks coming from the pool of `all_masks`. Two methods
    for combining masks are supported: 'AVG' uses a weighted average of
    overlapping mask pixels; 'UNION' takes the union of all mask pixels.
    """
    if len(top_masks) == 0:
        return

    all_not_crowd = [False] * len(all_masks)
    top_to_all_overlaps = mask_util.iou(top_masks, all_masks, all_not_crowd)
    decoded_all_masks = [
        np.array(mask_util.decode(rle), dtype=np.float32) for rle in all_masks
    ]
    decoded_top_masks = [
        np.array(mask_util.decode(rle), dtype=np.float32) for rle in top_masks
    ]
    all_boxes = all_dets[:, :4].astype(np.int32)
    all_scores = all_dets[:, 4]

    # Fill box support with weights
    mask_shape = decoded_all_masks[0].shape
    mask_weights = np.zeros((len(all_masks), mask_shape[0], mask_shape[1]))
    for k in range(len(all_masks)):
        ref_box = all_boxes[k]
        x_0 = max(ref_box[0], 0)
        x_1 = min(ref_box[2] + 1, mask_shape[1])
        y_0 = max(ref_box[1], 0)
        y_1 = min(ref_box[3] + 1, mask_shape[0])
        mask_weights[k, y_0:y_1, x_0:x_1] = all_scores[k]
    mask_weights = np.maximum(mask_weights, 1e-5)

    top_segms_out = []
    for k in range(len(top_masks)):
        # Corner case of empty mask
        if decoded_top_masks[k].sum() == 0:
            top_segms_out.append(top_masks[k])
            continue

        inds_to_vote = np.where(top_to_all_overlaps[k] >= iou_thresh)[0]
        # Only matches itself
        if len(inds_to_vote) == 1:
            top_segms_out.append(top_masks[k])
            continue

        masks_to_vote = [decoded_all_masks[i] for i in inds_to_vote]
        if method == 'AVG':
            ws = mask_weights[inds_to_vote]
            soft_mask = np.average(masks_to_vote, axis=0, weights=ws)
            mask = np.array(soft_mask > binarize_thresh, dtype=np.uint8)
        elif method == 'UNION':
            # Any pixel that's on joins the mask
            soft_mask = np.sum(masks_to_vote, axis=0)
            mask = np.array(soft_mask > 1e-5, dtype=np.uint8)
        else:
            raise NotImplementedError('Method {} is unknown'.format(method))
        rle = mask_util.encode(np.array(mask[:, :, np.newaxis], order='F'))[0]
        top_segms_out.append(rle)

    return top_segms_out


def rle_mask_nms(masks, dets, thresh, mode='IOU'):
    """Performs greedy non-maximum suppression based on an overlap measurement
    between masks. The type of measurement is determined by `mode` and can be
    either 'IOU' (standard intersection over union) or 'IOMA' (intersection over
    mininum area).
    """
    if len(masks) == 0:
        return []
    if len(masks) == 1:
        return [0]

    if mode == 'IOU':
        # Computes ious[m1, m2] = area(intersect(m1, m2)) / area(union(m1, m2))
        all_not_crowds = [False] * len(masks)
        ious = mask_util.iou(masks, masks, all_not_crowds)
    elif mode == 'IOMA':
        # Computes ious[m1, m2] = area(intersect(m1, m2)) / min(area(m1), area(m2))
        all_crowds = [True] * len(masks)
        # ious[m1, m2] = area(intersect(m1, m2)) / area(m2)
        ious = mask_util.iou(masks, masks, all_crowds)
        # ... = max(area(intersect(m1, m2)) / area(m2),
        #           area(intersect(m2, m1)) / area(m1))
        ious = np.maximum(ious, ious.transpose())
    elif mode == 'CONTAINMENT':
        # Computes ious[m1, m2] = area(intersect(m1, m2)) / area(m2)
        # Which measures how much m2 is contained inside m1
        all_crowds = [True] * len(masks)
        ious = mask_util.iou(masks, masks, all_crowds)
    else:
        raise NotImplementedError('Mode {} is unknown'.format(mode))

    scores = dets[:, 4]
    order = np.argsort(-scores)

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        ovr = ious[i, order[1:]]
        inds_to_keep = np.where(ovr <= thresh)[0]
        order = order[inds_to_keep + 1]

    return keep


def rle_masks_to_boxes(masks):
    """Computes the bounding box of each mask in a list of RLE encoded masks."""
    if len(masks) == 0:
        return []

    decoded_masks = [
        np.array(mask_util.decode(rle), dtype=np.float32) for rle in masks
    ]

    def get_bounds(flat_mask):
        inds = np.where(flat_mask > 0)[0]
        return inds.min(), inds.max()

    boxes = np.zeros((len(decoded_masks), 4))
    keep = [True] * len(decoded_masks)
    for i, mask in enumerate(decoded_masks):
        if mask.sum() == 0:
            keep[i] = False
            continue
        flat_mask = mask.sum(axis=0)
        x0, x1 = get_bounds(flat_mask)
        flat_mask = mask.sum(axis=1)
        y0, y1 = get_bounds(flat_mask)
        boxes[i, :] = (x0, y0, x1, y1)

    return boxes, np.where(keep)[0]


def xyxy_to_xywh(xyxy):
    """Convert [x1 y1 x2 y2] box format to [x1 y1 w h] format."""
    if isinstance(xyxy, (list, tuple)):
        # Single box given as a list of coordinates
        assert len(xyxy) == 4
        x1, y1 = xyxy[0], xyxy[1]
        w = xyxy[2] - x1 + 1
        h = xyxy[3] - y1 + 1
        return (x1, y1, w, h)
    elif isinstance(xyxy, np.ndarray):
        # Multiple boxes given as a 2D ndarray
        return np.hstack((xyxy[:, 0:2], xyxy[:, 2:4] - xyxy[:, 0:2] + 1))
    else:
        raise TypeError('Argument xyxy must be a list, tuple, or numpy array.')

def getLabelID(self, instID):
    if (instID < 1000):
        return instID
    else:
        return int(instID / 1000)


def detect(filepath):
    img = cv2.imread(filepath)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    corners = cv2.goodFeaturesToTrack(gray, 0, 0.04, 17)  # , blockSize=5
    # 返回的结果是[[ 311., 250.]] 两层括号的数组。
    corners = np.int0(corners)
    #
    point = corners.tolist()
    lists = [x[0] for x in point]  # 一行代码搞定！
    l = []
    for x in lists:
        l.append(x)
        l.append([x[0] + 1, x[1]])
        l.append([x[0] - 1, x[1]])
        l.append([x[0], x[1] + 1])
        l.append([x[0], x[1] - 1])

        l.append([x[0] - 1, x[1] - 1])
        l.append([x[0] + 1, x[1] - 1])
        l.append([x[0] - 1, x[1] + 1])
        l.append([x[0] + 1, x[1] + 1])
    return l


def polygon_area(points):
    """返回多边形面积

    """
    area = 0
    q = points[-1]
    for p in points:
        area += p[0] * q[1] - p[1] * q[0]
        q = p
    return area / 2


def on_one_line(point):
    res = []
    point.append(point[0])
    length = len(point)
    res.append(point[0])
    for i in range(2, length):
        x0, y0 = point[i - 2][0], point[i - 2][1]
        x1, y1 = point[i - 1][0], point[i - 1][1]
        x2, y2 = point[i][0], point[i][1]
        if (y1 - y0) * (x2 - x1) - (x1 - x0) * (y2 - y1) == 0:
            continue
        # elif point[i - 1] not in res:
        else:
            res.append(point[i - 1])
    if point[0] != res[-1]:
        res.append(point[0])
    return res

def convert_cityscapes_instance_only(data_dir, out_dir):
    """Convert from cityscapes format to COCO instance seg format - polygons"""

    # ends_in = '%s_polygons.json'
    '''
    {
		"city": "frankfurt",
		"img_path": "G:\\dataset\\cityspace\\leftImg8bit_trainvaltest\\leftImg8bit\\val/frankfurt/frankfurt_000001_051516_leftImg8bit.png",
		"img_width": 2048,
		"label": "road",
		"instance_id": "frankfurt_000001_051516_0",
		"image_id": "frankfurt_000001_051516",
		"img_height": 1024,
		"split": "val",
		"components": [
			{   "bbox": [5,556,2039,424],
				"poly": [[386,970],[557,949]],
				"area": 498202.5
			}
		],
		"bbox": [5,556,2038,423
		]
	}
    
    '''
    city = "building"
    img_path = r"G:\dataset\publicdataset\ISPRS\polyRNN++\image"  # image 路径
    img_width = 512
    img_height = 512
    split = "val"
    label = "building"
    ann_dir = os.path.join(data_dir)

    for root, _, files in os.walk(ann_dir):
        for filename in files:
            image_id = filename[:-4]
            json_name = str(image_id) + ".json"
            img_id = 0
            ann_id = 0
            cat_id = 11
            img_h = 512
            img_w = 512
            category_dict = {}
            max_len1 = 0
            max_len2 = 0
            category_instancesonly = [
                'building',
            ]

            # for data_set, ann_dir in zip(sets, ann_dirs):
            #     print('Starting %s' % data_set)
            ann_dict = []
            # ann = {}
            # ann['city'] = city
            # ann['img_path'] = img_path
            # ann['img_width'] = img_width
            # ann['img_height'] = img_height
            # ann['split'] = split
            # ann['label'] = label
            # ann['image_id'] = image_id
            images = []
            annotations = []
            if len(images) % 50 == 0:
                print("Processed %s images, %s annotations" % (len(images), len(annotations)))
            # json_ann = json.load(open(os.path.join(root, filename)) ,encoding='UTF-8')
            print('solving: {}'.format(os.path.join(root, filename)))
            image = {}
            image['id'] = img_id
            img_id += 1

            # image['width'] = json_ann['imgWidth']
            # image['height'] = json_ann['imgHeight']
            image['width'] = img_h
            image['height'] = img_w
            image['file_name'] = str(filename[:-4]) + ".png"  # image名字
            image['seg_file_name'] = filename  # label名字
            images.append(image)

            points = detect(os.path.join(root, filename))

            fullname = os.path.abspath(os.path.join(root, image['seg_file_name']))
            objects = cs.instances2dict_with_polygons([fullname], verbose=False)
            objects = objects[fullname]

            for object_cls in objects:
                if object_cls not in category_instancesonly:
                    continue  # skip non-instance categories

                for obj in objects[object_cls]:
                    if obj['contours'] == []:
                        print('Warning: empty contours.')
                        continue  # skip non-instance categories

                    len_p = [len(p) for p in obj['contours']]
                    for p in obj['contours']:
                        ann = {}
                        ann['city'] = city
                        repath = os.path.join(img_path, split, city)
                        ann['img_path'] = os.path.join(repath, filename)
                        ann['img_width'] = img_width
                        ann['img_height'] = img_height
                        ann['split'] = split
                        ann['label'] = label
                        ann['image_id'] = image_id
                        com_dic = {}
                        components = []
                        # p 一维数组 两个一组是一个坐标
                        length = len(p)
                        if length > 5:
                            p2points = []
                            step = 2
                            while step < length:
                                temp_point = p[step - 2: step]
                                if temp_point in points:
                                    p2points.append(temp_point)
                                step += 2
                            if len(p2points) > 5:
                                reduce_points = on_one_line(p2points)
                                # print("set len ===", len(list(set([tuple(t) for t in reduce_points]))))
                                # print("reduce_points len ===", len(reduce_points))
                                max_len1 = max(max_len1, len(reduce_points))
                                max_len2 = max(max_len2, length)
                                ann['instance_id'] = str(image_id) + "_" + str(ann_id)
                                ann_id += 1
                                com_dic["poly"] = reduce_points


                                if object_cls not in category_dict:
                                    category_dict[object_cls] = cat_id
                                    cat_id += 1
                                bbox = xyxy_to_xywh(polys_to_boxes([[p]])).tolist()[0]
                                # ann['bbox'] = xyxy_to_xywh(polys_to_boxes([[p]])).tolist()[0]
                                # com_dic["area"] = obj['pixelCount']
                                # a_list = p2points.copy()

                                a_area = polygon_area(reduce_points)
                                # print("c_area:", c_area)
                                # print("d_area:", d_area)
                                com_dic["area"] = int(a_area)
                                com_dic["bbox"] = bbox
                                components.append(com_dic)
                                ann['components'] = components
                                ann["bbox"] = bbox

                                # ann['segmentation'] = []#设为空
                                # annotations.append(ann)
                                ann_dict.append(ann)
            # print("Num categories: %s" % len(categories))
            print("Num images: %s" % len(images))
            print("Num annotations: %s" % len(annotations))
            print("缩减后max_len1 ", max_len1)
            print("缩减前max_len2 ", max_len2)
            os.makedirs(out_dir, exist_ok=True)
            with open(os.path.join(out_dir, json_name), 'w') as outfile:
                json.dump(ann_dict, outfile)


if __name__ == '__main__':
    def parse_args():
        parser = argparse.ArgumentParser(description='Convert dataset')
        parser.add_argument('--dataset', help="cocostuff, cityscapes", default='cityscapes_instance_only', type=str)
        parser.add_argument('--datadir', help="data dir", default=r'G:\dataset\publicdataset\ISPRS\test', type=str)
        parser.add_argument('--outdir', help="output dir", default=r'G:\dataset\publicdataset\ISPRS\polyRNN++\label\val\building', type=str)
        return parser.parse_args()
    args = parse_args()

    if args.dataset == "cityscapes_instance_only":
        convert_cityscapes_instance_only(args.datadir, args.outdir)
    # elif args.dataset == "cocostuff":
    #     convert_coco_stuff_mat(args.datadir, args.outdir)
    else:
        print("Dataset not supported: %s" % args.dataset)
'''
solving: G:\dataset\label2city\data\label\mask\top_potsdam_2_10_0_0.png
Num images: 1
Num annotations: 0
缩减后max_len1  420
缩减前max_len2  1482
Processed 0 images, 0 annotations
solving: G:\dataset\label2city\data\label\mask\top_potsdam_2_10_0_1.png
Num images: 1
Num annotations: 0
缩减后max_len1  302
缩减前max_len2  1642
Processed 0 images, 0 annotations
solving: G:\dataset\label2city\data\label\mask\top_potsdam_2_10_0_2.png
Num images: 1
Num annotations: 0
缩减后max_len1  921
缩减前max_len2  2972
Processed 0 images, 0 annotations
solving: G:\dataset\label2city\data\label\mask\top_potsdam_2_10_0_3.png
Num images: 1
Num annotations: 0
缩减后max_len1  396
缩减前max_len2  984
Processed 0 images, 0 annotations
solving: G:\dataset\label2city\data\label\mask\top_potsdam_2_10_1_0.png
Num images: 1
Num annotations: 0
缩减后max_len1  383
缩减前max_len2  1504
Processed 0 images, 0 annotations
solving: G:\dataset\label2city\data\label\mask\top_potsdam_2_10_1_1.png
Num images: 1
Num annotations: 0
缩减后max_len1  646
缩减前max_len2  2370

Process finished with exit code 0

'''