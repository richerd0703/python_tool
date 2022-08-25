import os

import cv2
import numpy as np
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt
'''
Create blended heat map with JET colormap 
'''


def create_heatmap(im_map, im_cloud, kernel_size=(1, 1), colormap=cv2.COLORMAP_JET, a1=0.5, a2=0.5):
    '''
    img is numpy array
    kernel_size must be odd ie. (5,5)
    '''

    # create blur image, kernel must be an odd number
    im_cloud_blur = cv2.GaussianBlur(im_cloud, kernel_size, 0)

    # If you need to invert the black/white data image
    # im_blur = np.invert(im_blur)
    # Convert back to BGR for cv2
    # im_cloud_blur = cv2.cvtColor(im_cloud_blur,cv2.COLOR_GRAY2BGR)

    # Apply colormap
    im_cloud_clr = cv2.applyColorMap(im_cloud_blur, colormap)

    # blend images 50/50
    return (a1 * im_map + a2 * im_cloud_clr).astype(np.uint8), im_cloud_clr.astype(np.uint8)


def create_heatmap2(im_cloud, kernel_size=(5,5), colormap=cv2.COLORMAP_JET):
    '''
    img is numpy array
    kernel_size must be odd ie. (5,5)
    '''

    # create blur image, kernel must be an odd number
    im_cloud_blur = cv2.GaussianBlur(im_cloud, kernel_size, 0)

    # If you need to invert the black/white data image
    # im_blur = np.invert(im_blur)
    # Convert back to BGR for cv2
    # im_cloud_blur = cv2.cvtColor(im_cloud_blur,cv2.COLOR_GRAY2BGR)

    # Apply colormap
    im_cloud_clr = cv2.applyColorMap(im_cloud_blur, colormap)
    # im_cloud_clr = cv2.cvtColor(im_cloud_blur,cv2.COLOR_BGR2GRAY)

    # blend images 50/50
    return im_cloud_clr.astype(np.uint8)

if __name__ == "__main__":
    # python create_heatmap.py -m G:\dataset\temp\image\3.png -c G:\dataset\temp\label\0-255\3.png
    ap = argparse.ArgumentParser()
    # ap.add_argument('-m', '--map_image', default=r'G:\dataset\temp\val\images\54.png', help="Path to map image")
    ap.add_argument('-c', '--cloud_image', default=r'G:\exp\crowdAI\my\test\draw\point', help="Path to cloud image (grayscale)")
    ap.add_argument('-s', '--save', default=r"G:\exp\crowdAI\my\test\draw\heatmap_point", help="Save image")


    # ap.add_argument('--cloud-pos',default=(0,0),required=False,help="Position of cloud, relative to map origin")
    args = ap.parse_args()

    for files in tqdm(os.listdir(args.cloud_image)):

        # im_map = cv2.imread(args.map_image)
        im_cloud = cv2.imread(os.path.join(args.cloud_image, files))
        # Normalize cloud image?
        # im_heatmap, heat = create_heatmap(im_map, im_cloud, a1=.5, a2=.5)
        # im_heatmap, heat = create_heatmap2(im_cloud)
        heat = create_heatmap2(im_cloud)

        # cv2.imwrite(os.path.join(args.save, "img1_heatmap.png"), im_heatmap)
        cv2.imwrite(os.path.join(args.save, files), heat)

    # plt.imshow(im_heatmap)
    # plt.show()
