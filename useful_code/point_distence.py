import numpy as np
from hausdorff import hausdorff_distance
import numba
from math import sqrt
import imutils
import cv2
from matplotlib import pyplot as plt

# two random 2D arrays (second dimension must match)
np.random.seed(0)
# X = np.random.random((3, 2))
# Y = np.random.random((2, 2))
x = [[1, 1],  [4, 4], [4,3], [10, 10]]
y = [[1, 1],[2, 1],[3,2]]

X = np.array(x)
Y = np.array(y)
# print(X)
# print(Y)
# X = cv2.imread("000000000007.png", 0)
# Y = cv2.imread("000000000027.png", 0)
# X = X / 255.0
# Y = Y / 255.0

# Test computation of Hausdorff distance with different base distances
# print(f"Hausdorff distance test: {hausdorff_distance(X, Y)}")
print(f"Hausdorff distance manhattan test: {hausdorff_distance(X, Y, distance='manhattan')}")
print(f"Hausdorff distance euclidean test: {hausdorff_distance(X, Y, distance='euclidean')}")
print(f"Hausdorff distance chebyshev test: {hausdorff_distance(X, Y, distance='chebyshev')}")
print(f"Hausdorff distance cosine test: {hausdorff_distance(X, Y, distance='cosine')}")


# For haversine, use 2D lat, lng coordinates
def rand_lat_lng(N):
    lats = np.random.uniform(-90, 90, N)
    lngs = np.random.uniform(-180, 180, N)
    return np.stack([lats, lngs], axis=-1)


def custom_dist(array_x, array_y):
    # print("=============")
    n = array_x.shape[0]
    ret = 0.
    for i in range(n):
        ret += (array_x[i] - array_y[i]) ** 2
    return sqrt(ret)


def dis(XA, XB):
    nA = XA.shape[0]
    nB = XB.shape[0]
    cmax = 0.
    l = []
    for j in range(nB):
        cmin = 10000
        for i in range(nA):
            d = custom_dist(XA[i, :], XB[j, :])
            if d == 0:
                cmin = 0
                break
            if d < cmin:
                cmin = d
        l.append(cmin)
    print(l)
    return cmax


print("Hausdorff custom dis test:", dis(X, Y))
# X = rand_lat_lng(100)
# Y = rand_lat_lng(250)
# print("Hausdorff custom custom_dist test:", custom_dist(X, Y))
# print(f"Hausdorff custom custom_dist test: {hausdorff_distance(X, Y, distance=custom_dist)}")
print("Hausdorff haversine test: {0}".format(hausdorff_distance(X, Y, distance="haversine")))

def get_center2(label):
    point = []
    image = cv2.imread(label, 0)
    cnts = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    # loop over the contours
    for c in cnts:
        # compute the center of the contour
        M = cv2.moments(c)
        try:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
        except:
            continue
        point.append([cX, cY])
    return point

    # cv2.imshow("Image", image)
    # cv2.waitKey(0)
