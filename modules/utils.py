import cv2
import imutils
import numpy as np


EDGE_KERNEL = np.array([
    [-1, -1, -1],
    [-1,  8, -1],
    [-1, -1, -1]
])

MIN_MATCH_COUNT = 10

def load_image(image_path: str, grayscale = True):
    if grayscale: image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    else: image = cv2.imread(image_path)
    return image


def normalize_image(image):
    return cv2.normalize(image, None, 0, 1.0, cv2.NORM_MINMAX, dtype=cv2.CV_32F)


def smooth_image(image):
    return cv2.bilateralFilter(image, 5, 75, 75)


def resize_image(image, width=None, height=None):
    return imutils.resize(image, width=width, height=height)


def convolve(image, kernel):
    return cv2.filter2D(image, -1, kernel)


def match_features_bf(a, b):
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    
    matches = bf.match(a, b)
    matches = sorted(matches, key=lambda x: x.distance)
        
    return matches

def match_features_flann(a, b):
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks = 50)

    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(a, b, k=2)

    return matches
