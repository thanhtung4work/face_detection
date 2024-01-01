import cv2
import imutils
import numpy as np


EDGE_KERNEL = np.array([
    [-1, -1, -1],
    [-1,  8, -1],
    [-1, -1, -1]
])


def load_image(image_path: str):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    return image


def normalize_image(image):
    return cv2.normalize(image, None, 0, 1.0, cv2.NORM_MINMAX, dtype=cv2.CV_32F)


def smooth_image(image):
    return cv2.bilateralFilter(image, 5, 75, 75)


def resize_image(image, width=None, height=None):
    return imutils.resize(image, width=width, height=height)


def convolve(image, kernel):
    return cv2.filter2D(image, -1, kernel)


def match_features(a, b):
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(a, b, k=2)

    good = []
    for m,n in matches: 
        if m.distance <= 0.9 * n.distance:
            good.append((m,))
        
    return good