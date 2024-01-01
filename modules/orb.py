import cv2
import numpy as np 

from .utils import load_image, match_features

sift = cv2.ORB_create()


def generate_sift_features(image):
    keypoints, descriptor = sift.detectAndCompute(image, None)
    return keypoints, descriptor


def perform_orb(img_path_1, img_path_2):
    im_1, im_2 = load_image(img_path_1), load_image(img_path_2)
    print(im_1.shape, im_2.shape)

    sift_features_1 = generate_sift_features(im_1)
    sift_features_2 = generate_sift_features(im_2) 

    matches = match_features(sift_features_1[1], sift_features_2[1])
    
    print(f"ORB Matches: {len(matches)}")