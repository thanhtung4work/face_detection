import cv2
import numpy as np 

from .utils import load_image, match_features_bf

orb = cv2.ORB_create()


def generate_orb_features(image):
    keypoints, descriptor = orb.detectAndCompute(image, None)
    return keypoints, descriptor


def perform_orb(img_path_1, img_path_2):
    im_1, im_2 = load_image(img_path_1), load_image(img_path_2)
    print(im_1.shape, im_2.shape)

    orb_features_1 = generate_orb_features(im_1)
    orb_features_2 = generate_orb_features(im_2) 

    matches = match_features_bf(orb_features_1[1], orb_features_2[1])
    
    print(f"ORB Matches: {len(matches)}")

    img3 = cv2.drawMatches(im_1, orb_features_1[0], im_2, orb_features_2[0], matches[:20], None)

    cv2.imshow('Match', img3)
    cv2.waitKey(0)
    cv2.destroyAllWindows()