import cv2
import numpy as np 

from .utils import load_image, match_features_bf, match_features_flann, MIN_MATCH_COUNT

sift = cv2.SIFT_create()


def generate_sift_features(image):
    keypoints, descriptor = sift.detectAndCompute(image, None)
    return keypoints, descriptor


def perform_sift(img_path_1, img_path_2):
    im_1, im_2 = load_image(img_path_1, grayscale=False), load_image(img_path_2, grayscale=False)
    print(im_1.shape, im_2.shape)

    sift_features_1 = generate_sift_features(im_1)
    print(f"SIFT keypoint of training image: {len(sift_features_1[0])}")
    sift_features_2 = generate_sift_features(im_2) 
    print(f"SIFT keypoint of query image: {len(sift_features_2[0])}")

    matches = match_features_flann(sift_features_1[1], sift_features_2[1])
    # store all the good matches as per Lowe's ratio test.
    
    good = []
    for m,n in matches:
        if m.distance < 0.7 * n.distance:
            good.append(m)

    img3 = cv2.drawMatches(im_1, sift_features_1[0], im_2, sift_features_2[0], good, None)

    cv2.imshow('Match', img3)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def perform_sift_homo(img_path_1, img_path_2):
    im_1, im_2 = load_image(img_path_1), load_image(img_path_2)
    print(im_1.shape, im_2.shape)

    sift_features_1 = generate_sift_features(im_1)
    sift_features_2 = generate_sift_features(im_2) 

    matches = match_features_flann(sift_features_1[1], sift_features_2[1])

    # store all the good matches as per Lowe's ratio test.
    good = []
    for m,n in matches:
        if m.distance < 1 * n.distance:
            good.append(m)
    
    if len(good) > MIN_MATCH_COUNT:
        src_pts = np.float32([ sift_features_1[0][m.queryIdx].pt for m in good ]).reshape(-1,1,2)
        dst_pts = np.float32([ sift_features_2[0][m.trainIdx].pt for m in good ]).reshape(-1,1,2)
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
        matchesMask = mask.ravel().tolist()
        h,w = im_1.shape
        pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
        dst = cv2.perspectiveTransform(pts,M)
        im_2 = cv2.polylines(im_2,[np.int32(dst)],True,255,3, cv2.LINE_AA)
    else:
        print( "Not enough matches are found - {}/{}".format(len(good), MIN_MATCH_COUNT) )
        matchesMask = None

    draw_params = dict(matchColor = (0,255,0), # draw matches in green color
                   singlePointColor = None,
                   matchesMask = matchesMask, # draw only inliers
                   flags = 2)
    
    img3 = cv2.drawMatches(im_1, sift_features_1[0], im_2, sift_features_2[0], good, None, **draw_params)
    cv2.imshow('Match', img3)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
