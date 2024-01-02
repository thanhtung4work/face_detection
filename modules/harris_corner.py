import cv2
import numpy as np

from . import utils

def perform_harris_corner(im_path_1, im_path_2):
    img_1, img_2 = utils.load_image(im_path_1), utils.load_image(im_path_2)
    img_1 = utils.resize_image(img_1, width=400)
    img_copy = img_2.copy()
    img_1 = np.float32(img_1)

    # apply the cv2.cornerHarris method 
    # to detect the corners with appropriate 
    # values as input parameters 
    dest = cv2.cornerHarris(img_2, 4, 9, 0.07) 
    
    # Results are marked through the dilated corners 
    dest = cv2.dilate(dest, None) 
    
    # Reverting back to the original image, 
    # with optimal threshold value 
    img_copy[dest > 0.01 * dest.max()] = [0] 
    
    # the window showing output image with corners 
    cv2.imshow('Image with Borders', img_copy) 
    
    # De-allocate any associated memory usage  
    if cv2.waitKey(0) & 0xff == 27: 
        cv2.destroyAllWindows() 

