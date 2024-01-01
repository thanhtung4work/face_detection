import cv2
import numpy as np

from . import utils


def perform_template_matching(im_path, templ_path):
    img = utils.load_image(im_path)
    template = utils.load_image(templ_path)

    img, template = utils.smooth_image(img), utils.smooth_image(template)
    # img = utils.convolve(img, utils.EDGE_KERNEL)
    
    print(img.max())
    
    for scale in [2, 3, 4, 5]:
        scaled_template = utils.resize_image(template, width=(img.shape[1]//scale))
        # scaled_template = utils.convolve(scaled_template, utils.EDGE_KERNEL)
        
        templ_h, templ_w = scaled_template.shape

        result = cv2.matchTemplate(img, scaled_template, cv2.TM_CCOEFF_NORMED)

        in_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

        top_left = max_loc
        bottom_right = (top_left[0] + templ_w, top_left[1] + templ_h)

        cv2.rectangle(img, top_left, bottom_right, 255, 5)
        cv2.imshow('Match', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()