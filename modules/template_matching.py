import cv2
import numpy as np

from . import utils


def perform_template_matching(im_path, templ_path):
    img = utils.load_image(im_path)
    template = utils.load_image(templ_path)

    # img, template = utils.smooth_image(img), utils.smooth_image(template)
    # img = utils.convolve(img, utils.EDGE_KERNEL)
    
    for scale in [2, 4, 6, 8]:
        img_copy = img.copy()
        img_copy = utils.resize_image(img_copy, width=600)

        scaled_template = utils.resize_image(template, width=(img_copy.shape[1]//scale))
        h, w = scaled_template.shape
        # scaled_template = utils.convolve(scaled_template, utils.EDGE_KERNEL)

        templ_h, templ_w = scaled_template.shape

        result = cv2.matchTemplate(img_copy, scaled_template, cv2.TM_CCOEFF_NORMED)

        
        in_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

        top_left = max_loc
        bottom_right = (top_left[0] + templ_w, top_left[1] + templ_h)

        cv2.rectangle(img_copy, top_left, bottom_right, 128, 2)
        

        """loc = np.where( result >= 0.5 )
        for pt in zip(*loc[::-1]):
            cv2.rectangle(img_copy, pt, (pt[0] + w, pt[1] + h), 0, 2)
        """
        
        cv2.imshow('Match', img_copy)
        cv2.waitKey(0)
        cv2.imshow('Match', scaled_template)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
