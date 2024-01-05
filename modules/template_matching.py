import cv2
import numpy as np

from . import utils


def perform_template_matching(im, templ):
    if isinstance(im, str):
        im = utils.load_image(im, grayscale=False)
    if isinstance(templ, str):
        templ = utils.load_image(templ, grayscale=False)
    
    img = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    template = cv2.cvtColor(templ, cv2.COLOR_BGR2GRAY)
    
    for scale in np.linspace(2, 8, 10):
        img_copy = img.copy()
        img_copy = utils.resize_image(img_copy, width=600)

        scaled_template = utils.resize_image(template, width=int(img_copy.shape[1]//scale))
        h, w = scaled_template.shape

        result = cv2.matchTemplate(img_copy, scaled_template, cv2.TM_SQDIFF_NORMED)

        
        in_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
        print(np.min(result))

        top_left = min_loc
        bottom_right = (top_left[0] + w, top_left[1] + h)

        cv2.rectangle(img_copy, top_left, bottom_right, 128, 2)
        
        """
        loc = np.where( result >= np.max(result) * 0.98 )
        for pt in zip(*loc[::-1]):
            cv2.rectangle(img_copy, pt, (pt[0] + w, pt[1] + h), 0, 2)
        """
        
        cv2.imshow('Match', img_copy)
        cv2.waitKey(0)
        cv2.imshow('Match', scaled_template)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
