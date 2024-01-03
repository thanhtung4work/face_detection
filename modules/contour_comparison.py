import cv2

from . import utils


def normalize_filled(img):
    ret, img = cv2.threshold(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY),
                        150, 255, cv2.THRESH_BINARY)

    cnt, _ = cv2.findContours(img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    
    # fill shape
    cv2.fillPoly(img, pts=cnt, color=(255,255,255))
    bounding_rect = cv2.boundingRect(cnt[0])
    img_cropped_bounding_rect = img[bounding_rect[1]:bounding_rect[1] + bounding_rect[3], bounding_rect[0]:bounding_rect[0] + bounding_rect[2]]
    
    # resize all to same size
    img_resized = cv2.resize(img_cropped_bounding_rect, (500, 500))
    return img_resized


def perform_cnt_comparison(query_im_path: str, compared_im_paths: list):
    query_im = utils.load_image(query_im_path, grayscale=False)
    query_im = normalize_filled(query_im)

    ims = [utils.load_image(path, grayscale=False) for path in compared_im_paths]
    ims = [normalize_filled(im) for im in ims]

    for i, im in enumerate(ims):
        diff_value = cv2.matchShapes(query_im, im, 1, 0.0)
        print(f"{compared_im_paths[i]}: {diff_value}")

        cv2.imshow('img', im)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

