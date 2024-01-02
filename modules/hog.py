import cv2
import imutils

from . import utils

hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

def detect_people(image, hog_detector, win_stride=(4, 4), scale=1.05):
    """
    Detect people in an image using the provided HOG detector.

    Parameters:
    - image (numpy.ndarray): The input image.
    - hog_detector (cv2.HOGDescriptor): The HOG detector.
    - win_stride (tuple): The window stride for the detection.
    - scale (float): The scale factor for the detection.

    Returns:
    - List[tuple]: A list of tuples representing the detected regions (x, y, w, h).
    """
    regions, _ = hog_detector.detectMultiScale(image, winStride=win_stride, scale=scale)
    return regions


def draw_regions(image, regions, color=(0, 0, 0), thickness=2):
    """
    Draw rectangles around the specified regions on the image.

    Parameters:
    - image (numpy.ndarray): The input image.
    - regions (List[tuple]): A list of tuples representing regions to be drawn (x, y, w, h).
    - color (tuple): The color of the rectangles.
    - thickness (int): The thickness of the rectangles.
    
    Returns:
    - numpy.ndarray: The image with drawn rectangles.
    """
    for (x, y, w, h) in regions:
        cv2.rectangle(image, (x, y), (x + w, y + h), color, thickness)
    return image


def perform_hog(im_path):
    im = utils.load_image(im_path, grayscale=False)
    im = imutils.resize(im, width=400)

    regions = detect_people(im, hog)
    im = draw_regions(im, regions)

    cv2.imshow('Match', im)
    cv2.waitKey(0)
    cv2.destroyAllWindows()