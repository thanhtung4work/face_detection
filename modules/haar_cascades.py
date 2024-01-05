import cv2

from . import utils




face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_alt2.xml")
eye_detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")

def perform_cascades(im, hierarchical=True, show_window=True):
    if isinstance(im, str):
        im = utils.load_image(im, grayscale=False)
        im_color = im.copy()
        im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    else:
        im_color = im.copy()
        im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

    face_results = face_detector.detectMultiScale(
        im, scaleFactor=1.2, 
        minNeighbors=8, minSize=(30, 30), 
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    for (x,y,w,h) in face_results:
        roi = im[y:y+h, x:x+w]
        roi_color = im_color[y:y+h, x:x+w]

        eye_results = eye_detector.detectMultiScale(
            roi, scaleFactor=1.1, 
            minNeighbors=8,
        )

        if len(eye_results) == 0 or not hierarchical: continue
        for (ex,ey,ew,eh) in eye_results:
            cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), 127, 2)
        
        cv2.rectangle(im_color, (x,y), (x+w, y+h),(0, 0, 0), 2)

    if show_window:
        cv2.imshow('img', im_color)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        return im_color
