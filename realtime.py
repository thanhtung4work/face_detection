import cv2
import numpy as np

from modules import utils

face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_alt2.xml")
eye_detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")

templ_path = "input\\facial_features\\eyes\\eyes_4.jpg"

# reading the input image now
cap = cv2.VideoCapture(0)
while cap.isOpened():
    _, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # _, gray = cv2.threshold(gray, 170, 255, cv2.THRESH_TOZERO_INV) 
    template = utils.load_image(templ_path)
    template = utils.smooth_image(template)
    # _, template = cv2.threshold(template, 170, 255, cv2.THRESH_TOZERO_INV) 

    """
    # FACE DETECTION CASCADE
    faces = face_detector.detectMultiScale(
        frame, scaleFactor=1.05, 
        minNeighbors=5, minSize=(30, 30), 
        flags=cv2.CASCADE_SCALE_IMAGE
    )
    
    for (x,y, w, h) in faces:
        cv2.rectangle(frame, pt1 = (x,y),pt2 = (x+w, y+h), color = (255,0,0),thickness =  3)
        roi_gray = gray[y:y+h,x:x+w]
        roi_color = frame[y:y+h, x:x+w]
        eyes = eye_detector.detectMultiScale(roi_gray)
        if len(eyes) == 0: continue
        for (ex,ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex,ey), (ex+ew, ey+eh), (0,255,0), 5)
        cv2.imshow("window", frame)
    """

    
    # TEMPLATE MATCHING
    for scale in range(4, 10):
        scaled_template = utils.resize_image(template, width=(gray.shape[1]//scale))
        h, w = scaled_template.shape
        # scaled_template = utils.convolve(scaled_template, utils.EDGE_KERNEL)

        result = cv2.matchTemplate(gray, scaled_template, cv2.TM_CCOEFF_NORMED)
        
        #in_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

        # top_left = max_loc
        # bottom_right = (top_left[0] + templ_w, top_left[1] + templ_h)

        # cv2.rectangle(frame, top_left, bottom_right, 128, 2)
        

        (yCoords, xCoords) = np.where( result >= np.max(result)*0.99 )

        # initialize our list of rectangles
        rects = []
        # loop over the starting (x, y)-coordinates again
        for (x, y) in zip(xCoords, yCoords):
            # update our list of rectangles
            rects.append((x, y + 50, x + w, y + h + 50))
            # frame[y+50:y+h+50, x:x+w, 0] = scaled_template

        pick = utils.non_max_suppression_fast(np.array(rects), 0.4)

        # loop over the final bounding boxes
        for (startX, startY, endX, endY) in pick:
            # draw the bounding box on the image
            cv2.rectangle(frame, (startX, startY), (endX, endY),
                (255, 0, 0), 3)
        
        
        cv2.imshow('Match', frame)
    

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()