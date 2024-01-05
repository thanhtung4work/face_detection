import time

import cv2
import numpy as np

from modules import utils, haar_cascades

face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_alt2.xml")
eye_detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")

templ_path = "input\\facial_features\\eyes\\eyes_4.jpg"

# used to record the time when we processed last frame 
prev_frame_time = 0
  
# used to record the time at which we processed current frame 
new_frame_time = 0

# font which we will be using to display FPS 
font = cv2.FONT_HERSHEY_SIMPLEX 

# reading the input image now
cap = cv2.VideoCapture(0)
while cap.isOpened():
    _, frame = cap.read()
    template = utils.load_image(templ_path)
    
    # time when we finish processing for this frame 
    new_frame_time = time.time() 
  
    # Calculating the fps 
  
    # fps will be number of frame processed in given time frame 
    # since their will be most of time error of 0.001 second 
    # we will be subtracting it to get more accurate result 
    fps = 1/(new_frame_time-prev_frame_time) 
    prev_frame_time = new_frame_time 

    # converting the fps into integer 
    fps = int(fps) 
  
    # converting the fps to string so that we can display it on frame 
    # by using putText function 
    fps = str(fps) 
  

    # FACE DETECTION CASCADE
    frame = haar_cascades.perform_cascades(frame, show_window=False)
    
    
    # putting the FPS count on the frame 
    cv2.putText(frame, fps, (7, 70), font, 3, (100, 255, 0), 3, cv2.LINE_AA) 
    cv2.imshow('Match', frame)
    

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()