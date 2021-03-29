import cv2
import mediapipe as mp
import numpy as np
from utils import *


mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic
holistic =  mp_holistic.Holistic( min_detection_confidence=0.5, min_tracking_confidence=0.5)

# canvas on which we are drawing
canvas = np.zeros([800,800,1],dtype=np.uint8)

def draw(canvas, threshold):
    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("video capture unsuccessful.")
            break

        image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = holistic.process(image)

        # Draw landmark annotation on the image.
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if results.left_hand_landmarks:
            hand_landmarks = results.left_hand_landmarks

            x,y,z = get_coordinates(hand_landmarks)
            canvas = draw_on_point(x,y,z,canvas, threshold)

            mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
            mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

        cv2.imshow('MediaPipe Holistic', image)
        cv2.imshow("Result", canvas)
        if cv2.waitKey(5) & 0xFF == ord('s'):
            break
    holistic.close()
    cv2.destroyAllWindows()
    cap.release()

draw(canvas,3)    
            


