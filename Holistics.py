import cv2
import mediapipe as mp
import numpy as np
from utils import *

mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic
holistic =  mp_holistic.Holistic( min_detection_confidence=0.5, min_tracking_confidence=0.5)
def init():
    count = 0

    list_points = []
    #list_avg = []

    return count, list_points
def distance_to_gesture(distance):
    if(distance < 50):
        gesture = 1
    else:
        gesture = 0
    return gesture

def find_gesture(results):
    if results.right_hand_landmarks:
        thumb  = results.right_hand_landmarks.landmark[4]
        distance = find_distance(results.right_hand_landmarks, thumb)
        gesture = distance_to_gesture(distance)
    else:
        gesture = -1

    return gesture

def execute_gesture(canvas, gesture):
    if(gesture == 1):
        canvas = np.zeros([512,512,1],dtype=np.uint8)
    return canvas

# canvas on which we are drawing
canvas = np.zeros([512,512,1],dtype=np.uint8)


def draw(canvas, color, k, threshold, thickness):
    cap = cv2.VideoCapture(0)
    count, list_points = init()

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

            #count = count+1
            hand_landmarks = results.left_hand_landmarks
            
            list_points = add_coordinates(list_points, hand_landmarks)

            # if(count == k):
            #     avg_x, avg_y, avg_z = find_average(list_points, k)
            #     list_avg = add_to_average_list(list_avg,avg_x,avg_y,avg_z)

            #     count = 0

            if(len(list_points) > threshold):
                
                canvas = draw_line(canvas,list_points,color,thickness)
        
            gesture = find_gesture(results)
            canvas = execute_gesture(canvas, gesture)
            
            
            mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
            mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
            
        
        cv2.imshow('MediaPipe Holistic', image)
        cv2.imshow("Result", canvas)
        if cv2.waitKey(5) & 0xFF == ord('s'):
            break
    holistic.close()
    cv2.destroyAllWindows()
    cap.release()

draw(canvas, color = (255,0,0), k =5, threshold = 1, thickness = 1)