import cv2

import mediapipe as mp
import numpy as np
from utils import *

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

hands = mp_hands.Hands(max_num_hands = 2, static_image_mode = False,min_detection_confidence = 0.5)

def init():
    count = 0
    toggle = 0
    list_points = []
    list_avg = []
    display = canvas.copy()
    
    return count, list_points, list_avg, display, toggle

def draw(canvas,color, radius, thickness, k , threshold):
    cap = cv2.VideoCapture(0)
    count, list_points, list_avg, display,toggle = init()

    while cap.isOpened():


        success, image = cap.read()

        if not success:
            print("video capture unsuccessful")
            break

        image = cv2.cvtColor(cv2.flip(image,1), cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = hands.process(image)

        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        key = cv2.waitKey(5)

        if key == 225:
            toggle = toggle^1
        elif key & 0xFF == ord('s'):
            break

        if results.multi_hand_landmarks:

                count = count+1
                hand_landmarks = results.multi_hand_landmarks[0]

                # add the coordinates to the list
                list_points = add_coordinates(list_points, hand_landmarks)
                
                if(count == k):
                    avg_x, avg_y, avg_z = find_average(list_points, k)
                    list_avg = add_to_average_list(list_avg,avg_x,avg_y,avg_z)
                    
                    count = 0

                if(len(list_avg) > threshold and toggle == 1):
                    
                    canvas = draw_line(canvas,list_avg,color,thickness)
                    display = canvas.copy()
                    

                elif(len(list_avg) > threshold and toggle == 0):
                    cursor = canvas.copy()
                    cursor = draw_circle(cursor, list_avg, radius, color,thickness)
                    display = cursor.copy()

                mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        # show the canvas and the webcam image   
        cv2.imshow("Detected Hands", image)
        cv2.imshow("Result", display)

    hands.close()
    cv2.destroyAllWindows()
    cap.release()

canvas = np.zeros([800,800,1],dtype=np.uint8)

draw(canvas, color = (255,0,0),radius = 2, thickness = 2, k =5, threshold = 4)