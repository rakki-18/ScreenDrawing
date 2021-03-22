import cv2

import mediapipe as mp
import numpy as np
from utils import *

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

hands = mp_hands.Hands(max_num_hands = 2, static_image_mode = False,min_detection_confidence = 0.5)

def init():
    buffer_count = 0
    base_depth = 0
    list_points = []
    list_avg = []
    display = canvas.copy()
    
    return buffer_count, list_points, list_avg, display, base_depth


def is_touch(list_avg, base_depth, threshold):
    if(list_avg[-1][2] - base_depth < threshold['touch']):
        return True
    else:
        return False

    
    return canvas

def draw(canvas, threshold,color, radius, thickness, k ):
    cap = cv2.VideoCapture(0)
    
    buffer_count, list_points, list_avg, display, base_depth = init()
    
    while cap.isOpened():
        success, image = cap.read()

        touching_check = False
        if not success:
            print("video capture unsuccessful")
            break



        image = cv2.cvtColor(cv2.flip(image,1), cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = hands.process(image)



        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if results.multi_hand_landmarks:



            buffer_count = buffer_count+1
            hand_landmarks = results.multi_hand_landmarks[0]

            # add the coordinates to the list
            list_points = add_coordinates(list_points, hand_landmarks)





            if(buffer_count == k):
                avg_x, avg_y, avg_z = find_average(list_points, k)
                list_avg = add_to_average_list(list_avg,avg_x,avg_y,avg_z)
                    
                buffer_count = 0



                if(len(list_avg) == threshold['base']):
                    base_depth = list_avg[-1][2]

                touching_check = is_touch(list_avg, base_depth, threshold)
                

                if(len(list_avg) > threshold['draw'] and touching_check == True):
                    
                    canvas = draw_line(canvas,list_avg,color,thickness)
                    display = canvas.copy()
                elif(len(list_avg) > threshold['draw'] and touching_check == False):

                    cursor = canvas.copy()
                    cursor = draw_circle(cursor, list_avg, radius, color,thickness)
                    display = cursor.copy()




            mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)




        # show the canvas and the webcam image   
        cv2.imshow("Detected Hands", image)
        cv2.imshow("Result", display)



        # exit when the user presses 's's
        if cv2.waitKey(5) & 0xFF == ord('s'):
            break
    hands.close()
    cv2.destroyAllWindows()
    cap.release()

canvas = np.zeros([800,800,1],dtype=np.uint8)

threshold = {'touch': -25,
             'base':  6,
             'draw':  8
            }

draw(canvas, threshold, color = (255,0,0),radius = 2, thickness = 2, k =5)